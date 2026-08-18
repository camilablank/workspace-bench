"""Modal launcher for the superposed write-position capture (mypy-excluded, like all launchers).

MIGRATION (2026-08-15): this path is ported to the sglang stack — see
scripts/oracle_lens_evals/olens_sglang/launch_suite.sh and
docs/project/experiments/oracle_lens/sglang_suite_unification.md. This launcher stays the
reference implementation until a GPU parity run confirms the port (it was last extended
2026-08-15 for the 3-ckpt eval audit: transform/alpha/prompt_kind/k_override).

ONE stage: items in, raw AO readouts out. No scoring, no region labelling, no gates — those are
pure-CPU recomputes over this file's output
(`global_workspace.olens_suite.superposed.{regions,score}`).

What it does per item, in the order the regime requires:
  1. render the frozen bare prompt (`spec.render_prompt`; no chat template);
  2. let the MODEL WRITE — greedy-generate the completion, then keep the prompt plus its first 20
     completion tokens. Reads must land on tokens the model wrote, not on prompt positions, or the
     lens can echo the dictated phrase from the read position itself;
  3. capture the residual at those 20 completion cells at every layer in `spec.LAYERS`;
  4. store the cell TOKENS alongside the readouts — the region rule needs them, and the
     `tokens[n + rel]` alignment assert is the only check that a cell is the cell it is labelled as;
  5. inject `OLENS_SCALE * h` into the verbalizer slot and sample k=6 per (cell, layer).

    modal run scripts/olens_suite/superposed_read_modal.py

The bank comes straight from the HF dataset (no manual fetch); the output lands at
results/superposed/read.json.
"""

from typing import Any

import modal
from runner_common import (
    HF_PATH,
    fetch_suite_bank,
    hf_cache_volume,
    hf_secret,
    suite_image,
    write_result,
)

app = modal.App("superposed-read")
image = suite_image()
hf_cache = hf_cache_volume()
GEN_NEW = 24  # completion tokens generated; the last 20 are the read cells


@app.function(
    image=image,
    gpu="H200",
    volumes={HF_PATH: hf_cache},
    secrets=[hf_secret()],
    # bullet-arm reads generate to max_new on every sample — the u64 arm fits 2 h but the
    # distill/RL arms timed out at 7200 s (observed 2026-08-14)
    timeout=8 * 60 * 60,  # k-parity u64 arm samples 4x
    memory=131072,
)
def read(
    items: list[dict[str, Any]],
    olens: str = "",
    olens_scale: float = 0.0,
    transform: str = "scaled",
    alpha: float = 0.0,
    prompt_kind: str = "explain",
    k_override: int = 0,
) -> dict[str, Any]:
    import torch

    from global_workspace.ola.verbalizer import renderer_for
    from global_workspace.olens_suite.runner import load_lens, make_sampler, merge_lens
    from global_workspace.olens_suite.superposed.spec import (
        DEFAULT_OLENS,
        LAYERS,
        MODEL_ID,
        OLENS_SCALE,
        READ_CELLS_REL,
        SAMPLING,
        render_control_prompt,
        render_prompt,
    )
    from jlens.hooks import ActivationRecorder

    dev = "cuda"
    # load_lens compile-wraps BEFORE PeftModel (this checkpoint has compile_blocks=on, so its keys
    # carry `._orig_mod.`; PEFT would otherwise load NOTHING and report no error — a base model in
    # a LoRA-shaped hat, which reads as "the oracle ignored the activation") and asserts the
    # adapter is live.
    olens = olens or DEFAULT_OLENS
    scale = olens_scale or OLENS_SCALE  # a NEW lens needs ITS OWN injection scale
    tok, model, probe, on, _delta = load_lens(MODEL_ID, olens, dev)
    inner = model.get_base_model().model
    blocks = inner.layers if hasattr(inner, "layers") else inner.language_model.layers

    rels = list(READ_CELLS_REL)
    caps = []
    for it in items:
        phrase = it.get("phrase", "")
        prompt = it.get("prompt") or (render_prompt(phrase) if phrase else render_control_prompt())
        pid = tok(prompt, return_tensors="pt").input_ids.to(dev)
        n_prompt = pid.shape[1]
        with torch.no_grad(), model.disable_adapter():
            gen = model.generate(
                pid, max_new_tokens=GEN_NEW, do_sample=False, pad_token_id=tok.eos_token_id
            )
        ids = gen[:, : n_prompt + len(rels)]  # prompt + the completion cells
        n = ids.shape[1]
        assert n + min(rels) >= n_prompt, "a read cell fell inside the prompt"
        with (
            torch.no_grad(),
            model.disable_adapter(),
            ActivationRecorder(blocks, at=list(LAYERS)) as rec,
        ):
            model(ids, use_cache=False)
            resid = {
                e: rec.activations[e][0, [n + r for r in rels]].float().clone() for e in LAYERS
            }
        caps.append(
            {
                "item": it,
                "prompt": prompt,
                "completion": tok.decode(ids[0, n_prompt:]),
                "tokens": [tok.decode(ids[0, n + r]) for r in rels],
                "resid": resid,
            }
        )
        print(f"[cap] {it['name']}", flush=True)

    model, _drift = merge_lens(model, probe, on)
    wv = {e: renderer_for(prompt_kind)(tok, layer=e) for e in LAYERS}
    sample_rows = make_sampler(model, tok, wv, scale, dev, transform=transform, alpha=alpha)
    k = k_override or int(SAMPLING["k"])

    def sample(layer: int, vec: Any, seed: int = 0) -> list[str]:
        return sample_rows(
            layer,
            vec.unsqueeze(0),
            k,
            seed,
            int(SAMPLING["max_new"]),
            float(SAMPLING["temperature"]),
            float(SAMPLING["top_p"]),
        )[0]

    out = []
    for cap in caps:
        it = cap["item"]
        rec = {
            "name": it["name"],
            "stratum": it.get("stratum", ""),
            "phrase": it.get("phrase", ""),
            "concepts": it.get("concepts", []),
            "prompt": cap["prompt"],
            "completion": cap["completion"],
            "rels": rels,
            "tokens": cap["tokens"],
            "ao": {
                str(e): [sample(e, cap["resid"][e][i]) for i in range(len(rels))] for e in LAYERS
            },
        }
        out.append(rec)
        print(f"[ao] {it['name']}", flush=True)
    return {
        "layers": list(LAYERS),
        "rels": rels,
        "sampling": SAMPLING,
        "records": out,
        "config": {
            "olens": olens,
            "scale": scale,
            "model": MODEL_ID,
            "transform": transform,
            "alpha": alpha,
            "prompt_kind": prompt_kind,
        },
    }


@app.local_entrypoint()
def main(
    olens: str = "",
    olens_scale: float = 0.0,
    transform: str = "scaled",
    alpha: float = 0.0,
    prompt_kind: str = "explain",
    k_override: int = 0,
) -> None:
    """--olens <repo>:<run> evaluates a different OLens; --olens-scale is ITS injection scale."""
    items = fetch_suite_bank("superposed")
    write_result(
        "superposed",
        "read",
        read.remote(items, olens, olens_scale, transform, alpha, prompt_kind, k_override),
    )
