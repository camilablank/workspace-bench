"""MIGRATION (2026-08-15): this path is ported to the sglang stack — see
scripts/oracle_lens_evals/olens_sglang/launch_suite.sh and
docs/project/experiments/oracle_lens/sglang_suite_unification.md. This launcher stays the
reference implementation until a GPU parity run confirms the port (it was last extended
2026-08-15 for the 3-ckpt eval audit: transform/alpha/prompt_kind/k). Note: the "mean"
condition below was never implemented — the code reads only the EOF cell.

AO readouts on dual-gated bugs, TWO injection conditions per layer:
  cell  — the activation at a chosen position (bug line, EOF)
  mean  — the mean of ALL positions' activations at that layer (the prompt 'gist')
Bare render (raw code). k=6 samples per (item, layer, condition). Raw text out.

    modal run scripts/olens_suite/buggy_code_read_modal.py

The bank comes straight from the HF dataset (no manual fetch); the output lands at
results/buggy_code/read.json.
"""

import random
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

app = modal.App("bug-read-final")
image = suite_image()
hf_cache = hf_cache_volume()
LAYERS = (56, 60)


@app.function(
    image=image, gpu="H200", volumes={HF_PATH: hf_cache}, secrets=[hf_secret()],
    timeout=7200, memory=131072,
)
def run(
    bank: dict[str, Any],
    olens: str = "",
    olens_scale: float = 0.0,
    transform: str = "scaled",
    alpha: float = 0.0,
    prompt_kind: str = "explain",
    k: int = 10,
) -> dict[str, Any]:
    import torch

    from global_workspace.ola.verbalizer import renderer_for
    from global_workspace.olens_suite.order_ops.spec import DEFAULT_OLENS, MODEL_ID, OLENS_SCALE

    # A NEW lens is two knobs: its checkpoint (repo:run) and ITS OWN injection scale — the
    # frozen OLENS_SCALE belongs to DEFAULT_OLENS (alpha / median||AR(p)|| of that run).
    olens = olens or DEFAULT_OLENS
    scale = olens_scale or OLENS_SCALE
    from global_workspace.olens_suite.runner import (
        load_lens,
        make_sampler,
        matched_noise,
        merge_lens,
    )
    from jlens.hooks import ActivationRecorder

    cells = bank["read_cells"]
    items = [dict(i, src="buggy") for i in bank["buggy"]] + [
        dict(i, src="clean") for i in bank["clean"]
    ]
    dev = "cuda"
    tok, model, probe, on, _delta = load_lens(MODEL_ID, olens, dev)
    inner = model.get_base_model().model
    blocks = inner.layers if hasattr(inner, "layers") else inner.language_model.layers

    caps = []
    for it in items:
        ids = tok(it["code"], return_tensors="pt").input_ids.to(dev)
        n = ids.shape[1]
        L = cells[it["lang_group"]]["layer"]  # noqa: N806
        with (
            torch.no_grad(),
            model.disable_adapter(),
            ActivationRecorder(blocks, at=list(LAYERS)) as rec,
        ):
            model(ids, use_cache=False)
            h = rec.activations[L][0, n - 1].float()  # frozen cell: EOF x lang layer
        caps.append((it, L, h))

    model, _drift = merge_lens(model, probe, on)
    wv = {e: renderer_for(prompt_kind)(tok, layer=e) for e in LAYERS}
    sample_rows = make_sampler(model, tok, wv, scale, dev, transform=transform, alpha=alpha)

    def sample(layer: int, vec: Any, k: int = 10, seed: int = 0) -> list[str]:
        return sample_rows(layer, vec.unsqueeze(0), k, seed, 44, 0.8, 0.95)[0]

    rng = random.Random(41)
    index = {it["name"]: h for it, _, h in caps}
    names = [it["name"] for it, _, _ in caps]
    out = []
    for it, L, h in caps:  # noqa: N806
        rec = {
            "name": it["name"],
            "src": it["src"],
            "lang_group": it["lang_group"],
            "layer": L,
            "cause": it.get("cause", ""),
        }
        rec["readout"] = sample(L, h, k, seed=0)
        rec["repeat1"] = sample(L, h, k, seed=1)
        rec["noise"] = sample(L, matched_noise(h), k, seed=101)
        pool = [n2 for n2 in names if n2 != it["name"]]
        dn = rng.choice(pool)
        rec["donor_from"] = dn
        rec["donor"] = sample(L, index[dn], k, seed=102)
        out.append(rec)
        print(f"[read] {it['name']}", flush=True)
    return {
        "cells": cells,
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
    k: int = 10,
) -> None:
    """--olens <repo>:<run> evaluates a different OLens; --olens-scale is ITS injection scale
    (required with --olens unless the new lens really shares the default's 33.152)."""
    bank = fetch_suite_bank("buggy_code")
    write_result(
        "buggy_code", "read", run.remote(bank, olens, olens_scale, transform, alpha, prompt_kind, k)
    )
