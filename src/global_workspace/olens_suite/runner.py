"""In-container GPU helpers shared by the olens_suite Modal runners.

Everything here runs INSIDE a Modal container (torch + CUDA + the pip deps present), reached via
the image's ``add_local_dir(src)`` mount. Heavy deps are imported lazily inside each function so
this module imports cleanly on CPU-only clients (mypy, tests, the local entrypoints).

Two of the suite's load-bearing guards live here, each protecting against a failure that produced
a plausible-looking but wrong result during development:

  INERT ADAPTER   The OLens checkpoint was trained with compile_blocks=on, so its keys carry
                  `._orig_mod.`. Loading onto an uncompiled model matches NOTHING and PEFT
                  reports no error — you get the base model wearing a LoRA-shaped hat, which
                  reads exactly like an interesting negative result. We compile-wrap first, then
                  assert the adapter measurably changes logits (`load_lens`).
  MERGE DRIFT     After capture the adapter is baked into the weights so generation runs eager
                  (~4x faster). Verified against pre-merge logits; aborts if it drifted
                  (`merge_lens`).
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

PROBE_TEXT = "The capital of France is"


def load_base(model_id: str, device: str = "cuda") -> tuple[Any, Any]:
    """Tokenizer (left-padding, eos as pad fallback) + bf16 base model on `device`."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map=device)
    return tok, base


def load_lens(model_id: str, olens: str, device: str = "cuda") -> tuple[Any, Any, Any, Any, float]:
    """Base model + OLens LoRA (``<repo>:<run>``), guarded against the INERT-ADAPTER failure.

    Returns ``(tok, model, probe_ids, on_logits, delta)``. Keep ``probe_ids`` and ``on_logits``:
    `merge_lens` re-checks the same probe against them after merging.
    """
    import torch
    import torch.nn as nn
    from huggingface_hub import snapshot_download
    from peft import PeftModel
    from safetensors import safe_open

    tok, base = load_base(model_id, device)

    # ---- OLens: compile-wrap iff the checkpoint needs it, then ASSERT it is live -----------
    repo, run_name = olens.split(":", 1)
    lora = Path(snapshot_download(repo, allow_patterns=[f"{run_name}/lora/*"])) / run_name / "lora"
    with safe_open(str(lora / "adapter_model.safetensors"), framework="pt") as f:
        needs_compile = any("_orig_mod." in k for k in f.keys())  # noqa: SIM118
    if needs_compile:
        for m in base.modules():
            ls = getattr(m, "layers", None)
            if isinstance(ls, nn.ModuleList) and hasattr(m, "norm"):
                blocks: Any = ls  # ModuleList __setitem__ wants a Module; compile returns a wrapper
                for i in range(len(ls)):
                    blocks[i] = torch.compile(blocks[i], dynamic=False)
                print(f"[olens] compiled {len(ls)} blocks (_orig_mod checkpoint)", flush=True)
                break
    model = PeftModel.from_pretrained(base, str(lora)).eval()
    probe = tok(PROBE_TEXT, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        on = model(probe).logits[0, -1].float()
        with model.disable_adapter():
            off = model(probe).logits[0, -1].float()
    delta = float((on - off).abs().max().item())
    print(f"[olens] logit_delta={delta:.3f}", flush=True)
    if delta < 1e-3:
        raise RuntimeError("INERT adapter: it changes nothing. Check the _orig_mod key handling.")
    return tok, model, probe, on, delta


def merge_lens(model: Any, probe_ids: Any, ref_logits: Any) -> tuple[Any, float]:
    """merge_and_unload + strip the ``_orig_mod`` compile wrappers, guarded against MERGE DRIFT.

    Merging bakes the adapter into the weights so generation runs eager (~4x faster). Verified
    against the pre-merge logits (`ref_logits` from `load_lens`); aborts if behaviour drifted.
    Returns ``(merged_model, drift)``.
    """
    import torch
    import torch.nn as nn

    ref = ref_logits.clone()
    model = model.merge_and_unload()
    for m in model.modules():
        ls = getattr(m, "layers", None)
        if isinstance(ls, nn.ModuleList) and hasattr(m, "norm"):
            blocks: Any = ls
            for i in range(len(ls)):
                if hasattr(blocks[i], "_orig_mod"):
                    blocks[i] = blocks[i]._orig_mod
            break
    model.eval()
    with torch.no_grad():
        drift = float((model(probe_ids).logits[0, -1].float() - ref).abs().max().item())
    print(f"[merge] drift={drift:.4f}", flush=True)
    if drift > 0.5:
        raise RuntimeError(f"merge changed behaviour (drift {drift}); refusing to proceed")
    return model, drift


def make_sampler(
    model: Any, tok: Any, wv_prompts: dict[int, Any], scale: float, device: str = "cuda"
) -> Callable[..., list[list[str]]]:
    """The shared injection sampler over a MERGED model.

    ``wv_prompts`` maps layer -> rendered verbalizer prompt (``render_wv_explain_prompt``), each
    carrying ``input_ids`` and the injection ``slot``. The returned ``sample`` embeds the prompt,
    writes ``scale * vec`` into the slot row (k rows per input vector), and samples:

        sample(layer, vecs_2d, k, seed, max_new, temperature, top_p, batch=240)
          -> [[k texts] for each row of vecs_2d]

    Single-vector callers pass ``vec.unsqueeze(0)`` and take ``[0]``. Generation is DETERMINISTIC
    given (activation, seed): the seed is set once per call and the batches consume the RNG
    sequentially, so re-sampling at the same seed reproduces the block byte-for-byte.
    """
    import torch

    embed = model.get_input_embeddings()

    def sample(
        layer: int,
        vecs_2d: Any,
        k: int,
        seed: int,
        max_new: int,
        temperature: float,
        top_p: float,
        batch: int = 240,
    ) -> list[list[str]]:
        p = wv_prompts[layer]
        pe0 = embed(torch.tensor([p.input_ids], device=device))[0]
        rows = (scale * vecs_2d.to(device).float()).repeat_interleave(k, dim=0)
        torch.manual_seed(seed)
        outs: list[str] = []
        for s in range(0, rows.shape[0], batch):
            ch = rows[s : s + batch]
            pe = pe0.unsqueeze(0).expand(ch.shape[0], -1, -1).clone()
            pe[:, p.slot, :] = ch.to(pe.dtype)
            at = torch.ones(pe.shape[0], pe.shape[1], dtype=torch.long, device=device)
            with torch.no_grad():
                g = model.generate(
                    inputs_embeds=pe,
                    attention_mask=at,
                    max_new_tokens=max_new,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=tok.eos_token_id,
                )
            outs += tok.batch_decode(g, skip_special_tokens=True)
        return [outs[i * k : (i + 1) * k] for i in range(vecs_2d.shape[0])]

    return sample


def matched_noise(vec: Any) -> Any:
    """Gaussian noise renormalized to `vec`'s norm — the noise control for the readouts."""
    import torch

    g = torch.randn_like(vec)
    return g / g.norm(dim=-1, keepdim=True) * vec.norm(dim=-1, keepdim=True)
