"""GPU stages. Produces raw readouts only — every metric lives in score.py.

MIGRATION (2026-08-15): the gate/read bench paths are ported to the sglang stack — see
scripts/oracle_lens_evals/olens_sglang/launch_suite.sh and
docs/project/experiments/oracle_lens/sglang_suite_unification.md. This launcher stays the
reference implementation until a GPU parity run confirms the port (it was last extended
2026-08-15 for the 3-ckpt eval audit: transform/alpha/prompt_kind/k). The exploratory
stages (grid/sweep/probe/bare) stay Modal-only.

    modal run scripts/olens_suite/order_ops_modal.py                       # read, all families
    modal run scripts/olens_suite/order_ops_modal.py --stage-name gate     # re-gate edited banks
    modal run scripts/olens_suite/order_ops_modal.py --stage-name sweep    # re-find read cells
    modal run scripts/olens_suite/order_ops_modal.py --olens <repo>:<run>  # a different OLens
    modal run scripts/olens_suite/order_ops_modal.py --families sqrt,sign  # subset

Banks come straight from the HF dataset (no manual fetch; a local order_ops_banks/ dir wins if
present). Results land under results/order_ops/<stage>_<family>.json.

Three load-bearing guards, each protecting against a failure that produced a plausible-looking
but wrong result during development (the first two live in global_workspace.olens_suite.runner):

  INERT ADAPTER   The OLens checkpoint was trained with compile_blocks=on, so its keys carry
                  `._orig_mod.`. Loading onto an uncompiled model matches NOTHING and PEFT
                  reports no error — you get the base model wearing a LoRA-shaped hat, which
                  reads exactly like an interesting negative result. We compile-wrap first, then
                  assert the adapter measurably changes logits.
  MERGE DRIFT     After capture the adapter is baked into the weights so generation runs eager
                  (~4x faster). Verified against pre-merge logits; aborts if it drifted.
  MISSING J LAYER The published J artifact spans layers 0..n-1 with target_layer=-1, so the
                  FINAL block has no Jacobian to itself. Recorded as absent, never as a null.
"""

import json
import random
from pathlib import Path
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

from global_workspace.olens_suite.order_ops.spec import (
    CONTINUE,
    DEFAULT_JLENS,
    DEFAULT_OLENS,
    FAMILIES,
    GATE,
    GRID_LAYERS,
    GRID_TOPK,
    JLENS_ABSENT_LAYERS,
    MODEL_ID,
    OLENS_SCALE,
    RENDER,
    REPEAT_SEEDS,
    SAMPLING,
    STOP_MARKERS,
    SWEEP_LAYERS,
    SWEEP_POSITIONS,
    SWEEP_SAMPLING,
)

app = modal.App("order-ops-eval")
image = suite_image()
hf_cache = hf_cache_volume()
BARE_LAYERS = (52, 56, 60)  # bare stage samples every position, so fewer layers


def truncate_at_stop(text: str) -> str:
    t = text.replace("<output>", "")
    for m in STOP_MARKERS:
        i = t.find(m)
        if i >= 0:
            t = t[:i]
    return t.strip()


@app.function(
    image=image,
    gpu="H200",
    volumes={HF_PATH: hf_cache},
    secrets=[hf_secret()],
    timeout=60 * 60 * 6,
    memory=131072,
)
def stage(
    kind: str,
    bank: list[dict[str, Any]],
    olens: str,
    jlens: str,
    topk: int,
    olens_scale: float = 0.0,
    transform: str = "scaled",
    alpha: float = 0.0,
    prompt_kind: str = "explain",
    k: int = 0,
) -> dict[str, Any]:
    import torch

    from global_workspace.lens import (
        cosine_readout,
        jlens_token_norms,
        logit_lens_logits,
        stacked_jacobians,
    )
    from global_workspace.ola.verbalizer import renderer_for
    from global_workspace.olens_suite.order_ops.anchors import find_anchors
    from global_workspace.olens_suite.runner import (
        load_base,
        load_lens,
        make_sampler,
        matched_noise,
        merge_lens,
    )
    from jlens.hooks import ActivationRecorder
    from jlens.lens import JacobianLens

    dev = "cuda"

    if kind == "gate":  # no lens needed
        # k-of-n gate per spec.GATE: k sampled rollouts at the EVAL temperature (that is the
        # distribution the activations are drawn from) plus one greedy rollout recorded for
        # reference. The old gate was one greedy rollout, which cannot distinguish "reliably
        # correct" from "correct that time" and cannot measure a leak rate at all.
        tok, base = load_base(MODEL_ID, dev)
        model = base.eval()
        out = []
        for i in range(0, len(bank), 8):
            ch = bank[i : i + 8]
            ps = [
                str(
                    tok.apply_chat_template(
                        [{"role": "user", "content": RENDER.format(expr=it["expr"])}],
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                )
                for it in ch
            ]
            enc = tok(ps, return_tensors="pt", padding=True).to(dev)
            k = GATE["k"]
            rep = {kk: vv.repeat_interleave(k, dim=0) for kk, vv in enc.items()}
            torch.manual_seed(0)
            with torch.no_grad():
                gs = model.generate(
                    **rep,
                    max_new_tokens=GATE["max_new"],
                    do_sample=True,
                    temperature=GATE["temperature"],
                    top_p=GATE["top_p"],
                    pad_token_id=tok.pad_token_id,
                )
                gg = model.generate(
                    **enc,
                    max_new_tokens=GATE["max_new"],
                    do_sample=False,
                    pad_token_id=tok.pad_token_id,
                )
            off = enc["input_ids"].shape[1]
            for j, it in enumerate(ch):
                rolls = [
                    tok.decode(gs[j * k + m][off:], skip_special_tokens=True) for m in range(k)
                ]
                out.append(
                    {
                        **it,
                        "gate_rollouts": [truncate_at_stop(r) for r in rolls],
                        "gate_greedy": truncate_at_stop(
                            tok.decode(gg[j][off:], skip_special_tokens=True)
                        ),
                    }
                )
            print(f"  gate {i + len(ch)}/{len(bank)}", flush=True)
        return {"stage": "gate", "items": out}

    if kind == "grid":
        # Token lenses ONLY, at EVERY layer x EVERY position — the funnel's second gate: where
        # (if anywhere) each step's value surfaces for a token lens, BEFORE any OLens GPU is
        # spent. No adapter is loaded (nothing here samples the OLens). Stored as token IDS
        # plus one per-item decode table — the whole grid decoded to strings is ~3x the bytes
        # for zero information.
        tok, base = load_base(MODEL_ID, dev)
        model = base.eval()
        inner = model.model
        gblocks = inner.layers if hasattr(inner, "layers") else inner.language_model.layers
        w_u = model.lm_head.weight.detach().to(dev, torch.float32)
        nw = inner.norm.weight.detach().to(dev, torch.float32)
        geps = float(getattr(inner.norm, "variance_epsilon", 1e-6))

        def gnorm(x: Any) -> Any:
            x = x.float()
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + geps) * nw

        grepo, gfile = jlens.split(":", 1)
        jac = stacked_jacobians(
            JacobianLens.from_pretrained(grepo, filename=gfile), device=dev, dtype=torch.float32
        )
        jden = jlens_token_norms(jac, w_u)
        glayers = [e for e in GRID_LAYERS if e < len(gblocks)]
        gitems = []
        for it in bank:
            text = str(
                tok.apply_chat_template(
                    [{"role": "user", "content": RENDER.format(expr=it["expr"])}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            )
            ids = tok(text, return_tensors="pt").input_ids.to(dev)
            with torch.no_grad(), ActivationRecorder(gblocks, at=glayers) as rec:
                model(ids, use_cache=False)
                jt, lt = {}, {}
                for e in glayers:
                    r = rec.activations[e][0].float().unsqueeze(0)
                    lt[str(e)] = (
                        logit_lens_logits(r, w_u, gnorm)[0].topk(GRID_TOPK, -1).indices.tolist()
                    )
                    jt[str(e)] = (
                        cosine_readout(jac[e : e + 1], r, w_u, denom=jden[e : e + 1])[0]
                        .topk(GRID_TOPK, -1)
                        .indices.tolist()
                        if e < jac.shape[0]
                        else []
                    )
            used = sorted({t for g in (jt, lt) for rows in g.values() for row in rows for t in row})
            gitems.append(
                {
                    "name": it["name"],
                    "meta": it,
                    "n_pos": ids.shape[1],
                    "tokens": [tok.decode(ids[0, p]) for p in range(ids.shape[1])],
                    "jlens_top": jt,
                    "logit_top": lt,
                    "vocab": {str(t): tok.decode(t) for t in used},
                }
            )
            print(f"  grid {it['name']}", flush=True)
        return {
            "stage": "grid",
            "config": {
                "model": MODEL_ID,
                "jlens": jlens,
                "layers": glayers,
                "topk": GRID_TOPK,
                "jlens_absent_layers": [x for x in GRID_LAYERS if x >= jac.shape[0]],
            },
            "items": gitems,
        }

    tok, model, probe, on, delta = load_lens(MODEL_ID, olens, dev)
    base = model.get_base_model()
    inner = base.model
    blocks = inner.layers if hasattr(inner, "layers") else inner.language_model.layers
    w_u = base.lm_head.weight.detach().to(dev, torch.float32)
    nw = inner.norm.weight.detach().to(dev, torch.float32)
    eps = float(getattr(inner.norm, "variance_epsilon", 1e-6))

    def norm_fn(x: Any) -> Any:
        x = x.float()
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * nw

    jrepo, jfile = jlens.split(":", 1)
    jac = stacked_jacobians(
        JacobianLens.from_pretrained(jrepo, filename=jfile), device=dev, dtype=torch.float32
    )
    jden = jlens_token_norms(jac, w_u)
    absent = [x for x in SWEEP_LAYERS if x >= jac.shape[0]]
    if sorted(absent) != sorted(x for x in JLENS_ABSENT_LAYERS if x in SWEEP_LAYERS):
        print(
            f"[jlens] WARNING declared absent {JLENS_ABSENT_LAYERS} but artifact "
            f"is missing {absent} of {list(SWEEP_LAYERS)}",
            flush=True,
        )
    print(f"[jlens] J{tuple(jac.shape)}; no Jacobian for {absent} (target_layer=-1)", flush=True)

    # Two different layer sets, deliberately:
    #   cap_layers    what we RECORD and read with the token lenses — always the full sweep set,
    #                 because a token-lens readout is one matmul and storing it at every
    #                 (layer, position) makes every later question a CPU recompute.
    #   olens_layers  where we SAMPLE the OLens — the expensive part (k generations per cell).
    #                 Sweep: everywhere. Read: only the pre-registered cells.
    cap_layers = list(SWEEP_LAYERS)
    olens_layers = (
        BARE_LAYERS
        if kind == "bare"
        else cap_layers
        if kind in ("sweep", "probe")
        else sorted(
            {
                FAMILIES[f]["cell"]["layer"]
                for f in {it["variant"] for it in bank}
                if FAMILIES[f]["cell"]
            }
        )
    )
    layers = sorted(set(cap_layers) | set(olens_layers))
    wv = {e: renderer_for(prompt_kind)(tok, layer=e) for e in olens_layers}
    samp = dict(SWEEP_SAMPLING if kind in ("sweep", "probe", "bare") else SAMPLING)
    if k:
        samp["k"] = k  # k-parity override (continuation arm at 4x bullet-arm k)

    caps = []
    for it in bank:
        # The bare stage is the no-chat-template control: does the intermediate become readable
        # at expression positions once there IS no template boundary for the model to summarize
        # at? Same prompt text, no im_start/im_end wrapper, AO sampled at EVERY position.
        if kind == "bare":
            text = RENDER.format(expr=it["expr"])
        else:
            text = str(
                tok.apply_chat_template(
                    [{"role": "user", "content": RENDER.format(expr=it["expr"])}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            )
        ids = tok(text, return_tensors="pt").input_ids.to(dev)
        n = ids.shape[1]
        if kind == "bare":
            rels = list(range(-n, 0))
        elif kind == "sweep":
            rels = list(SWEEP_POSITIONS)
        elif kind == "probe":
            # Expression-anchored positions, resolved PER ITEM (operand digit counts shift the
            # indices, so fixed rels cannot align here — that is why the sweep never covered
            # the expression region); logic lives in anchors.find_anchors ("single" reproduces
            # the original inline code; "dual" covers mulmul's two sub-expressions). The frozen
            # boundary cell rides along as the in-run reference.
            toks = tok.convert_ids_to_tokens(ids[0].tolist())
            strs = [tok.convert_tokens_to_string([t]) for t in toks]
            anchors = find_anchors(strs, FAMILIES[it["variant"]].get("anchors", "single"))
            cellr = FAMILIES[it["variant"]]["cell"]["pos"]
            rels = sorted({a - n for a in anchors} | {cellr})
        else:
            rels = [FAMILIES[it["variant"]]["cell"]["pos"]]
        keep = [n + r for r in rels]
        with torch.no_grad(), model.disable_adapter(), ActivationRecorder(blocks, at=layers) as rec:
            model(ids, use_cache=False)
            H = {e: rec.activations[e][0, keep].float() for e in layers}  # noqa: N806
            allpos = {e: rec.activations[e][0].float() for e in layers}
            g = model.generate(
                ids, max_new_tokens=28, do_sample=False, pad_token_id=tok.eos_token_id
            )
        # The `continue` reader (spec.CONTINUE): base model sampled from just AFTER each read
        # position — did it simply say the intermediate next? Layer-independent, so it lives on
        # the caps record once per position. OUTSIDE the recorder (its per-forward appends
        # across k x max_new generate steps are pure overhead) but still adapter-disabled.
        cont: dict[str, list[str]] = {}
        if kind in ("read", "probe", "sweep"):
            with torch.no_grad(), model.disable_adapter():
                for r in rels:
                    prefix = ids[:, : n + r + 1]
                    torch.manual_seed(int(CONTINUE["seed"]))
                    cg = model.generate(
                        prefix.repeat(int(CONTINUE["k"]), 1),
                        max_new_tokens=int(CONTINUE["max_new"]),
                        do_sample=True,
                        temperature=float(CONTINUE["temperature"]),
                        top_p=float(CONTINUE["top_p"]),
                        pad_token_id=tok.eos_token_id,
                    )
                    cont[str(r)] = [
                        tok.decode(row[prefix.shape[1] :], skip_special_tokens=True) for row in cg
                    ]
        jt, lt = {}, {}
        if kind == "read":  # token lenses at EVERY position x EVERY sweep layer: one matmul
            for e in cap_layers:
                r = allpos[e].unsqueeze(0)
                lt[str(e)] = [
                    [tok.decode(t) for t in row]
                    for row in logit_lens_logits(r, w_u, norm_fn)[0].topk(topk, -1).indices.tolist()
                ]
                jt[str(e)] = (
                    [
                        [tok.decode(t) for t in row]
                        for row in cosine_readout(jac[e : e + 1], r, w_u, denom=jden[e : e + 1])[0]
                        .topk(topk, -1)
                        .indices.tolist()
                    ]
                    if e < jac.shape[0]
                    else []
                )
        caps.append(
            {
                "name": it["name"],
                "meta": it,
                "n_pos": n,
                "keep_rel": rels,
                "tokens": [tok.decode(ids[0, p]) for p in range(n)],
                "committed": truncate_at_stop(tok.decode(g[0][n:], skip_special_tokens=True)),
                "jlens_top": jt,
                "logit_top": lt,
                "continue": cont,
                "H": {e: H[e].cpu() for e in layers},
            }
        )
        print(f"  captured {it['name']}", flush=True)

    # ---- merge for speed, then verify it preserved the adapter -----------------------------
    model, drift = merge_lens(model, probe, on)
    scale = olens_scale or OLENS_SCALE  # a NEW lens needs ITS OWN injection scale
    sample_rows = make_sampler(model, tok, wv, scale, dev, transform=transform, alpha=alpha)

    def sample(layer: int, vecs: Any, k: int, seed: int = 0) -> list[list[str]]:
        return sample_rows(
            layer, vecs, k, seed, samp["max_new"], samp["temperature"], samp["top_p"]
        )

    rng = random.Random(41)
    index = {c["name"]: i for i, c in enumerate(caps)}
    out = []
    for c in caps:
        fam = c["meta"]["variant"]
        per: dict[str, Any] = {"layers": {}}
        for e in olens_layers:
            got = sample(e, c["H"][e], samp["k"])
            per["layers"][str(e)] = got if kind in ("sweep", "probe", "bare") else got[0]
        if kind == "read":
            L = FAMILIES[fam]["cell"]["layer"]  # noqa: N806
            h = c["H"][L][0:1]
            # Repeat-run stability. sample() is DETERMINISTIC given (activation, seed) —
            # verified: re-sampling at seed 0 reproduces the scored block byte-for-byte,
            # 296/296. So only INDEPENDENT seeds are stored; the scored block (seed 0) is
            # arm A of the comparison.
            per["repeat"] = {str(s): sample(L, h, samp["k"], seed=s)[0] for s in REPEAT_SEEDS}
            per["noise"] = sample(L, matched_noise(h), samp["k"], seed=101)[0]
            pool = [m for m in c["meta"].get("null_set", []) if m in index and m != c["name"]]
            dn = rng.choice(pool) if pool else c["name"]
            per["donor_from"] = dn
            # Seed 102, NOT 0: at seed 0 this reproduces the donor's own scored block
            # byte-for-byte (generation is deterministic), making "donor recovers its own
            # target" a tautology instead of a control. Found by the qualitative audit.
            per["donor"] = sample(L, caps[index[dn]]["H"][L][0:1], samp["k"], seed=102)[0]
        out.append({"name": c["name"], "keep_rel": c["keep_rel"], "olens": per})
        print(f"[{c['name']}] done", flush=True)

    return {
        "stage": kind,
        "config": {
            "model": MODEL_ID,
            "olens": olens,
            "jlens": jlens,
            "scale": scale,
            "transform": transform,
            "alpha": alpha,
            "prompt_kind": prompt_kind,
            "layers": list(layers),
            "sampling": samp,
            "olens_logit_delta": delta,
            "merge_drift": drift,
            "jlens_absent_layers": absent,
            "cells": {f: FAMILIES[f]["cell"] for f in FAMILIES},
        },
        "items": [{k: v for k, v in c.items() if k != "H"} for c in caps],
        "readouts": out,
    }


BANKS_DIR = Path(__file__).parent / "order_ops_banks"


def load_order_ops_bank(fam: str) -> list[dict[str, Any]]:
    """Banks come from the HF dataset (fetch_suite_bank); a local order_ops_banks/ dir wins if
    present.

    The `variant` field MUST equal the bank's own name. It was not: signpair's 24 items carried
    variant="sign", and since the scorer groups by variant they would have been folded into the
    `sign` family — a structural family scored under a comparison family's cell and tolerance,
    with no error anywhere. Checked on load so a mislabelled bank cannot reach a GPU.

    Pre-freeze families (cell is None) may ALSO resolve from the staging dir written by
    gen_order_ops.py — that is what the gate and grid stages run on. The guard is the cell:
    once a family's cell is registered it must have been frozen by finalize_order_ops.py, and a
    stale staging copy silently winning over the frozen bank would un-freeze the eval.
    """
    # explicit scripts-local override > the committed evals copy / HF mirror (fetch_suite_bank)
    roots = [BANKS_DIR] if BANKS_DIR.exists() else [Path(fetch_suite_bank("order_ops"))]
    paths = [roots[0] / f"{fam}.json"]
    if FAMILIES[fam]["cell"] is None:
        # lazy: this module is re-imported inside the Modal container at /root/, where the
        # repo-relative path (and its parents) do not exist; the loader itself is local-only
        paths.append(
            Path(__file__).resolve().parents[2]
            / "evals"
            / "oracle_lens"
            / "staging_order_ops"
            / f"{fam}.json"
        )
    for p in paths:
        if p.exists():
            d = json.loads(p.read_text())
            items = [i.get("meta", i) for i in (d if isinstance(d, list) else d.get("items", d))]
            bad = {i.get("variant") for i in items} - {fam}
            if bad:
                raise ValueError(
                    f"{p.name}: items carry variant {bad}, expected {fam!r}. "
                    f"The scorer groups by variant, so this would mix families."
                )
            return items
    raise FileNotFoundError(f"no bank for {fam} under {roots[0]}")


@app.local_entrypoint()
def main(
    stage_name: str = "read",
    families: str = "",
    olens: str = DEFAULT_OLENS,
    jlens: str = DEFAULT_JLENS,
    topk: int = 10,
    olens_scale: float = 0.0,
    transform: str = "scaled",
    alpha: float = 0.0,
    prompt_kind: str = "explain",
    k: int = 0,
    limit: int = 0,
) -> None:
    """One GPU per family (Modal bills per GPU-second, so k families on k GPUs is ~k x faster at
    the same cost). The read stage refuses families whose cell has not been swept. --limit N
    slices each bank (the bare stage samples every position, so full banks are overkill)."""
    want = [f.strip() for f in families.split(",") if f.strip()] or list(FAMILIES)
    if stage_name == "read":
        ungated = [f for f in want if FAMILIES[f]["cell"] is None]
        if ungated:
            raise SystemExit(
                f"refusing to read {ungated} at a guessed cell — run --stage-name sweep first "
                f"and write the peak into spec.py (see sqrt: 0.992 at rel-7 vs 0.458 at rel-8)"
            )

    banks = [
        (
            stage_name,
            load_order_ops_bank(f)[: limit or None],
            olens,
            jlens,
            topk,
            olens_scale,
            transform,
            alpha,
            prompt_kind,
            k,
        )
        for f in want
    ]
    for f, (_, b, *_) in zip(want, banks, strict=True):
        print(f"  {f:<10} {len(b):>3} items")
    print(
        f"{sum(len(b) for _, b, *_ in banks)} items over {len(want)} GPUs; "
        f"stage={stage_name}; olens={olens}",
        flush=True,
    )

    for fam, payload in zip(want, stage.starmap(banks), strict=False):
        write_result("order_ops", stage_name, payload, family=fam)
