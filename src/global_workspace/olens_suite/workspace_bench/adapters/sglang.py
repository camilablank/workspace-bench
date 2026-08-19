"""Adapter: the olens_sglang bank pipeline -> the normalized bundle.

Inputs are an acts dir (its ``manifest.json`` carries every prompt's tokens/targets/family and
targeted eval positions) plus one or two generation dirs of the exact worker.py schema — the
oracle-lens run and, optionally, the J-lens baseline run (``jlens_eval.py``, identical
``L{layer:03d}.jsonl`` layout with the top-k tokens as ``samples``).

Hit semantics are the production ones: ``content_targets(exact_targets(...))`` then
:func:`hit_any` per grid point — imported from :mod:`global_workspace.olens_suite.bank.matching`
so this can never drift from ``score_targets.py``. A position "hits" if any of its samples
match; a layer hits if any of its positions hit; a question passes if any layer hits.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from global_workspace.judges.slop_judge import gated_hit
from global_workspace.olens_suite.bank.matching import (
    content_targets,
    exact_targets,
    hit_any,
)
from global_workspace.olens_suite.workspace_bench.schema import (
    FamilySummary,
    LayerCell,
    LensReadout,
    LensScore,
    PosEntry,
    Question,
)
from global_workspace.readout_text import strip_scaffolding

# Families whose headline pass is an LLM judge (score_targets.py's substring pass is only a
# mechanical proxy for these — the launcher's judge step overlays the real verdict later).
_FREETEXT_FAMILIES = {
    "sandbagging",
    "user-modeling",
    "directed-modulation",
    "directed-modulation-mt",
}


def build_from_sglang(
    acts_dir: Path,
    olens_gen: Path,
    jlens_gen: Path | None = None,
    *,
    families: set[str] | None = None,
) -> tuple[list[FamilySummary], dict[str, list[Question]]]:
    """Fold one sglang eval (olens + optional jlens) into (summaries, questions_by_family).

    The slop-gate precision condition (Camila 2026-08-19) rides in automatically when the
    free-text arm has a ``<gen>/judge/slop.json`` artifact (``judge_slop.py``): each hitting
    sample's slop score is attached to its cell and the gated pass (hit AND slop < the
    artifact's threshold) reports beside the recall headline. Token-bag arms are never
    slop-judged (rank is their precision instrument), so the J-lens gated column stays None.
    """
    manifest = _load_manifest(acts_dir)
    prompts = [
        p
        for p in manifest["prompts"]
        if p.get("targets") and (families is None or p["family"] in families)
    ]

    olens_layers = _present_layers(olens_gen, manifest["layers"])
    jlens_layers = _present_layers(jlens_gen, manifest["layers"]) if jlens_gen else []
    slop_index, slop_threshold = _load_slop(olens_gen)

    questions_by_family: dict[str, list[Question]] = {}
    for entry in prompts:
        tgts = content_targets(exact_targets(list(entry["targets"])))
        olens_ro = _read_lens(
            olens_gen,
            entry,
            olens_layers,
            tgts,
            kind="text",
            slop_index=slop_index,
            slop_threshold=slop_threshold,
        )
        jlens_ro = (
            _read_lens(jlens_gen, entry, jlens_layers, tgts, kind="tokens") if jlens_gen else None
        )
        q = Question(
            name=entry["label"],
            family=entry["family"],
            prompt="".join(entry.get("tokens", [])),
            targets=list(entry["targets"]),
            olens=olens_ro,
            jlens=jlens_ro,
            meta={"look_for": entry.get("look_for", ""), "n_pos": entry.get("n_pos")},
        )
        questions_by_family.setdefault(entry["family"], []).append(q)

    summaries = [
        _family_summary(fam, qs, slop_threshold=slop_threshold)
        for fam, qs in sorted(questions_by_family.items())
    ]
    return summaries, questions_by_family


def _read_lens(
    gen_dir: Path | None,
    entry: dict[str, Any],
    layers: list[int],
    tgts: list[str],
    *,
    kind: str,
    slop_index: dict[tuple[str, int, int, int], float] | None = None,
    slop_threshold: float | None = None,
) -> LensReadout | None:
    if gen_dir is None:
        return None
    allowed = entry.get("eval_positions")
    allowed_set = set(allowed) if allowed is not None else None
    by_layer: dict[str, LayerCell] = {}
    hitting: list[int] = []
    gated_hitting = False
    any_rows = False
    for layer in layers:
        rows = _unit_rows(gen_dir, entry["label"], layer, allowed_set)
        if not rows:
            continue
        any_rows = True
        entries: list[PosEntry] = []
        layer_hit = False
        for row in rows:
            samples = [strip_scaffolding(s) for s in row["samples"]]
            hit = hit_any(samples, tgts)
            slop: float | None = None
            if hit and slop_index is not None:
                # min over this position's judged samples: survival is judged by the cleanest
                # delivery of the answer (pass@k semantics, gated per sample)
                scores = [
                    slop_index[key]
                    for k in range(len(samples))
                    if (key := (entry["label"], layer, int(row["pos"]), k)) in slop_index
                ]
                slop = min(scores) if scores else None
                if slop_threshold is not None:
                    gated_hitting = gated_hitting or gated_hit(True, slop, slop_threshold)
            entries.append(PosEntry(row["pos"], row.get("token", ""), samples, hit, slop=slop))
            layer_hit = layer_hit or hit
        by_layer[str(layer)] = LayerCell(layer_hit, entries)
        if layer_hit:
            hitting.append(layer)
    if not any_rows:
        return None
    return LensReadout(
        passed=bool(hitting),
        kind=kind,  # type: ignore[arg-type]
        by_layer=by_layer,
        earliest_layer=min(hitting) if hitting else None,
        # ungated arms (no slop artifact) report None, never a silent copy of `passed`
        passed_gated=gated_hitting if slop_index is not None else None,
    )


def _family_summary(
    family: str, questions: list[Question], *, slop_threshold: float | None = None
) -> FamilySummary:
    judged = "freetext" if family in _FREETEXT_FAMILIES else "deterministic"
    metric = (
        "pass@k substring (proxy — LLM-judged family)"
        if family in _FREETEXT_FAMILIES
        else "pass@k substring (any layer/pos)"
    )
    return FamilySummary(
        family=family,
        judge_type=judged,  # type: ignore[arg-type]
        n_items=len(questions),
        metric=metric,
        olens=_score([q.olens for q in questions]),
        jlens=_score([q.jlens for q in questions]),
        slop_threshold=slop_threshold,
    )


def _score(readouts: list[LensReadout | None]) -> LensScore | None:
    present = [r for r in readouts if r is not None]
    if not present:
        return None
    n_pass = sum(1 for r in present if r.passed)
    gated = [r.passed_gated for r in present]
    n_gated = sum(1 for g in gated if g) if any(g is not None for g in gated) else None
    return LensScore(
        pass_rate=round(n_pass / len(present), 4),
        n_pass=n_pass,
        n_items=len(present),
        pass_rate_gated=round(n_gated / len(present), 4) if n_gated is not None else None,
        n_pass_gated=n_gated,
    )


# --- tiny readers for the stable acts-manifest / L*.jsonl formats (documented in
# --- scripts/oracle_lens_evals/olens_sglang/{common.py,README.md}); re-implemented here so the
# --- adapter imports no script-dir module and stays usable in a plain CPU/viz-build env. ------


def _load_manifest(acts_dir: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((acts_dir / "manifest.json").read_text())
    return data


def _load_slop(
    gen_dir: Path,
) -> tuple[dict[tuple[str, int, int, int], float] | None, float | None]:
    """``(name, layer, pos, sample_idx) -> slop`` from ``<gen>/judge/slop.json``, + threshold.

    ``(None, None)`` when the slop pass was never run — the adapter then reports no gated
    numbers at all instead of quietly equating gated with ungated.
    """
    path = gen_dir / "judge" / "slop.json"
    if not path.exists():
        return None, None
    payload = json.loads(path.read_text())
    index: dict[tuple[str, int, int, int], float] = {}
    for row in payload.get("rows", []):
        if row.get("slop") is None:
            continue  # judge-unavailable rows stay ungated
        key = (str(row["name"]), int(row["layer"]), int(row["pos"]), int(row["sample_idx"]))
        index[key] = float(row["slop"])
    threshold = float(payload.get("config", {}).get("threshold", 9.0))
    return index, threshold


def _present_layers(gen_dir: Path | None, layers: list[int]) -> list[int]:
    if gen_dir is None:
        return []
    found = {int(f.stem[1:]) for f in gen_dir.glob("*/L*.jsonl")}
    return [layer for layer in layers if layer in found]


def _unit_rows(
    gen_dir: Path, label: str, layer: int, allowed: set[int] | None
) -> list[dict[str, Any]]:
    f = gen_dir / label / f"L{layer:03d}.jsonl"
    if not f.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in f.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if allowed is not None and row["pos"] not in allowed:
            continue
        rows.append(row)
    return rows
