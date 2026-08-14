"""Adapter: the order-ops oracle-lens read results -> the normalized bundle.

Inputs are ``results/order_ops/read_<variant>.json`` for the 9 order-ops variants (one file per
family). Each file is the read-stage schema: top-level ``{stage, config, items, readouts}``. Every
``items[i]`` carries ``meta`` (variant, expr, answer, first_value, operands, intermediates, prompt,
...) and every ``readouts[r]`` (keyed by the same ``name``) carries ``keep_rel`` (the kept relative
position(s)) and ``olens.layers`` — a ``{"<layer>": block}`` map where ``block`` is either a flat
list of ``k`` sample strings (one kept position) or a ``[pos][k]`` nested list (a swept cell).

These files have NO per-readout J-lens arm, so the J-lens :class:`LensReadout` / :class:`LensScore`
is always ``None`` (never a fake 0 — see the schema's design rules).

Pass semantics are the PACKAGED order-ops ``value`` metric, reused verbatim so this can never drift
from the frozen scorer: the headline step target is :func:`step_targets(variant, meta)[0]` and a
sample "asserts" it within the family's frozen tolerance via
:func:`global_workspace.olens_suite.order_ops.score.asserts`. A question passes (its olens headline)
iff any sample at the kept cell asserts that never-written intermediate.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from global_workspace.olens_suite.order_ops.score import asserts
from global_workspace.olens_suite.order_ops.spec import FAMILIES, step_targets
from global_workspace.olens_suite.workspace_bench.schema import (
    FamilySummary,
    LayerCell,
    LensReadout,
    LensScore,
    PosEntry,
    Question,
)
from global_workspace.readout_text import strip_scaffolding


def build_order_ops(
    results_dir: Path, *, variants: set[str] | None = None
) -> tuple[list[FamilySummary], dict[str, list[Question]]]:
    """Fold the committed order-ops read results into (summaries, questions_by_family).

    One family per variant, named ``order_ops-<variant>``. ``variants`` (if given) filters by the
    bare variant name (e.g. ``{"sign", "dec16"}``), not the prefixed family name.
    """
    questions_by_family: dict[str, list[Question]] = {}
    summaries: list[FamilySummary] = []
    for read_path in sorted(results_dir.glob("read_*.json")):
        variant = read_path.stem[len("read_") :]
        if variants is not None and variant not in variants:
            continue
        if variant not in FAMILIES:
            continue
        summary, questions = _build_family(variant, read_path)
        family = summary.family
        questions_by_family[family] = questions
        summaries.append(summary)
    return summaries, questions_by_family


def _build_family(variant: str, read_path: Path) -> tuple[FamilySummary, list[Question]]:
    data = json.loads(read_path.read_text())
    metas = {item["name"]: item["meta"] for item in data["items"]}
    family = f"order_ops-{variant}"
    tol: str = FAMILIES[variant]["tol"]
    step_label: str = FAMILIES[variant]["steps"][0][0]

    questions: list[Question] = []
    for readout in data["readouts"]:
        meta = metas[readout["name"]]
        target = step_targets(variant, meta)[0]  # the headline never-written intermediate
        olens = _read_olens(readout, target, tol, step_label)
        intermediates = [str(x) for x in meta.get("intermediates", [])]
        questions.append(
            Question(
                name=readout["name"],
                family=family,
                prompt=meta.get("prompt") or meta["expr"],
                targets=[str(meta["first_value"]), *intermediates],
                olens=olens,
                jlens=None,  # no per-readout J-lens arm in these read files
                meta={
                    "expr": meta["expr"],
                    "answer": meta["answer"],
                    "committed": meta.get("committed", ""),
                },
            )
        )

    return _family_summary(family, questions), questions


def _read_olens(
    readout: dict[str, Any], target: float, tol: str, step_label: str
) -> LensReadout | None:
    """One olens :class:`LensReadout` from a read-stage readout: per-layer cells over the kept
    cell(s), each hit iff a sample asserts the headline target within the family tolerance."""
    kept: list[int] = list(readout.get("keep_rel") or [])
    by_layer: dict[str, LayerCell] = {}
    hitting: list[int] = []
    for layer, block in readout["olens"]["layers"].items():
        entries: list[PosEntry] = []
        layer_hit = False
        for pos, raw_samples in _positions(block, kept):
            hit = any(asserts(s, target, tol) for s in raw_samples)
            samples = [strip_scaffolding(s) for s in raw_samples]
            entries.append(PosEntry(pos=pos, token="", samples=samples, hit=hit))
            layer_hit = layer_hit or hit
        by_layer[str(layer)] = LayerCell(hit=layer_hit, entries=entries)
        if layer_hit:
            hitting.append(int(layer))
    if not by_layer:
        return None
    return LensReadout(
        passed=bool(hitting),
        kind="text",
        by_layer=by_layer,
        earliest_layer=min(hitting) if hitting else None,
        verdict={"metric": "value", "step": step_label, "target": target, "tol": tol},
    )


def _positions(block: list[Any], kept: list[int]) -> list[tuple[int, list[str]]]:
    """Yield ``(pos, samples)`` for a read-stage ``olens.layers`` block, handling both shapes:
    a nested ``[pos][k]`` sweep cell (one entry per kept position) and a flat ``[k]`` list (the
    single kept position). Mirrors ``order_ops.score._cell``."""
    if block and isinstance(block[0], list):  # nested [pos][k]
        return [(pos, list(block[i])) for i, pos in enumerate(kept)]
    pos = kept[0] if kept else 0  # flat [k]: the single kept position
    return [(pos, list(block))]


def _family_summary(family: str, questions: list[Question]) -> FamilySummary:
    return FamilySummary(
        family=family,
        judge_type="deterministic",
        n_items=len(questions),
        metric="value@k (headline intermediate asserted, packaged order_ops scorer)",
        olens=_score([q.olens for q in questions]),
        jlens=None,
        chance=None,
    )


def _score(readouts: list[LensReadout | None]) -> LensScore | None:
    present = [r for r in readouts if r is not None]
    if not present:
        return None
    n_pass = sum(1 for r in present if r.passed)
    return LensScore(
        pass_rate=round(n_pass / len(present), 4), n_pass=n_pass, n_items=len(present)
    )
