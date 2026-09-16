"""The single-token basic families: the bank judge with the any-layer item rule."""

import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from wsbench.banks import exact_targets, item_targets, load_bank
from wsbench.basic.judge import CellKey, JudgeOutcome, judge_cells
from wsbench.basic.prompts import PROMPT_VERSION
from wsbench.cache import Cache
from wsbench.judge_config import JudgeConfig
from wsbench.llm import Spend
from wsbench.mcjudge import Preflighter, base_config, is_subset, item_scope, with_readout_count
from wsbench.readouts import Cell, load_readouts
from wsbench.registry import REPO_ROOT, EvalSpec, JudgeArgs
from wsbench.results import FamilyResult, bootstrap_ci, completeness

GROUP = "basic"
CHANCE_LABEL = "no analytic floor (free text); the prompt-only baseline is the measured floor"


def bank_family(name: str, title: str, *, n_items: int = 100, n_layers: int = 11) -> EvalSpec:
    bank = Path("evals") / name / "items.json"
    return EvalSpec(
        name=name,
        title=title,
        group=GROUP,
        bank=bank,
        judge=JudgeConfig(prompt_version=PROMPT_VERSION),
        metric="pass_rate",
        higher_is_better=True,
        run=lambda args: run_family(args, name=name, bank=REPO_ROOT / bank),
        calls_per_arm=f"≈ {n_items * n_layers / 1000:.1f}k",
        sources="",
    )


def targets_of(item: dict[str, Any]) -> list[str]:
    return exact_targets(item_targets(item))


def group_cells(cells: list[Cell]) -> dict[CellKey, list[Cell]]:
    groups: dict[CellKey, list[Cell]] = defaultdict(list)
    for c in cells:
        groups[(c.id, c.layer)].append(c)
    return dict(groups)


def run_family(args: JudgeArgs, *, name: str, bank: Path) -> FamilyResult:
    _header, items = load_bank(bank)
    scope = item_scope(items, args)
    ids = [it["id"] for it in scope]
    targets = {it["id"]: targets_of(it) for it in scope}
    cells, report = load_readouts(args.readouts, ids=ids, layers=args.layers)
    layers = args.layers if args.layers is not None else report.layers
    kind = report.kind or "prose"
    groups = group_cells(cells)
    expected = [(i, layer) for i in ids for layer in layers]
    missing = [k for k in expected if k not in groups]
    if missing and not args.allow_missing and not args.dry_run:
        print(
            f"{name}: {len(missing)} of {len(expected)} (item, layer) cells have no readout; "
            "pass allow_missing=True to score the rest",
            file=sys.stderr,
        )
        raise SystemExit(2)

    spend = Spend()
    with Cache(args.out / "cells.jsonl") as cache:
        outcome = judge_cells(
            groups,
            targets,
            kind=kind,
            judge=args.judge,
            cache=cache,
            spend=spend,
            preflight=Preflighter(args.dry_run),
            concurrency=args.concurrency,
            rpm=args.rpm,
            dry_run=args.dry_run,
        )

    rows = [_item_row(i, layers, groups, targets[i], outcome) for i in ids]
    decided = [r for r in rows if r["pass"] is not None]
    n_pass = sum(1 for r in decided if r["pass"])
    passes = [1.0 if r["pass"] else 0.0 for r in decided]
    result = FamilyResult(
        family=name,
        metric="pass_rate",
        value=n_pass / len(decided) if decided else None,
        ci95=bootstrap_ci(passes) if decided else None,
        n_items=len(ids),
        higher_is_better=True,
        chance=None,
        chance_label=CHANCE_LABEL,
        complete=bool(expected)
        and completeness(
            pinned=args.judge.pinned,
            subset=is_subset(args),
            n_expected=len(expected),
            n_missing=len(missing),
            n_unjudged=outcome.n_unjudged,
            n_empty=outcome.n_empty,
        ),
        pinned_instrument=args.judge.pinned,
        config=base_config(args, PROMPT_VERSION, kind=kind, layers_judged=layers),
        # a cell is one (item, layer) judge call; empty = every sample at that layer is blank
        counts={
            "n_expected_cells": len(expected),
            "n_missing_cells": len(missing),
            "n_unjudged_cells": outcome.n_unjudged,
            "n_empty_cells": outcome.n_empty,
            "skipped_rows": sum(report.skipped.values()),
            "spend_usd": spend.usd,
        },
        extras={
            "n_calls": outcome.n_calls,
            "n_items_decided": len(decided),
            "n_items_undecided": len(rows) - len(decided),
        },
        rows=rows,
    )
    return with_readout_count(result, scope, cells)


def _item_row(
    item_id: str,
    layers: list[int],
    groups: dict[CellKey, list[Cell]],
    targets: list[str],
    outcome: JudgeOutcome,
) -> dict[str, Any]:
    """``pass`` is True on any expressed layer, False when every expected layer was judged
    negative, and None (undecided) when nothing is expressed and a layer is missing or
    unjudged."""
    per_layer: dict[str, dict[str, Any]] = {}
    expressed_layers: list[int] = []
    n_unjudged = n_missing = 0
    for layer in layers:
        cells = groups.get((item_id, layer))
        if cells is None:
            n_missing += 1
            continue
        verdict = outcome.verdicts.get((item_id, layer))
        if verdict is None:
            n_unjudged += 1
        elif verdict.expressed:
            expressed_layers.append(layer)
        per_layer[str(layer)] = verdict.to_json() if verdict is not None else {"expressed": None}
    undecided = n_unjudged > 0 or n_missing > 0 or not per_layer
    passed: bool | None = True if expressed_layers else (None if undecided else False)
    return {
        "id": item_id,
        "targets": targets,
        "pass": passed,
        "earliest_layer": min(expressed_layers) if expressed_layers else None,
        "n_unjudged": n_unjudged,
        "n_missing": n_missing,
        "layers": per_layer,
    }
