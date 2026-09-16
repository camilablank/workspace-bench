"""The hard multi-token families: conjunctive regex over units, no LLM judge.

Prose readouts are matched directly. Token readouts (a J-lens) first go through the shared
summarizer, since a top-10 token bag cannot hold a multi-token form and scores zero raw.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from wsbench.banks import load_bank
from wsbench.cache import Cache
from wsbench.hard.conjunctive import (
    LayerRows,
    family_columns,
    item_result,
    layer_unit_hits,
    permutation_chance,
)
from wsbench.hard.contract import BankContract, ScoredUnit, contract_for, scored_units, token_lens
from wsbench.judge_config import JudgeConfig
from wsbench.llm import Spend
from wsbench.mcjudge import Preflighter, base_config, is_subset, item_scope, with_readout_count
from wsbench.readouts import Cell, load_readouts
from wsbench.registry import REPO_ROOT, EvalSpec, JudgeArgs
from wsbench.results import FamilyResult, bootstrap_ci, completeness
from wsbench.summarizer import SUMMARIZER_PROMPT_VERSION, aux_judge, render_bag, summarize

GROUP = "basic"
SCORER_VERSION = "conjunctive-regex-2026-09-16"
CHANCE_LABEL = "permutation null: each item's units scored against a donor item's readouts"


def hard_family(name: str, title: str, *, n_items: int = 100) -> EvalSpec:
    return EvalSpec(
        name=name,
        title=title,
        group=GROUP,
        bank=Path("evals") / name / "items.json",
        judge=JudgeConfig(prompt_version=SCORER_VERSION),
        metric="pass_rate",
        higher_is_better=True,
        run=lambda args: run_family(args, name=name),
        calls_per_arm=f"0 (regex); tokens: ≈ {n_items * 11 / 1000:.1f}k summarizer",
        sources="",
    )


def load_units(
    name: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], BankContract, dict[str, list[ScoredUnit]]]:
    """The bank, its contract and every item's scoring units, with the multi-token filter
    applied from the bank's recorded token counts (and the family's ``target_token_lens.json``
    sidecar for ``target_alts``)."""
    root = REPO_ROOT / "evals" / name
    header, items = load_bank(root / "items.json")
    contract = contract_for(header)
    sidecar_path = root / "target_token_lens.json"
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8")) if sidecar_path.exists() else {}
    units = {}
    for it in items:
        ntok = (
            token_lens(it, sidecar.get(it["name"]), need_target=contract.include_target)
            if contract.multi_token
            else None
        )
        units[it["id"]] = scored_units(it, contract, ntok)
    return header, items, contract, units


def layer_rows(cells: list[Cell], texts: dict[str, str] | None) -> dict[str, LayerRows]:
    """item -> layer -> one sample list per readout cell. With ``texts`` (summarized token bags,
    keyed by cell) each cell contributes its one summary."""
    out: dict[str, LayerRows] = defaultdict(lambda: defaultdict(list))
    for c in cells:
        if texts is not None:
            t = texts.get(c.key)
            samples = [t] if t else []
        else:
            samples = list(c.samples or ())
        out[c.id][c.layer].append(samples)
    return {i: dict(v) for i, v in out.items()}


def _fail(msg: str) -> None:
    print(msg, file=sys.stderr)
    raise SystemExit(2)


def run_family(args: JudgeArgs, *, name: str) -> FamilyResult:
    _header, items, _contract, units = load_units(name)
    scope = item_scope(items, args)
    ids = [it["id"] for it in scope]
    # the null is measured over every bank item the file carries, not just the scored scope
    all_cells, rep = load_readouts(
        args.readouts, ids=[it["id"] for it in items], layers=args.layers
    )
    in_scope = set(ids)
    cells = [c for c in all_cells if c.id in in_scope]
    layers = args.layers if args.layers is not None else rep.layers
    positions: dict[tuple[str, int], set[int]] = defaultdict(set)
    for c in all_cells:
        positions[(c.id, c.layer)].add(c.pos)
    multi = sorted(k for k, ps in positions.items() if len(ps) > 1)
    if multi:
        _fail(
            f"{name}: {len(multi)} (item, layer) cells carry more than one read position "
            f"(first: {multi[0]}); this family reads the final prompt token only"
        )
    missing = [(i, layer) for i in ids for layer in layers if (i, layer) not in positions]
    if missing and not args.allow_missing and not args.dry_run:
        _fail(
            f"{name}: {len(missing)} of {len(ids) * len(layers)} (item, layer) cells have no "
            "readout; pass allow_missing=True to score the rest"
        )

    spend = Spend()
    texts: dict[str, str] | None = None
    n_unjudged = 0
    if rep.kind == "tokens":
        bundles = {c.key: render_bag(c.tokens or (), c.scores) for c in all_cells if c.tokens}
        sjudge = aux_judge(args.judge, args.aux_models, "summarizer")
        with Cache(args.out / "cells.jsonl") as cache:
            summ = summarize(
                bundles,
                judge=sjudge,
                cache=cache,
                spend=spend,
                concurrency=args.concurrency,
                rpm=args.rpm,
                dry_run=args.dry_run,
                preflight=Preflighter(args.dry_run).for_judge(sjudge),
            )
        texts = {k: v for k, v in summ.items() if v is not None}
        n_unjudged = sum(1 for c in cells if c.tokens and c.key not in texts)

    all_rows = layer_rows(all_cells, texts)
    results: dict[str, dict[str, Any]] = {}
    for i in ids:
        by_layer = {layer: layer_unit_hits(r, units[i]) for layer, r in all_rows.get(i, {}).items()}
        results[i] = item_result(by_layer, units[i])
    incomplete = {i for i, _layer in missing} if cells else set(ids)
    if texts is not None:  # a cell whose summary failed is unjudged; its item is undecided
        incomplete |= {c.id for c in cells if c.tokens and c.key not in texts}
    decided = {i: r for i, r in results.items() if r["pass"] or i not in incomplete}
    passes = [1.0 if r["pass"] else 0.0 for r in decided.values()]
    null = permutation_chance(
        [(units[it["id"]], all_rows[it["id"]]) for it in items if it["id"] in all_rows]
    )
    n_empty = sum(1 for c in cells if not any(s.strip() for s in (c.samples or c.tokens or ())))
    pinned = args.judge.pinned or rep.kind != "tokens"  # prose runs touch no model
    result = FamilyResult(
        family=name,
        metric="pass_rate",
        value=sum(passes) / len(passes) if passes else None,
        ci95=bootstrap_ci(passes) if passes else None,
        n_items=len(ids),
        higher_is_better=True,
        chance=null["rate"],
        chance_label=CHANCE_LABEL,
        complete=bool(cells)
        and completeness(
            pinned=pinned,
            subset=is_subset(args),
            n_expected=len(ids) * len(layers),
            n_missing=len(missing),
            n_unjudged=n_unjudged,
            n_empty=n_empty,
        ),
        pinned_instrument=pinned,
        config=base_config(
            args,
            SCORER_VERSION,
            kind=rep.kind or "prose",
            layers_judged=layers,
            summarizer=SUMMARIZER_PROMPT_VERSION if texts is not None else None,
            summarizer_model=aux_judge(args.judge, args.aux_models, "summarizer").model
            if texts is not None
            else None,
        ),
        # a cell is one (item, layer) readout at the single read position
        counts={
            "n_expected_cells": len(ids) * len(layers),
            "n_missing_cells": len(missing),
            "n_unjudged_cells": n_unjudged,
            "n_empty_cells": n_empty,
            "skipped_rows": sum(rep.skipped.values()),
            "spend_usd": spend.usd,
        },
        extras={
            "columns": family_columns([results[i] for i in decided]),
            "permutation_null": null,
            "n_items_decided": len(decided),
            "n_items_undecided": len(ids) - len(decided),
        },
        rows=[{"id": i, "units": [u.to_json() for u in units[i]], **results[i]} for i in ids],
    )
    return with_readout_count(result, scope, cells)
