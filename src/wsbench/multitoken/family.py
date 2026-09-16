"""Multi-token basic families: one forced-choice judge call per (item, layer, unit); a layer
passes when every unit is picked correctly with a verbatim quote, an item at any layer."""

import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from wsbench.banks import load_bank
from wsbench.cache import Cache
from wsbench.judge_config import JudgeConfig
from wsbench.llm import Spend
from wsbench.mc import fold, letter_index
from wsbench.mcjudge import (
    Call,
    Preflighter,
    base_config,
    is_subset,
    item_scope,
    run_calls,
    with_readout_count,
)
from wsbench.multitoken.options import judged_roles, option_sets
from wsbench.multitoken.prompts import LETTERS, PROMPT_VERSION, SCHEMA, SYSTEM, render_user
from wsbench.readouts import Cell, load_readouts
from wsbench.registry import REPO_ROOT, EvalSpec, JudgeArgs
from wsbench.results import FamilyResult, bootstrap_ci, completeness
from wsbench.summarizer import SUMMARIZER_PROMPT_VERSION, aux_judge, render_bag, summarize

GROUP = "basic_mt"
CHANCE_LABEL = (
    "no analytic floor for an any-layer conjunction (per call 1/6 per unit); the measured floors "
    "are the lucky-guessing and prompt-only baselines"
)


def mt_family(name: str, title: str, *, calls_per_arm: str) -> EvalSpec:
    return EvalSpec(
        name=name,
        title=title,
        group=GROUP,
        bank=Path("evals") / name / "items.json",
        judge=JudgeConfig(prompt_version=PROMPT_VERSION),
        metric="pass_rate",
        higher_is_better=True,
        run=lambda args: run_family(args, name=name),
        calls_per_arm=calls_per_arm,
        sources="",
    )


def verdict(
    res: dict[str, Any] | None, gold_idx: int, options: list[str], readout: str
) -> dict[str, Any]:
    """One judge answer: ``correct`` / ``distractor`` / ``cannot`` / ``invalid``; a correct pick
    counts only with a verbatim (folded) quote from the readout. ``invalid`` (no lone letter, or
    a letter past the option list) is judged and negative: the instrument answered, wrongly."""
    if res is None:
        return {"judged": False, "kind": "unavailable", "correct": False}
    idx = letter_index(res.get("choice"), options)
    if idx is None:
        return {
            "judged": True,
            "pick": str(res.get("choice", "")).strip() or None,
            "kind": "invalid",
            "correct": False,
            "quote_ok": False,
        }
    letter = LETTERS[idx]
    if idx == len(options) - 1:
        return {
            "judged": True,
            "pick": letter,
            "kind": "cannot",
            "correct": False,
            "quote_ok": False,
        }
    quote = str(res.get("quote", ""))
    ok = bool(fold(quote.strip())) and fold(quote.strip()) in fold(readout)
    kind = "correct" if idx == gold_idx else "distractor"
    return {
        "judged": True,
        "pick": letter,
        "kind": kind,
        "correct": idx == gold_idx and ok,
        "quote": quote,
        "quote_ok": ok,
    }


def _fail(msg: str) -> None:
    print(msg, file=sys.stderr)
    raise SystemExit(2)


def run_family(args: JudgeArgs, *, name: str) -> FamilyResult:
    _header, items = load_bank(REPO_ROOT / "evals" / name / "items.json")
    scope = item_scope(items, args)
    ids = [it["id"] for it in scope]
    by_id = {it["id"]: it for it in scope}
    options = option_sets(items)  # over the whole bank, never the scored subset
    cells, rep = load_readouts(args.readouts, ids=ids, layers=args.layers)
    layers = args.layers if args.layers is not None else rep.layers
    positions: dict[tuple[str, int], list[Cell]] = defaultdict(list)
    for c in cells:
        positions[(c.id, c.layer)].append(c)
    multi = sorted(k for k, cs in positions.items() if len({c.pos for c in cs}) > 1)
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
    pre = Preflighter(args.dry_run)
    texts: dict[str, str] = {}
    empty: set[tuple[str, int]] = set()  # (item, layer) cells with nothing to judge: negatives
    with Cache(args.out / "cells.jsonl") as cache:
        if rep.kind == "tokens":
            bundles = {c.key: render_bag(c.tokens or (), c.scores) for c in cells if c.tokens}
            sjudge = aux_judge(args.judge, args.aux_models, "summarizer")
            summ = summarize(
                bundles,
                judge=sjudge,
                cache=cache,
                spend=spend,
                concurrency=args.concurrency,
                rpm=args.rpm,
                dry_run=args.dry_run,
                preflight=pre.for_judge(sjudge),
            )
            texts = {k: v for k, v in summ.items() if v is not None}
            empty = {(c.id, c.layer) for c in cells if not c.tokens}
        else:
            for c in cells:
                joined = "\n".join(s for s in (c.samples or ()) if s.strip())
                if joined:
                    texts[c.key] = joined
                else:
                    empty.add((c.id, c.layer))
        calls: list[Call] = []
        meta: dict[
            str, tuple[str, int, str, int, list[str], str]
        ] = {}  # key -> item, layer, role, gold, options, readout
        for (item_id, layer), cs in sorted(positions.items()):
            cell = cs[0]
            readout = texts.get(cell.key)
            if readout is None:
                continue
            for role, (opts, gold_idx) in options[item_id].items():
                key = f"{item_id}|L{layer:03d}|{role}"
                meta[key] = (item_id, layer, role, gold_idx, opts, readout)
                calls.append(
                    Call(
                        key=key,
                        system=SYSTEM,
                        user=render_user(readout, role, opts),
                        meta={"item": item_id, "layer": layer, "role": role},
                    )
                )
        results = run_calls(
            calls,
            schema=SCHEMA,
            judge=args.judge,
            prompt_version=PROMPT_VERSION,
            cache=cache,
            spend=spend,
            concurrency=args.concurrency,
            rpm=args.rpm,
            dry_run=args.dry_run,
            preflight=pre.for_judge(args.judge),
            temperature=0.0,
        )

    verdicts: list[dict[str, Any]] = []
    for key, (item_id, layer, role, gold_idx, opts, readout) in meta.items():
        v = verdict(results.get(key), gold_idx, opts, readout)
        verdicts.append(
            {
                "item": item_id,
                "layer": layer,
                "role": role,
                "gold": options[item_id][role][0][gold_idx],
                **v,
            }
        )
    unsummarized = {
        (c.id, c.layer) for c in cells if rep.kind == "tokens" and c.tokens and c.key not in texts
    }
    rows = [
        _item_row(i, judged_roles(by_id[i]), verdicts, missing, unsummarized, empty) for i in ids
    ]
    decided = [r for r in rows if r["pass"] is not None]
    passes = [1.0 if r["pass"] else 0.0 for r in decided]
    n_unjudged_units = sum(1 for v in verdicts if not v["judged"])
    unjudged_cells = {(v["item"], v["layer"]) for v in verdicts if not v["judged"]} | unsummarized
    kinds: dict[str, int] = defaultdict(int)
    for v in verdicts:
        kinds[v["kind"]] += 1
    roles = sorted({r for i in ids for r in judged_roles(by_id[i])})
    role_any = {
        r: _rate([row["unit_correct"][r] for row in decided if r in row["unit_correct"]])
        for r in roles
    }
    result = FamilyResult(
        family=name,
        metric="pass_rate",
        value=sum(passes) / len(passes) if passes else None,
        ci95=bootstrap_ci(passes) if passes else None,
        n_items=len(ids),
        higher_is_better=True,
        chance=None,
        chance_label=CHANCE_LABEL,
        complete=bool(cells)
        and completeness(
            pinned=args.judge.pinned,
            subset=is_subset(args),
            n_expected=len(ids) * len(layers),
            n_missing=len(missing),
            n_unjudged=len(unjudged_cells),
            n_empty=len(empty),
        ),
        pinned_instrument=args.judge.pinned,
        config=base_config(
            args,
            PROMPT_VERSION,
            kind=rep.kind or "prose",
            layers_judged=layers,
            summarizer=SUMMARIZER_PROMPT_VERSION if rep.kind == "tokens" else None,
        ),
        # a cell is one (item, layer) readout at the single read position; a call is one unit
        counts={
            "n_expected_cells": len(ids) * len(layers),
            "n_missing_cells": len(missing),
            "n_unjudged_cells": len(unjudged_cells),
            "n_empty_cells": len(empty),
            "skipped_rows": sum(rep.skipped.values()),
            "spend_usd": spend.usd,
        },
        extras={
            "n_calls": len(calls),
            "n_unjudged_units": n_unjudged_units,
            "n_items_decided": len(decided),
            "n_items_undecided": len(ids) - len(decided),
            "unit_any_layer": role_any,
            "kinds": dict(kinds),
            "abstain_rate": _rate([v["kind"] == "cannot" for v in verdicts if v["judged"]]),
        },
        rows=rows,
    )
    return with_readout_count(result, scope, cells)


def _rate(flags: list[bool]) -> float | None:
    return sum(flags) / len(flags) if flags else None


def _item_row(
    item_id: str,
    roles: list[str],
    verdicts: list[dict[str, Any]],
    missing: list[tuple[str, int]],
    unsummarized: set[tuple[str, int]],
    empty: set[tuple[str, int]],
) -> dict[str, Any]:
    """``pass`` = some layer where every judged unit is correct; None (undecided) when no layer
    passes and a unit verdict is unjudged, a summary failed or a layer is missing. An empty cell
    is a judged negative at that layer (nothing was read), never a reason to be undecided."""
    mine = [v for v in verdicts if v["item"] == item_id]
    by_layer: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for v in mine:
        by_layer[v["layer"]][v["role"]] = v
    for i, layer in sorted(empty):
        if i == item_id:
            by_layer[layer] = {
                r: {"judged": True, "kind": "empty", "correct": False, "quote_ok": False}
                for r in roles
            }
    passing = sorted(
        layer for layer, vs in by_layer.items() if all(vs.get(r, {}).get("correct") for r in roles)
    )
    unjudged = any(not v["judged"] for v in mine)
    incomplete = (
        unjudged
        or any(i == item_id for i, _l in unsummarized)
        or any(i == item_id for i, _l in missing)
        or not by_layer
    )
    passed: bool | None = True if passing else (None if incomplete else False)
    return {
        "id": item_id,
        "roles": roles,
        "pass": passed,
        "earliest_layer": passing[0] if passing else None,
        "passing_layers": passing,
        "unit_correct": {
            r: any(vs.get(r, {}).get("correct", False) for vs in by_layer.values()) for r in roles
        },
        "layers": {
            str(layer): {
                r: {k: v for k, v in vs[r].items() if k not in ("item", "layer", "role")}
                for r in vs
            }
            for layer, vs in sorted(by_layer.items())
        },
    }
