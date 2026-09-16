"""Brew intermediates: a blind multi-select colour judge over the pinned read cells; an item passes
when the gold intermediate is named in more emission cells than the mean off-trajectory colour.
Regions, the null and the baseline are explained in the family README."""

import hashlib
import random
import sys
from fractions import Fraction
from typing import Any

from wsbench.banks import load_bank
from wsbench.cache import Cache
from wsbench.llm import Spend
from wsbench.mcjudge import (
    Call,
    Preflighter,
    base_config,
    is_subset,
    item_scope,
    run_calls,
    with_readout_count,
)
from wsbench.readouts import load_readouts
from wsbench.registry import REPO_ROOT, JudgeArgs
from wsbench.results import FamilyResult, bootstrap_ci, completeness
from wsbench.summarizer import render_bag

from .prompts import PROMPT_VERSION, SCHEMA, SEED, SYSTEM, mentions_any_colour, render_user

NAME = "brew_intermediates"
BANK = REPO_ROOT / "evals" / NAME / "items.json"
EMIT = ("emit_asst", "emit_think", "emit_colon")
STIR = ("stir",)
REGION_SETS = {
    "headline": EMIT + STIR,  # the emission cells (headline) and the stir cells (control)
    "all": EMIT + STIR + ("start", "question"),
}
CHANCE_LABEL = (
    "no analytic floor; the measured floors are the role-swap null and the no-information "
    "baseline (extras), the stir-region control and the lucky-guessing baseline"
)


def shuffled(options: list[str], key: str) -> list[str]:
    opts = list(options)
    random.Random(int(hashlib.sha256(f"{SEED}|{key}".encode()).hexdigest(), 16)).shuffle(opts)
    return opts


def verdict(
    res: dict[str, Any] | None, options: list[str], gold: str, offs: list[str]
) -> dict[str, Any]:
    """One judge answer: the candidate colours the readout names (filtered to the options),
    whether the gold is among them, and which off colours are."""
    if res is None:
        return {"judged": False, "named": [], "gold": False, "offs": [False] * len(offs)}
    raw = res.get("named")
    named = sorted({str(c) for c in raw if str(c) in options}) if isinstance(raw, list) else []
    primary = res.get("primary")
    return {
        "judged": True,
        "named": named,
        "gold": gold in named,
        "offs": [c in named for c in offs],
        "primary": primary if primary in options else "none",
        "basis": res.get("basis"),
    }


def rule(cells: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The source adapter's rule over a set of judged cells: G = cells naming the gold, O_j =
    cells naming off colour j. ``pass`` iff K*G > sum(O) (gold named in more cells than the mean
    off colour; ties fail). ``null`` = the same rule with each off colour in the gold's role,
    averaged; ``baseline`` = the fraction of the K+1 colours that clear the rule (what a lens
    with no idea which colour is the intermediate scores)."""
    if not cells:
        return None
    g = sum(c["gold"] for c in cells)
    k = len(cells[0]["offs"])
    o = [sum(c["offs"][j] for c in cells) for j in range(k)]
    total = g + sum(o)
    return {
        "n_cells": len(cells),
        "G": g,
        "O": o,
        "pass": k * g > sum(o),
        "null": float(Fraction(sum((k + 1) * oj > total for oj in o), k)),
        "baseline": float(Fraction(sum((k + 1) * c > total for c in [g, *o]), k + 1)),
    }


def _fail(msg: str) -> None:
    print(msg, file=sys.stderr)
    raise SystemExit(2)


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def run(args: JudgeArgs) -> FamilyResult:
    _header, items = load_bank(BANK)
    scope = item_scope(items, args)
    ids = [it["id"] for it in scope]
    by_id = {it["id"]: it for it in scope}
    region_set = args.extra.get("regions", "headline")
    if region_set not in REGION_SETS:
        _fail(f"{NAME}: opts=regions= must be one of {sorted(REGION_SETS)}, got {region_set!r}")
    wanted = set(REGION_SETS[region_set])
    # the read cells: every pinned position whose region is judged
    positions = {
        it["id"]: sorted(int(p) for p, r in it["regions"].items() if r in wanted) for it in scope
    }
    cells, rep = load_readouts(args.readouts, ids=ids, layers=args.layers, positions=positions)
    layers = args.layers if args.layers is not None else rep.layers
    have = {(c.id, c.layer, c.pos): c for c in cells}
    expected = [(i, layer, p) for i in ids for layer in layers for p in positions[i]]
    missing = [k for k in expected if k not in have]
    if missing and not args.allow_missing and not args.dry_run:
        _fail(
            f"{NAME}: {len(missing)} of {len(expected)} (item, layer, cell) readouts are missing; "
            "pass allow_missing=True to score the rest"
        )
    judged_cells = [have[k] for k in expected if k in have]

    texts: dict[tuple[str, int, int], str] = {}
    empty: set[tuple[str, int, int]] = set()
    for c in judged_cells:
        text = (
            render_bag(c.tokens or (), c.scores)
            if c.tokens is not None
            else "\n".join(s for s in (c.samples or ()) if s.strip())
        )
        if text.strip():
            texts[(c.id, c.layer, c.pos)] = text
        else:
            empty.add((c.id, c.layer, c.pos))
    options = {it["id"]: [str(c) for c in it["options_adjacent"][0]] for it in scope}
    gold = {it["id"]: str(it["intermediates"][0]) for it in scope}
    offs = {
        i: sorted(
            c for c in options[i] if c not in (gold[i], by_id[i]["start"], by_id[i]["answer"])
        )
        for i in ids
    }
    screened = {k for k, t in texts.items() if not mentions_any_colour(t)}

    spend = Spend()
    pre = Preflighter(args.dry_run)
    with Cache(args.out / "cells.jsonl") as cache:
        calls = [
            Call(
                key=f"{i}|L{layer:03d}|p{pos}",
                system=SYSTEM,
                user=render_user(text, shuffled(options[i], f"{i}|{layer}|{pos}")),
                meta={"item": i, "layer": layer, "pos": pos},
            )
            for (i, layer, pos), text in sorted(texts.items())
            if (i, layer, pos) not in screened
        ]
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
    for i, layer, pos in sorted(texts):
        key = f"{i}|L{layer:03d}|p{pos}"
        if (i, layer, pos) in screened:
            v: dict[str, Any] = {
                "judged": True,
                "named": [],
                "gold": False,
                "offs": [False] * len(offs[i]),
                "primary": "none",
                "basis": "none",
                "screened": True,
            }
        else:
            v = verdict(results.get(key), options[i], gold[i], offs[i])
        verdicts.append(
            {"item": i, "layer": layer, "pos": pos, "region": by_id[i]["regions"][str(pos)], **v}
        )
    rows = [_item_row(by_id[i], gold[i], offs[i], verdicts, missing, empty) for i in ids]
    decided = [r for r in rows if r["pass"] is not None]
    passes = [1.0 if r["pass"] else 0.0 for r in decided]
    unjudged = {(v["item"], v["layer"], v["pos"]) for v in verdicts if not v["judged"]}
    result = FamilyResult(
        family=NAME,
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
            n_expected=len(expected),
            n_missing=len(missing),
            n_unjudged=len(unjudged),
            n_empty=len(empty),
        ),
        pinned_instrument=args.judge.pinned,
        config=base_config(
            args, PROMPT_VERSION, kind=rep.kind or "prose", layers_judged=layers, regions=region_set
        ),
        # a cell is one (item, layer, pinned position) readout; one call per cell not screened
        counts={
            "n_expected_cells": len(expected),
            "n_missing_cells": len(missing),
            "n_unjudged_cells": len(unjudged),
            "n_empty_cells": len(empty),
            "skipped_rows": sum(rep.skipped.values()),
            "spend_usd": spend.usd,
        },
        extras={
            "n_calls": len(calls),
            "n_screened_cells": len(screened),
            "n_items_decided": len(decided),
            "n_items_undecided": len(ids) - len(decided),
            "regions_judged": sorted(wanted),
            "null": _mean([r["emit"]["null"] for r in decided if r["emit"]]),
            "baseline": _mean([r["emit"]["baseline"] for r in decided if r["emit"]]),
            "stir_pass_rate": _mean([float(r["stir"]["pass"]) for r in decided if r["stir"]]),
            "stir_null": _mean([r["stir"]["null"] for r in decided if r["stir"]]),
            "cell_gold_rate": {
                reg: _mean(
                    [float(v["gold"]) for v in verdicts if v["judged"] and v["region"] == reg]
                )
                for reg in sorted(wanted)
            },
            "cell_off_rate": {
                reg: _mean(
                    [
                        sum(v["offs"]) / len(v["offs"])
                        for v in verdicts
                        if v["judged"] and v["region"] == reg and v["offs"]
                    ]
                )
                for reg in sorted(wanted)
            },
        },
        rows=rows,
    )
    return with_readout_count(result, scope, cells)


def _item_row(
    item: dict[str, Any],
    gold: str,
    offs: list[str],
    verdicts: list[dict[str, Any]],
    missing: list[tuple[str, int, int]],
    empty: set[tuple[str, int, int]],
) -> dict[str, Any]:
    """``pass`` = the rule over the item's judged emission cells (empty cells name nothing);
    None (undecided) when an emission cell is unjudged or missing. ``stir`` is the same rule
    over the stir cells, the negative control."""
    item_id = item["id"]
    mine = [v for v in verdicts if v["item"] == item_id]
    regions = item["regions"]
    blank = [
        {"gold": False, "offs": [False] * len(offs), "region": regions[str(p)]}
        for i, _l, p in empty
        if i == item_id
    ]
    emit = [v for v in mine + blank if v["region"] in EMIT]
    stir = [v for v in mine + blank if v["region"] in STIR]
    incomplete = any(not v["judged"] for v in mine if v["region"] in EMIT) or any(
        i == item_id and regions[str(p)] in EMIT for i, _l, p in missing
    )
    emit_stats = rule(emit) if emit else None
    passed: bool | None = None if incomplete or emit_stats is None else emit_stats["pass"]
    return {
        "id": item_id,
        "gold": gold,
        "start": item.get("start"),
        "answer": item.get("answer"),
        "offs": offs,
        "pass": passed,
        "emit": emit_stats,
        "stir": rule(stir) if stir else None,
        "gold_named_at": sorted(f"L{v['layer']}:{v['region']}" for v in mine if v.get("gold")),
        "cells": [
            {k: x for k, x in v.items() if k != "item"}
            for v in sorted(mine, key=lambda v: (v["layer"], v["pos"]))
        ],
    }
