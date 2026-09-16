"""Chained intermediates: a blind free-recall judge names the value(s) a readout presents as
computed; an item passes when the top-named value is one of its never-written intermediates at
some layer. Floors and the decoy null are explained in the family README."""

import re
import sys
from collections import defaultdict
from typing import Any

from wsbench.banks import load_bank
from wsbench.cache import Cache
from wsbench.llm import Spend
from wsbench.mc import fold
from wsbench.mcjudge import (
    Call,
    Preflighter,
    base_config,
    is_subset,
    item_scope,
    run_calls,
    with_readout_count,
)
from wsbench.readouts import Cell, load_readouts
from wsbench.registry import REPO_ROOT, JudgeArgs
from wsbench.results import FamilyResult, bootstrap_ci, completeness
from wsbench.summarizer import render_bag

from .prompts import MAX_VALUES, NEAR, PROMPT_VERSION, SCHEMA, SYSTEM, render_user

NAME = "chain_intermediates"
BANK = REPO_ROOT / "evals" / NAME / "items.json"
CHANCE_LABEL = (
    "no analytic floor (free recall); the measured floors are the magnitude-matched decoy null "
    "(extras.null_top1_near) and the prompt-only baseline"
)


def near_set(gold: list[int], exclude: tuple[int, ...] = ()) -> list[int]:
    """Decoys within +-NEAR of an intermediate, excluding the intermediates themselves and
    ``exclude`` (the item's start and answer, which a lens may echo for other reasons)."""
    out: list[int] = []
    for v in gold:
        out += [
            x
            for x in range(max(1, v - NEAR), v + NEAR + 1)
            if x not in gold and x not in exclude and x not in out
        ]
    return out


def names_number(n: int, text: str) -> bool:
    """``n`` appears in ``text`` as a whole number (not inside 10, 2019 or 10.83)."""
    return re.search(rf"(?<!\d)(?<!\d\.){n}(?!\d)(?!\.\d)", text) is not None


def verdict(
    res: dict[str, Any] | None, gold: list[int], near: list[int], readout: str
) -> dict[str, Any]:
    """One judge answer. ``values`` are the judge's ranked integers capped at three; the top value
    is credited only when it appears in ``readout`` as a whole number or the quote is verbatim in
    it (a digit-only quote must also be a whole number there). ``readout`` is the plain text (a
    token bag without its scores). ``any_of_3`` checks the ranked values against the gold without
    verifying the lower ranks: a diagnostic, not a pass rule."""
    if res is None:
        return {"judged": False, "kind": "unavailable", "values": [], "hit": False}
    raw = res.get("values")
    values = (
        [int(v) for v in raw if isinstance(v, int) and not isinstance(v, bool)][:MAX_VALUES]
        if isinstance(raw, list)
        else []
    )
    if not values:
        return {
            "judged": True,
            "kind": "none",
            "values": [],
            "hit": False,
            "basis": res.get("basis"),
        }
    top = values[0]
    quote = str(res.get("quote", "")).strip()
    if quote.isdigit():
        quote_ok = names_number(int(quote), readout)
    else:
        quote_ok = bool(quote) and fold(quote) in fold(readout)
    verified = names_number(top, readout) or quote_ok
    if not verified:
        kind = "unverified"
    elif top in gold:
        kind = "hit"
    elif top in near:
        kind = "near"
    else:
        kind = "other"
    return {
        "judged": True,
        "kind": kind,
        "values": values,
        "hit": kind == "hit",
        "any_of_3": verified and any(v in gold for v in values),
        "basis": res.get("basis"),
        "quote": quote,
    }


def _fail(msg: str) -> None:
    print(msg, file=sys.stderr)
    raise SystemExit(2)


def _rate(flags: list[bool]) -> float | None:
    return sum(flags) / len(flags) if flags else None


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def run(args: JudgeArgs) -> FamilyResult:
    _header, items = load_bank(BANK)
    scope = item_scope(items, args)
    ids = [it["id"] for it in scope]
    by_id = {it["id"]: it for it in scope}
    cells, rep = load_readouts(args.readouts, ids=ids, layers=args.layers)
    layers = args.layers if args.layers is not None else rep.layers
    # the read cell is the last prompt token: per (item, layer) the max-pos row; other rows are
    # ignored and counted
    groups: dict[tuple[str, int], list[Cell]] = defaultdict(list)
    for c in cells:
        groups[(c.id, c.layer)].append(c)
    chosen = {k: max(cs, key=lambda c: c.pos) for k, cs in groups.items()}
    n_extra_rows = sum(len(cs) - 1 for cs in groups.values())
    missing = [(i, layer) for i in ids for layer in layers if (i, layer) not in chosen]
    if missing and not args.allow_missing and not args.dry_run:
        _fail(
            f"{NAME}: {len(missing)} of {len(ids) * len(layers)} (item, layer) cells have no "
            "readout; pass allow_missing=True to score the rest"
        )

    # a token lens is judged as its scored bag (the prompt handles loose numerals) but verified
    # against the score-free token text, so a score like 10.83 can never vouch for a value
    texts: dict[tuple[str, int], str] = {}
    plain: dict[tuple[str, int], str] = {}
    empty: set[tuple[str, int]] = set()
    for k, c in chosen.items():
        if c.tokens is not None:
            text, bare = render_bag(c.tokens, c.scores), " | ".join(c.tokens)
        else:
            text = bare = "\n".join(s for s in (c.samples or ()) if s.strip())
        if text.strip():
            texts[k], plain[k] = text, bare
        else:
            empty.add(k)

    gold = {it["id"]: [int(v) for v in it["intermediates"]] for it in scope}
    near = {
        i: near_set(g, (int(by_id[i]["start"]), int(by_id[i]["answer"]))) for i, g in gold.items()
    }
    spend = Spend()
    pre = Preflighter(args.dry_run)
    with Cache(args.out / "cells.jsonl") as cache:
        calls = [
            Call(
                key=f"{i}|L{layer:03d}",
                system=SYSTEM,
                user=render_user(text),
                meta={"item": i, "layer": layer},
            )
            for (i, layer), text in sorted(texts.items())
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
    for call in calls:
        i, layer = call.meta["item"], call.meta["layer"]
        v = verdict(results.get(call.key), gold[i], near[i], plain[(i, layer)])
        verdicts.append({"item": i, "layer": layer, **v})
    rows = [_item_row(by_id[i], gold[i], near[i], verdicts, missing, empty) for i in ids]
    decided = [r for r in rows if r["pass"] is not None]
    passes = [1.0 if r["pass"] else 0.0 for r in decided]
    unjudged = {(v["item"], v["layer"]) for v in verdicts if not v["judged"]}
    judged = [v for v in verdicts if v["judged"]]
    kinds: dict[str, int] = defaultdict(int)
    basis: dict[str, int] = defaultdict(int)
    for v in judged:
        kinds[v["kind"]] += 1
        basis[str(v.get("basis"))] += 1
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
            n_expected=len(ids) * len(layers),
            n_missing=len(missing),
            n_unjudged=len(unjudged),
            n_empty=len(empty),
        ),
        pinned_instrument=args.judge.pinned,
        config=base_config(args, PROMPT_VERSION, kind=rep.kind or "prose", layers_judged=layers),
        # a cell is one (item, layer) readout at the last prompt token; one call per cell
        counts={
            "n_expected_cells": len(ids) * len(layers),
            "n_missing_cells": len(missing),
            "n_unjudged_cells": len(unjudged),
            "n_empty_cells": len(empty),
            "skipped_rows": sum(rep.skipped.values()) + n_extra_rows,
            "spend_usd": spend.usd,
        },
        extras={
            "n_calls": len(calls),
            "n_rows_not_last_token": n_extra_rows,
            "n_items_decided": len(decided),
            "n_items_undecided": len(ids) - len(decided),
            # mean over decided items (the source averaged every item with a null)
            "null_top1_near": _mean(
                [r["null_top1_near"] for r in decided if r["null_top1_near"] is not None]
            ),
            "any_of_3_rate": _rate([r["any_of_3"] for r in decided]),
            "committed_rate": _rate([bool(v["values"]) for v in judged]),
            "kinds": dict(kinds),
            "basis": dict(basis),
            "per_depth": {
                str(d): _rate([bool(r["pass"]) for r in decided if r["depth"] == d])
                for d in sorted({r["depth"] for r in decided})
            },
        },
        rows=rows,
    )
    return with_readout_count(result, scope, cells)


def _item_row(
    item: dict[str, Any],
    gold: list[int],
    near: list[int],
    verdicts: list[dict[str, Any]],
    missing: list[tuple[str, int]],
    empty: set[tuple[str, int]],
) -> dict[str, Any]:
    """``pass`` = some layer whose top-named value is an intermediate; None (undecided) when no
    layer hits and a cell is unjudged or missing. An empty cell is a judged negative.
    ``null_top1_near`` is the item's decoy rate: layers whose top value is a decoy within +-NEAR
    of an intermediate, weighted by len(gold)/len(near), over the judged layers."""
    item_id = item["id"]
    mine = sorted((v for v in verdicts if v["item"] == item_id), key=lambda v: v["layer"])
    hits = [v["layer"] for v in mine if v["hit"]]
    incomplete = any(not v["judged"] for v in mine) or any(i == item_id for i, _l in missing)
    passed: bool | None = True if hits else (None if incomplete else False)
    judged = [v for v in mine if v["judged"]]
    n_judged = len(judged) + sum(1 for i, _l in empty if i == item_id)
    near_hits = sum(1 for v in judged if v["kind"] == "near")
    return {
        "id": item_id,
        "depth": item.get("depth"),
        "intermediates": gold,
        "answer": item.get("answer"),
        "start": item.get("start"),
        "pass": passed,
        "earliest_layer": min(hits) if hits else None,
        "hitting_layers": hits,
        "any_of_3": any(v.get("any_of_3") for v in mine),
        "null_top1_near": (near_hits * len(gold) / len(near) / n_judged)
        if n_judged and near
        else None,
        "named_by_layer": {str(v["layer"]): v["values"] for v in mine},
        "kind_by_layer": {str(v["layer"]): v["kind"] for v in mine},
        "basis_by_layer": {str(v["layer"]): v.get("basis") for v in mine},
    }
