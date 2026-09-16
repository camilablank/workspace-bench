"""Arithmetic intermediates: a blind free-recall judge names the values a readout presents as
computed at one frozen cell per variant; an item passes when a named value lands within its
variant's tolerance of the never-written intermediate. The permutation null (``cross``) and the
per-variant table are explained in the family README."""

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

from .prompts import MAX_VALUES, PROMPT_VERSION, SCHEMA, SYSTEM, render_user

NAME = "arithmetic_intermediates"
BANK = REPO_ROOT / "evals" / NAME / "items.json"
TOLERANCES = {"exact": 0.0, "rel0.5pct": 0.005, "rel1pct": 0.01, "rel2pct": 0.02, "rel5pct": 0.05}
CHANCE_LABEL = (
    "no analytic floor (free recall); the measured floors are the permutation null over each "
    "item's null set (extras.cross) and the prompt-only baseline"
)
_NUMERAL = re.compile(r"(?<![\d.])-?\d[\d,]*(?:\.\d+)?(?!\d)(?!\.\d)")
# fullwidth digits, full stop, comma and hyphen-minus (U+FF10.., U+FF0E, U+FF0C, U+FF0D) -> ASCII
_FULLWIDTH = str.maketrans(
    "".join(chr(0xFF10 + d) for d in range(10)) + "\uff0e\uff0c\uff0d", "0123456789.,-"
)


def tolerance_ok(got: float, target: float, tol: str) -> bool:
    """The source spec's single numeric match: exact, or within a relative tolerance; sign-aware."""
    if tol == "exact":
        return got == target
    return abs(got - target) <= TOLERANCES[tol] * abs(target)


def quantities(text: str) -> list[float]:
    """The numerals a readout writes (ASCII or fullwidth), with list markers ("2. foo") and step
    labels stripped as the source scorer does; used only to VERIFY that a value the judge names
    is in the text, never to score."""
    t = text.translate(_FULLWIDTH)
    t = re.sub(r"(?m)^[ \t]*-?[ \t]*\d+\.[ \t]+(?=[A-Za-z*_#`(])", " ", t)
    t = re.sub(r"(?i)step\s*\d+", " ", t)
    out: list[float] = []
    for m in _NUMERAL.finditer(t):
        s = m.group(0).replace(",", "")
        try:
            out.append(float(s))
        except ValueError:
            continue
    return out


def verified(value: float, text: str, quote: str) -> bool:
    """A named value counts when the readout writes it (to six decimals) or the judge's quote is
    verbatim in the readout (a Chinese numeral is credited that way)."""
    if any(abs(q - value) <= 1e-6 * max(1.0, abs(value)) for q in quantities(text)):
        return True
    return (
        bool(quote)
        and not quote.strip().lstrip("-").replace(".", "").isdigit()
        and fold(quote) in fold(text)
    )


def verdict(
    res: dict[str, Any] | None, target: float, tol: str, nulls: list[float], text: str
) -> dict[str, Any]:
    """One judge answer. ``values`` = the judge's ranked numbers, capped at three, kept only when
    verified in the readout. ``hit`` = some kept value within tolerance of the intermediate;
    ``cross`` = the fraction of the item's null set whose intermediate some kept value hits."""
    if res is None:
        return {"judged": False, "kind": "unavailable", "values": [], "hit": False, "cross": None}
    raw = res.get("values")
    named = (
        [float(v) for v in raw if isinstance(v, int | float) and not isinstance(v, bool)][
            :MAX_VALUES
        ]
        if isinstance(raw, list)
        else []
    )
    quote = str(res.get("quote", "")).strip()
    kept = [v for v in named if verified(v, text, quote)]
    hit = any(tolerance_ok(v, target, tol) for v in kept)
    cross = (
        sum(any(tolerance_ok(v, n, tol) for v in kept) for n in nulls) / len(nulls)
        if nulls
        else None
    )
    if not named:
        kind = "none"
    elif not kept:
        kind = "unverified"
    elif hit:
        kind = "hit"
    else:
        kind = "other"
    return {
        "judged": True,
        "kind": kind,
        "values": named,
        "kept": kept,
        "hit": hit,
        "top1_hit": bool(kept) and tolerance_ok(kept[0], target, tol),
        "cross": cross,
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
    header, items = load_bank(BANK)
    scope = item_scope(items, args)
    ids = [it["id"] for it in scope]
    by_id = {it["id"]: it for it in scope}
    by_name = {it["name"]: it for it in items}
    if args.layers and len(args.layers) > 1:
        _fail(f"{NAME}: one frozen cell per item; pass a single layer or none, not {args.layers}")
    layer_of = {i: (args.layers[0] if args.layers else int(by_id[i]["cell"]["layer"])) for i in ids}
    pos_of = {i: int(by_id[i]["cell"]["pos"]) for i in ids}
    cells, rep = load_readouts(args.readouts, ids=ids, layers=args.layers)
    have: dict[str, Cell] = {}
    n_off_cell = 0
    for c in cells:
        if c.layer == layer_of[c.id] and c.pos == pos_of[c.id]:
            have[c.id] = c
        else:
            n_off_cell += 1
    missing = [i for i in ids if i not in have]
    if missing and not args.allow_missing and not args.dry_run:
        _fail(
            f"{NAME}: {len(missing)} of {len(ids)} items have no readout at their frozen cell "
            "(layer, pos); pass allow_missing=True to score the rest"
        )
    texts: dict[str, str] = {}
    plain: dict[str, str] = {}
    empty: set[str] = set()
    for i, c in have.items():
        if c.tokens is not None:
            text, bare = render_bag(c.tokens, c.scores), " | ".join(c.tokens)
        else:
            text = bare = "\n".join(s for s in (c.samples or ()) if s.strip())
        if text.strip():
            texts[i], plain[i] = text, bare
        else:
            empty.add(i)
    target = {i: float(by_id[i]["intermediates"][0]) for i in ids}
    tol = {i: str(by_id[i]["tolerance"]) for i in ids}
    nulls = {
        i: [float(by_name[n]["intermediates"][0]) for n in by_id[i]["null_set"] if n in by_name]
        for i in ids
    }

    spend = Spend()
    pre = Preflighter(args.dry_run)
    with Cache(args.out / "cells.jsonl") as cache:
        calls = [
            Call(
                key=f"{i}|L{layer_of[i]:03d}|p{pos_of[i]}",
                system=SYSTEM,
                user=render_user(texts[i]),
                meta={"item": i, "layer": layer_of[i], "pos": pos_of[i]},
            )
            for i in ids
            if i in texts
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

    rows = []
    for i in ids:
        it = by_id[i]
        if i in empty:
            v: dict[str, Any] = {
                "judged": True,
                "kind": "empty",
                "values": [],
                "hit": False,
                "cross": 0.0 if nulls[i] else None,
            }
        elif i in texts:
            v = verdict(
                results.get(f"{i}|L{layer_of[i]:03d}|p{pos_of[i]}"),
                target[i],
                tol[i],
                nulls[i],
                plain[i],
            )
        else:
            v = {"judged": False, "kind": "missing", "values": [], "hit": False, "cross": None}
        rows.append(
            {
                "id": i,
                "variant": it["variant"],
                "role": it["role"],
                "expr": it["expr"],
                "intermediate": target[i],
                "tolerance": tol[i],
                "layer": layer_of[i],
                "pos": pos_of[i],
                "pass": v["hit"] if v["judged"] else None,
                **v,
            }
        )
    decided = [r for r in rows if r["pass"] is not None]
    passes = [1.0 if r["pass"] else 0.0 for r in decided]
    unjudged = [r for r in rows if not r["judged"] and r["kind"] != "missing"]
    kinds: dict[str, int] = defaultdict(int)
    for r in rows:
        kinds[r["kind"]] += 1
    cross_rates = [r["cross"] for r in decided if r["cross"] is not None]
    value_rate = _rate([bool(r["pass"]) for r in decided])
    cross_rate = _mean(cross_rates)
    per_variant = {}
    for v in header["variants"]:
        vs = [r for r in decided if r["variant"] == v]
        if not vs:
            continue
        val = _rate([bool(r["pass"]) for r in vs])
        crs = _mean([r["cross"] for r in vs if r["cross"] is not None])
        per_variant[v] = {
            "n": len(vs),
            "value": val,
            "cross": crs,
            "net": (val - crs) if val is not None and crs is not None else None,
            "tolerance": header["variants"][v]["tolerance"],
            "role": header["variants"][v]["role"],
        }
    result = FamilyResult(
        family=NAME,
        metric="pass_rate",
        value=value_rate,
        ci95=bootstrap_ci(passes) if passes else None,
        n_items=len(ids),
        higher_is_better=True,
        chance=None,
        chance_label=CHANCE_LABEL,
        complete=bool(cells)
        and completeness(
            pinned=args.judge.pinned,
            subset=is_subset(args),
            n_expected=len(ids),
            n_missing=len(missing),
            n_unjudged=len(unjudged),
            n_empty=len(empty),
        ),
        pinned_instrument=args.judge.pinned,
        config=base_config(
            args,
            PROMPT_VERSION,
            kind=rep.kind or "prose",
            layers_judged=sorted(set(layer_of.values())),
        ),
        # a cell is one item at its frozen (layer, pos); one call per non-empty cell
        counts={
            "n_expected_cells": len(ids),
            "n_missing_cells": len(missing),
            "n_unjudged_cells": len(unjudged),
            "n_empty_cells": len(empty),
            "skipped_rows": sum(rep.skipped.values()) + n_off_cell,
            "spend_usd": spend.usd,
        },
        extras={
            "n_calls": len(calls),
            "n_rows_off_cell": n_off_cell,
            "n_items_decided": len(decided),
            "n_items_undecided": len(ids) - len(decided),
            "cross": cross_rate,
            "net": (value_rate - cross_rate)
            if value_rate is not None and cross_rate is not None
            else None,
            "top1_rate": _rate([bool(r.get("top1_hit")) for r in decided]),
            "committed_rate": _rate([bool(r["values"]) for r in rows if r["judged"]]),
            "kinds": dict(kinds),
            "per_variant": per_variant,
            "per_role": {
                role: _rate([bool(r["pass"]) for r in decided if r["role"] == role])
                for role in sorted({r["role"] for r in decided})
            },
        },
        rows=rows,
    )
    return with_readout_count(result, scope, cells)
