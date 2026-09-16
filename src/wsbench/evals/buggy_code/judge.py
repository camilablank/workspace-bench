"""Buggy code: a consequence-ladder judge per program (buggy and clean twins alike) at one read
cell; headline ``net_S2`` = P(consequence read | buggy) - P(consequence read | clean). Rubric,
strata and the quote gate are explained in the family README."""

import random
from collections import defaultdict
from typing import Any

from wsbench.banks import load_bank
from wsbench.cache import Cache
from wsbench.family import cell_text, fail, quote_in, rate, require_cells
from wsbench.llm import Spend
from wsbench.mcjudge import (
    Call,
    Preflighter,
    base_config,
    base_counts,
    is_subset,
    item_scope,
    run_calls,
    with_readout_count,
)
from wsbench.readouts import Cell, load_readouts
from wsbench.registry import REPO_ROOT, JudgeArgs
from wsbench.results import FamilyResult, completeness

from .prompts import CONSEQUENCE, PROMPT_VERSION, RUNGS, SCHEMA, SYSTEM, render_user

NAME = "buggy_code"
BANK = REPO_ROOT / "evals" / NAME / "items.json"
CHANCE_LABEL = (
    "difference of two rates; 0 = the lens reads bug consequences into clean programs as often "
    "as into buggy ones. The clean twins are the false-alarm control"
)


def verdict(res: dict[str, Any] | None, samples: list[str]) -> dict[str, Any]:
    """One judge answer. An S2+ rung counts as a consequence read only with a verbatim quote from
    a sample; without one it is recorded as ``unverified`` and scored below S2."""
    if res is None or res.get("rung") not in RUNGS:
        return {"judged": False, "rung": None, "consequence": False}
    rung = str(res["rung"])
    quote = str(res.get("quote", "")).strip()
    quote_ok = any(quote_in(quote, s) for s in samples)  # ONE sample
    consequence = rung in CONSEQUENCE and quote_ok
    return {
        "judged": True,
        "rung": rung,
        "consequence": consequence,
        "unverified": rung in CONSEQUENCE and not quote_ok,
        "anti": bool(res.get("anti", False)),
        "quote": quote,
        "quote_ok": quote_ok,
        "why": str(res.get("why", "")),
    }


def net_ci(
    buggy: list[bool], clean: list[bool], *, n: int = 1000, seed: int = 0
) -> tuple[float, float] | None:
    """Bootstrap CI of the rate difference, resampling the buggy and clean sets separately."""
    if not buggy or not clean:
        return None
    rng = random.Random(seed)
    diffs = sorted(
        sum(rng.choice(buggy) for _ in buggy) / len(buggy)
        - sum(rng.choice(clean) for _ in clean) / len(clean)
        for _ in range(n)
    )
    return diffs[round(0.025 * (n - 1))], diffs[round(0.975 * (n - 1))]


def run(args: JudgeArgs) -> FamilyResult:
    header, items = load_bank(BANK)
    scope = item_scope(items, args)
    ids = [it["id"] for it in scope]
    by_id = {it["id"]: it for it in scope}
    read_layer = {it["id"]: int(header["read_cells"][it["lang_group"]]["layer"]) for it in scope}
    cells, rep = load_readouts(args.readouts, ids=ids, layers=args.layers)
    # one cell per item: its read layer (or the override), the max-pos row = the EOF anchor
    layer_of = {i: (args.layers[0] if args.layers else read_layer[i]) for i in ids}
    if args.layers and len(args.layers) > 1:
        fail(f"{NAME}: one read layer per item; pass a single layer or none, not {args.layers}")
    groups: dict[tuple[str, int], list[Cell]] = defaultdict(list)
    for c in cells:
        groups[(c.id, c.layer)].append(c)
    chosen = {
        i: max(groups[(i, layer_of[i])], key=lambda c: c.pos)
        for i in ids
        if (i, layer_of[i]) in groups
    }
    n_extra = len(cells) - len(chosen)  # rows at other layers or positions
    missing = [i for i in ids if i not in chosen]
    require_cells(NAME, missing, len(ids), args)
    samples: dict[str, list[str]] = {}
    empty: set[str] = set()
    for i, c in chosen.items():
        # the judge reads each prose sample separately (the one-sample quote rule); a token
        # lens is one scored bag
        ss = (
            [cell_text(c)[0]]
            if c.tokens is not None
            else [s for s in (c.samples or ()) if s.strip()]
        )
        if ss:
            samples[i] = ss
        else:
            empty.add(i)

    spend = Spend()
    with Cache(args.out / "cells.jsonl") as cache:
        calls = [
            Call(
                key=f"{i}|L{layer_of[i]:03d}",
                system=SYSTEM,
                user=render_user(by_id[i], samples[i]),
                meta={"item": i, "layer": layer_of[i], "src": by_id[i]["src"]},
            )
            for i in ids
            if i in samples
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
            preflight=Preflighter(args.dry_run).for_judge(args.judge),
            temperature=0.0,
        )

    rows = []
    for i in ids:
        it = by_id[i]
        if i in empty:
            v: dict[str, Any] = {"judged": True, "rung": "S0", "consequence": False, "empty": True}
        elif i in samples:
            v = verdict(results.get(f"{i}|L{layer_of[i]:03d}"), samples[i])
        else:
            v = {"judged": False, "rung": None, "consequence": False, "missing": True}
        rows.append(
            {
                "id": i,
                "src": it["src"],
                "lang_group": it["lang_group"],
                "consequence_class": it.get("consequence_class"),
                "layer": layer_of[i],
                **v,
            }
        )
    judged = [r for r in rows if r["judged"]]
    buggy = [r["consequence"] for r in judged if r["src"] == "buggy"]
    clean = [r["consequence"] for r in judged if r["src"] == "clean"]
    b_rate, c_rate = rate(buggy), rate(clean)
    value = (b_rate - c_rate) if b_rate is not None and c_rate is not None else None
    n_unjudged = sum(1 for r in rows if not r["judged"] and not r.get("missing"))
    rungs: dict[str, dict[str, int]] = {"buggy": defaultdict(int), "clean": defaultdict(int)}
    for r in judged:
        rungs[r["src"]][str(r["rung"])] += 1
    strata = sorted(
        {(r["consequence_class"] or "none", r["lang_group"]) for r in judged if r["src"] == "buggy"}
    )
    per_stratum = {}
    for cls, lg in strata:
        b = [
            r["consequence"]
            for r in judged
            if r["src"] == "buggy"
            and (r["consequence_class"] or "none") == cls
            and r["lang_group"] == lg
        ]
        c = [r["consequence"] for r in judged if r["src"] == "clean" and r["lang_group"] == lg]
        per_stratum[f"{cls}/{lg}"] = {
            "n_buggy": len(b),
            "buggy_S2": rate(b),
            "clean_S2": rate(c),
            "net_S2": (rate(b) - rate(c)) if b and c else None,
        }
    result = FamilyResult(
        family=NAME,
        metric="net_S2",
        value=value,
        ci95=net_ci(buggy, clean),
        n_items=len(ids),
        higher_is_better=True,
        chance=0.0,
        chance_label=CHANCE_LABEL,
        complete=bool(cells)
        and completeness(
            pinned=args.judge.pinned,
            subset=is_subset(args),
            n_expected=len(ids),
            n_missing=len(missing),
            n_unjudged=n_unjudged,
            n_empty=len(empty),
        ),
        pinned_instrument=args.judge.pinned,
        config=base_config(
            args,
            PROMPT_VERSION,
            kind=rep.kind or "prose",
            layers_judged=sorted(set(layer_of.values())),
        ),
        # a cell is one item at its read layer; one call per non-empty cell
        counts=base_counts(
            n_expected=len(ids),
            n_missing=len(missing),
            n_unjudged=n_unjudged,
            n_empty=len(empty),
            skipped_rows=sum(rep.skipped.values()) + n_extra,
            spend=spend,
        ),
        extras={
            "n_calls": len(calls),
            "n_rows_not_at_read_cell": n_extra,
            "buggy_S2_rate": b_rate,
            "clean_S2_rate": c_rate,
            "n_buggy_judged": len(buggy),
            "n_clean_judged": len(clean),
            "anti_rate_buggy": rate(bool(r.get("anti")) for r in judged if r["src"] == "buggy"),
            "corrective_rate_buggy": rate(
                r["rung"] == "corrective" for r in judged if r["src"] == "buggy"
            ),
            "unverified_S2_rate": rate(bool(r.get("unverified")) for r in judged),
            "rungs": {k: dict(v) for k, v in rungs.items()},
            "per_stratum": per_stratum,
        },
        rows=rows,
    )
    return with_readout_count(result, scope, cells)
