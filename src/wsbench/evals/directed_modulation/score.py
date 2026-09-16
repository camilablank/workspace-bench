"""directed_modulation: item-level rates over every judged (layer, position, sample) row."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from wsbench.llm import Spend
from wsbench.mcjudge import base_config, is_subset
from wsbench.registry import JudgeArgs
from wsbench.results import FamilyResult, bootstrap_ci, completeness

from .prompts import N_DISTRACTORS, PROMPT_VERSION

CHANCE_LABEL = (
    f"per call 1/{N_DISTRACTORS + 2} over the shown options incl. the escape; the measured "
    "floors are the lucky-guessing and prompt-only baselines"
)

Pred = Callable[[dict[str, Any]], bool]


def _any_by_item(verdicts: list[dict[str, Any]], pred: Pred) -> dict[str, bool]:
    """One bool per item: ANY judged row qualifies (the any-cell rule the family uses)."""
    out: dict[str, bool] = {}
    for v in verdicts:
        if not v["judged"]:
            out.setdefault(v["item"], False)
            continue
        out[v["item"]] = out.get(v["item"], False) or bool(pred(v))
    return out


def _rate(flags: dict[str, bool]) -> float | None:
    return sum(flags.values()) / len(flags) if flags else None


def summarize(items: list[dict[str, Any]], verdicts: list[dict[str, Any]]) -> dict[str, Any]:
    """``expressed`` = the gold concept picked at any row; ``content_bound`` = picked AND engaged as
    content (the headline); ``instruction_narration`` = picked only as narration of the task;
    ``distractor`` = a same-stratum wrong concept picked (the false-alarm channel); ``hinted`` =
    gold picked OR the gold concept's domain present; ``hint_distractor`` = a distractor's domain
    present (the hint channel's built-in null). ``white_bear`` is the per-pair think vs
    don't-think contrast on the headline rate."""
    channels: dict[str, Pred] = {
        "expressed": lambda v: v["pick"] == "gold",
        "content_bound": lambda v: v["pick"] == "gold" and v["basis"] == "content_bound",
        "instruction_narration": lambda v: (
            v["pick"] == "gold" and v["basis"] == "instruction_narration"
        ),
        "distractor": lambda v: v["pick"] == "distractor",
        "hinted": lambda v: v["pick"] == "gold" or v["gold_overlap"],
        "hint_distractor": lambda v: v["distractor_overlap"],
    }
    flags = {name: _any_by_item(verdicts, pred) for name, pred in channels.items()}
    sub_of = {it["id"]: str(it["subfamily"]) for it in items}

    def block(ids: list[str]) -> dict[str, Any]:
        keep = set(ids)
        out: dict[str, Any] = {"n_items": len(keep)}
        for name, f in flags.items():
            out[name] = _rate({k: v for k, v in f.items() if k in keep})
        return out

    out: dict[str, Any] = {"overall": block(list(sub_of))}
    for sub in sorted(set(sub_of.values())):
        out[sub] = block([i for i, s in sub_of.items() if s == sub])
    pairs: dict[str, dict[str, bool]] = {}
    strict = flags["content_bound"]
    for it in items:
        if it["subfamily"] == "pair" and it.get("pair_id"):
            pairs.setdefault(str(it["pair_id"]), {})[str(it["polarity"])] = strict.get(
                it["id"], False
            )
    both = {p: d for p, d in pairs.items() if "think" in d and "dont_think" in d}
    if both:
        think = sum(d["think"] for d in both.values()) / len(both)
        dont = sum(d["dont_think"] for d in both.values()) / len(both)
        out["white_bear"] = {
            "n_pairs": len(both),
            "think_rate": think,
            "dont_think_rate": dont,
            "delta": think - dont,
        }
    picks: dict[str, int] = {}
    forms: dict[str, int] = {}
    for v in verdicts:
        if not v["judged"]:
            continue
        picks[v["pick"]] = picks.get(v["pick"], 0) + 1
        if v["pick"] == "gold":
            forms[v["form"]] = forms.get(v["form"], 0) + 1
    out["picks"] = picks
    out["form_counts"] = forms
    out["n_voided"] = sum(1 for v in verdicts if v["judged"] and v["voided"])
    out["n_choice_invalid"] = sum(1 for v in verdicts if v["judged"] and v["choice_invalid"])
    return out


def score(
    items: list[dict[str, Any]],
    verdicts: list[dict[str, Any]],
    *,
    args: JudgeArgs,
    kind: str,
    layers: list[int],
    n_empty: int,
    skipped_rows: int,
    spend: Spend,
) -> FamilyResult:
    summary = summarize(items, verdicts)
    strict = _any_by_item(verdicts, lambda v: v["pick"] == "gold" and v["basis"] == "content_bound")
    judged_items = {v["item"] for v in verdicts if v["judged"]}
    unjudged_items = {v["item"] for v in verdicts if not v["judged"]}
    # an item with an unjudged row and no positive is undecided: out of the denominator
    decided = {i: p for i, p in strict.items() if p or i not in unjudged_items}
    passes = [1.0 if p else 0.0 for p in decided.values()]
    n_unjudged = sum(1 for v in verdicts if not v["judged"])
    ids = [it["id"] for it in items]
    return FamilyResult(
        family="directed_modulation",
        metric="pass_rate",
        value=sum(passes) / len(passes) if passes else None,
        ci95=bootstrap_ci(passes) if passes else None,
        n_items=len(ids),
        higher_is_better=True,
        chance=None,
        chance_label=CHANCE_LABEL,
        complete=bool(verdicts)
        and completeness(
            pinned=args.judge.pinned,
            subset=is_subset(args),
            n_expected=len(verdicts),
            n_missing=0,
            n_unjudged=n_unjudged,
            n_empty=n_empty,
        ),
        pinned_instrument=args.judge.pinned,
        config=base_config(
            args, PROMPT_VERSION, kind=kind, layers_judged=layers, n_distractors=N_DISTRACTORS
        ),
        # a cell is one judged readout row (item, layer, position, sample)
        counts={
            "n_expected_cells": len(verdicts),
            "n_missing_cells": 0,
            "n_unjudged_cells": n_unjudged,
            "n_empty_cells": n_empty,
            "skipped_rows": skipped_rows,
            "spend_usd": spend.usd,
        },
        extras={
            "headline": "content_bound (gold picked and engaged as content, any row)",
            "n_calls": len(verdicts),
            "n_items_decided": len(decided),
            "n_items_undecided": len(ids) - len(decided),
            "n_items_judged": len(judged_items),
            "summary": summary,
        },
        rows=verdicts,
    )
