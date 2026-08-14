"""Typed analyst verdicts: the durable half of the bank-eval audit.

The analyst agents' FP/FN judgments used to evaporate into a chat message — a second run
could never diff against the previous correction, and the skill's ±0.03 flag threshold had
nothing durable to compare to. Now each analyst writes ``<gen>/verdicts/<family>.json`` in
this schema, and folding into corrected pass rates is deterministic code, not prose.
"""


import json
from pathlib import Path
from typing import Any

__all__ = ["REASONS_FN", "REASONS_FP", "fold_family", "load_verdicts", "validate_verdict"]

REASONS_FP = frozenset(
    {
        "prompt_echo",
        "unrelated_sense",
        "boilerplate",
        "degenerate_repetition",
        "think_block_only",
        "scaffold_text",
        "other",
    }
)
REASONS_FN = frozenset(
    {
        "synonym",
        "morphology",
        "other_language",
        "paraphrase",
        "linebreak_split",
        "truncated",
        "partial_compound",
        "other",
    }
)
_CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}


def validate_verdict(v: dict[str, Any]) -> None:
    """Hard-error on schema violations — a silently-skipped verdict is a lost audit."""
    verdict = v.get("verdict")
    if verdict not in ("false_positive", "false_negative", "confirmed"):
        raise ValueError(f"bad verdict {verdict!r} in {v.get('item')!r}")
    reason = v.get("reason_code", "other")
    allowed = REASONS_FP if verdict == "false_positive" else REASONS_FN
    if verdict != "confirmed" and reason not in allowed:
        raise ValueError(f"unknown reason_code {reason!r} for {verdict} on {v.get('item')!r}")
    if reason == "other" and verdict != "confirmed" and not v.get("note"):
        raise ValueError(f"reason_code 'other' needs a note ({v.get('item')!r})")
    if v.get("confidence", "medium") not in _CONFIDENCE_RANK:
        raise ValueError(f"bad confidence {v.get('confidence')!r} on {v.get('item')!r}")


def load_verdicts(verdicts_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """``{family: [verdict, ...]}`` from every ``<family>.json`` under ``verdicts_dir``."""
    out: dict[str, list[dict[str, Any]]] = {}
    for f in sorted(verdicts_dir.glob("*.json")):
        payload = json.loads(f.read_text())
        family = payload.get("family") or f.stem
        vs = list(payload.get("verdicts", []))
        for v in vs:
            validate_verdict(v)
        out[family] = vs
    return out


def fold_family(
    prompts: dict[str, dict[str, Any]],
    family: str,
    verdicts: list[dict[str, Any]],
    *,
    min_confidence: str = "medium",
) -> dict[str, Any]:
    """Corrected pass rate for one family: mechanical hits minus all-FP items, plus FN items.

    An item flips to corrected-miss only when EVERY one of its hit verdicts is a
    false_positive at >= ``min_confidence``; any false_negative verdict flips a miss to a
    corrected-hit. ``coverage_frac`` reports how many of the family's items received any
    verdict — a partial audit can never masquerade as a full one.
    """
    floor = _CONFIDENCE_RANK[min_confidence]
    fam_items = {label: row for label, row in prompts.items() if row.get("family") == family}
    strong = [
        v
        for v in verdicts
        if _CONFIDENCE_RANK[v.get("confidence", "medium")] >= floor and v.get("item") in fam_items
    ]
    fp_items = {v["item"] for v in strong if v["verdict"] == "false_positive"}
    fp_contested = {v["item"] for v in strong if v["verdict"] == "confirmed"}
    fn_items = {v["item"] for v in strong if v["verdict"] == "false_negative"}
    corrected_hits = 0
    for label, row in fam_items.items():
        hit = bool(row.get("hit"))
        if hit and label in fp_items and label not in fp_contested:
            hit = False
        elif not hit and label in fn_items:
            hit = True
        corrected_hits += int(hit)
    n = len(fam_items)
    mechanical = sum(1 for row in fam_items.values() if row.get("hit")) / n if n else None
    corrected = corrected_hits / n if n else None
    delta = corrected - mechanical if mechanical is not None and corrected is not None else None
    return {
        "n_items": n,
        "mechanical_pass_rate": round(mechanical, 4) if mechanical is not None else None,
        "corrected_pass_rate": round(corrected, 4) if corrected is not None else None,
        "delta": round(delta, 4) if delta is not None else None,
        "over_threshold": bool(delta is not None and abs(delta) > 0.03),
        "n_fp_items": len(fp_items - fp_contested),
        "n_fn_items": len(fn_items),
        "n_verdicts": len(verdicts),
        "coverage_frac": round(len({v["item"] for v in strong}) / n, 4) if n else None,
    }
