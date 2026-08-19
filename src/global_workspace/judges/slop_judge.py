"""Slop judge: the precision condition for workspace-bench readouts (Camila 2026-08-19).

The mechanical pass and the family judges are recall instruments: a readout that buries the
correct answer under fabricated detail scores identically to one that delivers it cleanly —
the NLA failure mode (hallucinatory free-text readouts that assert extra incorrect details on
the subject alongside the right answer). This judge rates that junk on a 1.0-10.0 scale (one
decimal), given the CONTEXT the model read and the TARGET concept the readout should deliver;
downstream scorers count an answer only when its slop is below a threshold. The threshold is a
CONFIG, not a prompt constant — it is chosen qualitatively from the gate artifact
(``scripts/oracle_lens_evals/gate_slop_judge.py``) after reading how strict the judge runs.

Rubric (Camila 2026-08-19, verbatim intent):

- **1.0** — the only concept meaningfully asserted is the precise target (the desired
  intermediate); any extra text is a reasonable readout of the prompt text itself.
- **< 5** — any extra information is harmless and it is clear the precise answer is actually
  present, even if there's other stuff going on in the readout.
- **> 5** — unnecessary information that is NOT inferrable from the context and is related to
  the target (hallucinated extra detail about the target that can be confusing), or completely
  hallucinated unrelated content that is also distracting.
- **10.0** — a complete contradiction, or other plausible distractor answers contained in the
  readout. A very high score means it is so unclear what the answer is that the question
  should be invalidated entirely.

Scope: slop is a FREE-TEXT failure mode. Token-bag readouts (J-lens top-k lists) are not
"asserting" anything, so judge them only after the item-blind summarizer
(``--readout-format tokens-llm``) has turned them into prose — never rate a raw token bag.

The judge itself never decides pass/fail: ``gated_hit(hit, slop, threshold)`` applies the cut
per grid row, so the item-level pass@any aggregation stays exactly the published one with a
per-row precision veto layered in.
"""

from typing import Any

from .llm_client import CLAUDE_JUDGE, async_json, schema_block

__all__ = [
    "SLOP_SCHEMA",
    "SLOP_SYSTEM",
    "SLOP_USER",
    "async_json",
    "gated_hit",
    "judge_slop",
    "mechanical_targets",
    "slop_context",
    "slop_target",
    "summarize_slop",
]

SLOP_SYSTEM = (
    "You rate the SLOP of an interpretability readout: how much content beyond the precise "
    "target concept it asserts, and how much that extra content obscures or contradicts the "
    "target. You are not re-deciding whether the target is correct — you are rating how "
    "cleanly the readout delivers it. Judge extra content against the provided context, never "
    "against your own world knowledge. Output only the JSON verdict."
)

SLOP_USER = """\
You are rating the output of an interpretability lens that reads a language model's internal \
activation while the model processed the context below. A clean readout delivers the target \
concept precisely; a sloppy readout buries it under extra assertions.

<context>
{context}
</context>

Target concept (the desired intermediate the readout should deliver): {target}

<lens_output>
{readout}
</lens_output>

Rate the readout's slop from 1.0 to 10.0 (one decimal place):

- 1.0: the only concept meaningfully asserted is the precise target. Any extra text is a \
reasonable reading of the context itself — restating or paraphrasing the prompt, describing \
the task or the expected answer format, hedging.
- 1.1-4.9: extra information is present but harmless: it is inferrable from the context or \
generically about the task, and the precise target is still clearly the answer the readout \
delivers. More extra material, or extra material that is only loosely licensed by the \
context, pushes the score toward 5.
- 5.0-7.9: unnecessary information that is NOT inferrable from the context: invented details \
about the target's subject (attributes, quantities, names, dates, or facts the context does \
not support — confusing even when the target is present), or fabricated unrelated content \
that distracts from the answer.
- 8.0-9.9: the extra content seriously obscures the answer — several fabricated claims, or \
plausible alternative answers to the question appearing alongside the target.
- 10.0: the readout contradicts the target, or asserts other plausible distractor answers so \
prominently that it is unclear what the answer is. Scores this high mean the readout should \
be invalidated as an answer entirely.

Rules:

- Judge extra content against the CONTEXT, not world knowledge: a detail that happens to be \
true in the world but is not inferrable from the context still counts as hallucinated extra \
content.
- Do NOT penalize: restating/paraphrasing the context; naming the expected answer format; \
uncertainty or hedging; the target expressed as a synonym, morphological variant, or in \
another language.
- Only asserted content counts. Noise tokens, fragments, repeated boilerplate, and formatting \
are not assertions (they may still pad the readout, but padding alone is not slop).
- If the precise target is absent, rate the slop of what IS asserted (an absent target with \
clean context-only content can still be low slop; an absent target amid fabricated answers is \
10.0).

Also report:
- target_present: true if the readout delivers the precise target in any surface form.
- extra_claims: up to 5 short verbatim quotes of the extraneous assertions you counted \
(empty list if none).

slop must be a number from 1.0 to 10.0 with at most one decimal place."""

SLOP_SCHEMA: dict[str, Any] = schema_block(
    "slop",
    {
        # (no minimum/maximum: the structured-output schema rejects numeric bounds; the prompt
        # states the 1-10 range and judge_slop clamps)
        "slop": {"type": "number"},
        "target_present": {"type": "boolean"},
        "extra_claims": {"type": "array", "items": {"type": "string"}},
        "rationale": {"type": "string"},
    },
    ["slop", "target_present", "extra_claims", "rationale"],
)


def slop_target(item: dict[str, Any]) -> str:
    """The item's canonical answer, family-agnostic.

    Judged families carry it as a field (``concept``/``attr``/``correct_text``). Mechanical
    families carry answer strings in the headline unit's ``match`` list plus the latent
    ``intermediates`` (multihop's "iron" for target "Fe") — deduped case-insensitively through
    the canonical alias filters so the judge sees each acceptable form once. Aliases are joined
    with " / ": any one of them is the precise target.
    """
    for key in ("concept", "attr", "correct_text"):
        if item.get(key):
            return str(item[key]).replace("_", " ")
    from ..olens_suite.bank.matching import content_targets, exact_targets

    raw: list[str] = []
    for unit in item.get("units", []):
        if unit.get("headline"):
            raw += [str(m) for m in unit.get("match", [])]
    raw += [str(m) for m in item.get("intermediates") or []]
    if item.get("target"):
        raw.append(str(item["target"]))
    kept = exact_targets(content_targets(raw))
    seen: dict[str, str] = {}
    for t in kept:
        seen.setdefault(t.lower(), t)
    return " / ".join(seen.values())


def mechanical_targets(item: dict[str, Any]) -> list[str]:
    """The matcher-facing target strings, mirroring ``capture_acts._bank_targets``:
    headline-unit matches + latent ``intermediates`` (+ ``target`` fallback), through the
    canonical alias filters — so a 'hit' computed from the bank matches the published
    matcher's inputs."""
    from ..olens_suite.bank.matching import content_targets, exact_targets

    units = list(item.get("units") or [])
    chosen = [u for u in units if u.get("headline")] or units
    out: list[str] = []
    for u in chosen:
        for m in u.get("match", []):
            if str(m) not in out:
                out.append(str(m))
    for m in item.get("intermediates") or ([] if out else [item.get("target")]):
        if m and str(m) not in out:
            out.append(str(m))
    return exact_targets(content_targets(out))


def slop_context(item: dict[str, Any]) -> str:
    """The prompt text the model read (rendered messages for chat-shaped items)."""
    if item.get("prompt"):
        return str(item["prompt"])
    if item.get("messages"):
        return "\n\n".join(f"[{m['role']}] {m['content']}" for m in item["messages"])
    return ""


def judge_slop(
    rows: list[tuple[dict[str, Any], str]],
    *,
    model: str = CLAUDE_JUDGE,
    concurrency: int = 256,
) -> list[dict[str, Any]]:
    """Rate ``(bank_item, readout_text)`` pairs; one row per input.

    Rows carry ``{name, target, slop, target_present, extra_claims, rationale, judge}``; a
    judge outage yields ``judge="unavailable"`` with no ``slop`` — downstream gating treats a
    missing slop as UNGATED (the precision condition can only remove credit when it actually
    ran, never silently pass/fail an unjudged row).
    """
    if not rows:
        return []
    prompts = [
        (
            SLOP_SYSTEM,
            SLOP_USER.format(context=slop_context(it), target=slop_target(it), readout=readout),
        )
        for it, readout in rows
    ]
    verdicts = async_json(prompts, schema=SLOP_SCHEMA, model=model, concurrency=concurrency)
    out: list[dict[str, Any]] = []
    for (it, _), v in zip(rows, verdicts, strict=True):
        row: dict[str, Any] = {"name": it["name"], "target": slop_target(it)}
        if v is None:
            row["judge"] = "unavailable"
        else:
            raw = v.get("slop")
            slop = round(float(raw), 1) if isinstance(raw, int | float) else None
            if slop is not None:
                slop = min(10.0, max(1.0, slop))
            row.update(
                {
                    "slop": slop,
                    "target_present": bool(v.get("target_present")),
                    "extra_claims": [str(c) for c in v.get("extra_claims") or []][:5],
                    "rationale": str(v.get("rationale", "")),
                    "judge": model,
                }
            )
        out.append(row)
    return out


def gated_hit(hit: bool, slop: float | None, threshold: float) -> bool:
    """The precision condition on ONE grid row: a hit counts only when its slop is below the
    threshold. A missing slop (judge unavailable, or the row was never judged) leaves the hit
    untouched — the gate can only ever remove credit it actually measured."""
    if not hit or slop is None:
        return hit
    return slop < threshold


def summarize_slop(
    rows: list[dict[str, Any]], thresholds: tuple[float, ...] = (3.0, 4.0, 5.0, 6.0, 7.0)
) -> dict[str, Any]:
    """Distribution + would-be-gated fractions at candidate thresholds (for picking the cut)."""
    scores = sorted(float(r["slop"]) for r in rows if r.get("slop") is not None)
    n = len(scores)
    hist = {f"{lo}-{lo + 1}": sum(1 for s in scores if lo <= s < lo + 1) for lo in range(1, 10)}
    hist["10"] = sum(1 for s in scores if s == 10.0)
    return {
        "n_judged": n,
        "n_unavailable": sum(1 for r in rows if r.get("judge") == "unavailable"),
        "mean": round(sum(scores) / n, 2) if n else None,
        "median": scores[n // 2] if n else None,
        "histogram": hist,
        "frac_gated_at": {
            str(t): round(sum(1 for s in scores if s >= t) / n, 4) if n else None
            for t in thresholds
        },
        "target_present_frac": (
            round(sum(1 for r in rows if r.get("target_present")) / n, 4) if n else None
        ),
    }
