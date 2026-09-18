"""Ask a model its own bank questions and grade the answers: the gate every bank was built with,
re-run on a new model. See AGENTS.md."""

import hashlib
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from wsbench.cache import Cache
from wsbench.judge_config import JudgeConfig, ResolvedJudge, resolve
from wsbench.llm import Spend, schema_block
from wsbench.mcjudge import Call, Preflighter, run_calls

from .questions import Question, build

PROMPT_VERSION = "capable-2026-09-18"
DEFAULT_DRAWS = 10
DEFAULT_TEMPERATURE = 0.7
DEFAULT_THRESHOLD = 0.8  # the source repo's >= 8/10 gate
MAX_TOKENS = 2048  # room for a reasoning model's hidden tokens before the answer

ANSWER_SYSTEM = (
    "You are answering benchmark questions. Answer the question as asked, as briefly as it "
    "allows, with no preamble, no explanation and no restatement of the question. If the "
    "question asks you to continue a text, reply with the continuation only."
)
ANSWER_SCHEMA = schema_block("answer", {"answer": {"type": "string"}}, ["answer"])

GRADE_SYSTEM = (
    "You grade a model's ANSWER against the EXPECTED answer to the same question. Mark it "
    "correct when the answer gives the expected one: the same word or value, an inflection of "
    "it, an established synonym or alias, a faithful translation into another language, or the "
    "same quantity written differently. A longer answer that contains the expected one is "
    "correct; an answer that merely mentions the topic, or names something related but "
    "different, is not. Judge only the answer in front of you."
)
GRADE_SCHEMA = schema_block(
    "grade", {"correct": {"type": "boolean"}, "why": {"type": "string"}}, ["correct", "why"]
)


def render_grade(question: Question, answer: str) -> str:
    expected = "\n".join(f"- {g}" for g in question.golds)
    return f"QUESTION:\n{question.ask}\n\nEXPECTED (any one of):\n{expected}\n\nANSWER:\n{answer}"


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def run_family(
    family: str,
    *,
    model: str,
    judge: ResolvedJudge,
    out: Path,
    draws: int = DEFAULT_DRAWS,
    temperature: float = DEFAULT_TEMPERATURE,
    threshold: float = DEFAULT_THRESHOLD,
    limit: int = 0,
    concurrency: int = 64,
    rpm: float = 240.0,
    dry_run: bool = False,
) -> dict[str, Any]:
    """``draws`` answers per item from ``model``, each graded by ``judge``. Identical answers to
    one item are graded once. Returns the per-item records and the aggregate; both are written by
    the caller. Answers and grades are cached under ``out/cells.jsonl``, so a re-run resumes."""
    questions = build(family)[: limit or None]
    answerer = resolve(JudgeConfig(model=model, prompt_version=PROMPT_VERSION), flag=model)
    spend = Spend()
    pre = Preflighter(dry_run)
    answers: dict[tuple[str, int], str | None] = {}
    grades: dict[str, bool | None] = {}

    with Cache(out / "cells.jsonl") as cache:
        calls = [
            Call(
                key=f"{family}|{q.id}|d{d}",
                system=ANSWER_SYSTEM,
                user=q.ask,
                meta={"item": q.id, "draw": d},
            )
            for q in questions
            for d in range(draws)
        ]
        replies = run_calls(
            calls,
            schema=ANSWER_SCHEMA,
            judge=answerer,
            prompt_version=PROMPT_VERSION,
            cache=cache,
            spend=spend,
            concurrency=concurrency,
            rpm=rpm,
            dry_run=dry_run,
            preflight=pre.for_judge(answerer),
            temperature=temperature,
            max_tokens=MAX_TOKENS,
        )
        if dry_run:
            return {"family": family, "model": model, "dry_run": True, "n_items": len(questions)}
        for q in questions:
            for d in range(draws):
                r = replies.get(f"{family}|{q.id}|d{d}")
                text = str(r.get("answer", "")).strip() if isinstance(r, dict) else None
                answers[(q.id, d)] = text or None

        by_id = {q.id: q for q in questions}
        wanted = {
            (q_id, text)
            for (q_id, _d), text in answers.items()
            if text is not None  # one grade per distinct answer to an item
        }
        grade_calls = [
            Call(
                key=f"{family}|{q_id}|g{_digest(text)}",
                system=GRADE_SYSTEM,
                user=render_grade(by_id[q_id], text),
                meta={"item": q_id, "answer": text},
            )
            for q_id, text in sorted(wanted)
        ]
        verdicts = run_calls(
            grade_calls,
            schema=GRADE_SCHEMA,
            judge=judge,
            prompt_version=PROMPT_VERSION,
            cache=cache,
            spend=spend,
            concurrency=concurrency,
            rpm=rpm,
            dry_run=False,
            preflight=pre.for_judge(judge),
            temperature=0.0,
        )
        for c in grade_calls:
            v = verdicts.get(c.key)
            grades[c.key] = bool(v["correct"]) if isinstance(v, dict) else None

    rows = []
    for q in questions:
        drawn = [answers[(q.id, d)] for d in range(draws)]
        graded = [grades.get(f"{family}|{q.id}|g{_digest(t)}") for t in drawn if t is not None]
        decided = [g for g in graded if g is not None]
        rate = statistics.fmean(float(g) for g in decided) if decided else None
        rows.append(
            {
                "id": q.id,
                "golds": q.golds,
                "answers": drawn,
                "n_decided": len(decided),
                "n_correct": sum(decided),
                "rate": rate,
                "passes_gate": None if rate is None else rate >= threshold,
                **({"meta": q.meta} if q.meta else {}),
            }
        )

    decided_rows = [r for r in rows if r["rate"] is not None]
    n_fail = sum(1 for r in rows for a in r["answers"] if a is None)
    return {
        "family": family,
        "model": model,
        "grader": judge.model,
        "prompt_version": PROMPT_VERSION,
        "draws": draws,
        "temperature": temperature,
        "threshold": threshold,
        "aggregate": {
            "gate_rate": _mean(float(r["passes_gate"]) for r in decided_rows),
            "mean_accuracy": _mean(r["rate"] for r in decided_rows),
            "n_items": len(rows),
            "n_items_decided": len(decided_rows),
            "n_answers_failed": n_fail,
            "spend_usd": round(spend.usd, 4),
        },
        "rows": rows,
    }


def _mean(values: Any) -> float | None:
    vals = [float(v) for v in values]
    return statistics.fmean(vals) if vals else None


def report_line(result: dict[str, Any]) -> str:
    a = result["aggregate"]
    gate = "—" if a["gate_rate"] is None else f"{a['gate_rate']:.3f}"
    acc = "—" if a["mean_accuracy"] is None else f"{a['mean_accuracy']:.3f}"
    return (
        f"{result['family']} on {result['model']}: gate={gate} accuracy={acc} "
        f"n={a['n_items_decided']}/{a['n_items']} failed={a['n_answers_failed']} "
        f"spend=${a['spend_usd']:.2f}"
    )


def subfamily_rates(result: dict[str, Any], key: str) -> dict[str, float]:
    """Mean accuracy split by a ``meta`` key (buggy code's arm, user modelling's subfamily)."""
    by: dict[str, list[float]] = {}
    for r in result["rows"]:
        value = (r.get("meta") or {}).get(key)
        if value is not None and r["rate"] is not None:
            by.setdefault(str(value), []).append(r["rate"])
    return {k: statistics.fmean(v) for k, v in sorted(by.items())}


def answer_histogram(result: dict[str, Any], top: int = 5) -> list[tuple[str, int]]:
    """The most common answers across the run: a quick read on a model that refuses or loops."""
    counts: Counter[str] = Counter()
    for r in result["rows"]:
        counts.update(a for a in r["answers"] if a)
    return counts.most_common(top)
