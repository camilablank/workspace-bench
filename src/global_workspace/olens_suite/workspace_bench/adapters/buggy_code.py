"""Adapter: the committed buggy_code oracle-lens run -> the normalized bundle.

buggy_code is a gated free-text family. Each item is a short program (buggy or a matched
clean control); the oracle lens is read at one EOF-anchored layer and asked to say what the
code *does*. A human/LLM judge then places each item's best read on a consequence ladder
(``S0``..``S4``): a question **passes** when its rung is ``S2`` or higher — i.e. the lens read
the executed *consequence* of the code, not merely its surface topic. There is no J-lens
baseline for this family, so every ``jlens`` arm is ``None``.

Inputs (all committed, no GPU needed):

* ``<results_dir>/read.json`` — the raw reads: ``config`` (olens/scale/model), per-lang ``cells``
  (anchor + layer), and ``records[]`` with the free-text ``readout`` samples per item.
* ``<results_dir>/judged_verdicts.json`` — one verdict per item: the ladder ``rung`` plus the
  judge's ``quote``/``why``/``notable`` and the ``noise``/``donor`` control rungs.
* ``bank_file`` — the read bank (``{"buggy": [...], "clean": [...]}``); each entry carries the
  ``code`` (the prompt), the executed-truth ``verified`` string, and the ``cause``.

Items with no matching verdict fail (``pass=False``). Free-text samples are stripped of their
``<explanation>`` scaffolding via :func:`global_workspace.readout_text.strip_scaffolding`, the
same helper the sglang adapter uses, so the emitted samples match the visualizer's expectations.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from global_workspace.olens_suite.workspace_bench.schema import (
    FamilySummary,
    LayerCell,
    LensReadout,
    LensScore,
    PosEntry,
    Question,
)
from global_workspace.readout_text import strip_scaffolding

FAMILY = "buggy_code"

# Ladder rungs at which the executed consequence was read (vs mere topic/domain echo).
_PASS_RUNGS = {"S2", "S3", "S4"}

# Default read bank shipped with the eval (override via the ``bank_file`` argument).
_DEFAULT_BANK = Path(
    "hillclimbing_evals/buggy_code/read_bank.json"
)

# Judge-verdict fields carried through onto each question's ``verdict`` extras.
_VERDICT_FIELDS = ("rung", "quote", "why", "notable", "noise_rung", "donor_rung")


def build_buggy_code(
    results_dir: Path,
    bank_file: Path = _DEFAULT_BANK,
) -> tuple[list[FamilySummary], dict[str, list[Question]]]:
    """Fold the committed buggy_code results into ``(summaries, questions_by_family)``.

    One family, ``"buggy_code"`` (src / lang_group live in each question's ``meta``, not as
    separate families). ``pass`` is the judged rung being ``>= S2``.
    """
    read = _load_json(results_dir / "read.json")
    verdicts = _index_verdicts(_load_json(results_dir / "judged_verdicts.json"))
    bank = _index_bank(_load_json(bank_file))

    questions: list[Question] = []
    for record in read["records"]:
        name = record["name"]
        questions.append(_build_question(record, verdicts.get(name), bank.get(name)))

    questions_by_family = {FAMILY: questions}
    summary = FamilySummary(
        family=FAMILY,
        judge_type="gated_freetext",
        n_items=len(questions),
        metric="rung ≥ S2 (consequence read)",
        olens=_score([q.olens for q in questions]),
        jlens=None,
    )
    return [summary], questions_by_family


def _build_question(
    record: dict[str, Any],
    verdict: dict[str, Any] | None,
    bank_entry: dict[str, Any] | None,
) -> Question:
    passed = verdict is not None and verdict.get("rung") in _PASS_RUNGS
    layer = int(record["layer"])

    bank_entry = bank_entry or {}
    cause = bank_entry.get("cause") or record.get("cause") or ""
    prompt = bank_entry.get("code") or cause

    targets: list[str] = []
    for candidate in (bank_entry.get("verified"), cause):
        if candidate and candidate not in targets:
            targets.append(candidate)

    samples = [strip_scaffolding(s) for s in record.get("readout", [])]
    entry = PosEntry(pos=0, token="", samples=samples, hit=passed)
    verdict_extras: dict[str, Any] = {"src": record["src"]}
    if verdict is not None:
        verdict_extras.update({k: verdict.get(k) for k in _VERDICT_FIELDS})

    olens = LensReadout(
        passed=passed,
        kind="text",
        by_layer={str(layer): LayerCell(hit=passed, entries=[entry])},
        earliest_layer=layer if passed else None,
        verdict=verdict_extras,
    )
    return Question(
        name=record["name"],
        family=FAMILY,
        prompt=prompt,
        targets=targets,
        olens=olens,
        jlens=None,
        meta={
            "src": record["src"],
            "lang_group": record["lang_group"],
            "cause": cause,
        },
    )


def _score(readouts: list[LensReadout | None]) -> LensScore | None:
    present = [r for r in readouts if r is not None]
    if not present:
        return None
    n_pass = sum(1 for r in present if r.passed)
    return LensScore(
        pass_rate=round(n_pass / len(present), 4), n_pass=n_pass, n_items=len(present)
    )


def _index_verdicts(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row["name"]: row for row in rows}


def _index_bank(bank: dict[str, Any]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for src in ("buggy", "clean"):
        for entry in bank.get(src, []):
            index[entry["name"]] = entry
    return index


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())
