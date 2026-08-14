"""Adapter: the committed ordered_association + entity_binding MC eval -> the normalized bundle.

This folds the ``oracle_lens_evals/oa_eb_eval`` checkpoint (``ao28500``) into two
:class:`FamilySummary` rows + per-question :class:`LensReadout`\\ s. The families are the two
multiple-choice readout-judge families from :data:`~..schema.CHANCE`:
``ordered_association`` and ``entity_binding``, both scored as "all 3 MC sub-questions correct"
(5 options + "cannot tell", 3 questions) at a single read site.

Sources (all under ``eval_dir``):

* ``verdicts_ao28500.json`` — the ORACLE-LENS arm, one row per ``(item, layer, pos)``::

      {key, id, family, layer, pos, correct:[3 bools], pass, evidence, foil_intrusion}

  ``pass`` is ``all(correct)`` for that read site; ``family`` is ``"oa"`` / ``"eb"``.
* ``verdicts_jlens.json`` — the J-LENS baseline arm, *identical* schema.
* ``prompts_pilot.json`` — a list of ``{label, family, user, prefill, targets, look_for}``; the
  ``label`` is the item id, so the stimulus text is ``user`` (``prefill`` is unused here).

Pass derivation (both arms, one metric so the arms are apples-to-apples): a question PASSES iff
some ``(layer, pos)`` read site is all-3-MC-correct (``pass == True``); its ``earliest_layer`` is
the lowest layer holding such a site. A :class:`LayerCell` hits iff any position at that layer is
all-3-correct; each :class:`PosEntry` carries the judge's ``evidence`` quote as its single sample.

IMPORTANT — why the J-lens arm is read from ``verdicts_jlens.json`` and NOT the
``eb_scores_jlens.json`` per_item map:
the ``eb_scores_*`` files carry a DIFFERENT metric (per read site: ``name`` correct, ``city``
correct, ``foil`` intrusion — effectively a 2-of-3 name&city&!foil check OR'd across positions),
which is far more lenient than the 3-MC-question ``pass`` this adapter reports (e.g. oracle
``eb_scores`` says ~39/48 pass at L60 vs 2/48 all-3-correct in the verdicts), and they cover only
``entity_binding`` (no ``ordered_association``). Reading both arms from the matched ``verdicts_*``
files keeps the metric identical across arms and covers both families. If an arm's verdicts file
is absent, that arm degrades to ``None`` per question.
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

# id-prefix / verdict-``family`` -> normalized family name (matches schema.CHANCE keys).
_FAMILY_OF: dict[str, str] = {"oa": "ordered_association", "eb": "entity_binding"}
_ORACLE_VERDICTS = "verdicts_ao28500.json"
_JLENS_VERDICTS = "verdicts_jlens.json"
_PROMPTS = "prompts_pilot.json"


def build_oa_eb(
    eval_dir: Path,
) -> tuple[list[FamilySummary], dict[str, list[Question]]]:
    """Fold the oa_eb_eval (oracle + optional J-lens) into (summaries, questions_by_family).

    ``eval_dir`` is ``outputs/oracle_lens_evals/oa_eb_eval``. The oracle arm comes from
    ``verdicts_ao28500.json`` and the J-lens arm from ``verdicts_jlens.json`` (both optional and
    degraded to ``None`` if missing); prompts come from ``prompts_pilot.json`` when present.
    """
    prompts = _load_prompts(eval_dir / _PROMPTS)
    olens = _load_arm(eval_dir / _ORACLE_VERDICTS, kind="text")
    jlens = _load_arm(eval_dir / _JLENS_VERDICTS, kind="tokens")

    # union of item ids seen in either arm, grouped by normalized family
    ids_by_family: dict[str, list[str]] = {}
    seen: set[str] = set()
    for arm in (olens, jlens):
        for item_id, (family, _ro) in arm.items():
            if item_id in seen:
                continue
            seen.add(item_id)
            ids_by_family.setdefault(family, []).append(item_id)

    questions_by_family: dict[str, list[Question]] = {}
    for family, item_ids in ids_by_family.items():
        questions: list[Question] = []
        for item_id in sorted(item_ids):
            prompt_meta = prompts.get(item_id, {})
            questions.append(
                Question(
                    name=item_id,
                    family=family,
                    prompt=str(prompt_meta.get("user", "")),
                    targets=[],  # MC family: no substring target
                    olens=olens.get(item_id, (family, None))[1],
                    jlens=jlens.get(item_id, (family, None))[1],
                    meta={
                        "look_for": str(prompt_meta.get("look_for", "")),
                        "prompt_found": item_id in prompts,
                    },
                )
            )
        questions_by_family[family] = questions

    summaries = [_family_summary(family, qs) for family, qs in sorted(questions_by_family.items())]
    return summaries, questions_by_family


def _load_arm(path: Path, *, kind: str) -> dict[str, tuple[str, LensReadout]]:
    """Read a ``verdicts_*.json`` arm into ``item_id -> (family, LensReadout)``.

    Missing file -> empty dict (arm not run). Robust to absent optional fields on a row.
    """
    if not path.exists():
        return {}
    doc: dict[str, Any] = json.loads(path.read_text())
    rows_by_item: dict[str, list[dict[str, Any]]] = {}
    family_of_item: dict[str, str] = {}
    for row in doc.get("verdicts", []):
        item_id = row.get("id")
        if item_id is None:
            continue
        rows_by_item.setdefault(item_id, []).append(row)
        raw_family = str(row.get("family", ""))
        family_of_item.setdefault(item_id, _FAMILY_OF.get(raw_family, _family_from_id(item_id)))
    return {
        item_id: (family_of_item[item_id], _readout(rows, kind=kind))
        for item_id, rows in rows_by_item.items()
    }


def _readout(rows: list[dict[str, Any]], *, kind: str) -> LensReadout:
    """Build one lens's readout from that item's ``(layer, pos)`` verdict rows."""
    by_layer_rows: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        layer = _as_int(row.get("layer"))
        if layer is None:
            continue
        by_layer_rows.setdefault(layer, []).append(row)

    by_layer: dict[str, LayerCell] = {}
    hitting: list[int] = []
    for layer in sorted(by_layer_rows):
        entries: list[PosEntry] = []
        layer_hit = False
        for row in sorted(by_layer_rows[layer], key=lambda r: _as_int(r.get("pos")) or 0):
            hit = bool(row.get("pass"))
            evidence = row.get("evidence")
            samples = [str(evidence)] if evidence else []
            entries.append(
                PosEntry(
                    pos=_as_int(row.get("pos")) or 0,
                    token="",
                    samples=samples,
                    hit=hit,
                )
            )
            layer_hit = layer_hit or hit
        by_layer[str(layer)] = LayerCell(hit=layer_hit, entries=entries)
        if layer_hit:
            hitting.append(layer)

    earliest = min(hitting) if hitting else None
    return LensReadout(
        passed=bool(hitting),
        kind=kind,  # type: ignore[arg-type]
        by_layer=by_layer,
        earliest_layer=earliest,
        verdict=_verdict(rows, earliest),
    )


def _verdict(rows: list[dict[str, Any]], earliest: int | None) -> dict[str, Any]:
    """Judge-specific extras: the representative ``correct`` triple + foil-intrusion flag.

    The triple is taken from the best read site — a passing row at ``earliest_layer`` if the item
    passed, else the row with the most sub-questions correct.
    """
    best = _best_row(rows, earliest)
    correct = best.get("correct", []) if best else []
    return {
        "correct": list(correct) if isinstance(correct, list) else [],
        "foil_intrusion": any(bool(r.get("foil_intrusion")) for r in rows),
        "n_read_sites": len(rows),
    }


def _best_row(rows: list[dict[str, Any]], earliest: int | None) -> dict[str, Any] | None:
    if not rows:
        return None
    if earliest is not None:
        passing = [r for r in rows if _as_int(r.get("layer")) == earliest and r.get("pass")]
        if passing:
            return passing[0]
    return max(rows, key=lambda r: sum(1 for c in r.get("correct", []) if c))


def _family_summary(family: str, questions: list[Question]) -> FamilySummary:
    return FamilySummary(
        family=family,
        judge_type="mc",
        n_items=len(questions),
        metric="all-3-MC-correct (best layer)",
        olens=_score([q.olens for q in questions]),
        jlens=_score([q.jlens for q in questions]),
        # chance auto-fills from schema.CHANCE for both families
    )


def _score(readouts: list[LensReadout | None]) -> LensScore | None:
    present = [r for r in readouts if r is not None]
    if not present:
        return None
    n_pass = sum(1 for r in present if r.passed)
    return LensScore(pass_rate=round(n_pass / len(present), 4), n_pass=n_pass, n_items=len(present))


def _load_prompts(path: Path) -> dict[str, dict[str, Any]]:
    """Read ``prompts_pilot.json`` (a list of ``{label, ...}``) into ``label -> entry``."""
    if not path.exists():
        return {}
    doc: Any = json.loads(path.read_text())
    entries = doc if isinstance(doc, list) else doc.get("prompts", [])
    out: dict[str, dict[str, Any]] = {}
    for entry in entries:
        label = entry.get("label")
        if isinstance(label, str):
            out[label] = entry
    return out


def _family_from_id(item_id: str) -> str:
    return "entity_binding" if item_id.startswith("eb") else "ordered_association"


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
