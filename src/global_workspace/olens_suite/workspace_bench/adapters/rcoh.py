"""Adapter: the readout-coherence (rcoh) eval -> the normalized bundle.

rcoh reads at EVERY prompt position (acts-rcoh: eval_positions = null, 137-357 positions x
7 layers x 3 samples per conversation), so unlike the bank families the full grid cannot live
in ``families/readout_coherence.json`` without bloating the always-fetched file. The adapter
therefore emits TWO artifacts per question:

* a trimmed in-family readout — only sumtok-flagged positions (effective_level >= 2 from the
  arm's ``sumtok_ao.jsonl`` verdicts), capped per layer like the safety_cases adapter, and
* a full-grid detail file ``families/readout_coherence.details/<label>.json`` =
  ``{"tokens": [...], "olens": LensReadout, "jlens": LensReadout}`` that the visualizer
  fetches lazily on card expand (``Question.meta["detail"]`` carries the relative path; the
  tokens list powers the clickable full-prompt position strip).

Pass semantics: a conversation "passes" when the arm's sumtok judge flagged at least one
position (L >= 2) — the same card criterion the standalone rcoh sites use. A missing verdicts
file degrades to passed=False everywhere (the grid still renders; a note lands in verdict).
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

FAMILY = "readout_coherence"
DETAILS_SUBDIR = f"{FAMILY}.details"


def build_rcoh(
    acts_dir: Path,
    olens_gen: Path,
    jlens_gen: Path | None = None,
    verdicts_ao: Path | None = None,
    verdicts_jlens: Path | None = None,
    *,
    max_flagged_per_layer: int = 12,
) -> tuple[list[FamilySummary], dict[str, list[Question]], dict[str, dict[str, Any]]]:
    """Fold one rcoh arm -> (summaries, questions_by_family, detail_json_by_label).

    ``verdicts_ao`` / ``verdicts_jlens`` are the ``sumtok_ao.jsonl`` / ``sumtok_jlens.jsonl``
    files (jlens flags are model-independent — the frozen run's file serves every arm).
    """
    manifest = json.loads((acts_dir / "manifest.json").read_text())
    prompts = [p for p in manifest["prompts"] if p["family"] == FAMILY]
    layers: list[int] = list(manifest["layers"])

    ao_flags = _load_flags(verdicts_ao)
    jl_flags = _load_flags(verdicts_jlens)

    questions: list[Question] = []
    details: dict[str, dict[str, Any]] = {}
    for entry in prompts:
        label = entry["label"]
        flags = ao_flags.get(label, {})
        jflags = jl_flags.get(label, {})
        olens_full = _read_lens(olens_gen, label, layers, flags)
        jlens_full = (
            _read_lens(jlens_gen, label, layers, jflags, kind="tokens") if jlens_gen else None
        )
        if olens_full is None:
            continue
        olens_trim = _trim(olens_full, flags, cap=max_flagged_per_layer)
        jlens_trim = _trim(jlens_full, jflags, cap=max_flagged_per_layer) if jlens_full else None
        n_flag = sum(1 for lv in flags.values() if lv >= 2)
        olens_trim.verdict = {
            "judge": "sumtok (AO strict)",
            "n_pos": entry.get("n_pos"),
            "flagged": n_flag,
            "l3": sum(1 for lv in flags.values() if lv >= 3),
        } | ({} if flags else {"note": "no sumtok verdicts for this arm — hits unavailable"})
        details[label] = {
            "tokens": list(entry.get("tokens", [])),
            "olens": olens_full.to_json(),
            "jlens": jlens_full.to_json() if jlens_full else None,
        }
        questions.append(
            Question(
                name=label,
                family=FAMILY,
                prompt="".join(entry.get("tokens", [])),
                targets=[],
                olens=olens_trim,
                jlens=jlens_trim,
                meta={
                    "detail": f"{DETAILS_SUBDIR}/{label}.json",
                    "n_pos": entry.get("n_pos"),
                    "flagged": n_flag,
                },
            )
        )

    summary = FamilySummary(
        family=FAMILY,
        judge_type="freetext",
        n_items=len(questions),
        metric="convs with >=1 sumtok-flagged (L>=2) readout position",
        olens=_score([q.olens for q in questions]),
        jlens=_score([q.jlens for q in questions]),
    )
    return [summary], {FAMILY: questions}, details


def write_details(run_dir: Path, details: dict[str, dict[str, Any]]) -> Path:
    """Write the per-question full-grid files next to the family JSON."""
    out = run_dir / "families" / DETAILS_SUBDIR
    out.mkdir(parents=True, exist_ok=True)
    for label, payload in details.items():
        tmp = out / f"{label}.json.tmp"
        tmp.write_text(json.dumps(payload, ensure_ascii=False))
        tmp.replace(out / f"{label}.json")
    return out


def _load_flags(verdicts: Path | None) -> dict[str, dict[int, int]]:
    """sumtok verdicts jsonl -> {label -> {pos -> effective_level}} (api_error rows dropped)."""
    if verdicts is None or not verdicts.exists():
        return {}
    flags: dict[str, dict[int, int]] = {}
    for line in verdicts.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("api_error"):
            continue
        flags.setdefault(row["label"], {})[int(row["pos"])] = int(row.get("effective_level") or 0)
    return flags


def _read_lens(
    gen_dir: Path | None,
    label: str,
    layers: list[int],
    flags: dict[int, int],
    *,
    kind: str = "text",
) -> LensReadout | None:
    if gen_dir is None:
        return None
    by_layer: dict[str, LayerCell] = {}
    any_rows = False
    for layer in layers:
        f = gen_dir / label / f"L{layer:03d}.jsonl"
        if not f.exists():
            continue
        entries: list[PosEntry] = []
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            hit = flags.get(int(row["pos"]), 0) >= 2
            entries.append(PosEntry(row["pos"], row.get("token", ""), row["samples"], hit))
        if not entries:
            continue
        any_rows = True
        by_layer[str(layer)] = LayerCell(any(e.hit for e in entries), entries)
    if not any_rows:
        return None
    return LensReadout(
        passed=any(cell.hit for cell in by_layer.values()),
        kind=kind,  # type: ignore[arg-type]
        by_layer=by_layer,
        earliest_layer=None,  # flags are per-position (judge pools layers), so no layer onset
    )


def _trim(readout: LensReadout, flags: dict[int, int], *, cap: int) -> LensReadout:
    """Flagged-positions-only copy of a full readout (highest level first, then position)."""
    by_layer: dict[str, LayerCell] = {}
    for layer, cell in readout.by_layer.items():
        kept = [e for e in cell.entries if flags.get(e.pos, 0) >= 2]
        kept.sort(key=lambda e: (-flags.get(e.pos, 0), e.pos))
        if kept:
            by_layer[layer] = LayerCell(cell.hit, kept[:cap])
    return LensReadout(
        passed=readout.passed,
        kind=readout.kind,
        by_layer=by_layer,
        earliest_layer=None,
    )


def _score(readouts: list[LensReadout | None]) -> LensScore | None:
    present = [r for r in readouts if r is not None]
    if not present:
        return None
    n_pass = sum(1 for r in present if r.passed)
    return LensScore(pass_rate=round(n_pass / len(present), 4), n_pass=n_pass, n_items=len(present))
