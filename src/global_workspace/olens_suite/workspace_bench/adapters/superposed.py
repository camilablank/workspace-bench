"""Adapter: the committed *superposed* directed-modulation readouts -> the normalized bundle.

The superposed domain tells the model to hold a phrase in mind while it writes an unrelated
dictated sentence, then reads the activations at the write positions. Two lenses are folded here:

* **Oracle lens** (free text) — ``read.json`` (combined dictation + binding) or the split
  ``dictation_ao.json`` + ``binding_ao.json``. Each record is ``{name, rels, tokens, ao}`` with
  ``ao["<layer>"][i]`` the ``k`` free-text samples read at cell ``rels[i]`` (negative offsets,
  stored nearest-last-token-first; ``tokens[i]`` is the token at that cell).
* **J-lens** (top-k tokens) — ``binding_jlens.json`` + ``dictation_jlens.json``. Each record is
  ``{meta:{name,prompt,concepts,stratum}, tokens, jlens:{"<layer>":[per-pos [top-k tokens]]}}``.
  Audit 2026-08-15: the capture was verified to be WRITE cells (negative rels indexing into the
  last-20 generated dictation tokens), not prompt positions — but it carries no region labels,
  so no IN_SENTENCE gate is applied.

Pass semantics reuse the PACKAGED scorer :mod:`global_workspace.olens_suite.superposed.score`:
:func:`~...score.load_reads` labels every oracle cell's region and
:func:`~...score.in_sentence_activations` restricts headline claims to IN_SENTENCE cells; a concept
is "surfaced" by :func:`~...score.concept_hits` (word-boundary match over its
:func:`~...spec.content_words`). A question **passes** a lens iff at least one of its concepts is
surfaced by that lens:

* oracle: any concept word-matched by a sample of an IN_SENTENCE cell (identical to the scorer's
  ``per_item_union > 0``; off-task items such as ``d-petrichor`` have no IN_SENTENCE cell and so
  never pass);
* J-lens: any concept word-matched by the top-k tokens at any (layer, position) — the same
  word-boundary rule, applied at every captured write cell (no region gate available).

Control items (``d-none`` / ``b-none``) dictate nothing, so their concept list is empty and both
arms are ``pass=False`` by construction; they are kept as questions, not dropped.

The oracle metadata (prompt / concepts / stratum) lives on the ``read.json`` records only in the
raw ``results/superposed/`` copy; the finalized readout copies (``read.json`` and the ``*_ao``
splits alike) carry only ``{name, rels, tokens, ao}``. Metadata is therefore resolved per record
name from the oracle record when present, else from the J-lens ``meta`` block.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from global_workspace.olens_suite.superposed.regions import Region
from global_workspace.olens_suite.superposed.score import (
    ItemRead,
    concept_hits,
    load_reads,
)
from global_workspace.olens_suite.superposed.spec import content_words
from global_workspace.olens_suite.workspace_bench.schema import (
    FamilySummary,
    LayerCell,
    LensReadout,
    LensScore,
    PosEntry,
    Question,
)
from global_workspace.readout_text import strip_scaffolding

_FAMILY = "superposed"
_METRIC = "≥1 concept surfaced"  # ">=1 concept surfaced"


def build_superposed(
    readouts_dir: Path,
) -> tuple[list[FamilySummary], dict[str, list[Question]]]:
    """Fold the committed superposed readouts into ``([FamilySummary], {family: [Question]})``.

    ``readouts_dir`` holds the oracle readouts — ``read.json`` (combined) or ``dictation_ao.json``
    + ``binding_ao.json`` — and the J-lens readouts ``dictation_jlens.json`` +
    ``binding_jlens.json``. Either lens may be absent; a missing lens sets that arm to ``None``
    for every question (and its :class:`LensScore` to ``None``), never a fake 0.
    """
    olens_blob, olens_layers = _load_olens(readouts_dir)
    jlens_recs, jlens_layers = _load_jlens(readouts_dir)

    meta = _resolve_metadata(olens_blob, jlens_recs)
    olens_reads = _load_reads(olens_blob, meta)

    questions: list[Question] = []
    for name in meta:
        prompt, concepts, stratum = meta[name]
        olens_ro = _olens_readout(olens_reads[name], olens_layers) if name in olens_reads else None
        jlens_ro = (
            _jlens_readout(jlens_recs[name], concepts, jlens_layers)
            if jlens_recs is not None and name in jlens_recs
            else None
        )
        questions.append(
            Question(
                name=name,
                family=_FAMILY,
                prompt=prompt,
                targets=[str(c) for c in concepts],
                olens=olens_ro,
                jlens=jlens_ro,
                meta={"stratum": stratum, "concepts": list(concepts)},
            )
        )

    summary = FamilySummary(
        family=_FAMILY,
        judge_type="deterministic",
        n_items=len(questions),
        metric=_METRIC,
        olens=_score([q.olens for q in questions]),
        jlens=_score([q.jlens for q in questions]),
        chance=None,
    )
    return [summary], {_FAMILY: questions}


# --- oracle lens (free text): region-gated concept surfacing via the packaged scorer -----------


def _olens_readout(read: ItemRead, layers: list[int]) -> LensReadout:
    """One oracle-lens readout: a cell hits iff IN_SENTENCE and a sample surfaces a concept."""
    by_layer: dict[str, LayerCell] = {}
    hitting: list[int] = []
    for layer in layers:
        entries: list[PosEntry] = []
        layer_hit = False
        for i, rel in enumerate(read.rels):
            samples = [strip_scaffolding(s) for s in read.samples[layer][i]]
            in_sentence = read.regions[i] is Region.IN_SENTENCE
            hit = in_sentence and any(
                concept_hits(s, words) for s in samples for words in read.concepts
            )
            entries.append(PosEntry(pos=rel, token="", samples=samples, hit=hit))
            layer_hit = layer_hit or hit
        by_layer[str(layer)] = LayerCell(layer_hit, entries)
        if layer_hit:
            hitting.append(layer)
    return LensReadout(
        passed=bool(hitting),
        kind="text",
        by_layer=by_layer,
        earliest_layer=min(hitting) if hitting else None,
    )


# --- J-lens (top-k tokens): same word rule, no region gate (reads the prompt) ------------------


def _jlens_readout(rec: dict[str, Any], concepts: list[str], layers: list[int]) -> LensReadout:
    """One J-lens readout: a cell hits iff its top-k tokens word-match any concept."""
    concept_words = [content_words(c) for c in concepts]
    tokens: list[str] = list(rec.get("tokens", []))
    jlens: dict[str, list[list[str]]] = rec["jlens"]
    by_layer: dict[str, LayerCell] = {}
    hitting: list[int] = []
    for layer in layers:
        col = jlens.get(str(layer))
        if col is None:
            continue
        entries: list[PosEntry] = []
        layer_hit = False
        for i, topk in enumerate(col):
            joined = " ".join(topk)
            hit = any(concept_hits(joined, words) for words in concept_words)
            token = tokens[i] if i < len(tokens) else ""
            entries.append(PosEntry(pos=i, token=token, samples=list(topk), hit=hit))
            layer_hit = layer_hit or hit
        by_layer[str(layer)] = LayerCell(layer_hit, entries)
        if layer_hit:
            hitting.append(layer)
    return LensReadout(
        passed=bool(hitting),
        kind="tokens",
        by_layer=by_layer,
        earliest_layer=min(hitting) if hitting else None,
    )


def _score(readouts: list[LensReadout | None]) -> LensScore | None:
    present = [r for r in readouts if r is not None]
    if not present:
        return None
    n_pass = sum(1 for r in present if r.passed)
    return LensScore(pass_rate=round(n_pass / len(present), 4), n_pass=n_pass, n_items=len(present))


# --- loaders for the committed readout files (no I/O beyond reading the dir) --------------------


def _load_olens(readouts_dir: Path) -> tuple[dict[str, Any] | None, list[int]]:
    """The oracle readouts as a single ``{layers, records}`` blob, or ``(None, [])`` if absent.

    Prefers the combined ``read.json``; otherwise concatenates the ``*_ao`` splits.
    """
    combined = readouts_dir / "read.json"
    if combined.exists():
        blob: dict[str, Any] = json.loads(combined.read_text())
        return blob, [int(x) for x in blob["layers"]]
    records: list[dict[str, Any]] = []
    layers: list[int] = []
    for fname in ("dictation_ao.json", "binding_ao.json"):
        path = readouts_dir / fname
        if path.exists():
            part = json.loads(path.read_text())
            layers = [int(x) for x in part["layers"]]
            records.extend(part["records"])
    if not records:
        return None, []
    return {"layers": layers, "records": records}, layers


def _load_jlens(readouts_dir: Path) -> tuple[dict[str, dict[str, Any]] | None, list[int]]:
    """The J-lens readouts keyed by record name, or ``(None, [])`` if neither file is present."""
    records: dict[str, dict[str, Any]] = {}
    layers: list[int] = []
    for fname in ("dictation_jlens.json", "binding_jlens.json"):
        path = readouts_dir / fname
        if path.exists():
            part = json.loads(path.read_text())
            layers = [int(x) for x in part["layers"]]
            for rec in part["records"]:
                records[rec["meta"]["name"]] = rec
    if not records:
        return None, []
    return records, layers


def _resolve_metadata(
    olens_blob: dict[str, Any] | None,
    jlens_recs: dict[str, dict[str, Any]] | None,
) -> dict[str, tuple[str, list[str], str]]:
    """Per record name, resolve ``(prompt, concepts, stratum)`` — oracle record first, else J-lens.

    Insertion order is oracle records first (dictation then binding), then any J-lens-only names.
    """
    meta: dict[str, tuple[str, list[str], str]] = {}
    if olens_blob is not None:
        for rec in olens_blob["records"]:
            name = rec["name"]
            meta[name] = (
                str(rec.get("prompt", "")),
                [str(c) for c in rec.get("concepts", [])],
                str(rec.get("stratum", "")),
            )
    if jlens_recs is not None:
        for name, rec in jlens_recs.items():
            jmeta = rec.get("meta", {})
            prompt, concepts, stratum = meta.get(name, ("", [], ""))
            meta[name] = (
                prompt or str(jmeta.get("prompt", "")),
                concepts or [str(c) for c in jmeta.get("concepts", [])],
                stratum or str(jmeta.get("stratum", "")),
            )
    return meta


def _load_reads(
    olens_blob: dict[str, Any] | None,
    meta: dict[str, tuple[str, list[str], str]],
) -> dict[str, ItemRead]:
    """Region-labelled :class:`ItemRead` per oracle record name (empty if no oracle arm)."""
    if olens_blob is None:
        return {}
    items = [
        {"name": name, "prompt": prompt, "concepts": concepts, "stratum": stratum}
        for name, (prompt, concepts, stratum) in meta.items()
    ]
    reads = load_reads(olens_blob, items, ao_path="read.json", items_path="<jlens meta>")
    return {r.name: r for r in reads}
