"""Merge several adapter outputs into one bundle's (summaries, questions_by_family).

Every adapter returns the same pair type — ``(list[FamilySummary], dict[family -> list[Question]])``
— so composing a multi-source bundle (e.g. the sglang bank families + the bespoke order_ops /
buggy_code / superposed families, all for one checkpoint) is a signature-agnostic concat with a
family-name collision guard. Family names are the merge key and must be unique across sources.
"""

from __future__ import annotations

from global_workspace.olens_suite.workspace_bench.schema import FamilySummary, Question

Source = tuple[list[FamilySummary], dict[str, list["Question"]]]


def merge(*sources: Source) -> Source:
    """Concatenate adapter outputs. Raises on a duplicate family name across sources (they must
    not silently overwrite — a collision means two adapters claim the same family and the caller
    must rename or pick one)."""
    summaries: list[FamilySummary] = []
    questions: dict[str, list[Question]] = {}
    seen: set[str] = set()
    for fam_summaries, qbf in sources:
        for fs in fam_summaries:
            if fs.family in seen:
                raise ValueError(
                    f"duplicate family {fs.family!r} across sources — rename or drop one"
                )
            seen.add(fs.family)
            summaries.append(fs)
        for fam, qs in qbf.items():
            questions[fam] = qs
    summaries.sort(key=lambda s: s.family)
    return summaries, questions
