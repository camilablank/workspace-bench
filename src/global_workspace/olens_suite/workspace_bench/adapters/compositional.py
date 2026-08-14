"""Adapter: the compositional_association latent-eval (10-way MC) -> the normalized bundle.

The instrument is ``scripts/oracle_lens/latent_eval/judge_mc.py``: for each item it aggregates
the readout across all captured layers into ONE blob, then asks the judge a single 10-way (+1
"cannot tell" escape = 11 lines) multiple-choice question — which frozen ``mc_options`` the
readout states. Item PASS = the judge picks ``gold_label`` (``pick == "gold"``).

Its outputs are per-item verdict files (``{"tag","n","pass","breakdown","per_item"}``) with
``per_item[id] = {choice, gold_pos, pick, correct, quote, quote_ok}``. Two arms live side by
side in ``latent_eval_dir``:

* ``verdicts_mc_teacher.json`` — the ORACLE-LENS arm (``tag == "teacher"``); folded as ``olens``.
* ``verdicts_mc_jlens.json``   — the J-LENS baseline (``tag == "jlens"``);   folded as ``jlens``.

The sibling ``contrastive_*.json`` files are a DIFFERENT (2AFC, ``{gold,flipped}`` yes/no)
instrument, not the MC verdicts, and ``gpu_gates_full*.json`` are upstream GPU gates — neither
is read here.

Because the judge already collapses layers into one verdict per item, this data is *not*
layered: item pass is ``pick == "gold"`` taken directly (no best-layer search), a question's
readout is a single cell, and ``earliest_layer`` stays ``None``. Any arm without a verdict for
an item -> ``None`` for that arm.

Self-contained on purpose (mirrors :mod:`sglang`): it re-implements the judge's seeded option
shuffle so it can recover the chosen option's *label* without importing the script-dir module,
keeping the adapter usable in a plain CPU/viz-build env.
"""

from __future__ import annotations

import hashlib
import json
import random
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

FAMILY = "compositional_association"

# Must match scripts/oracle_lens/latent_eval/judge_mc.py exactly so the reconstructed option
# order (and hence the chosen label) is identical to what the judge saw.
_COMP_SEED = 20260805
_CANNOT = "cannot tell from the readout"

# Verdict file name -> bundle arm key, in the order the judge_mc tags name them.
_OLENS_FILE = "verdicts_mc_teacher.json"
_JLENS_FILE = "verdicts_mc_jlens.json"

_METRIC = "MC choice == gold (1/11 chance)"
# Single non-layered cell key (the judge collapsed all layers into one readout per item).
_CELL = "all"


def build_compositional(
    latent_eval_dir: Path,
    items_file: Path,
) -> tuple[list[FamilySummary], dict[str, list[Question]]]:
    """Fold the compositional_association MC latent-eval into (summaries, questions_by_family).

    ``latent_eval_dir`` holds the judge_mc verdict files (``verdicts_mc_teacher.json`` = olens,
    ``verdicts_mc_jlens.json`` = jlens); ``items_file`` is the frozen ``items.json`` bank used to
    recover each item's prompt and gold label. Missing verdict files degrade to a ``None`` arm.
    """
    items = json.loads(Path(items_file).read_text())
    olens_verdicts = _load_verdicts(latent_eval_dir / _OLENS_FILE)
    jlens_verdicts = _load_verdicts(latent_eval_dir / _JLENS_FILE)

    questions: list[Question] = []
    for it in items:
        iid = it["id"]
        olens_ro = _readout(it, olens_verdicts.get(iid))
        jlens_ro = _readout(it, jlens_verdicts.get(iid))
        if olens_ro is None and jlens_ro is None:
            continue
        questions.append(
            Question(
                name=iid,
                family=FAMILY,
                prompt=str(it.get("stimulus", "")),
                targets=[str(it["gold_label"])] if it.get("gold_label") else [],
                olens=olens_ro,
                jlens=jlens_ro,
                meta={
                    "cluster": it.get("cluster"),
                    "level": it.get("level"),
                    "anchor": it.get("anchor"),
                    "contrast_label": it.get("contrast_label"),
                },
            )
        )

    summary = FamilySummary(
        family=FAMILY,
        judge_type="mc",
        n_items=len(questions),
        metric=_METRIC,
        olens=_score([q.olens for q in questions]),
        jlens=_score([q.jlens for q in questions]),
    )
    return [summary], {FAMILY: questions}


def _readout(item: dict[str, Any], verdict: dict[str, Any] | None) -> LensReadout | None:
    """One arm's readout for one item, or None if that arm has no verdict."""
    if verdict is None:
        return None
    pick = verdict.get("pick")
    passed = pick == "gold"
    choice = verdict.get("choice")
    quote = (verdict.get("quote") or "").strip()
    label = _choice_label(item, choice)

    entries = [PosEntry(pos=0, token="", samples=[quote] if quote else [], hit=passed)]
    return LensReadout(
        passed=passed,
        kind="text",
        by_layer={_CELL: LayerCell(hit=passed, entries=entries)},
        earliest_layer=None,  # judge_mc collapses all layers into one verdict — not layered
        verdict={"choice": choice, "label": label, "pick": pick, "quote": quote},
    )


def _choice_label(item: dict[str, Any], choice: Any) -> str | None:
    """Recover the shown option text for a 1-based ``choice`` (mirrors judge_mc.build_question)."""
    opts = list(dict.fromkeys(item.get("mc_options", [])))  # dedupe, preserve order
    if not opts:
        return None
    shown = [*_seeded_shuffle(item["id"], opts), _CANNOT]
    if not isinstance(choice, int) or not (1 <= choice <= len(shown)):
        return None
    return shown[choice - 1]


def _seeded_shuffle(key: str, opts: list[str]) -> list[str]:
    seed = int.from_bytes(hashlib.sha256(f"{_COMP_SEED}:{key}".encode()).digest()[:8], "big")
    order = list(range(len(opts)))
    random.Random(seed).shuffle(order)
    return [opts[j] for j in order]


def _score(readouts: list[LensReadout | None]) -> LensScore | None:
    present = [r for r in readouts if r is not None]
    if not present:
        return None
    n_pass = sum(1 for r in present if r.passed)
    return LensScore(
        pass_rate=round(n_pass / len(present), 4), n_pass=n_pass, n_items=len(present)
    )


def _load_verdicts(path: Path) -> dict[str, dict[str, Any]]:
    """Read a judge_mc verdict file's ``per_item`` map, or {} if the file is absent."""
    if not path.exists():
        return {}
    data: dict[str, Any] = json.loads(path.read_text())
    per_item = data.get("per_item", {})
    return {str(k): v for k, v in per_item.items()}
