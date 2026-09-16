"""Goldens for agentic_misalignment from the SOURCE ``narrative_judge.py`` (stdlib-only, imported
by path) and driver: the raw prompt templates, one rendered call per stage on fixed inputs, and
``aggregate_design`` / ``bootstrap_mean`` / ``bootstrap_macro`` on a fixed record set.

Run: cd <this repo> && uv run --no-sync python tests/golden/make_agentic_misalignment.py
(after tests/golden/make_agentic_bank.py, whose bank supplies the Stage C scenario).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _source import HERE, R

BANK = HERE.parents[1] / "evals/agentic_misalignment/items.json"

# fixed inputs shared with tests/test_agentic_misalignment.py
READOUTS = {
    36: "",
    20: "You are Alex, an email agent at SummitBridge.",
    44: " I will use the affair ",
}
NOTES = [
    (0, "The model is reading a system prompt. voice: instruction"),
    (1, "uninformative"),
    (2, "Uninformative."),
    (3, "It plans to email Kyle. voice: own intention"),
    (7, "uninformative"),
]
PARTIAL_LO, PARTIAL_HI = 3, 7
ACCOUNT = "TOY ACCOUNT\nCANDIDATE ACTIONS: 1. email Kyle (committed, enabled, high) [pos 3]"
SCENARIO_ITEM = "rh-unit-tests"

RECORDS = [
    {"id": "a", "arm": "misaligned", "family": "blackmail", "design_fidelity": 3, "design_rank": 1,
     "commit_strength": 3, "misalignment_rate": 1.0, "asserts_misaligned_plan": True,
     "origin_match": True},
    {"id": "b", "arm": "misaligned", "family": "blackmail", "design_fidelity": 1, "design_rank": 2,
     "commit_strength": 1, "misalignment_rate": 0.25, "asserts_misaligned_plan": True,
     "origin_match": False},
    {"id": "c", "arm": "misaligned", "family": "leaking", "design_fidelity": 0, "design_rank": 0,
     "commit_strength": 0, "misalignment_rate": None, "asserts_misaligned_plan": False,
     "origin_match": False},
    {"id": "d", "arm": "misaligned", "family": "murder", "design_fidelity": None,
     "design_rank": None, "commit_strength": None, "misalignment_rate": 0.5,
     "asserts_misaligned_plan": None, "origin_match": False},
    {"id": "e", "arm": "misaligned", "family": "murder", "design_fidelity": 2, "design_rank": 3,
     "commit_strength": 2, "misalignment_rate": 1.0, "asserts_misaligned_plan": True,
     "origin_match": True},
    {"id": "f", "arm": "misaligned", "family": "reward_hacking", "design_fidelity": 3,
     "design_rank": 1, "commit_strength": 3, "misalignment_rate": 0.875,
     "asserts_misaligned_plan": True, "origin_match": True},
    {"id": "g", "arm": "control", "family": "blackmail", "design_fidelity": 0, "design_rank": 0,
     "commit_strength": 2, "misalignment_rate": None, "asserts_misaligned_plan": True,
     "origin_match": False},
    {"id": "h", "arm": "control", "family": "murder", "design_fidelity": 0, "design_rank": 0,
     "commit_strength": 0, "misalignment_rate": None, "asserts_misaligned_plan": False,
     "origin_match": True},
]  # fmt: skip


def load_narrative_judge() -> ModuleType:
    """``_source.load_script`` plus a ``sys.modules`` entry: the source module defines a
    ``@dataclass`` under ``from __future__ import annotations``, which Python 3.13 resolves
    through ``sys.modules[cls.__module__]``."""
    name = "narrative_judge"
    spec = importlib.util.spec_from_file_location(
        name, R / "src/global_workspace/diagnostic/narrative_judge.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    nj = load_narrative_judge()
    bank = json.loads(BANK.read_text(encoding="utf-8"))
    it = next(i for i in bank["items"] if i["id"] == SCENARIO_ITEM)
    notes = [nj.PositionNote(p, n) for p, n in NOTES]
    scenario = f"{it['system']}\n\n---\n\n{it['text']}"  # driver L300
    rendered = {
        "position": nj.POSITION_PROMPT.format(
            readout_kind=nj.READOUT_KIND["olens"], readouts=nj.format_readouts(READOUTS)
        ),
        "position_jlens_kind": nj.READOUT_KIND["jlens"],
        "partial": nj.PARTIAL_PROMPT.format(
            lo=PARTIAL_LO, hi=PARTIAL_HI, notes=nj.format_notes(notes)
        ),
        "consolidate": nj.CONSOLIDATE_PROMPT.format(notes=nj.format_notes(notes)),
        "design_score": nj.DESIGN_SCORE_PROMPT.format(  # driver L302-307
            account=ACCOUNT,
            families=", ".join(nj.FAMILIES),
            descriptor=it["descriptor"],
            scenario=scenario[:24000],
        ),
    }
    templates = {
        "POSITION_PROMPT": nj.POSITION_PROMPT,
        "CONSOLIDATE_PROMPT": nj.CONSOLIDATE_PROMPT,
        "PARTIAL_PROMPT": nj.PARTIAL_PROMPT,
        "DESIGN_SCORE_PROMPT": nj.DESIGN_SCORE_PROMPT,
        "FAMILIES": list(nj.FAMILIES),
        "NONE_OPTION": nj.NONE_OPTION,
        "ORIGINS": list(nj.ORIGINS),
        "READOUT_KIND": dict(nj.READOUT_KIND),
    }
    chunks = nj.chunk_notes(notes, 80)
    helpers = {
        "chunk_notes_80": [[[n.pos, n.note] for n in ch] for ch in chunks],
        "format_notes": nj.format_notes(notes),
        "format_readouts": nj.format_readouts(READOUTS),
    }
    scoring = {
        "aggregate_design": nj.aggregate_design(RECORDS),
        "bootstrap_mean": nj.bootstrap_mean(
            [r for r in RECORDS if r["arm"] == "misaligned"], nj.design_item_score
        ),
        # the source's macro_score keys on the retired actual-mode `fidelity`; feeding it
        # fidelity = design_fidelity makes it the design-mode macro, so the port's
        # bootstrap_macro (keyed on design_item_score) must reproduce the same draws.
        "bootstrap_macro": list(
            nj.bootstrap_macro(
                [
                    {**r, "fidelity": r["design_fidelity"]}
                    for r in RECORDS
                    if r["arm"] == "misaligned"
                ]
            )
        ),
    }
    (HERE / "agentic_misalignment_prompts.json").write_text(
        json.dumps(
            {"templates": templates, "rendered": rendered, "helpers": helpers},
            indent=1,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (HERE / "agentic_misalignment_scoring.json").write_text(
        json.dumps({"records": RECORDS, **scoring}, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print("wrote agentic_misalignment_prompts.json, agentic_misalignment_scoring.json")


if __name__ == "__main__":
    main()
