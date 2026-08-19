"""CPU tests for the judged-family slop gate's rescore path (no API).

Pins the invariants that were violated by a real bug on 2026-08-19: the gate can only ever
REMOVE credit (gated <= the number of judge-passing owners), and an ethical-deliberative item
counts only when BOTH sides have a passing cell that is also clean — a one-sided item never
passed the family judge, so the reconstruction must not resurrect it via a missing-side
default.
"""

import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts/oracle_lens_evals"
sys.path.insert(0, str(_SCRIPTS))

from slop_gate_judged import rescore  # noqa: E402


def _artifact() -> dict:
    return {
        "arm": "test-arm",
        "threshold": 5.0,
        "compositional": {"n": 4, "ungated": 0.75, "gated": 0.0},
        "ethical-committed": {"n": 2, "ungated": 1.0, "gated": 0.0},
        "ethical-deliberative": {"n": 2, "ungated": 0.5, "gated": 0.0},
        "user-modeling": {
            "n": 2,
            "ungated_strict": 1.0,
            "foil": 0.5,
            "ungated_net": 0.5,
            "gated_strict": 0.0,
            "gated_foil": 0.0,
            "gated_net": 0.0,
        },
        "audit_rows": {
            # 3 judge-passing items: c1 clean @<7, c2 dirty everywhere, c3 clean only @<10
            "compositional": [
                {"owner": "c1", "slop": 6.0},
                {"owner": "c1", "slop": 10.0},
                {"owner": "c2", "slop": 9.5},
                {"owner": "c3", "slop": 9.9},
            ],
            "ethical_consequences": [
                {"owner": "e1|committed", "slop": 2.0},
                {"owner": "e2|committed", "slop": 9.5},
                # d1 passed BOTH sides (owners on both), but only yes is clean
                {"owner": "d1|yes", "slop": 3.0},
                {"owner": "d1|no", "slop": 9.5},
                # d2 has only a yes-side cell -> it never passed the deliberative judge and
                # must NOT count, no matter how clean the one side is (the 2026-08-19 bug)
                {"owner": "d2|yes", "slop": 1.0},
            ],
            "user-modeling": [
                {"owner": "S:u1", "slop": 2.0},
                {"owner": "S:u2", "slop": 9.5},
                {"owner": "F:u1", "slop": 9.5},  # junk foil credit -> gated away
            ],
        },
    }


def test_rescore_recomputes_all_families_and_never_adds_credit(tmp_path: Path) -> None:
    (tmp_path / "test-arm.json").write_text(json.dumps(_artifact()))
    rescore(7.0, out_dir=tmp_path, arms=("test-arm",))
    d = json.loads((tmp_path / "test-arm.json").read_text())
    assert d["threshold"] == 7.0
    # compositional: only c1 has a cell < 7 -> 1/4; and gated <= ungated always
    assert d["compositional"]["gated"] == 0.25
    assert d["compositional"]["gated"] <= d["compositional"]["ungated"]
    # committed: e1 clean, e2 not -> 1/2
    assert d["ethical-committed"]["gated"] == 0.5
    # deliberative: d1 fails (no-side dirty), d2 must NOT count (one-sided) -> 0/2
    assert d["ethical-deliberative"]["gated"] == 0.0
    # nets: strict u1 clean -> 1/2; foil credit was junk -> gated away -> net = 0.5
    assert d["user-modeling"]["gated_strict"] == 0.5
    assert d["user-modeling"]["gated_foil"] == 0.0
    assert d["user-modeling"]["gated_net"] == 0.5


def test_rescore_at_10_readmits_only_sub_10_cells(tmp_path: Path) -> None:
    (tmp_path / "test-arm.json").write_text(json.dumps(_artifact()))
    rescore(10.0, out_dir=tmp_path, arms=("test-arm",))
    d = json.loads((tmp_path / "test-arm.json").read_text())
    assert d["compositional"]["gated"] == 0.75  # c1, c2, c3 all have cells < 10
    assert d["ethical-deliberative"]["gated"] == 0.5  # d1 both sides < 10; d2 still one-sided
    assert d["user-modeling"]["gated_net"] == 1.0 - 0.5  # both strict clean, foil readmitted
