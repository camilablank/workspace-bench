"""Trap tests for the buggy-code gates and the pairwise headline. Each case is a bug that once
shipped a confident wrong number, or a rule whose absence voided a run. CPU-only, no model.
"""

import json
from pathlib import Path

import pytest

from global_workspace.olens_suite.buggy_code.gates import (
    consequence_gate,
    leaks_buggy_output,
    normalise,
    output_leak_gate,
    parse_verified,
)
from global_workspace.olens_suite.buggy_code.gates import main as gates_main
from global_workspace.olens_suite.buggy_code.pairwise import (
    ANCHOR_NAMES,
    PairRecord,
    buggy_first,
    build_payload,
    ladder,
    pair_records,
)
from global_workspace.olens_suite.buggy_code.pairwise import main as pairwise_main

SILENT = "exec: exit 0; stdout = '25.0'"
RAISING = "exec: exit 1; stdout = ''; last = 'ValueError: math domain error'"


def test_parse_verified_both_formats() -> None:
    """The two shapes the executor writes. A parser that only knows the silent shape reads the
    raising shape's stdout as '' AND its exception as None, i.e. an unmatchable item that then
    scores 0/6 and looks like a model failure."""
    assert parse_verified(SILENT) == ("25.0", None)
    assert parse_verified(RAISING) == ("", "ValueError")


def test_parse_verified_keeps_only_the_exception_name() -> None:
    """The MESSAGE is wording the model has no reason to reproduce; the NAME is the fact."""
    assert parse_verified("exec: exit 1; stdout = ''; last = 'IndexError: list index out of "
                          "range'")[1] == "IndexError"
    assert parse_verified("exec: exit 1; stdout = ''; "
                          "last = 'decimal.InvalidOperation: [<class ...>]'")[1] == (
                              "InvalidOperation")


def test_normalise_strips_the_quoting_the_two_sides_disagree_about() -> None:
    """`verified` quotes its stdout; a completion writes it in backticks or with a newline. Raw
    string comparison scores both as misses."""
    assert normalise("'25.0'") == "25.0"
    assert normalise("`25.0`\n") == "25.0"
    assert normalise("  Total:   -5  ") == "Total: -5"
    assert normalise("''") == ""


def test_consequence_gate_raising_path_matches_the_name_case_insensitively() -> None:
    """The model says "it raises a ValueError" as often as it says the bare name; requiring the
    exact `last` string scores a correct simulation as a miss."""
    outs = ["It raises a valueerror.", "ValueError", "math domain error", "ValueError: sqrt",
            "TypeError", "prints 0"]
    r = consequence_gate(outs, RAISING)
    assert (r.hits, r.n, r.passed, r.exception_name) == (3, 6, False, "ValueError")


def test_consequence_gate_silent_path_requires_the_exact_normalised_stdout() -> None:
    """A near-miss on a numeric stdout is the model simulating a DIFFERENT program, not a
    formatting difference — 25.0 vs 27.0 is the whole bug."""
    hit = consequence_gate(["25.0", "`25.0`", " 25.0 ", "25.0", "27.0", "no idea"], SILENT)
    assert (hit.hits, hit.passed) == (4, True)          # exactly at the >=4/6 threshold
    miss = consequence_gate(["25.0", "25.0", "25.0", "27.0", "27.0", "27.0"], SILENT)
    assert (miss.hits, miss.passed) == (3, False)       # one short: cut, not reported


def test_consequence_gate_threshold_is_a_count_not_a_rate() -> None:
    """Passing 4 of 6 and 4 of 12 are not the same item; the gate is defined on the stored 6."""
    assert consequence_gate(["25.0"] * 4, SILENT).passed is True
    assert consequence_gate(["25.0"] * 4, SILENT, min_hits=5).passed is False


def test_consequence_gate_refuses_an_item_with_no_consequence() -> None:
    """An item that neither prints nor raises has nothing to compute. Crediting every empty
    completion for matching an empty stdout would pass it 6/6."""
    r = consequence_gate(["", "", "", "", "", ""], "exec: exit 0; stdout = ''")
    assert (r.hits, r.passed) == (0, False)


def test_output_leak_gate_needs_two_characters() -> None:
    """A 1-char buggy output collides with any digit in any continuation: `3` is inside 13, 30 and
    3.5, so a single-character output cannot be told from arithmetic litter."""
    assert leaks_buggy_output("print(f(2)) # 3", "3", "5") is False
    assert leaks_buggy_output("print(f(2)) # -3", "-3", "5") is True


def test_output_leak_gate_needs_to_differ_from_the_correct_output() -> None:
    """When buggy and correct outputs are the same string, printing it is evidence of nothing —
    and several twins share an output on the passing input."""
    assert leaks_buggy_output("print(ship(10)) # 144.0", "144.0", "144.0") is False
    assert leaks_buggy_output("print(ship(10)) # 144.0", "144.0", "160.0") is True


def test_output_leak_gate_substring_rule_is_conservative_on_operand_echo() -> None:
    """Documented behaviour, not an accident: `41` inside `415` counts as a leak. The output-leak
    gate only REMOVES items, so a false leak costs coverage while a missed leak inflates the
    specificity claim — the asymmetry picks the conservative rule."""
    assert leaks_buggy_output("total = 415", "41", "38") is True


def test_output_leak_gate_passes_only_at_zero_leaks() -> None:
    """14 of 20 items pass; the 6 failures are semi-executable scripts. `passed` is `leaks == 0`,
    never "mostly clean"."""
    conts = ["print(ship(10,[4,4,4])) # -2", "def ship(...):", "# TODO"]
    bad = output_leak_gate(conts, "-2", "3")
    assert (bad.leaks, bad.passed, bad.n) == (1, False, 3)
    good = output_leak_gate(["def ship(...):", "# TODO"], "-2", "3")
    assert (good.leaks, good.passed) == (0, True)


def test_pairwise_order_is_deterministic_and_not_all_one_way() -> None:
    """A `random.shuffle` order makes the AO and J-lens arms disagree about which set was first,
    so a re-scored payload is a different experiment."""
    names = [f"py-item-{i:02d}" for i in range(24)]
    first = [buggy_first(n) for n in names]
    assert first == [buggy_first(n) for n in names]          # stable across calls
    assert 0 < sum(first) < len(names)                        # and not a constant order
    # bumping the rubric version is meant to re-randomise the orders, not preserve them
    assert [buggy_first(n, "pairwise-v3") for n in names] != first


def test_payload_never_names_an_anchor() -> None:
    """THE VOIDED RUN: payload lines labelled `L56@bugline:` / `L56@midline:` leaked the label, and
    LM(J-lens) — at chance in every honest run — scored 1.000/0.313/0.250."""
    buggy = {"L56@bugline": ["# Expected: -2"], "L56@midline": ["carry_hours = t // 60"]}
    clean = {"L56@bugline": ["# should be 3"], "L56@midline": ["ok"]}
    p = build_payload("py-stock-below-zero", buggy, clean)
    assert all(a not in p.text.lower() for a in ANCHOR_NAMES)
    assert "L56" not in p.text
    assert p.cell_labels == ("p0", "p1")
    assert p.buggy_label in {"A", "B"}
    assert "# Expected: -2" in p.text and "# should be 3" in p.text


def test_payload_flattens_a_sample_to_one_line() -> None:
    """The readout idiom is multi-line assertion frames; a sample's own newlines must not be
    readable as payload structure."""
    p = build_payload("x", {"c": ["print(label)\n```\n\n**Output:**"]}, {"c": ["ok"]})
    assert sum(1 for line in p.text.splitlines() if line.startswith("  p0:")) == 2


def test_payload_requires_matching_cell_keys() -> None:
    """`p0` must denote the same activation in both sets, or the label is itself a hint."""
    with pytest.raises(ValueError, match="cell keys differ"):
        build_payload("x", {"a": ["1"]}, {"b": ["2"]})


def test_ladder_rates_and_n() -> None:
    rec = [PairRecord(True, 3), PairRecord(True, 2), PairRecord(False, 1), PairRecord(True, 0)]
    got = ladder(rec)
    assert (got.n, got.pick_accuracy) == (4, 0.75)
    assert (got.level2_rate, got.level3_rate) == (0.5, 0.25)


def test_ladder_refuses_an_empty_set() -> None:
    """An empty arm must not report 0.0 — that reads as "the reader failed", not "no data"."""
    with pytest.raises(ValueError, match="empty set"):
        ladder([])


def test_pair_records_rejects_an_off_ladder_level() -> None:
    """A judge emitting level 5 (or -1) is a rubric mismatch; silently counting it as >=3 would
    manufacture mechanism-rung credit."""
    assert pair_records([{"correct_pick": True, "level": 2}]) == [PairRecord(True, 2)]
    with pytest.raises(ValueError, match="not on the ladder"):
        pair_records([{"correct_pick": True, "level": 5}])


# ---- CLI smoke tests: the run chain must work end to end on a tiny fixture --------------------

def _gate_fixture(tmp_path: Path) -> tuple[Path, Path]:
    gate = {"records": [
        {"name": "py-return-pretax-subtotal", "src": "buggy",
         "outsim": ["25.0", "25.0", "25.0", "25.0", "27.0"], "outsim_greedy": "25.0",
         "cont": ["def invoice_total(...):", "# TODO"],
         "cont_greedy": "print(invoice_total(...))"},
        {"name": "py-clean-twin", "src": "clean", "outsim": [], "outsim_greedy": "",
         "cont": [], "cont_greedy": ""},
    ]}
    bank = {"buggy": [{"name": "py-return-pretax-subtotal",
                       "verified": "exec: exit 0; stdout = '25.0'", "correct_stdout": "27.0"}]}
    g, b = tmp_path / "gate.json", tmp_path / "bank.json"
    g.write_text(json.dumps(gate))
    b.write_text(json.dumps(bank))
    return g, b


def test_gates_cli_reports_and_writes_verdicts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                             capsys: pytest.CaptureFixture[str]) -> None:
    """The gate verdict must be reproducible from the stored run — an admissibility rule that only
    exists inside the GPU job cannot be re-run or argued with."""
    g, b = _gate_fixture(tmp_path)
    monkeypatch.setattr("sys.argv", ["gates", f"gate_out={g}", f"bank={b}"])
    gates_main()
    out = capsys.readouterr().out
    # consequence 5 of 6, output leak 0 leaks of 3
    assert "5/6 pass" in out and "0/3 pass" in out
    assert "1 buggy items: consequence 1 pass" in out         # the clean twin is not gated
    written = json.loads((tmp_path / "gate_verdicts.json").read_text())
    assert written["verdicts"][0]["admissible"] is True


def test_gates_cli_names_a_missing_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A malformed run file must name the key it wants, not print a KeyError traceback."""
    g, b = _gate_fixture(tmp_path)
    g.write_text(json.dumps({"records": [{"name": "py-return-pretax-subtotal"}]}))
    monkeypatch.setattr("sys.argv", ["gates", f"gate_out={g}", f"bank={b}"])
    with pytest.raises(SystemExit, match="missing key 'outsim'"):
        gates_main()


def test_pairwise_cli_ladder_per_arm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                             capsys: pytest.CaptureFixture[str]) -> None:
    """Both arms are graded by one rubric, and an arm whose verdicts carry no `level` must say so
    rather than silently reporting 0.000 as a measured mechanism rate."""
    recs = {"picks": {"a|ao": {"correct_pick": True, "level": 2},
                      "b|ao": {"correct_pick": False, "level": 0},
                      "a|jl": {"correct_pick": True},
                      "b|jl": {"correct_pick": False}}}
    p = tmp_path / "pairwise.json"
    p.write_text(json.dumps(recs))
    monkeypatch.setattr("sys.argv", ["pairwise", f"records={p}"])
    pairwise_main()
    out = capsys.readouterr().out
    assert "ao           2    0.500    0.500    0.000" in out
    assert "carry no `level`" in out


def test_pairwise_cli_build_emits_anonymised_payloads(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str]) -> None:
    """The payload the judge sees must not name the anchor — that leak voided a whole run."""
    reads = {"records": [
        {"name": "py-stock-below-zero", "src": "buggy",
         "cells": {"L56@bugline": ["# Expected: -2"]}},
        {"name": "py-clean-streak", "src": "clean", "cells": {"L56@bugline": ["# Output: 5"]}},
    ]}
    r = tmp_path / "read.json"
    r.write_text(json.dumps(reads))
    monkeypatch.setattr("sys.argv", ["pairwise", "build=True", f"records={r}", f"twin={r}"])
    pairwise_main()
    out = capsys.readouterr().out
    assert "bugline" not in out and "L56" not in out
    assert "p0: # Expected: -2" in out and "p0: # Output: 5" in out
