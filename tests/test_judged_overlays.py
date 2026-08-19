"""CPU tests pinning the audit-2026-08-19 scoring fixes in the bundle adapters + bank judge.

What these pin:
* overlay_judged ignores the permutation-null *foil* probes, mirrors each family's documented
  strict predicate, folds screen verdicts for non-escalated grid points, and never flips
  judge-uncovered items to fail (they keep the mechanical proxy).
* build_ethical / oa_eb draw the ANY-over-grid chance floor, not the per-call chance.
* build_superposed excludes controls and write-non-compliant items from BOTH arms' rates
  (matching the packaged scorer), verified on the shipped readout fixture.
* bank_judge verifies quotes against the readout samples only — a hallucinated 'quote' of the
  target (verbatim in the prompt's targets header) must not pass the quote gate.
"""

import json
from pathlib import Path
from typing import Any

from global_workspace.olens_suite.workspace_bench.adapters.judged import (
    build_ethical,
    overlay_judged,
)
from global_workspace.olens_suite.workspace_bench.adapters.oa_eb import _family_summary
from global_workspace.olens_suite.workspace_bench.adapters.superposed import build_superposed
from global_workspace.olens_suite.workspace_bench.bank_judge import quote_verified
from global_workspace.olens_suite.workspace_bench.schema import (
    FamilySummary,
    LensReadout,
    Question,
)

REPO = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------ overlay_judged


def _question(name: str, family: str, proxy_pass: bool) -> Question:
    return Question(
        name=name,
        family=family,
        prompt="",
        targets=[],
        olens=LensReadout(passed=proxy_pass, kind="text", by_layer={}, earliest_layer=None),
        jlens=None,
        meta={},
    )


def _write_judge_file(
    gen_dir: Path, family: str, verdicts: list[dict[str, Any]],
    screen_verdicts: list[dict[str, Any]] | None = None,
) -> None:
    d = gen_dir / "judge"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{family}.json").write_text(
        json.dumps({"verdicts": verdicts, "screen_verdicts": screen_verdicts or [],
                    "summary": {}})
    )


def _um_row(name: str, probe: str, basis: str, layer: int = 44) -> dict[str, Any]:
    return {"name": name, "probe": probe, "encoded": "CORRECT", "basis": basis,
            "layer": layer, "pos": 0, "sample_idx": 0}


def test_overlay_ignores_foil_probes_and_echo_basis(tmp_path: Path) -> None:
    fam = "user-modeling"
    questions = {fam: [_question("um-a", fam, proxy_pass=False),
                       _question("um-b", fam, proxy_pass=False)]}
    summaries = [FamilySummary(family=fam, judge_type="deterministic", n_items=2, metric="proxy", olens=None, jlens=None)]
    _write_judge_file(
        tmp_path, fam,
        [
            # um-a: only the FOIL false-alarms + a verbatim echo — neither may count
            _um_row("um-a", "um_attribute_foil", "inferred_characterization"),
            _um_row("um-a", "um_attribute", "verbatim_echo"),
            # um-b: the real probe, strict basis — counts
            _um_row("um-b", "um_attribute", "inferred_characterization"),
        ],
    )
    overlay_judged(summaries, questions, tmp_path, families=(fam,))
    by_name = {q.name: q for q in questions[fam]}
    assert by_name["um-a"].olens.passed is False
    assert by_name["um-b"].olens.passed is True
    assert summaries[0].olens.n_pass == 1


def test_overlay_sandbagging_strict_includes_generic_boilerplate(tmp_path: Path) -> None:
    fam = "sandbagging"
    questions = {fam: [_question("sb-a", fam, proxy_pass=False)]}
    summaries = [FamilySummary(family=fam, judge_type="deterministic", n_items=1, metric="proxy", olens=None, jlens=None)]
    _write_judge_file(
        tmp_path, fam,
        [{"name": "sb-a", "probe": "sb_motivation", "motivation": "SAFETY",
          "basis": "generic_boilerplate", "layer": 44, "pos": 0, "sample_idx": 0}],
    )
    overlay_judged(summaries, questions, tmp_path, families=(fam,))
    # post-PR-181 safety_strict: generic_boilerplate is IN the strict basis set
    assert questions[fam][0].olens.passed is True


def test_overlay_folds_screen_verdicts_and_keeps_proxy_for_uncovered(tmp_path: Path) -> None:
    fam = "user-modeling"
    questions = {fam: [
        _question("um-frontier", fam, proxy_pass=False),
        _question("um-screen-only", fam, proxy_pass=False),   # screen-negative, not escalated
        _question("um-uncovered", fam, proxy_pass=True),      # judge never saw it: keep proxy
    ]}
    summaries = [FamilySummary(family=fam, judge_type="deterministic", n_items=3, metric="proxy", olens=None, jlens=None)]
    frontier = [_um_row("um-frontier", "um_attribute", "inferred_characterization")]
    screen = [
        # same grid point as the frontier row -> superseded, must NOT double in
        _um_row("um-frontier", "um_attribute", "verbatim_echo"),
        # a non-escalated point: folds in as a (negative) verdict -> item stays failed
        {**_um_row("um-screen-only", "um_attribute", "inferred_characterization"),
         "encoded": "WRONG"},
    ]
    _write_judge_file(tmp_path, fam, frontier, screen)
    overlay_judged(summaries, questions, tmp_path, families=(fam,))
    by_name = {q.name: q for q in questions[fam]}
    assert by_name["um-frontier"].olens.passed is True
    assert by_name["um-screen-only"].olens.passed is False
    assert by_name["um-screen-only"].olens.verdict == {"judged": True, "basis": ""}
    assert by_name["um-uncovered"].olens.passed is True  # proxy preserved
    assert not by_name["um-uncovered"].olens.verdict  # no judge verdict attached
    assert "1 items on proxy" in summaries[0].metric


# ------------------------------------------------------------------ chance floors


def test_build_ethical_chance_is_any_of_grid_floor(tmp_path: Path) -> None:
    # committed item judged at 30 grid points: the floor is 1-(5/6)^30 ~= 0.9958, not 1/6
    verdicts = [
        {"id": "ec-1", "layer": 44, "pos": p, "side": "committed", "correct": False,
         "reason_class": "committed"}
        for p in range(30)
    ]
    blob = {"aggregate": {"per_item": {"ec-1": {"class": "committed", "pass_any": False}}},
            "verdicts": verdicts}
    path = tmp_path / "verdicts_ao.json"
    path.write_text(json.dumps(blob))
    summaries, _questions = build_ethical(path)
    (s,) = summaries
    assert s.family == "ethical-committed"
    assert abs(s.chance - (1 - (5 / 6) ** 30)) < 1e-9
    assert "floor" in (s.chance_label or "")


def test_oa_eb_chance_is_any_of_grid_floor() -> None:
    ro = LensReadout(passed=False, kind="text", by_layer={}, earliest_layer=None,
                     verdict={"n_read_sites": 36})
    q = Question(name="oa-1", family="ordered_association", prompt="", targets=[],
                 olens=ro, jlens=None, meta={})
    s = _family_summary("ordered_association", [q])
    per_site = (1 / 6) ** 3
    assert abs(s.chance - (1 - (1 - per_site) ** 36)) < 1e-9
    assert s.chance > per_site  # strictly above the per-site line it replaces


# ------------------------------------------------------------------ superposed adapter


def test_build_superposed_excludes_controls_and_noncompliant_from_both_arms() -> None:
    readouts = REPO / "hillclimbing_evals/superposed/readouts"
    summaries, questions = build_superposed(readouts)
    (s,) = summaries
    qs = questions["superposed"]
    excluded = {q.name: q.meta["excluded_from_rate"] for q in qs
                if "excluded_from_rate" in q.meta}
    # the two dictate-nothing controls + the write-non-compliant off-task item
    assert set(excluded) == {"d-none", "b-none", "d-petrichor"}
    assert "control" in excluded["d-none"]
    assert "write-non-compliant" in excluded["d-petrichor"]
    # all 27 items remain browsable, but the rates run over the 24 scored items — both arms
    assert s.n_items == 27
    assert s.olens is not None and s.olens.n_items == 24
    assert s.jlens is not None and s.jlens.n_items == 24


# ------------------------------------------------------------------ bank_judge quote gate


def test_bank_judge_quote_verified_against_samples_only() -> None:
    samples_text = "the sky was clear\na cat sat outside"
    assert quote_verified("cat sat", samples_text)
    assert quote_verified("CAT   sat", samples_text)  # whitespace/case-normalized fallback
    # the target word appears verbatim in the prompt's targets header but NOT in any sample —
    # quoting it must fail the gate
    assert not quote_verified("february", samples_text)
    assert not quote_verified("", samples_text)
