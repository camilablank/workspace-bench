"""Trap tests for the superposed domain: region labelling, word matching, capacity, specificity.
Each case is a bug that once produced a confident wrong number. CPU-only, no model, no network.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from global_workspace.olens_suite.superposed.regions import (
    Region,
    classify_regions,
    headline_indices,
    label_cells,
    write_compliance,
)
from global_workspace.olens_suite.superposed.score import (
    Activation,
    capacity,
    concept_hits,
    main,
    matches_word,
    specificity_screen,
)
from global_workspace.olens_suite.superposed.spec import content_words, render_prompt

# The shape every binding item's completion has: sentence, blank line, empty think block, then the
# model re-writing the prompt. rel -1..-4 land in that re-write for 14 of the 17 binding items.
WITH_THINK = ["review", " the", " annual", " budget", '."', "\n\n", "<think>", "\n\n", "</think>",
              "\n\n", "Think", " about"]
# d-cranes: no think block at all, and a self-added meta-continuation as the tail.
NO_THINK = ["review", " the", " annual", " budget", '."', "\n\n", "Now", " write", " this"]
# d-petrichor: the model ignored the instruction and wrote its own sentence instead.
OFF_TASK = ["is", " called", " petrichor", ".", " It", " is", " a", " scent", " that", " many"]


def test_classify_regions_with_a_think_block() -> None:
    """POST_THINK is contaminated on the SAME footing as IN_THINK: of the 47 leaking Gate-A
    continuations, the first restatement lands after `</think>` in 29 cases (13 inside, 5 before).
    Labelling the re-write region as in-sentence would put those cells under the headline."""
    got = classify_regions(WITH_THINK)
    assert got[:5] == [Region.IN_SENTENCE] * 5
    assert got[5:9] == [Region.IN_THINK] * 4      # the blank line before <think> is NOT in-sentence
    assert got[9:] == [Region.POST_THINK] * 3
    assert headline_indices(got) == [0, 1, 2, 3, 4]


def test_classify_regions_without_a_think_block() -> None:
    """d-cranes never opens a think block; its tail is instruction-shaped text the model added
    itself. Scoring that tail as in-sentence credits the lens for reading text that is there."""
    got = classify_regions(NO_THINK)
    assert got == [Region.IN_SENTENCE] * 5 + [Region.SELF_ADDED_META] * 4


def test_classify_regions_window_entirely_inside_the_sentence() -> None:
    """A 20-cell window that never leaves the written sentence is all headline-usable."""
    assert classify_regions(["will", " meet", " on", " Thursday"]) == [Region.IN_SENTENCE] * 4


def test_word_boundary_rejects_cross_inside_across() -> None:
    """THE BUG: a substring rule matched `cross` inside "across" and inflated a chain item's union
    from 2 concepts to 3. The leading \\b rejects it; the missing trailing one keeps stem
    tolerance, which is how the dictated element actually comes back."""
    assert matches_word("she looked across the room", "cross") is False
    assert matches_word("then cross the bridge", "cross") is True
    assert matches_word("crossing the bridge", "cross") is True     # stem-tolerant on purpose
    assert matches_word("a violin FLOATING in honey", "float") is True
    assert matches_word("the librarian's umbrella", "librar") is True


def test_concept_hits_is_any_word_of_the_concept() -> None:
    assert concept_hits("imagery of a floating violin", ["violin", "floating"]) is True
    assert concept_hits("the committee will meet", ["violin", "floating"]) is False


def test_capacity_separates_its_three_levels() -> None:
    """The headline "capacity ~ 2" is the per-activation MAX (1.88 of 3), not the typical
    activation (mean 0.16; 363 of 412 in-sentence activations decode nothing). Reporting one
    number for all three levels is the error this decomposition exists to prevent."""
    concepts = [["violin"], ["floating"], ["honey"]]
    acts = [
        Activation(cell=-12, layer=44, samples=("imagery of a violin",)),
        Activation(cell=-12, layer=52, samples=("something floating, unrelated task",)),
        Activation(cell=-14, layer=44, samples=("a jar of honey",)),
        Activation(cell=-14, layer=52, samples=("the committee will meet",)),
    ]
    got = capacity(acts, concepts)
    assert got.per_activation_max == 1        # no single activation ever decodes two
    assert got.per_cell_union == 2            # pooling the layers of one cell buys the second
    assert got.per_item_union == 3            # pooling every activation buys the third
    assert got.per_activation_mean == 0.75    # and the typical activation holds well under one
    assert got.n_activations == 4


def test_capacity_refuses_an_item_with_no_activations() -> None:
    """d-petrichor has NO in-sentence cells — it wrote its own petrichor sentence instead of the
    dictated one. It must be excluded, not scored 0 (its raw hit count is 401 of 480)."""
    with pytest.raises(ValueError, match="no activations"):
        capacity([], [["petrichor"]])


def test_specificity_screen_returns_both_rates() -> None:
    """The screen killed a "blue ladder read back as a RED ladder" drift claim: `blue` has a base
    rate of 13/1000 in this template's context, so the ratio, not the raw count, is the evidence —
    and both rates are reported because a "hit" can be a single occurrence."""
    own = ["blue ladder", "a blue rung", "blue", "the committee will meet"]
    other = ["blue skies over the budget", *["the committee will meet"] * 9]
    got = specificity_screen("blue", own, other)
    assert (got.hits, got.n, got.rate) == (3, 4, 0.75)
    assert (got.base_hits, got.base_n, got.base_rate) == (1, 10, 0.1)
    assert got.passed is True


def test_specificity_screen_min_hits_binds_when_the_base_rate_is_zero() -> None:
    """A zero base rate makes the ratio test vacuous, so a 2-hit word must NOT pass on it: at k=6
    per cell, two hits is inside sampling noise."""
    got = specificity_screen("kite", ["a paper kite", "kite"], ["the committee will meet"] * 10)
    assert (got.hits, got.base_rate, got.passed) == (2, 0.0, False)


def test_content_word_rule_drops_the_short_nouns_it_says_it_drops() -> None:
    """Stated cost, applied uniformly: `fox` (3 letters) is dropped, so "a silver fox" is scored
    through `silver` alone. A rule that silently kept some short words would make the per-item
    capacity numbers incomparable."""
    assert content_words("a silver fox chasing a paper kite") == ("silver", "chasing", "paper",
                                                                 "kite")
    assert content_words("the plumber's blue ladder leaning against the mango tree") == (
        "plumber", "blue", "ladder", "mango", "tree")


def test_write_compliance_accepts_a_window_inside_the_dictated_sentence() -> None:
    """The window is 20 tokens and may start mid-sentence, so the test is a contiguous RUN of the
    target's words, not equality — and not bag overlap, which any English text passes on "the"."""
    got = write_compliance(NO_THINK)
    assert got.complied is True
    assert got.matched == "review the annual budget"
    assert got.run == 4


def test_write_compliance_rejects_petrichor_style_off_task_text() -> None:
    """THE BUG: d-petrichor wrote its OWN petrichor sentence, so all 480 of its readouts are the
    lens reading text that is literally present. Counting it inflated the typical-activation mean
    from 0.16 to 0.27 — a number that had to be retracted."""
    got = write_compliance(OFF_TASK)
    assert got.complied is False
    assert got.run < got.min_run


def test_off_task_item_has_no_in_sentence_cells() -> None:
    """An off-task window is OFF_TASK throughout — a distinct label from SELF_ADDED_META, which
    means the sentence WAS written and its cells are still valid (d-cranes scores normally)."""
    regions, compliance = label_cells(OFF_TASK)
    assert compliance.complied is False
    assert set(regions) == {Region.OFF_TASK}
    assert headline_indices(regions) == []
    on_task, ok = label_cells(NO_THINK)
    assert ok.complied is True
    assert Region.IN_SENTENCE in on_task and Region.OFF_TASK not in on_task


def test_render_prompt_is_the_frozen_head_placement() -> None:
    """Head placement measured 46 surfacings against 3 for the tail; the phrasing is frozen so no
    per-item prompt tuning can enter the comparison."""
    got = render_prompt("a violin floating in a jar of honey")
    assert got.startswith("Think about a violin floating in a jar of honey. Now write")
    assert "The committee will meet on Thursday to review the annual budget." in got
    assert "%s" not in got


# ---- CLI smoke tests: the run chain must work, and must not re-admit an off-task item ---------

def _record(name: str, tokens: list[str], samples: list[str]) -> dict[str, Any]:
    """One item's stored read: n cells x 1 layer, 1 sample each — the shape the runner writes.

    `tokens`/`samples` are given in FORWARD reading order and stored reversed, because the runner
    stores rel -1 (the last token) first. A CLI that forgets to reverse mislabels every region.
    """
    return {"name": name,
            "rels": [-(i + 1) for i in range(len(tokens))],
            "tokens": list(reversed(tokens)),
            "ao": {"44": [[s] for s in reversed(samples)]}}


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    ao = {"layers": [44], "records": [
        _record("d-violin", [" review", " the", " annual"],
                ["imagery of a violin", "floating", "the committee will meet"]),
        _record("d-petrichor", ["is", " called", " petrichor"],
                ["the smell of rain", "petrichor on dry ground", "a scent"]),
    ]}
    items = [{"name": "d-violin", "concepts": ["a violin", "floating"], "stratum": "novel"},
             {"name": "d-petrichor", "concepts": ["petrichor", "dry ground"], "stratum": "anchor"}]
    ao_p, it_p = tmp_path / "ao.json", tmp_path / "items.json"
    ao_p.write_text(json.dumps(ao))
    it_p.write_text(json.dumps(items))
    return ao_p, it_p


def test_cli_excludes_an_off_task_item_from_the_mean(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str]) -> None:
    """The aggregate must cover compliant items only: d-petrichor's self-echo is what turned the
    typical-activation mean from 0.16 into 0.27, and 0.27 had to be retracted once already."""
    ao_p, it_p = _fixture(tmp_path)
    monkeypatch.setattr("sys.argv", ["score", f"ao_out={ao_p}", f"items={it_p}"])
    main()
    out = capsys.readouterr().out
    assert "EXCLUDED write non-compliance" in out
    assert "MEAN over 1 items" in out
    assert "excluded from the MEAN row: d-petrichor" in out


def test_cli_specificity_screen_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                             capsys: pytest.CaptureFixture[str]) -> None:
    """The screen is reachable from the command line, over in-sentence readouts only."""
    ao_p, it_p = _fixture(tmp_path)
    monkeypatch.setattr("sys.argv", ["score", f"ao_out={ao_p}", f"items={it_p}",
                                     "specificity=violin"])
    main()
    out = capsys.readouterr().out
    assert "specificity screen for 'violin'" in out
    assert "own   1/3" in out


def test_cli_names_a_missing_key_instead_of_raising_keyerror(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A malformed run file must name the key it wants, not print a KeyError traceback."""
    bad, items = tmp_path / "ao.json", tmp_path / "items.json"
    bad.write_text(json.dumps({"records": []}))
    items.write_text("[]")
    monkeypatch.setattr("sys.argv", ["score", f"ao_out={bad}", f"items={items}"])
    with pytest.raises(SystemExit, match="missing key 'layers'"):
        main()
