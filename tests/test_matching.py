"""Unit-form matching for the regex-scored families."""

from __future__ import annotations

import pytest

from wsbench.matching import (
    answer_number_matcher,
    fold,
    hit_forms,
    parse_cjk_numeral,
    standalone_number_matcher,
    unicode_word_matcher,
)


def test_fold():
    assert fold("México") == "mexico" and fold("Rubik’s") == "rubik's"  # noqa: RUF001


@pytest.mark.parametrize(
    "form,text,want",
    [
        ("México", "in mexico city", True),
        ("Марс", "на марс", True),  # noqa: RUF001
        ("Марс", "на марсе", False),  # noqa: RUF001
        ("mars", "the martian", False),
        ("서울", "서울에서", True),
        ("肾上腺素", "分泌肾上腺素激素", True),
        ("Simula 67", "it was simula-67 that", True),
        ("Simula 67", "simula 670", False),
    ],
)
def test_unicode_word_matcher(form, text, want):
    assert unicode_word_matcher(form)(text) is want


@pytest.mark.parametrize(
    "text,want",
    [
        ("the answer is 435", True),
        ("answer: 435", True),
        ("Answer: 435", False),  # markers are matched as written, as in the source scorer
        ("= 435.", True),
        ("435", True),
        ("4350", False),
        ("435 + 1 = 436", False),
        ("答案是四百三十五", True),
        ("４３５", True),  # noqa: RUF001
    ],
)
def test_answer_number_matcher(text, want):
    assert answer_number_matcher("435")(text) is want


def test_parse_cjk_numeral():
    assert parse_cjk_numeral("四百三十五") == 435 and parse_cjk_numeral("两百") == 200
    assert parse_cjk_numeral("七十二个") is None and parse_cjk_numeral("") is None


def test_standalone_number_and_hit_forms():
    m = standalone_number_matcher("52")
    assert m("sets qux to 52") and m("(52)") and not m("152") and not m("52.5") and not m("1,052")
    forms = {"en": ["iron ore"], "zh": ["铁矿"]}
    assert hit_forms(["it is iron ore"], forms) == ["en"]
    assert hit_forms(["这是铁矿"], forms) == ["zh"]
    assert hit_forms(["iron", "ore"], forms) == []  # never assembled across samples
    assert hit_forms(["52"], {"n": ["52"]}, numeric_match="standalone") == ["n"]
