"""Unit-form matching for the regex-scored families: any script, folded, word-bounded."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable
from typing import Literal

NumericMatch = Literal["context", "standalone"]

FULLWIDTH_DIGITS = str.maketrans("０１２３４５６７８９．", "0123456789.")  # noqa: RUF001
_CJK_DIGIT = {
    "零": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}
_CJK_UNIT = {"十": 10, "百": 100, "千": 1000}
_CJK_HANGUL = re.compile(r"[぀-ヿ㐀-鿿가-힯]")
_WORDY = re.compile(r"[^\W_]+(?:[\s\-'][^\W_]+)*")
_WRAP = r"[\s$*{(\\`：（＊]*"  # noqa: RUF001
_ANSWER_CONTEXT = (
    r"(?:=|->|→|\bequals\b|\b(?:answer|result|value|total|sum|product|quotient)"
    r"\b(?:\s+is)?\s*:?"
    r"|(?:答案|结果|总和|总数)(?:是|为|：|:)?"  # noqa: RUF001
    r"|(?:值|积|商|差|和)(?:是|为|：|:)"  # noqa: RUF001
    r"|等于|得出|即|共计|共)"
)
_CJK_NUMERAL = r"([零一二两三四五六七八九十百千]{1,6})(?![一-鿿])"


def parse_cjk_numeral(s: str) -> int | None:
    """Standard CJK numerals up to 9999 (十五 = 15, 两百 = 200); ``None`` for anything else."""
    if not s or any(c not in _CJK_DIGIT and c not in _CJK_UNIT for c in s):
        return None
    total, current = 0, 0
    for c in s:
        if c in _CJK_DIGIT:
            current = _CJK_DIGIT[c]
        else:
            total += (current or 1) * _CJK_UNIT[c]
            current = 0
    return total + current


def fold(s: str) -> str:
    """NFKD, strip combining marks, straighten apostrophes, casefold."""
    s = unicodedata.normalize("NFKD", s).replace("\u2019", "'")
    return "".join(c for c in s if not unicodedata.combining(c)).casefold()


def answer_number_matcher(digits: str) -> Callable[[str], bool]:
    """The number in answer position (after ``=`` / ``answer:`` / ``等于`` ...) or alone at the
    start of the text, with full-number identity (84 is not 84.5 or 184); CJK numerals and
    fullwidth digits count in the same positions."""
    num = re.escape(digits) + r"(?![\d.,]?\d)(?!\.\d)"
    after_context = re.compile(_ANSWER_CONTEXT + _WRAP + num)
    at_start = re.compile(r"\A[\s$*#>\-]*" + num + r"(?!\s*[+\-*/×÷^=]\s*\d)")  # noqa: RUF001
    cjk = re.compile(r"(?:" + _ANSWER_CONTEXT + _WRAP + r"|\A[\s$*#>\-]*)" + _CJK_NUMERAL)
    value = int(digits)

    def numeric(text: str) -> bool:
        folded = text.lower().translate(FULLWIDTH_DIGITS)
        if after_context.search(folded) or at_start.search(folded):
            return True
        return any(parse_cjk_numeral(m) == value for m in cjk.findall(folded))

    return numeric


def unicode_word_matcher(form: str) -> Callable[[str], bool]:
    """Boundary match on folded text for a unit form in any script (Mexico with or without the
    accent, pointed and unpointed Hebrew). CJK and Hangul forms are substring (no word spaces;
    Korean particles attach); purely numeric forms use the answer-position rule."""
    f = fold(form)
    if re.fullmatch(r"\d+", f):
        return answer_number_matcher(f)
    if _CJK_HANGUL.search(form) or not _WORDY.fullmatch(f):
        return lambda text: f in fold(text)
    parts = [re.escape(w) for w in re.split(r"[\s\-]+", f) if w]
    pat = re.compile(r"(?<![^\W_])" + r"[\s\-]+".join(parts) + r"(?![^\W_])")
    return lambda text: bool(pat.search(fold(text)))


def standalone_number_matcher(form: str) -> Callable[[str], bool]:
    """``form`` occurs as a standalone integer anywhere: "52" hits "(52)" but not "152", "52.5",
    "-52" or "1,052"."""
    pat = re.compile(r"(?<![\w.,-])" + re.escape(form.strip()) + r"(?![\w.,]?\d)(?!\w)")
    return lambda text: bool(pat.search(fold(text).replace(",", "")))


def hit_forms(
    samples: list[str],
    forms: dict[str, list[str]],
    *,
    numeric_match: NumericMatch = "context",
) -> list[str]:
    """The language keys of ``forms`` whose strings hit any SINGLE sample, in dict order (a phrase
    is never assembled across two samples)."""

    def matcher(f: str) -> Callable[[str], bool]:
        if numeric_match == "standalone" and re.fullmatch(r"-?\d+", f.strip()):
            return standalone_number_matcher(f)
        return unicode_word_matcher(f)

    out: list[str] = []
    for lang, fs in forms.items():
        matchers = [matcher(f) for f in fs]
        if matchers and any(m(s) for m in matchers for s in samples):
            out.append(lang)
    return out
