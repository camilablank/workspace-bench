"""The single source of hit semantics for the oracle-lens bank evals.

Every consumer — ``score_targets.py`` (the headline metric), ``digest_eval.py`` (the analyst
digests), ``fve_score.py`` (the ablation report) — matches through these functions, so a
"hit" means the same thing everywhere. Before this module, ``fve_score.hits_of`` used plain
lowercase substring while the headline path used word+exact: the ablation table's numbers
were computed under a different metric than the caption claimed.
"""


import re
from collections.abc import Callable

__all__ = [
    "component_of",
    "content_targets",
    "exact_targets",
    "hit_any",
    "lenient_matcher",
    "normalize_text",
    "sample_equals_number",
    "word_matcher",
]

# Operation NAMES are task identity, not computed content. When an item's real deliverable
# is a number (order-ops: result + numeric steps), crediting "multiplication" as a full hit
# is guaranteed collision — walkthrough-genre readouts contain these words at massive base
# rate for entirely unrelated computations (2026-07-31 audit).
_TASK_WORDS = frozenset(
    {"addition", "subtraction", "multiplication", "division", "modulo", "remainder",
     "exponentiation", "arithmetic", "parentheses"}
)  # fmt: skip


def content_targets(targets: list[str]) -> list[str]:
    """Drop operation-name targets when the item also has numeric targets: for those items
    the names document HOW the answer is computed, and only the computed values are
    content. Items with no numeric target (e.g. basic-readout "boxing") are untouched."""
    if not any(t.strip().isdigit() for t in targets):
        return targets
    kept = [t for t in targets if t.strip().lower() not in _TASK_WORDS]
    return kept or targets


def _word_seq(s: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", s.lower())


def component_of(t: str, u: str) -> bool:
    """True when ``t`` is a strict word-subsequence of ``u`` ("moon" of "full moon").

    Word-level, so "ice cube" is NOT a component of "ice cubes" (plural sibling) —
    singular/plural variants both survive. Wordless targets (CJK) fall back to strict
    character containment.
    """
    tw, uw = _word_seq(t), _word_seq(u)
    if not tw or not uw:
        tl, ul = t.lower(), u.lower()
        return tl != ul and tl in ul
    if len(tw) >= len(uw):
        return False
    return any(uw[i : i + len(tw)] == tw for i in range(len(uw) - len(tw) + 1))


def exact_targets(targets: list[str]) -> list[str]:
    """Drop component aliases: a target that is a word-subsequence of a LONGER sibling
    ("moon" in "full moon", "bowling" in "bowling alley") is a partial alias and does not
    count as an exact concept match. Distinct synonyms ("carousel" vs "merry-go-round")
    and inflection variants ("ice cube" vs "ice cubes") survive."""
    return [t for t in targets if not any(component_of(t, u) for u in targets)]


def word_matcher(target: str) -> Callable[[str], bool]:
    """Word-boundary match (lowercased) for wordy targets; answer-context match for purely
    numeric targets; plain substring otherwise.

    The bank targets include short tokens like "3" or "led" where a raw substring hit is
    noise ("3" matches almost any arithmetic output; "led" matches "called"). Targets with
    non-word material (CJK, punctuation-bearing phrases) keep substring semantics — word
    boundaries are meaningless there.

    Purely numeric targets are the worst false-alarm surface (2026-07-31 audit: every one of
    the GT lens's order-ops "hits" was a digit inside an unrelated walkthrough, a number
    list, a decimal like 6.75, even a JS character class). A bare number is not a concept
    word, so presence-anywhere is not evidence of a readout. A numeric target must appear
    in ANSWER position: after "=", "answer/result/value (is/:)", "equals", "->", or alone
    (possibly bold/mathjax-wrapped) at the start of the text — and never as a prefix of a
    longer decimal ("84" != "84.5" != "6" in "6.75").
    """
    t = target.lower()
    if re.fullmatch(r"\d+", t):
        num = re.escape(t) + r"(?![\d.,]?\d)(?!\.\d)"  # full-number identity: 84 ∉ 84.5/184
        after_eq = re.compile(
            r"(?:=|->|→|\bequals\b|\b(?:answer|result|value|total|sum|product|quotient)"
            r"\b(?:\s+is)?\s*:?)[\s$*{(\\`]*" + num
        )
        # standalone at the start of the readout ("23.", "**84**") — but a leading number
        # that OPENS an expression ("6 x 7 = 42...") is an operand, not an answer
        # (the class holds the REAL multiplication/division signs models emit, not x):
        at_start = re.compile(r"\A[\s$*#>\-]*" + num + r"(?!\s*[+\-*/×÷^=]\s*\d)")  # noqa: RUF001
        return lambda text: bool(after_eq.search(text) or at_start.search(text))
    if re.fullmatch(r"[a-z0-9 '\-]+", t):
        # Hyphen/space equivalence between words ("tug of war" ≡ "tug-of-war" — 2026-07-31
        # scan: every markdown-split FN was a hyphenation) and curly apostrophes match
        # ascii ones (rubik's with U+2019 or U+0027).
        parts = [re.escape(w).replace("'", "['’]") for w in re.split(r"[\s\-]+", t) if w]  # noqa: RUF001
        pat = re.compile(r"(?<![a-z0-9])" + r"[\s\-]+".join(parts) + r"(?![a-z0-9])")
        return lambda text: bool(pat.search(text))
    return lambda text: t in text


def sample_equals_number(sample: str, target: str) -> bool:
    """True when one SAMPLE, stripped of markdown/punctuation wrapping, IS the number.

    The answer-context rule in ``word_matcher`` is prose-shaped; token-bank readouts
    (J-lens top-k lists) and terse AO readouts deliver a bare number per sample, where
    "the sample equals the target" is exactly the claim being scored. Without this, the
    v2 numeric rule silently broke the baseline arm (J-lens order-ops 44 → 7 hits)."""
    s = sample.strip().strip("*`$#!()[]{}\"' ").rstrip(".,:;")
    return s.lower() == target


def hit_any(samples: list[str], targets: list[str]) -> bool:
    """The headline pass criterion for one grid point: any target word-matches ANY SINGLE
    lowercased sample — or, for numeric targets, any single sample IS the number.

    Per-sample on purpose: the old ``" \\n ".join(samples)`` let a multi-word target match
    ACROSS the seam between two samples, because ``word_matcher`` joins target words with
    ``[\\s\\-]+`` (which matches the seam's newline). Harmless for prose readouts, but a
    J-lens grid point's samples are its top-k vocabulary TOKENS — adjacent ranks glued into
    fake phrase hits (['Gotham','city'] passed target "Gotham City"; audit 2026-08-15: every
    J-lens -mt "pass" was this artifact). A phrase spanning two different rollouts was never
    a real readout either."""
    lowered = [s.lower() for s in samples]
    for t in targets:
        matcher = word_matcher(t)
        if any(matcher(s) for s in lowered):
            return True
        tl = t.lower()
        if re.fullmatch(r"\d+", tl) and any(sample_equals_number(s, tl) for s in samples):
            return True
    return False


_SUBSCRIPTS = str.maketrans("₀₁₂₃₄₅₆₇₈₉⁰¹²³⁴⁵⁶⁷⁸⁹", "01234567890123456789")


def normalize_text(s: str) -> str:
    """Fold the surface variation the 2026-07-30 audit found hiding real readouts:
    diacritics (jalapeño→jalapeno), sub/superscript digits (H₂SO₄→H2SO4), hyphens→spaces
    (sixty-first ≡ sixty first), casefold."""
    import unicodedata

    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.translate(_SUBSCRIPTS).replace("-", " ").casefold()


def lenient_matcher(target: str) -> Callable[[str], bool]:
    """A second, LOOSER tier reported beside word+exact — never the headline. Adds, on
    normalized text: last-word ±s/es inflection and a relaxed right boundary so fused forms
    ("kidneygen", "governmentment") still register. Cross-language/synonym recall stays with
    the analyst/judge pass — no regex can do it honestly."""
    t = normalize_text(target)
    if not re.fullmatch(r"[a-z0-9 ']+", t):
        return lambda text: t in normalize_text(text)
    words = t.split()
    last = words[-1]
    stem = last[:-2] if last.endswith("es") else (last[:-1] if last.endswith("s") else last)
    body = (" ".join(re.escape(w) for w in words[:-1]) + " " if len(words) > 1 else "")
    pat = re.compile(r"(?<![a-z0-9])" + body + re.escape(stem) + r"(?:s|es)?(?![0-9])")
    return lambda text: bool(pat.search(normalize_text(text)))
