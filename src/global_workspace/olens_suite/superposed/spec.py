"""Frozen specification for the superposed-directed-modulation eval. Single source of truth.

The question: tell the model to hold something in mind while it writes an unrelated dictated
sentence, then read the activations at the positions it is WRITING at. Is the held thing there,
how much of it, and does the binding survive?

Nothing here reads data or runs a model. The Modal runner, the region labeller and the scorer all
import from here, so the phrasing, the read cells and the concept-word rule can only change in one
place — and the phrasing in particular must not change per item (see below).

WHY EACH FIELD EXISTS:

HEAD_TEMPLATE     The think-clause goes FIRST, before the write instruction, and then the whole
                  phrasing is frozen. A rephrasing search over placements measured 46 surfacings
                  with the think-clause first against 3 when it followed the write instruction.
                  Freezing it after that search is the point: with a per-item prompt tunable, any
                  cross-item or cross-lens comparison is a comparison of prompt tuning.
CONTROL_TEMPLATE  Same shape, no think-clause (`d-none` / `b-none`). Its in-sentence readouts
                  contain 0 hits for any dictated concept word (0/312). The two exceptions ARE the
                  measurement of lexical loading in this target sentence: "key" fires at 10 of 52
                  control cells ("key points", "Key Changes") and "owe/debt" at 6 of 52 (budget
                  context) — which is why per-word claims need `score.specificity_screen`.
TARGET_SENTENCE   Shares no content word with any dictated concept, so there is nothing at a read
                  position to echo. This is what makes write-position reading interpretable at all.
RENDER            Bare, no chat template. (The stored J-lens/logit records carry `render: "chat"`
                  in their metadata while their token sequences contain no chat-template tokens —
                  that field is stale; the rendering is bare.)
READ_CELLS_REL    The last 20 tokens of the model's own completion. Every read cell is inside the
                  completion for every item — no read cell is a prompt position, and alignment
                  (`tokens[n + rel] == read_cell_token`) is asserted per item by the runner.
LAYERS            4 layers only. The frame-to-lexical crossover (L44 is 76 frame / 14 lexical
                  readouts; L60 is 93 / 75) is therefore measured at 4 points, not resolved.
"""

from typing import Any

MODEL_ID = "Qwen/Qwen3.6-27B"
DEFAULT_OLENS = "agu18dec/qwen3.6-27b-mlayer-ao-checkpoints:ao.asst.alldata-v2cont.k2.g8"
OLENS_SCALE = 33.152003804062936      # from the checkpoint's own run.json; never guess
OLENS_PROMPT_KIND = "explain"         # NOT "explanation", a different and more verbose template
DATASET = "agu18dec/olens_eval_suite"  # banks live under superposed/ on the dataset

# Frozen 2026-08-05 by the placement search (46 head-first surfacings vs 3 tail-first).
# %s placeholders, in order: the dictated phrase, then the sentence to write.
HEAD_TEMPLATE = 'Think about %s. Now write this sentence: "%s"\n'
CONTROL_TEMPLATE = 'Read carefully. Now write this sentence: "%s"\n'
TARGET_SENTENCE = "The committee will meet on Thursday to review the annual budget."

RENDER = "bare"
LAYERS: tuple[int, ...] = (44, 52, 56, 60)
READ_CELLS_REL: tuple[int, ...] = tuple(range(-1, -21, -1))
# k=1 is the cross-arm sample-parity contract (2026-08-19): every text lens gets ONE sampled
# readout per cell, matching the bank audit contract and the external-lens workers. The metric
# is a per-item UNION over samples, so k>1 inflates scores vs k=1 arms (RL measured 1.58@k=6
# vs 1.34@k=1). Raise k only if EVERY compared arm is re-run at the same k.
SAMPLING: dict[str, Any] = {"k": 1, "max_new": 44, "temperature": 0.8, "top_p": 0.95}

# ---- the concept-word rule ------------------------------------------------------------------
# A dictated phrase is scored through its content words. Words of <=3 letters are dropped along
# with the stopwords, which is a real cost, stated rather than hidden: it drops `fox`, `egg`, `key`,
# `jar` and `pan`, so "silver fox" rides on `silver` alone even though `fox` itself scores 18/312.
# The rule was fixed BEFORE counting and applied uniformly to every item and every base rate.
#
# It is also not sufficient on its own: "key" has a base rate of 25/1000 and "blue" 13/1000 in this
# target sentence's context, so any per-word claim must clear `score.specificity_screen` as well.
MIN_CONTENT_WORD_LEN = 4
STOPWORDS: frozenset[str] = frozenset({
    "about", "against", "another", "been", "being", "between", "both", "each", "from", "have",
    "into", "itself", "leaning", "next", "onto", "over", "same", "some", "something", "that",
    "their", "them", "then", "there", "these", "they", "this", "those", "through", "under",
    "upon", "very", "what", "when", "which", "while", "will", "with", "your",
})

# The screen's thresholds. Chosen AFTER seeing the base-rate distribution, not pre-registered —
# it is a stated caveat, not a hidden one: "mango tree" passes on 3 hits of `tree` while `mango`
# itself is 0, and that single call moves d-plumber's capacity score from 2 to 3.
SPECIFICITY_MIN_HITS = 3
SPECIFICITY_MIN_RATIO = 3.0


def render_prompt(phrase: str, target: str = TARGET_SENTENCE) -> str:
    """The frozen dictation/binding prompt. Only `phrase` varies across the two banks."""
    return HEAD_TEMPLATE % (phrase, target)


def render_control_prompt(target: str = TARGET_SENTENCE) -> str:
    """The no-think-clause control (`d-none` / `b-none`): same shape, nothing to hold."""
    return CONTROL_TEMPLATE % (target,)


def content_words(phrase: str) -> tuple[str, ...]:
    """Scoreable words of a dictated phrase, in order, deduplicated, lowercased.

    See MIN_CONTENT_WORD_LEN above for what this deliberately drops.
    """
    out: list[str] = []
    for raw in "".join(c if c.isalpha() else " " for c in phrase.lower()).split():
        if len(raw) >= MIN_CONTENT_WORD_LEN and raw not in STOPWORDS and raw not in out:
            out.append(raw)
    return tuple(out)
