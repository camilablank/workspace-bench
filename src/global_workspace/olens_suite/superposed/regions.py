"""Which completion cells may carry a claim. The load-bearing provenance rule of this domain.

For these items the model writes the dictated sentence and THEN opens a `<think>` block, often
followed by a re-write of the whole prompt. A read cell's meaning depends entirely on which of
those regions it sits in, so every cell is labelled and only IN_SENTENCE cells carry claims.

WHY, AS MEASURED (the verbalization-ban gate, 176 continuations of the same prompts):
  * 47 of 176 continuations (26.7%) restate the dictated phrase verbatim;
  * the FIRST restatement lands after `</think>` in 29 cases, inside `<think>` in 13, and before
    `<think>` in 5.
So the dominant leak is NOT the think block — it is the model re-emitting the prompt after
`</think>`. POST_THINK is therefore contaminated on exactly the same footing as IN_THINK: it is a
region where the phrase may be being reproduced FOR OUTPUT rather than held off-site. And it is
where the read cells rel -1 to rel -4 land for 14 of the 17 binding items.

The reads used one rollout in which no reproduction happened inside the 20-token window, so this is
a region-level risk that was not eliminated by construction — which is the whole reason the labels
exist rather than a note saying "probably fine". Worked example, `d-fox-a`: its in-sentence cell is
rel -12, token ' the' inside "review THE annual budget"; cell rel -3 is ' a' inside
`<think>\\nHere's a`, and that in-think cell is exactly where "fox" surfaces 11 times. Those 11 are
excluded from every claim; the item's in-sentence count is 18.
"""

from collections.abc import Sequence
from enum import StrEnum
from typing import NamedTuple

from global_workspace.olens_suite.superposed.spec import TARGET_SENTENCE

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


class Region(StrEnum):
    IN_SENTENCE = "in_sentence"
    IN_THINK = "in_think"
    POST_THINK = "post_think"
    SELF_ADDED_META = "self_added_meta"
    OFF_TASK = "off_task"         # the dictated sentence was never written (see `label_cells`)


def _is_blank_line(token: str) -> bool:
    """A paragraph break: whitespace only, containing a newline. This is the boundary token."""
    return token.strip() == "" and "\n" in token


def classify_regions(tokens: Sequence[str]) -> list[Region]:
    """Label each completion cell, given the cells' tokens in FORWARD order.

    The rule, in one pass:
      * everything before the first paragraph break is IN_SENTENCE — the model is writing the
        dictated sentence. The blank line itself is NOT in-sentence (it is the break that precedes
        `<think>`), which is why it is labelled on the contaminated side; dropping it changes none
        of the headline capacity numbers (they hold at 384 activations as at 412).
      * if a `<think>` token occurs, the break through `</think>` inclusive is IN_THINK and
        anything after `</think>` is POST_THINK — the re-write region.
      * if no `<think>` occurs, everything from the break on is SELF_ADDED_META: the model finished
        the sentence and appended instruction-shaped text of its own (`d-cranes`).
      * a window with no break at all is entirely IN_SENTENCE.

    Takes the tokens of the read window only, so the sentence may start mid-window; the labels are
    positional relative to the break, never to an absolute index.
    """
    open_at = next((i for i, t in enumerate(tokens) if THINK_OPEN in t), None)
    close_at = next((i for i, t in enumerate(tokens) if THINK_CLOSE in t), None)
    break_at = next((i for i, t in enumerate(tokens) if _is_blank_line(t)), None)
    if open_at is not None and (break_at is None or break_at > open_at):
        break_at = open_at        # window opens after the sentence, or the break is not a cell
    if open_at is None and close_at is not None:
        break_at = 0              # the window opens INSIDE the think block
    if break_at is None:
        return [Region.IN_SENTENCE] * len(tokens)
    tail = (Region.IN_THINK if open_at is not None or close_at is not None
            else Region.SELF_ADDED_META)
    out = [Region.IN_SENTENCE if i < break_at else tail for i in range(len(tokens))]
    if close_at is not None:
        for i in range(close_at + 1, len(tokens)):
            out[i] = Region.POST_THINK
    return out


def headline_indices(regions: Sequence[Region]) -> list[int]:
    """Indices of the cells a headline claim may use. There is one, and it is IN_SENTENCE."""
    return [i for i, r in enumerate(regions) if r is Region.IN_SENTENCE]


class WriteCompliance(NamedTuple):
    """Did the model write the DICTATED sentence in this read window?"""

    complied: bool
    matched: str      # the longest contiguous fragment of the target found in the window
    run: int          # its length in words
    min_run: int


def _words(text: str) -> list[str]:
    return "".join(c if c.isalnum() else " " for c in text.lower()).split()


def write_compliance(cell_tokens: Sequence[str], target: str = TARGET_SENTENCE,
                     *, min_run: int = 3) -> WriteCompliance:
    """Write-compliance gate: `min_run` consecutive words of the target must appear in the window.

    `d-petrichor` is why this exists, and why it is a gate rather than a note. It scores 401 of 480
    readouts containing a dictated word — an order of magnitude above every other item — because
    the model never wrote the committee sentence: its completion is "The smell of rain on dry
    ground is called petrichor. It is a scent that many people…", so all 20 read cells sit in the
    model's OWN text and the lens is reading words that are literally present. Region labelling
    cannot catch that (no think block, no paragraph break to key on), and including the item
    inflated the typical-activation mean from 0.16 to **0.27** — a number that had to be retracted.

    A CONTIGUOUS RUN, not word overlap: the window is 20 tokens and may start mid-sentence, so
    equality is wrong, but bag overlap is too weak — off-task English shares "the" and "a" with any
    target. A run of three consecutive target words is not something unrelated prose produces.
    """
    want, got = _words(target), _words(" ".join(cell_tokens))
    best, at = 0, 0
    for i in range(len(want)):
        for j in range(len(got)):
            n = 0
            while i + n < len(want) and j + n < len(got) and want[i + n] == got[j + n]:
                n += 1
            if n > best:
                best, at = n, i
    return WriteCompliance(best >= min_run, " ".join(want[at:at + best]), best, min_run)


def label_cells(cell_tokens: Sequence[str], *, target: str = TARGET_SENTENCE,
                min_run: int = 3) -> tuple[list[Region], WriteCompliance]:
    """Region label per cell (forward order) plus the write-compliance verdict.

    A window that fails write-compliance is labelled `OFF_TASK` throughout, so **an off-task item
    has no IN_SENTENCE cells by construction** and cannot reach a headline number. This is a
    distinct label from `SELF_ADDED_META` on purpose: self-added meta means the model DID write the
    dictated sentence and then appended text of its own, so its sentence cells are still valid
    (`d-cranes` is scored normally); OFF_TASK means the dictated sentence was never written, so no
    cell in the window is a read of something held off-site.
    """
    compliance = write_compliance(cell_tokens, target, min_run=min_run)
    if not compliance.complied:
        return [Region.OFF_TASK] * len(cell_tokens), compliance
    return classify_regions(cell_tokens), compliance
