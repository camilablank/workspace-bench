"""The catalog of prompts the lens runs on — one source of truth for the script and the UI."""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Prompt:
    """A named prompt for the visualizer's preset dropdown."""

    name: str
    text: str


# Categories for the verbal-report probes ("think of a <category>").
WORKSPACE_CATEGORIES: tuple[str, ...] = (
    "sport",
    "language",
    "country",
    "animal",
    "color",
    "fruit",
    "instrument",
    "profession",
    "city",
    "metal",
    "planet",
    "vegetable",
    "tree",
    "emotion",
)

# Candidate items per category, for the verbal-report correlation eval. Hand-curated raw lists;
# the eval filters to single-token (leading-space, output-capitalized) items at runtime against the
# model's own tokenizer and logs the survivor count per category (drops categories with <5). The
# eval ranks these two ways — by the model's output prob vs by the J-lens score — and correlates.
CATEGORY_ITEMS: dict[str, tuple[str, ...]] = {
    "sport": (
        "Soccer",
        "Football",
        "Tennis",
        "Golf",
        "Boxing",
        "Cricket",
        "Rugby",
        "Basketball",
        "Baseball",
        "Hockey",
        "Swimming",
        "Cycling",
        "Running",
        "Skiing",
        "Surfing",
        "Volleyball",
        "Badminton",
        "Wrestling",
        "Rowing",
        "Fencing",
        "Archery",
        "Karate",
    ),
    "language": (
        "English",
        "French",
        "Spanish",
        "German",
        "Italian",
        "Chinese",
        "Japanese",
        "Russian",
        "Arabic",
        "Portuguese",
        "Korean",
        "Dutch",
        "Hindi",
        "Greek",
        "Latin",
        "Swedish",
        "Polish",
        "Turkish",
        "Hebrew",
        "Thai",
        "Finnish",
        "Danish",
    ),
    "country": (
        "France",
        "China",
        "Japan",
        "Brazil",
        "India",
        "Germany",
        "Italy",
        "Spain",
        "Russia",
        "Canada",
        "Mexico",
        "Egypt",
        "Australia",
        "Norway",
        "Sweden",
        "Poland",
        "Greece",
        "Turkey",
        "Kenya",
        "Peru",
        "Chile",
        "Cuba",
        "Iran",
        "Iraq",
    ),
    "animal": (
        "Dog",
        "Cat",
        "Lion",
        "Tiger",
        "Elephant",
        "Horse",
        "Bear",
        "Wolf",
        "Fox",
        "Rabbit",
        "Mouse",
        "Snake",
        "Eagle",
        "Shark",
        "Whale",
        "Dolphin",
        "Frog",
        "Deer",
        "Sheep",
        "Goat",
        "Cow",
        "Pig",
        "Monkey",
        "Zebra",
    ),
    "color": (
        "Red",
        "Blue",
        "Green",
        "Yellow",
        "Orange",
        "Purple",
        "Pink",
        "Black",
        "White",
        "Brown",
        "Gray",
        "Grey",
        "Cyan",
        "Violet",
        "Indigo",
        "Gold",
        "Silver",
    ),
    "fruit": (
        "Apple",
        "Banana",
        "Orange",
        "Grape",
        "Mango",
        "Cherry",
        "Peach",
        "Pear",
        "Lemon",
        "Lime",
        "Melon",
        "Plum",
        "Kiwi",
        "Strawberry",
        "Pineapple",
        "Apricot",
        "Fig",
        "Date",
        "Coconut",
    ),
    "instrument": (
        "Piano",
        "Guitar",
        "Violin",
        "Drums",
        "Flute",
        "Trumpet",
        "Cello",
        "Saxophone",
        "Harp",
        "Clarinet",
        "Trombone",
        "Banjo",
        "Tuba",
        "Viola",
        "Bass",
        "Organ",
    ),
    "profession": (
        "Doctor",
        "Teacher",
        "Lawyer",
        "Engineer",
        "Nurse",
        "Chef",
        "Pilot",
        "Artist",
        "Writer",
        "Scientist",
        "Architect",
        "Dentist",
        "Farmer",
        "Actor",
        "Soldier",
        "Painter",
        "Carpenter",
    ),
    "city": (
        "Paris",
        "London",
        "Tokyo",
        "Berlin",
        "Madrid",
        "Rome",
        "Moscow",
        "Beijing",
        "Cairo",
        "Sydney",
        "Toronto",
        "Boston",
        "Chicago",
        "Miami",
        "Seattle",
        "Denver",
        "Vienna",
        "Athens",
        "Dublin",
        "Oslo",
    ),
    "metal": (
        "Gold",
        "Silver",
        "Iron",
        "Copper",
        "Steel",
        "Aluminum",
        "Aluminium",
        "Bronze",
        "Lead",
        "Zinc",
        "Nickel",
        "Tin",
        "Platinum",
        "Titanium",
        "Mercury",
        "Brass",
        "Cobalt",
        "Chromium",
        "Uranium",
        "Tungsten",
    ),
    "planet": (
        "Earth",
        "Mars",
        "Venus",
        "Jupiter",
        "Saturn",
        "Mercury",
        "Neptune",
        "Uranus",
        "Pluto",
        "Moon",
    ),
    "vegetable": (
        "Carrot",
        "Potato",
        "Tomato",
        "Onion",
        "Cabbage",
        "Spinach",
        "Broccoli",
        "Lettuce",
        "Pepper",
        "Cucumber",
        "Garlic",
        "Celery",
        "Corn",
        "Pumpkin",
        "Beet",
        "Kale",
    ),
    "tree": (
        "Oak",
        "Pine",
        "Maple",
        "Birch",
        "Willow",
        "Cedar",
        "Elm",
        "Palm",
        "Spruce",
        "Fir",
        "Ash",
        "Beech",
        "Redwood",
        "Cypress",
        "Walnut",
        "Chestnut",
        "Aspen",
        "Sycamore",
    ),
    "emotion": (
        "Happiness",
        "Sadness",
        "Anger",
        "Fear",
        "Joy",
        "Love",
        "Hate",
        "Surprise",
        "Disgust",
        "Pride",
        "Shame",
        "Guilt",
        "Hope",
        "Envy",
        "Excitement",
        "Anxiety",
        "Grief",
    ),
}

# One template so the wording lives in one place; the lens is read at the trailing colon.
_PROBE_PREFIX = "Human: Think of a {category}. Answer in one word.\n\nAssistant:"
# Qwen3.6 is a reasoning model: after a bare "Assistant:" it emits its <think> control token
# rather than naming the item, so the immediate next-token distribution is dominated by <think>.
# An empty think block disables that — the plaintext equivalent of enable_thinking=False (a control
# sequence, NOT the role-tagged chat template) — and the trailing "Answer:" keeps a colon read-site
# while eliciting a high-probability, leading-space, single-token item.
_NO_THINK_SUFFIX = " <think></think>\n\nAnswer:"


def category_think_answer_probe(category: str) -> Prompt:
    """Open a ``<think>`` block and cue ``Answer:`` *inside* it; the lens is read at that colon.

    Tests whether the reasoning model names the item inside its own scratchpad — a read-site that
    lives mid-reasoning, unlike the post-``</think>`` no-think sites.
    """
    return Prompt(
        name=f"think-answer {category}",
        text=_PROBE_PREFIX.format(category=category) + " <think>Answer:",
    )


def category_user_message(category: str) -> str:
    """The bare user turn for the chat-template framing (no role scaffold, no think suffix).

    The chat framing feeds this to the tokenizer's chat template, which injects the role tokens
    and the generation prompt itself — so unlike :func:`category_probe` it must NOT carry the
    plaintext ``Human:``/``Assistant:`` markers. Same wording as the ``_PROBE_PREFIX`` question.
    """
    return f"Think of a {category}. Answer in one word."


# Assistant-turn stem appended after the chat template. A chat answer starts at column
# zero, where Qwen's vocab fragments many words ('Tennis' -> 'T'+'ennis'); a stem ending
# at a colon makes the next token the leading-space whole-word form (' Tennis') — the
# same handle class, and the same colon readout position, as the paper's "Assistant:".
ANSWER_STEM = "My one-word answer is:"


# --------------------------------------------------------------------------- DRUMMER suite
# Verbatim DRUMMER_REPORT prompts (App. A.2 / A.5 / A.6); the trailing newline is part of each.
# Canonical home for the drummer line — ``drummer.DRUMMER_LINE`` re-exports this one.

# DRUMMER_LINE drops the report's trailing comma on purpose. "…line,\n" tokenizes as
# "…line" · "," · "\n": the intervening "," token suppresses the meta-token gate at the final-"\n"
# readout (是什么意思 falls out of the top-10 entirely, 这句话 collapses to 3 layers). The bare
# "…line\n" — newline directly after "line" — recovers all three meta tokens, matching report A.4.
# See docs/project/experiments/drummer/suite.md ("What changed").
DRUMMER_LINE: str = "The drummer kept the marching column in line\n"

# A.5 injection target lines — ordinary lines that don't naturally evoke the meta tokens
# (base INTERPRET ~6.3%). NOTE: §4 says "six target lines" but A.5 prints only these five;
# the suite augments them with GPT-5.5-generated ordinary lines for power (and flags the gap).
INJECTION_TARGETS: tuple[str, ...] = (
    "She danced alone beneath the broken moon,\n",
    "The old man sold his shadow to the sea,\n",
    "We marched all night through fields of frozen rain,\n",
    "A single candle lit the silent hall,\n",
    "Oh baby, don't you leave me here alone,\n",
)

# A.5 control tokens: norm/subspace-matched injection controls (not interpret-evoking).
RANDOM_CONTROL_ZH: tuple[str, ...] = ("自行车", "图书馆", "番茄")  # bicycle, library, tomato
RANDOM_CONTROL_EN: tuple[str, ...] = ("bicycle", "library", "tomato")

# A.6 disambiguating prefixes for §6 (prepended to DRUMMER_LINE). "none" is the bare line.
DRUMMER_PREFIXES: dict[str, str] = {
    "none": "",
    "rhyming_couplet_lower": "a rhyming couplet\n",
    "rhyming_couplet_colon": "A rhyming couplet:\n",
    "song_lyrics": "Song lyrics:\n",
}


def category_probe(category: str, *, no_think: bool = True) -> Prompt:
    """The verbal-report probe for one category (e.g. "sport" -> "think of a sport").

    ``no_think`` (default) appends the empty-think + "Answer:" suffix so a reasoning model names
    the item directly; ``no_think=False`` ends at the bare "Assistant:" colon (base-model style).
    """
    text = _PROBE_PREFIX.format(category=category)
    if no_think:
        text += _NO_THINK_SUFFIX
    return Prompt(name=f"think of a {category}", text=text)


def chat_category_probe(tokenizer: Any, category: str, stem: str = "") -> str:
    """The same probe through the model's chat template, with thinking disabled.

    Reasoning models (e.g. Qwen3.6) answer the raw Human/Assistant probe with a
    ``<think>`` tag instead of a word; ``enable_thinking=False`` closes the think block
    in the template so the next token is the answer. The lens reads at the final prompt
    position (the analogue of the colon). Models whose template ignores
    ``enable_thinking`` are unaffected by the extra kwarg.

    A non-empty ``stem`` (e.g. :data:`ANSWER_STEM`) is appended as the start of the
    assistant turn, so the model's answer begins with a leading space — derive token ids
    with ``leading_space=True`` to match.
    """
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": f"Think of a {category}. Answer in one word."}],
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False,
    )
    return str(text) + stem


def first_word(text: str) -> str:
    """The first word of a continuation (the model's one-word answer); "" if there is none.

    Words keep internal apostrophes and hyphens (``it's``, ``well-being``). Shared by the
    verbal- and swap-report scripts so answer parsing can't drift between them.
    """
    words = re.findall(r"[\w'-]+", text)
    return words[0] if words else ""


def first_token_id(tokenizer: Any, item: str, leading_space: bool) -> int:
    """First token of ``item`` as the model would emit it at the readout position.

    After ``Assistant:`` or a mid-sentence stem the answer starts with a space; after a
    bare chat template's ``</think>\\n\\n`` it starts at column zero. The two tokenize
    differently (`' Tennis'` is one token, column-zero `'Tennis'` is `'T'+'ennis'`).
    """
    text = (" " + item) if leading_space else item
    ids: list[int] = tokenizer.encode(text, add_special_tokens=False)
    return ids[0]


def item_id_problems(items: Sequence[str], ids: Sequence[int], tokenizer: Any) -> list[str]:
    """Why these candidate first-token ids would corrupt rank/Spearman/swap metrics.

    Flags duplicate ids (forced Spearman ties; ambiguous swap targets) and fragment ids
    that cover less than half of their item (orthographic handles like 'T' for 'Tennis'
    carry no concept content — the single-token limitation, paper section 8.1).
    """
    problems: list[str] = []
    by_id: dict[int, list[str]] = {}
    for item, token_id in zip(items, ids, strict=True):
        by_id.setdefault(token_id, []).append(item)
    for token_id, sharers in by_id.items():
        if len(sharers) > 1:
            problems.append(f"duplicate first token {tokenizer.decode([token_id])!r}: {sharers}")
    for item, token_id in zip(items, ids, strict=True):
        text = str(tokenizer.decode([token_id]))
        if len(text.strip()) * 2 < len(item):
            problems.append(f"fragment handle {text!r} for {item!r}")
    return problems


# Factual probes: the answer is a single late-layer token.
EXAMPLE_PROMPTS: list[Prompt] = [
    Prompt("eiffel tower", "The Eiffel Tower is in the city of"),
    Prompt("capital of japan", "The capital of Japan is"),
    Prompt("opposite of hot", "The opposite of hot is"),
    Prompt("2+2", "2 + 2 ="),
]

# Verbal-report probes.
WORKSPACE_PROBES: list[Prompt] = [category_probe(c) for c in WORKSPACE_CATEGORIES]

# Broadcast probes: the same argument under many functions.
BROADCAST_ARGUMENT: str = "France"
BROADCAST_PROBES: list[Prompt] = [
    Prompt("capital of France", "The capital of France is"),
    Prompt("language of France", "Most people in France speak"),
    Prompt("continent of France", "France is on the continent of"),
    Prompt("currency of France", "The currency used in France is the"),
]

# Sections, in dropdown order. The UI renders one <optgroup> per section.
PROMPT_SECTIONS: list[tuple[str, list[Prompt]]] = [
    ("Factual", EXAMPLE_PROMPTS),
    ("Verbal report (think of a …)", WORKSPACE_PROBES),
    ("Broadcast (same argument, many functions)", BROADCAST_PROBES),
]

# Flattened catalog.
ALL_PROMPTS: list[Prompt] = [p for _, prompts in PROMPT_SECTIONS for p in prompts]

# Shown on first page load.
DEFAULT_PROMPT: str = EXAMPLE_PROMPTS[0].text
