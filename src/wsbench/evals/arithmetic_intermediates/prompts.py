"""Prompts of the arithmetic-intermediates free-recall judge: the chained-intermediates judge
(source repo ``judge_free_modal.py``) with its task sentence changed to a bare arithmetic
expression; printed verbatim in the family README."""

from wsbench.llm import schema_block

PROMPT_VERSION = "arith-free-2026-09-23"
MAX_VALUES = 3  # the prompt caps the ranked list at three; the schema cannot (no maxItems)

SYSTEM = """You are reading the output of a "lens" that tries to verbalize what a language model is
computing internally. The model is silently evaluating a short arithmetic expression: two or
three operations on given numbers, answered with no chain of thought. You do NOT get to see the
expression — only the lens output.

The lens output may be free text, bullet points, or a bag of loose tokens, possibly in several
languages (Chinese numerals like 十四 mean fourteen). It is often vague, generic, or about
something else entirely, and it frequently contains numbers that are not results of anything.

Your job: name the number(s) that this output presents as a COMPUTED VALUE — a result, an
intermediate, a total, a "the answer is X". Rank them, most clearly a computed result first,
at most three. Decimals and negative numbers are values too; report them exactly as written.

Rules:
- Do NOT list numbers that appear as operands, quantities, labels, list indices, years, counts
  of items, or parts of unrelated prose. Only numbers the output presents as something that was
  worked out.
- If the output states arithmetic with both operands and a result ("16 - 2 = 14"), list the
  RESULT (14), not the operands.
- A bag of loose numerals with no arithmetic around them still counts: if the bag is dominated
  by one value or a tight cluster, list what it points at, most frequent or most prominent first.
- If the output presents no number as a computed value, set states_value false and return an
  empty list. That is a normal and common answer — do not invent one.
- Never do any arithmetic of your own. You do not know the task, so you cannot know what the
  right answer is; only report what the text itself puts forward."""

USER = (
    "Lens output:\n<<<\n{readout}\n>>>\n\n"
    "Which number(s) does this output present as a computed value?"
)

PROMPTS: dict[str, str] = {"SYSTEM": SYSTEM, "USER": USER}

SCHEMA = schema_block(
    "arith_free",
    {
        "states_value": {"type": "boolean"},
        "values": {
            "type": "array",
            "items": {"type": "number"},
            "description": "ranked, most clearly a computed result first, at most three",
        },
        "basis": {"type": "string", "enum": ["arithmetic", "stated_result", "numeral_bag", "none"]},
        "quote": {
            "type": "string",
            "description": "shortest verbatim span supporting the top value",
        },
    },
    ["states_value", "values", "basis", "quote"],
)


def render_user(readout: str) -> str:
    return USER.replace("{readout}", readout.strip())
