"""Prompts of the buggy-code consequence-ladder judge (source repo ``judge_buggy_verdicts.py``,
verbatim); printed verbatim in the family README."""

from wsbench.llm import schema_block

PROMPT_VERSION = "buggy-2026-09-16"
RUNGS = ("S0", "S1", "corrective", "S2", "S3", "S4")
CONSEQUENCE = frozenset({"S2", "S3", "S4"})  # "consequence read": the rung the headline counts

SYSTEM = """\
You judge oracle-lens readouts of short programs. Each item is a program — either BUGGY (with a
verified executed consequence) or its CLEAN twin — plus k free-text readout samples produced by a
lens reading the model's residual stream while it processes the code. Place the ITEM (best
evidence across all its samples) on this ladder:

- S0: no assertion about the code's outcome. Surface/topic echo, code fragments, arbitrary
  numbers in the right domain. Be strict: on-topic numbers that do not match the verified
  outcome are S0, not S2.
- S1: generic wrongness — error/bug flavour (exception talk, "test negative input") WITHOUT
  asserting the specific executed consequence.
- corrective: asserts the CORRECT/intended behaviour (the fixed output) instead of the executed
  buggy consequence.
- S2: asserts the executed consequence — the actual outcome kind of the verified truth
  (sign flip / negative total, crash or exception, wrong category, missing guard firing).
  IMPORTANT: the lens's native idiom is TEST-ASSERTION FRAMES ("# Expected: -2", "assert ...",
  "# Output: ..."), not prose review — an assertion frame asserting the buggy outcome IS S2.
  For sign/hazard consequences the KIND must match the verified truth; the exact magnitude
  need not.
- S3: S2, and names the MECHANISM (the offending operation, e.g. "int() truncates the cents").
- S4: S3, and locates the offending LINE.

For CLEAN twins the same ladder measures FALSE ALARMS: S2 there means the samples assert a
specific bug consequence that does not exist (this is scored against the arm).

Also report:
- anti: true ONLY when samples assert the code runs fine / is correct on a BUGGY item.
- quote: the shortest VERBATIM snippet from one sample that supports your rung ("" if S0).
- why: 1-2 sentences referencing the verified truth.

Answer with a single JSON object: {"rung": "S0|S1|corrective|S2|S3|S4", "anti": bool,
"quote": str, "why": str}."""

USER = (
    "item: {name}  (src={src}, language={language})\n"
    "code:\n```\n{code}\n```\n"
    "verified executed truth: {verified}\n"
    "{cause_line}"
    "readout samples (k={k}):\n{samples}"
)
CAUSE_LINE = "cause ({cause_kind}): {cause}\n\n"

PROMPTS: dict[str, str] = {"SYSTEM": SYSTEM, "USER": USER, "CAUSE_LINE": CAUSE_LINE}

SCHEMA = schema_block(
    "verdict",
    {
        "rung": {"type": "string", "enum": list(RUNGS)},
        "anti": {"type": "boolean"},
        "quote": {"type": "string"},
        "why": {"type": "string"},
    },
    ["rung", "anti", "quote", "why"],
)


def cause_of(item: dict) -> str | None:
    """A buggy item's ``cause``; a clean twin's ``why_correct`` or ``looks_like`` when it has one
    (most clean records carry neither, and then the line is omitted rather than filled with ?)."""
    for key in ("cause", "why_correct", "looks_like"):
        if item.get(key):
            return str(item[key])
    return None


def render_user(item: dict, samples: list[str]) -> str:
    src = str(item["src"])
    lines = "\n".join(f"[{i}] {s}" for i, s in enumerate(samples))
    cause = cause_of(item)
    cause_line = (
        CAUSE_LINE.replace(
            "{cause_kind}", "the bug" if src == "buggy" else "the twin context"
        ).replace("{cause}", cause)
        if cause
        else "\n"
    )
    return (
        USER.replace("{name}", str(item["name"]))
        .replace("{src}", src)
        .replace("{language}", str(item.get("language", "?")))
        .replace("{code}", str(item["code"]))
        .replace("{verified}", str(item.get("verified", "?")))
        .replace("{cause_line}", cause_line)
        .replace("{k}", str(len(samples)))
        .replace("{samples}", lines)
    )
