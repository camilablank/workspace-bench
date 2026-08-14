"""The consequence gate and the output-leak gate for the buggy-code eval: the two gates that are
pure string work. (Formerly "Gate B" and "Gate E" — older artifacts, run logs and records use the
letters. The full old-to-new mapping: Gate A = the verbalization ban, Gate B = the consequence
gate, Gate C = the j-lens hint tag, Gate D = the articulation gate, Gate E = the output-leak
gate.)

They live here, not inside the Modal runner, because they decide which items are ADMISSIBLE — and
an admissibility rule that only exists inside a GPU job cannot be re-run, unit-tested or argued
with. The GPU stages store raw completions; these functions turn completions into pass/fail.

CONSEQUENCE GATE — computes-consequence. An item is interpretable only if the model can simulate
what its bug does: if Qwen cannot predict the buggy behaviour, the consequence is not in its
activations and a zero readout says nothing about the lens. It is the expensive gate — on the
24-item expansion bank it cut 9 of 24: the model mentally auto-fixes an off-by-one, cannot
simulate list aliasing, and mislabels a positional-after-keyword SyntaxError as a TypeError.
Pass = >=4 of 6 (5 sampled + 1 greedy output-sim completions under the chat render).

OUTPUT-LEAK GATE — continuations must not PRINT the buggy output. The verbalization ban stops
the model from *verbalising* the bug; it does not stop it from *executing* the code onto the
page.
Base-model continuations semi-execute: they annotate their own test calls with the computed output
(`print(f(...))  # -2`), and a judge reconstructs the bug from that annotation alone — 4 of the
continuation baseline's 5 level-3 verdicts came from items that fail this gate. Pass = 0 leaks;
any AO bug-DESCRIPTION claim must be scoped to the output-leak survivors.

Silent items are matched on stdout, raising items on the exception NAME: the exact message is
wording the model has no reason to reproduce ("math domain error"), while the name is the fact.

RUN (stage 1 of the chain; see scripts/olens_suite/README.md):

    modal run scripts/olens_suite/buggy_code_gate_modal.py      # -> results/buggy_code/gate.json
    uv run python -m global_workspace.olens_suite.buggy_code.gates \\
        gate_out=results/buggy_code/gate.json bank=bug_bank.json      # -> *_verdicts.json

CLI CONFIG (pydra; pass `--show` to print the resolved config and exit)

  gate_out    REQUIRED  the Modal gate stage's raw output JSON
  bank        REQUIRED  the item bank (executed truth per item)
  json_out    verdicts output path (default: <gate_out stem>_verdicts.json)

INPUT FORMAT

  gate_out.json   {"records": [{"name": str, "src": "buggy"|"clean",
                                "cont": [str, ...],        # raw continuations, k=10 (the
                                                           # verbalization-ban / output-leak
                                                           # material)
                                "cont_greedy": str,        # the greedy continuation
                                "outsim": [str, ...],      # consequence-gate output-sims, k=5
                                "outsim_greedy": str}]}    # the greedy output-sim
  bank.json       {"buggy": [{"name": str,
                              "verified": "exec: exit 0; stdout = '25.0'",
                              "correct_stdout": "27.0"}],  # the twin's stdout; the output-leak
                   "clean": [...]}                         # gate needs it
Only `src == "buggy"` records are gated. An item whose bank entry has no `correct_stdout` gets
the output-leak gate reported as `-` rather than a silent pass.
"""

import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any, NamedTuple

import pydra

# `verified` is written by actually EXECUTING the file, and has exactly two shapes:
#   silent  : "exec: exit 0; stdout = '25.0'"
#   raising : "exec: exit 1; stdout = ''; last = 'ValueError: math domain error'"
# Parse it, never re-derive it: the executed truth is the only thing the gate is allowed to
# compare against (sql-null-not-equal was cut because the model predicted the INTENDED semantics).
_STDOUT = re.compile(r"stdout\s*=\s*(.*?)(?=;\s*last\s*=|$)", re.DOTALL)
_LAST = re.compile(r"last\s*=\s*(.*)$", re.DOTALL)
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_.]*")
_WHITESPACE = re.compile(r"\s+")
_QUOTES = "\"'`"

CONSEQUENCE_N = 6            # 5 sampled + 1 greedy
CONSEQUENCE_MIN_HITS = 4     # >=4 of 6. Below this the item is "sometimes right", which cannot
#                              distinguish "the model does not simulate the bug" from sampling
#                              noise.
LEAK_MIN_CHARS = 2    # a 1-char buggy output collides with any digit in any continuation


def normalise(text: str) -> str:
    """Whitespace collapsed, surrounding quotes/backticks stripped.

    Both sides of every comparison in this module go through here, because the two sides come from
    different places: `verified` quotes its stdout (`'25.0'`), while a completion may write it in
    backticks or with a trailing newline. Comparing raw strings scores those as misses.
    """
    out = _WHITESPACE.sub(" ", text).strip()
    while len(out) >= 2 and out[0] in _QUOTES and out[-1] in _QUOTES:
        out = out[1:-1].strip()
    return out.strip(_QUOTES).strip()


def parse_verified(verified: str) -> tuple[str, str | None]:
    """`(normalised stdout, exception name or None)` from an item's executed-truth string.

    The exception name is the leading identifier of `last`, dotted tail last: `ValueError: math
    domain error` -> `ValueError`, `decimal.InvalidOperation: ...` -> `InvalidOperation`.
    """
    m = _STDOUT.search(verified)
    stdout = normalise(m.group(1)) if m else ""
    tail = _LAST.search(verified)
    if tail is None:
        return stdout, None
    head = normalise(tail.group(1)).split(":", 1)[0].strip()
    name = _IDENT.match(head)
    return stdout, name.group(0).rsplit(".", 1)[-1] if name else None


class ConsequenceGate(NamedTuple):
    passed: bool
    hits: int
    n: int
    expected_stdout: str
    exception_name: str | None


def consequence_gate(completions: Sequence[str], verified: str, *,
                     min_hits: int = CONSEQUENCE_MIN_HITS) -> ConsequenceGate:
    """Does the model compute this bug's consequence? Pass = `hits >= min_hits`.

    Raising items match the exception NAME case-insensitively anywhere in the completion (the
    model says "it raises a ValueError" as often as it says the bare name). Silent items must match
    the normalised stdout EXACTLY — a near-miss is the model computing a different program.

    An item that neither prints nor raises has no consequence to compute and scores 0 hits rather
    than crediting every empty completion for matching an empty stdout.
    """
    stdout, exc = parse_verified(verified)
    if exc is not None:
        needle = exc.lower()
        hits = sum(1 for c in completions if needle in normalise(c).lower())
    elif stdout:
        hits = sum(1 for c in completions if normalise(c) == stdout)
    else:
        hits = 0
    return ConsequenceGate(hits >= min_hits, hits, len(completions), stdout, exc)


class OutputLeakGate(NamedTuple):
    passed: bool
    leaks: int
    n: int
    buggy: str
    correct: str


def leaks_buggy_output(continuation: str, buggy_stdout: str, correct_stdout: str) -> bool:
    """Does this raw continuation print the buggy output?

    Two guards, both of which exist because the buggy output is often a small number:
      1. the normalised buggy string must be >=2 characters — `3` occurs in any continuation
         containing 13, 30 or 3.5, so a 1-char output cannot be told from arithmetic litter;
      2. it must DIFFER from the correct output — when both are the same string, printing it is
         not evidence of anything (and several twins share an output on the passing input).

    Deliberately conservative beyond that: the check is substring, so an operand echo like `41`
    inside `415` counts as a leak. The output-leak gate only ever REMOVES an item, so a false leak
    costs coverage while a missed leak inflates the specificity claim — the asymmetry decides the
    rule.
    """
    buggy, correct = normalise(buggy_stdout), normalise(correct_stdout)
    if len(buggy) < LEAK_MIN_CHARS or buggy == correct:
        return False
    return buggy in normalise(continuation)


def output_leak_gate(continuations: Sequence[str], buggy_stdout: str,
                     correct_stdout: str) -> OutputLeakGate:
    """Pass = NO stored continuation prints the buggy output (see `leaks_buggy_output`)."""
    leaks = sum(1 for c in continuations
                if leaks_buggy_output(c, buggy_stdout, correct_stdout))
    return OutputLeakGate(leaks == 0, leaks, len(continuations),
                          normalise(buggy_stdout), normalise(correct_stdout))


# ---- CLI: parse, call the functions above, print. No scoring logic lives below this line. -----

def _need(obj: dict[str, Any], key: str, where: str) -> Any:
    if key not in obj:
        raise SystemExit(f"{where}: missing key {key!r} (have {sorted(obj)}) — see the module "
                         f"docstring for the expected shape")
    return obj[key]


def bank_index(blob: Any, path: str) -> dict[str, dict[str, Any]]:
    """`{"buggy": [...]}` -> `{name: entry}`. Accepts a bare list of entries too."""
    entries = blob if isinstance(blob, list) else _need(blob, "buggy", path)
    return {_need(e, "name", path): e for e in entries}


def verdicts(records: Sequence[dict[str, Any]], bank: dict[str, dict[str, Any]],
             *, path: str = "gate_out.json") -> list[dict[str, Any]]:
    """Field routing only: output-sims -> consequence gate, raw continuations -> output-leak
    gate. Row keys `"consequence"` / `"output_leak"` were `"gate_b"` / `"gate_e"` in older
    verdict files."""
    out: list[dict[str, Any]] = []
    for rec in records:
        name = _need(rec, "name", path)
        if rec.get("src", "buggy") != "buggy":
            continue
        if name not in bank:
            raise SystemExit(f"{path}: item {name!r} is not in the bank")
        entry = bank[name]
        b = consequence_gate([*_need(rec, "outsim", path), _need(rec, "outsim_greedy", path)],
                             _need(entry, "verified", path))
        row: dict[str, Any] = {"name": name, "consequence": b._asdict()}
        correct = entry.get("correct_stdout")
        if correct is None:
            row["output_leak"] = None
        else:
            row["output_leak"] = output_leak_gate(
                [*_need(rec, "cont", path), _need(rec, "cont_greedy", path)],
                b.expected_stdout or "", str(correct))._asdict()
        row["admissible"] = bool(
            b.passed and (row["output_leak"] is None or row["output_leak"]["passed"]))
        out.append(row)
    return out


class Config(pydra.Config):  # type: ignore[misc]
    """See the module docstring for the input shapes; `--show` prints the resolved config."""

    def __init__(self) -> None:
        super().__init__()
        self.gate_out = ""      # REQUIRED: the Modal gate stage's raw output JSON
        self.bank = ""          # REQUIRED: the item bank (executed truth per item)
        self.json_out = ""      # verdicts output path (default: <gate_out>_verdicts.json)


@pydra.main(Config)  # type: ignore[untyped-decorator]
def main(config: Config) -> None:
    if not config.gate_out:
        raise SystemExit("gate_out= is required (the Modal gate stage's raw output JSON)")
    if not config.bank:
        raise SystemExit("bank= is required (the item bank with the executed truth per item)")

    blob = json.loads(Path(config.gate_out).read_text())
    records = blob if isinstance(blob, list) else _need(blob, "records", config.gate_out)
    bank = bank_index(json.loads(Path(config.bank).read_text()), config.bank)
    rows = verdicts(records, bank, path=config.gate_out)

    print(f"{'item':<34}{'executed truth':<24}{'consequence':<12}{'output leak':<12}admissible")
    print("-" * 96)
    for r in rows:
        b, e = r["consequence"], r["output_leak"]
        truth = (f"raises {b['exception_name']}" if b["exception_name"]
                 else f"stdout {b['expected_stdout']!r}")
        truth = truth if len(truth) <= 23 else truth[:20] + "..."
        b_txt = f"{b['hits']}/{b['n']}"
        e_txt = f"{e['leaks']}/{e['n']}" if e else "-"
        e_mark = ("pass" if e["passed"] else "FLAG") if e else "-"
        print(f"{r['name']:<34}{truth:<24}"
              f"{b_txt:>5} {'pass' if b['passed'] else 'CUT':<6}"
              f"{e_txt:>5} {e_mark:<6}"
              f"{'yes' if r['admissible'] else 'NO'}")
    n = len(rows)
    b_pass = sum(r["consequence"]["passed"] for r in rows)
    e_scored = [r for r in rows if r["output_leak"]]
    e_pass = sum(r["output_leak"]["passed"] for r in e_scored)
    print(f"\n{n} buggy items: consequence {b_pass} pass / {n - b_pass} cut"
          f" · output leak {e_pass} pass / {len(e_scored) - e_pass} flagged"
          f" ({n - len(e_scored)} without a twin stdout)"
          f" · admissible {sum(r['admissible'] for r in rows)}")
    dest = Path(config.json_out) if config.json_out else Path(config.gate_out).with_name(
        Path(config.gate_out).stem + "_verdicts.json")
    dest.write_text(json.dumps({"gate_out": config.gate_out, "bank": config.bank,
                                "verdicts": rows}, indent=1))
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
