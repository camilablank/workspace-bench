"""The headline metric: pairwise blind inference. Payload construction + ladder aggregation, pure.

A judge LM sees TWO readout sets — one from a buggy item, one from its surface-matched clean twin —
picks the buggy one, and describes the bug; a second judge grades the description on a 0-4
specificity ladder. Chance on the pick is exactly 0.5. This replaced the single-set yes/no, which
is nearly uninformative here: asked about one set at a time the judge says "there is a bug" on 0.64
of buggy items AND 0.50 of clean twins, because the AO's idiom is debugging-flavoured for both.
The twin cancels that idiom by construction; what survives is specificity.

WHY THE PAYLOAD IS BUILT HERE, AND WHY LABELS ARE ANONYMISED. A pooled-payload variant of this run
was VOIDED and kept in the record: its lines were labelled `L56@bugline:` / `L56@midline:`, so the
ANCHOR NAME told the judge which lines were read at the bug line — and LM(J-lens), at chance in
every honest run, scored 1.000/0.313/0.250. Scaffolding is part of the payload; audit it like one.
Cell labels here are positional (`p0`, `p1`, ...) assigned in sorted key order, so a caller's real
cell keys (`L56@bugline`) never reach the text.

WHY THE ORDER IS A STABLE HASH, NOT `random.shuffle`. Two arms (AO readouts and J-lens token lists)
are graded on the same pairs, and a re-scored payload must be the same experiment: hashing
(item name, rubric version) fixes the order per item, keeps the arms in step, and still leaves it
unguessable from the item. Bumping `rubric_version` re-randomises every order on purpose.

RUN (stages 3-4 of the chain; see scripts/olens_suite/README.md):

    # build the anonymised payloads for the judge
    uv run python -m global_workspace.olens_suite.buggy_code.pairwise \\
        build=True records=bug_read_out.json twin=bug_read_out.json out=payloads.txt
    # ... judge the payloads, store one verdict per (item, arm) ...
    uv run python -m global_workspace.olens_suite.buggy_code.pairwise records=pairwise_out.json

CLI CONFIG (pydra; pass `--show` to print the resolved config and exit)

  records         judge verdicts JSON (score mode), or the buggy readouts JSON (build mode)
  build           bool; emit anonymised payloads instead of scoring
  twin            the clean twins' readouts JSON (build mode)
  out             write payloads here instead of stdout (build mode)
  rubric_version  defaults to RUBRIC_VERSION; bumping it re-randomises every set order

INPUT FORMAT

  records (score) {"picks": {"<item>|<arm>": {"correct_pick": bool, "level": int,
                                              "guess": str}}}      # `level` may be absent for an
                  or a bare list of {"correct_pick", "level"}      # ungraded arm -> counted as 0
  records (build) {"records": [{"name": str, "src": "buggy"|"clean",
                                "cells": {"<key>": [str, ...]}}]}  # or "readout": [str, ...] for
  twin            same shape; the clean twins                      # the single frozen cell
                  Pairing is by the buggy record's "twin" field when present, else by position.
"""

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from statistics import mean
from typing import Any, NamedTuple

import pydra

RUBRIC_VERSION = "pairwise-v2"
SET_LABELS = ("A", "B")

# The specificity ladder. 2 is where the metric starts being about the BUG rather than about the
# AO's tone; 3 is the checkpoint's ceiling (0.06) and 4 has never been observed at the frozen cell.
LEVELS: dict[int, str] = {
    0: "wrong content",
    1: "generic trouble",
    2: "consequence",
    3: "mechanism",
    4: "mechanism tied to the line",
}

# Anchor names from READ_CELLS/SWEEP_ANCHORS. Listed so a payload can be asserted free of them;
# see the voided run above.
ANCHOR_NAMES = ("eof", "bugline", "innocent", "midline")


def buggy_first(item_name: str, rubric_version: str = RUBRIC_VERSION) -> bool:
    """Deterministic set order: same (item, rubric) -> same order, on any machine, on a re-run."""
    digest = hashlib.sha256(f"{rubric_version}\x00{item_name}".encode()).digest()
    return digest[0] % 2 == 0


class Payload(NamedTuple):
    text: str
    buggy_label: str
    cell_labels: tuple[str, ...]


def build_payload(item_name: str,
                  buggy_cells: Mapping[str, Sequence[str]],
                  clean_cells: Mapping[str, Sequence[str]],
                  *, rubric_version: str = RUBRIC_VERSION) -> Payload:
    """One anonymised two-set payload for `item_name` and its clean twin.

    `*_cells` map a cell key (whatever the runner called it) to that cell's readout samples. Both
    sides must carry the SAME key set: `p0` has to mean the same activation in both sets, or the
    label itself becomes a hint about which set is which.

    Each sample is flattened to one line so that a sample's own newlines cannot be mistaken for
    payload structure — the readout idiom here is multi-line assertion frames.
    """
    if set(buggy_cells) != set(clean_cells):
        raise ValueError(
            f"{item_name}: buggy and clean cell keys differ "
            f"({sorted(buggy_cells)} vs {sorted(clean_cells)}) — positional labels would then "
            f"mean different activations in the two sets."
        )
    keys = sorted(buggy_cells)
    labels = tuple(f"p{i}" for i in range(len(keys)))

    def block(label: str, cells: Mapping[str, Sequence[str]]) -> str:
        lines = [f"SET {label}"]
        for key, lab in zip(keys, labels, strict=True):
            lines += [f"  {lab}: {' '.join(sample.split())}" for sample in cells[key]]
        return "\n".join(lines)

    first_is_buggy = buggy_first(item_name, rubric_version)
    order = (buggy_cells, clean_cells) if first_is_buggy else (clean_cells, buggy_cells)
    text = "\n\n".join(block(lab, cells) for lab, cells in zip(SET_LABELS, order, strict=True))
    return Payload(text, SET_LABELS[0] if first_is_buggy else SET_LABELS[1], labels)


class PairRecord(NamedTuple):
    correct_pick: bool
    level: int


def pair_records(rows: Iterable[Mapping[str, Any]]) -> list[PairRecord]:
    """Judge verdicts (as stored) -> typed records, validating the level."""
    out: list[PairRecord] = []
    for row in rows:
        level = int(row["level"])
        if level not in LEVELS:
            raise ValueError(f"level {level} is not on the ladder {sorted(LEVELS)}")
        out.append(PairRecord(bool(row["correct_pick"]), level))
    return out


class Ladder(NamedTuple):
    n: int
    pick_accuracy: float
    level2_rate: float
    level3_rate: float


def ladder(records: Sequence[PairRecord]) -> Ladder:
    """Pick accuracy plus the two ladder rates. Chance on the pick is 0.5, NOT 0.

    Reported as rates over PAIRS, so `n` travels with them: the scored set is 16 pairs and the
    Gate-E survivor subset is 3, where 0.667 and 0.333 are one verdict apart.
    """
    if not records:
        raise ValueError("no pair records — an empty set must not report a rate of 0.0")
    return Ladder(
        n=len(records),
        pick_accuracy=mean(r.correct_pick for r in records),
        level2_rate=mean(r.level >= 2 for r in records),
        level3_rate=mean(r.level >= 3 for r in records),
    )


# ---- CLI: parse, call the functions above, print. No metric logic lives below this line. ------

def _need(obj: Mapping[str, Any], key: str, where: str) -> Any:
    if key not in obj:
        raise SystemExit(f"{where}: missing key {key!r} (have {sorted(obj)}) — see the module "
                         f"docstring for the expected shape")
    return obj[key]


def arms(blob: Any, path: str) -> dict[str, list[dict[str, Any]]]:
    """`{"picks": {"item|arm": verdict}}` -> `{arm: [verdict, ...]}`. A bare list is one arm."""
    if isinstance(blob, list):
        return {"all": list(blob)}
    picks = _need(blob, "picks", path)
    out: dict[str, list[dict[str, Any]]] = {}
    for key, verdict in picks.items():
        arm = key.split("|", 1)[1] if "|" in key else "all"
        out.setdefault(arm, []).append({"item": key.split("|", 1)[0], **verdict})
    return out


def _cells(record: Mapping[str, Any], path: str) -> dict[str, list[str]]:
    if "cells" in record:
        return {k: list(v) for k, v in record["cells"].items()}
    return {"cell": list(_need(record, "readout", path))}


def _report_ladder(name: str, rows: Sequence[Mapping[str, Any]]) -> None:
    ungraded = sum(1 for r in rows if "level" not in r)
    got = ladder(pair_records([{"correct_pick": r["correct_pick"], "level": r.get("level", 0)}
                               for r in rows]))
    print(f"{name:<10}{got.n:>4}{got.pick_accuracy:>9.3f}{got.level2_rate:>9.3f}"
          f"{got.level3_rate:>9.3f}"
          + (f"   ({ungraded} verdicts carry no `level` — counted as 0)" if ungraded else ""))


class Config(pydra.Config):  # type: ignore[misc]
    """See the module docstring for the input shapes; `--show` prints the resolved config."""

    def __init__(self) -> None:
        super().__init__()
        self.records = ""       # judge verdicts (score mode) or buggy readouts (build mode)
        self.build = False      # emit anonymised payloads instead of scoring
        self.twin = ""          # the clean twins' readouts (build mode)
        self.out = ""           # write payloads here instead of stdout (build mode)
        self.rubric_version = RUBRIC_VERSION


@pydra.main(Config)  # type: ignore[untyped-decorator]
def main(config: Config) -> None:
    if config.build:
        if not config.records:
            raise SystemExit("build=True needs records= (the buggy readouts JSON)")
        if not config.twin:
            raise SystemExit("build=True needs twin= (the clean twins' readouts)")
        buggy = [r for r in _need(json.loads(Path(config.records).read_text()), "records",
                                  config.records)
                 if r.get("src", "buggy") == "buggy"]
        twins = [r for r in _need(json.loads(Path(config.twin).read_text()), "records",
                                  config.twin)
                 if r.get("src", "clean") == "clean"]
        by_name = {_need(t, "name", config.twin): t for t in twins}
        if len(buggy) != len(twins) and not all("twin" in b for b in buggy):
            raise SystemExit(f"{len(buggy)} buggy vs {len(twins)} clean records and no `twin` "
                             f"field to pair on — pairing would be a guess")
        blocks = []
        for i, b in enumerate(buggy):
            name = _need(b, "name", config.records)
            twin = by_name[b["twin"]] if b.get("twin") in by_name else twins[i]
            p = build_payload(name, _cells(b, config.records), _cells(twin, config.twin),
                              rubric_version=config.rubric_version)
            blocks.append(f"### PAIR {name}  (buggy set = {p.buggy_label}, "
                          f"cells {', '.join(p.cell_labels)})\n{p.text}")
            print(f"[pair] {name:<32} buggy={p.buggy_label}  twin={twin['name']}")
        text = "\n\n".join(blocks) + "\n"
        if config.out:
            Path(config.out).write_text(text)
            print(f"wrote {config.out} ({len(blocks)} payloads)")
        else:
            print(text)
        return

    if not config.records:
        raise SystemExit("give records= (a verdicts JSON), or build=True records=... twin=... "
                         "to emit payloads")
    print(f"{'arm':<10}{'n':>4}{'pick':>9}{'lvl>=2':>9}{'lvl>=3':>9}   (pick chance = 0.5)")
    print("-" * 60)
    for arm, rows in arms(json.loads(Path(config.records).read_text()), config.records).items():
        _report_ladder(arm, rows)


if __name__ == "__main__":
    main()
