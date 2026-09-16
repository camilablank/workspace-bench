"""Golden options + prompt for user_modeling from the SOURCE library's option functions.

``pick_options`` / ``build_options`` of ``global_workspace.judges.oracle_lens_judge`` run over
all 100 bank items (seed 0, 4 distractors, ``UM_OPTION_SUPPLEMENT``); the prompt is the
source's ``UM_ATTRIBUTE_USER.format(...)`` for the pinned example cell (item 1, sample 0).

Run: cd <this repo> && uv run --no-sync python tests/golden/make_user_modeling.py
"""

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _source import read_bank, write_golden

READOUT = "The user is Danish."  # examples/readouts/user_modeling.jsonl, item 1, sample 0


def main() -> None:
    olj = importlib.import_module("global_workspace.judges.oracle_lens_judge")
    items = read_bank("multi_token/lens-eval-user-modeling.json")["items"]
    names = [str(it["name"]) for it in items]
    distractors = olj.pick_options(
        names,
        [str(it["attr_class"]) for it in items],
        [str(it["attr"]) for it in items],
        n_distractors=4,
        seed=0,
        supplement=olj.UM_OPTION_SUPPLEMENT,
    )
    golden: dict = {}
    for it in items:
        name = str(it["name"])
        options, gold_position = olj.build_options(
            name,
            olj._display_value(str(it["attr"])),
            [olj._display_value(v) for v in distractors[name]],
            seed=0,
        )
        golden[name] = {"options": options, "gold_position": gold_position}
    first = items[0]
    user_turn = "\n\n".join(f"[{m['role']}] {m['content']}" for m in first["messages"])
    prompt = olj.UM_ATTRIBUTE_USER.format(
        user_turn=user_turn,
        readout=READOUT,
        options_block=olj._option_lines(golden[first["name"]]["options"]),
        cannot=olj.CANNOT_TELL,
    )
    write_golden("user_modeling", golden, prompt)


if __name__ == "__main__":
    main()
