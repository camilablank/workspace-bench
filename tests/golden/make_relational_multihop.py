"""Golden options for relational_multihop from the SOURCE script's option functions.

Run: cd <this repo> && uv run --no-sync python tests/golden/make_relational_multihop.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _source import (
    TOY_READOUT,
    load_script,
    read_bank,
    shown_from_listing,
    write_golden,
)


def main() -> None:
    src = load_script(
        "judge_relational_multihop", "scripts/oracle_lens_evals/judge_relational_multihop.py"
    )
    items = read_bank("relational_multihop/cloze_items_final.json")
    prof, kin = src.pools(items)
    golden: dict = {}
    for it in items:
        listing, p1, p2 = src.build_mc(it["id"], it["hop1"], it["hop2"], prof, kin)
        golden[it["id"]] = {
            "shown": shown_from_listing(listing),
            "x_gold_pos": p1,
            "y_gold_pos": p2,
        }
    body, _gold = src.question(items[0], TOY_READOUT, "X", prof, kin)
    write_golden("relational_multihop", golden, body)


if __name__ == "__main__":
    main()
