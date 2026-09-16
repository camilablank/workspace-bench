"""Golden question blocks for role_bound_association from the SOURCE script's ``oa_question``.

Run: cd <this repo> && uv run --no-sync python tests/golden/make_role_bound_association.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _source import TOY_READOUT, load_script, read_bank, write_golden


def main() -> None:
    src = load_script("oa_eb_readout_judge", "scripts/oracle_lens_evals/oa_eb_readout_judge.py")
    items = read_bank("role_bound_association/items.json")
    golden: dict = {}
    for it in items:
        block, golds = src.oa_question(it, items)
        golden[it["id"]] = {"block": block, "golds": list(golds)}
    block0 = golden[items[0]["id"]]["block"]
    write_golden("role_bound_association", golden, f"READOUT:\n{TOY_READOUT}\n\n{block0}")


if __name__ == "__main__":
    main()
