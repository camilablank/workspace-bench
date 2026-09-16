"""Golden options for moral_rationale from the SOURCE script's option functions.

Run: cd <this repo> && uv run --no-sync python tests/golden/make_moral_rationale.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _source import TOY_READOUT, load_script, read_bank, write_golden


def main() -> None:
    src = load_script("ec_readout_judge", "scripts/oracle_lens_evals/ec_readout_judge.py")
    items = read_bank("moral_rationale/items.json")
    comm, yes, no = src.build_pools(items)
    golden: dict = {}
    prompt = None
    for it in items:
        mcs = src.build_mcs(it, comm, yes, no)
        golden[it["id"]] = {
            side: {"shown": shown, "gold_pos": gold_pos, "short_pool": short}
            for side, (_q, gold_pos, shown, _nd, short) in mcs.items()
        }
        if prompt is None:
            q = next(iter(mcs.values()))[0]
            prompt = f"READOUT:\n{TOY_READOUT}\n\n{q}"  # source L303
    assert prompt is not None
    write_golden("moral_rationale", golden, prompt)


if __name__ == "__main__":
    main()
