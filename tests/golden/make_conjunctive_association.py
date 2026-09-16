"""Golden options for conjunctive_association from the SOURCE script's ``build_question``.

Run: cd <this repo> && uv run --no-sync python tests/golden/make_conjunctive_association.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _source import TOY_READOUT, load_script, read_bank, write_golden


def main() -> None:
    src = load_script("judge_mc", "scripts/oracle_lens/latent_eval/judge_mc.py")
    items = read_bank("multi_token/conjunctive_association/items.json")
    golden: dict = {}
    prompt = None
    for it in items:
        q, gold_pos, shown = src.build_question(it)
        golden[it["id"]] = {"shown": shown, "gold_pos": gold_pos}
        if prompt is None:
            prompt = f"READOUTS:\n{TOY_READOUT}\n\n{q}"  # source L190
    assert prompt is not None
    write_golden("conjunctive_association", golden, prompt)


if __name__ == "__main__":
    main()
