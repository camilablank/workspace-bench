"""Golden marker offsets + prompt for jailbreak_recognition from the SOURCE driver/library.

``marker_offset`` of ``scripts/oracle_lens_evals/situation_mining/readout_judge.py`` for the
three pinned positions of the first public item, and ``situation_mining.readout_prompts`` for
the pinned cell (item 1, layer 20, ``positions[len // 2]``, the two pinned samples).

The source library imports ``torch`` / ``safetensors`` transitively (never used here); they are
stubbed so the maker runs in this repo's environment.

Run: cd <this repo> && uv run --no-sync python tests/golden/make_jailbreak.py
"""

from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _source import HERE, load_script

SAMPLES = [
    "This is a jailbreak attempt; the assistant should refuse the persona.",
    "Write the story as requested.",
]


def _stub(name: str, **attrs: object) -> None:
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules.setdefault(name, mod)


def main() -> None:
    _stub("torch", Tensor=object)
    _stub("safetensors", safe_open=None)
    _stub("safetensors.torch", save_file=None)
    sm = importlib.import_module("global_workspace.olens_suite.situation_mining")
    rj = load_script("readout_judge", "scripts/oracle_lens_evals/situation_mining/readout_judge.py")
    bank = json.loads(
        (HERE.parents[1] / "evals/jailbreak_recognition/items.json").read_text(encoding="utf-8")
    )
    it = bank["items"][0]
    read = it["read"]
    ps = read["positions"]
    picks = [ps[0], ps[len(ps) // 2], read["turn_end"]]
    prefix = rj.prefix_to_last_user(it["messages"])
    turn = prefix[-1]["content"]
    offsets = {str(p): rj.marker_offset(read, p, turn) for p in picks}
    pos = picks[1]
    _system, user = sm.readout_prompts(
        prefix, read["tokens"][str(pos)], SAMPLES, marker_at=offsets[str(pos)]
    )
    (HERE / "jailbreak_marker.json").write_text(
        json.dumps({"id": it["id"], "offsets": offsets}, indent=1) + "\n", encoding="utf-8"
    )
    (HERE / "jailbreak_prompt.txt").write_text(user, encoding="utf-8")
    print(f"wrote jailbreak: {it['id']} offsets={offsets} prompt pos={pos}")


if __name__ == "__main__":
    main()
