"""Build the public jailbreak_recognition bank from the source repo's bank (one-off, committed).

Keeps the items of the public bank (``KEEP_IDS``) and, per item, exactly the five fields the judge
reads: ``id``, ``source``, ``source_id``, ``messages`` (verbatim WildChat turns, including the
trailing assistant turn) and ``read`` (``positions``, ``turn_end``, ``n_tokens``, ``tokens``).

Run: cd <this repo> && uv run --no-sync python tests/golden/make_jailbreak_bank.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _source import HERE, read_bank

KEEP_IDS = frozenset(
    (
        "as-wildchat-20f6d664",
        "as-wildchat-574b06eb",
        "as-wildchat-45ebc413",
        "as-wildchat-5f8d0852",
        "as-wildchat-f0b3805b",
        "as-wildchat-cef2ae82",
        "as-wildchat-d480bc6f",
        "as-wildchat-a2522c5d",
        "as-wildchat-f8394621",
        "as-wildchat-de3e86e8",
        "as-wildchat-409351aa",
        "as-wildchat-9335de39",
        "as-wildchat-6b7287b7",
        "as-wildchat-413fb74c",
        "as-wildchat-dec48b7a",
        "as-wildchat-2f5a7c0a",
        "as-wildchat-5bbdabce",
        "as-wildchat-dbd80d72",
        "as-wildchat-d176cce9",
        "as-wildchat-9ec5de3b",
        "as-wildchat-c37715f4",
        "as-wildchat-020ee08c",
        "as-wildchat-da96b1a7",
        "as-wildchat-636ec994",
        "as-wildchat-426f6a17",
        "as-wildchat-8d95aaaf",
        "as-wildchat-e6f3117c",
        "as-wildchat-ecf896d0",
        "as-wildchat-37bd7a5e",
        "as-wildchat-c2296210",
        "as-wildchat-12175fa1",
        "as-wildchat-d8c1ef04",
        "as-wildchat-d40b14ae",
        "as-wildchat-ee0a0cf2",
        "as-wildchat-b81fb6c7",
        "as-wildchat-9d58ab80",
        "as-wildchat-74b41fef",
        "as-wildchat-e28fdb46",
        "as-wildchat-a21d0d47",
        "as-wildchat-1c3d52d2",
        "as-wildchat-3363efdd",
        "as-wildchat-4ba04577",
        "as-wildchat-a3658671",
        "as-wildchat-6ddaf185",
        "as-wildchat-01dedd49",
        "as-wildchat-c76e00e6",
        "as-wildchat-2d228371",
        "as-wildchat-b1040287",
        "as-wildchat-b15f9497",
        "as-wildchat-58b52f86",
        "as-wildchat-8e4ce349",
        "as-wildchat-ce04ae81",
        "as-wildchat-33ca0cee",
        "as-wildchat-69a00830",
        "as-wildchat-28778d0b",
        "as-wildchat-1d31a034",
        "as-wildchat-e30d2c04",
        "as-wildchat-c09ac99c",
        "as-wildchat-f91fc123",
        "as-wildchat-cb71c751",
        "as-wildchat-9b4b5457",
        "as-wildchat-4565f5ef",
        "as-wildchat-b6def3c9",
        "as-wildchat-bed3a766",
        "as-wildchat-74461479",
        "as-wildchat-8b43ea2a",
        "as-wildchat-b32ac395",
        "as-wildchat-80211947",
        "as-wildchat-9d835d4c",
        "as-wildchat-57f50a27",
        "as-wildchat-7307a8fb",
        "as-wildchat-582d98c7",
        "as-wildchat-e92ab591",
        "as-wildchat-12fe1503",
        "as-wildchat-45ba96c4",
        "as-wildchat-52502410",
        "as-wildchat-ab493c9a",
        "as-wildchat-8b5bf344",
        "as-wildchat-958b22ce",
        "as-wildchat-af8a915c",
        "as-wildchat-358488a2",
        "as-wildchat-4bb3f791",
        "as-wildchat-72956bf9",
        "as-wildchat-b412dc71",
        "as-wildchat-2b22e0a1",
        "as-wildchat-da4c7a15",
    )
)


READ_KEYS = ("positions", "turn_end", "n_tokens", "tokens")
OUT = HERE.parents[1] / "evals/jailbreak_recognition/items.json"


def main() -> None:
    src = read_bank("jailbreak_recognition/items.json")
    items = [
        {
            "id": it["id"],
            "source": it["source"],
            "source_id": it["source_id"],
            "messages": it["messages"],
            "read": {k: it["read"][k] for k in READ_KEYS},
        }
        for it in src["items"]
        if it["id"] in KEEP_IDS
    ]
    bank = {"family": "jailbreak_recognition", "n_items": len(items), "items": items}
    OUT.write_text(json.dumps(bank, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {OUT}: {len(items)} items")


if __name__ == "__main__":
    main()
