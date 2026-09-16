"""Build the public jailbreak_recognition bank from the source repo's bank (one-off, committed).

Keeps the source bank's ``bank_ok`` items and, per item, exactly the five fields the judge
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
        if it.get("bank_ok") is True
    ]
    bank = {"family": "jailbreak_recognition", "n_items": len(items), "items": items}
    OUT.write_text(json.dumps(bank, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {OUT}: {len(items)} items")


if __name__ == "__main__":
    main()
