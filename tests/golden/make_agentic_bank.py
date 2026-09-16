"""Build ``evals/agentic_misalignment/items.json`` from the two SOURCE files (plan 0005).

Merges the 32-item pointer manifest (``{label, category, group, arm}``) with the frozen
scenario bank (``misalignment_exhaustive_bank.json``). Per item: ``id`` (= ``label``),
``label``, ``category``, ``arm``, ``group``, ``system``, ``text``, ``prefill``, ``descriptor``
(= ``source.note`` if present else ``look_for``), ``misalignment_rate`` (nullable),
``source_bank``, ``source_group``, ``rollout_pick``. Capture-time fields (``render``,
``gen_answer``, ``targets``, ``intermediates``) are dropped.

Run: cd <this repo> && uv run --no-sync python tests/golden/make_agentic_bank.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _source import R

POINTER = R / "evals/workspace-bench/baseline_evals/multi_token/agentic_misalignment/items.json"
SCENARIOS = R / "evals/diagnostic/misalignment_exhaustive_bank.json"
OUT = Path(__file__).resolve().parents[2] / "evals/agentic_misalignment/items.json"


def build() -> dict:
    pointer = json.loads(POINTER.read_text(encoding="utf-8"))
    scenarios = {
        it["label"]: it for it in json.loads(SCENARIOS.read_text(encoding="utf-8"))["items"]
    }
    items = []
    for p in pointer["items"]:
        s = scenarios[p["label"]]
        src = s.get("source", {})
        descriptor = src.get("note") or s.get("look_for", "")
        assert descriptor, p["label"]
        assert src.get("group", p["group"]) == p["group"], p["label"]
        items.append(
            {
                "id": p["label"],
                "label": p["label"],
                "category": p["category"],
                "arm": p["arm"],
                "group": p["group"],
                "system": s.get("system", ""),
                "text": s.get("text", ""),
                "prefill": s.get("prefill", ""),
                "descriptor": descriptor,
                "misalignment_rate": src.get("misalignment_rate"),
                "source_bank": src.get("bank"),
                "source_group": src.get("group"),
                "rollout_pick": src.get("rollout_pick"),
            }
        )
    return {"family": "agentic_misalignment", "n_items": len(items), "items": items}


def main() -> None:
    bank = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(bank, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {OUT}: {bank['n_items']} items")


if __name__ == "__main__":
    main()
