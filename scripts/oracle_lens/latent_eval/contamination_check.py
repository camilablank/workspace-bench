# ruff: noqa  (one-off pilot gate; not linted)
"""Gate 3 (+ anchors): Opus 5 judge, 10-way FORCED choice, judge never sees the stimulus.

Two runs over every item that has an MC option set:
  anchors:       readout = the item's gold label       -> judge must pick it (expect ~100%).
  contamination: readout = a light stimulus paraphrase -> judge must NOT pick it (pass = miss;
                 dataset-level correct-pick rate should sit near chance = 1/10).

Option order is a per-item seeded shuffle (seed = sha256(DATASET_SEED:item_id)), the same
convention the eval-time judge will use. Results are written back into each item's qa dict.

Usage: uv run --no-sync python scripts/oracle_lens/latent_eval/contamination_check.py <items.json>
"""

import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from global_workspace.judges.llm_client import CLAUDE_JUDGE, async_json, schema_block  # noqa: E402

DATASET_SEED = 20260804

SYSTEM = """You are scoring readouts from an interpretability lens. A "readout" is text decoded \
from a language model's internal activations while it read a passage you cannot see. Your job is \
semantic matching only: decide which ONE of the numbered candidate descriptions of a character's \
latent mental state the readout itself expresses or describes. Do not try to guess or infer what \
the hidden passage might imply beyond what the readout states — match the readout's own content \
to the closest description. You MUST choose exactly one option number."""

USER_TMPL = """READOUT:
{readout}

CANDIDATE DESCRIPTIONS:
{options}

Which one option does the readout express? Answer with the option number."""

SCHEMA = (
    schema_block(  # Anthropic structured outputs reject min/max on integers -> validate in code
        "mc_choice",
        {"choice": {"type": "integer"}},
        ["choice"],
    )
)


def option_order(item_id: str, n: int) -> list[int]:
    seed = int.from_bytes(hashlib.sha256(f"{DATASET_SEED}:{item_id}".encode()).digest()[:8], "big")
    order = list(range(n))
    random.Random(seed).shuffle(order)
    return order  # order[k] = original index of the option shown at position k


def build_prompt(readout: str, options: list[str], order: list[int]) -> str:
    lines = [f"{k + 1}. {options[j]}" for k, j in enumerate(order)]
    return USER_TMPL.format(readout=readout, options="\n".join(lines))


def main() -> None:
    path = Path(sys.argv[1])
    items = json.loads(path.read_text())
    mc_items = [it for it in items if it.get("mc_options")]
    print(f"{len(mc_items)} items with MC option sets (of {len(items)})")

    prompts: list[tuple[str, str]] = []
    meta: list[tuple[dict, str, int]] = []  # (item, mode, shown_position_of_gold)
    for it in mc_items:
        opts = it["mc_options"]
        order = option_order(it["id"], len(opts))
        gold_pos = order.index(opts.index(it["gold_label"])) + 1  # 1-based shown position
        for mode, readout in (
            ("anchor_gold", it["gold_label"]),
            ("contamination", it["paraphrase"]),
        ):
            prompts.append((SYSTEM, build_prompt(readout, opts, order)))
            meta.append((it, mode, gold_pos))

    results = async_json(prompts, schema=SCHEMA, model=CLAUDE_JUDGE)

    n_anchor_ok = n_anchor = n_cont_ok = n_cont = 0
    for (it, mode, gold_pos), res in zip(meta, results):
        choice = res["choice"] if res else -1
        picked_gold = choice == gold_pos
        if mode == "anchor_gold":
            n_anchor += 1
            ok = picked_gold
            n_anchor_ok += ok
        else:
            n_cont += 1
            ok = not picked_gold
            n_cont_ok += ok
        it.setdefault("qa", {})[mode] = "pass" if ok else "FAIL"
        print(
            f"{it['id']:8s} {mode:13s} choice={choice:2d} gold_at={gold_pos:2d} "
            f"{'pass' if ok else 'FAIL'}"
        )

    print(f"\nanchors (gold-as-readout, expect ~100% correct): {n_anchor_ok}/{n_anchor}")
    print(
        f"contamination (paraphrase-as-readout, pass = judge misses): {n_cont_ok}/{n_cont} "
        f"(correct-pick rate {(n_cont - n_cont_ok) / max(n_cont, 1):.0%}, chance = 10%)"
    )
    path.write_text(json.dumps(items, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
