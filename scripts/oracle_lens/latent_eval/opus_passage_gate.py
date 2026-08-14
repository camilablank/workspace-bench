# ruff: noqa  (gate; typed at release)
"""Opus-reads-the-passage gate (Camila, 2026-08-05): an item passes iff the Opus judge, shown
the FULL stimulus, answers YES to the gold composition and NO to the flipped sibling. This is
the item-validity ceiling promoted to a per-item gate: if a frontier reader with full text
can't endorse the gold conjunction (or can't reject the near-neighbor), the item is unfair to
any readout method.

API 400s are retried (3 rounds); an item is only marked on definitive answers.

Usage: uv run --no-sync python scripts/oracle_lens/latent_eval/opus_passage_gate.py \
    <items.json> [--only id1,id2,...]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from global_workspace.judges.llm_client import CLAUDE_JUDGE, async_json, schema_block  # noqa: E402

SYSTEM = """You are shown a short passage and one PROPOSITION describing a mental-state \
composition. Answer yes ONLY if the passage, read carefully, conveys ALL components of the \
proposition (the state/intent, what it concerns, and the person/relation where the proposition \
names one) — as the best reading of what the passage implies about the character. Components may be conveyed by implication — through behavior, contrasts between what is \
said and what is done, or the character's reactions — count a component as conveyed when it \
is the best reading the passage supports, even if never stated outright. Merely related or \
partially matching content is still NOT enough — answer no. Answer strictly yes or no."""
SCHEMA = schema_block("yn", {"answer": {"type": "string"}}, ["answer"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("items")
    ap.add_argument("--only", default=None)
    args = ap.parse_args()
    items_path = Path(args.items)
    items = json.loads(items_path.read_text())
    only = set(args.only.split(",")) if args.only else None

    todo = [
        (it, mode)
        for it in items
        if (only is None or it["id"] in only)
        for mode in ("gold", "flipped")
    ]
    votes: dict[tuple[str, str], list[str]] = {k: [] for k in ((it["id"], m) for it, m in todo)}
    for _round in range(5):  # majority-of-5 (boundary items flap under 3)
        prompts = []
        for it, mode in todo:
            label = it["gold_label"] if mode == "gold" else it["contrast_label"]
            prompts.append(
                (
                    SYSTEM,
                    f"PASSAGE:\n{it['stimulus']}\n\nPROPOSITION: {label}\n\n"
                    f"Does the passage convey all components of the "
                    f"proposition? Answer yes or no.",
                )
            )
        res = async_json(prompts, schema=SCHEMA, model=CLAUDE_JUDGE)
        for (it, mode), r in zip(todo, res):
            if r is not None:
                ans = (r.get("answer") or "").strip().lower()
                votes[(it["id"], mode)].append("yes" if ans.startswith("y") else "no")
    answers = {k: max(set(v), key=v.count) if v else None for k, v in votes.items()}

    n_pass = n = 0
    for it in items:
        if only is not None and it["id"] not in only:
            continue
        n += 1
        g = answers.get((it["id"], "gold"))
        f = answers.get((it["id"], "flipped"))
        ok = g == "yes" and f == "no"
        n_pass += ok
        it.setdefault("qa", {})["opus_gate"] = "pass" if ok else "FAIL"
        if not ok:
            print(f"{it['id']:8s} FAIL  gold->{g}  flipped[{it['contrast_axis']}]->{f}")
    print(f"\n{n_pass}/{n} items pass the opus-passage gate")
    items_path.write_text(json.dumps(items, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
