# ruff: noqa  (analysis; typed at release)
"""Contrastive judging of LENS readouts: per item, the judge sees the readout bundle and one
composition proposition, and answers yes/no — "does the readout express this?". Run twice per
item (gold -> want yes, flipped sibling -> want no), mirroring the Qwen contrastive gate.

Usage:
  uv run --no-sync python scripts/oracle_lens/latent_eval/contrastive_lens_judge.py \
      <items.json> <gen_dir> --tag <name> --out <results.json> [--layers 20,44] [--max-chars N]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_lens_readouts import load_readouts  # noqa: E402

from global_workspace.judges.llm_client import CLAUDE_JUDGE, async_json, schema_block  # noqa: E402

SYSTEM = """You judge readouts from an interpretability lens: text decoded from a language \
model's internal activations while it read a passage you cannot see (possibly many readouts, \
labeled by layer; tokens may include noise, fragments, or other languages). You are given one \
PROPOSITION describing a mental-state composition. Answer yes ONLY if EACH component of the \
proposition is EXPLICITLY PRESENT in the readouts as stated content — a word or phrase that \
itself names the state/intent, the event it concerns, and the person/relation where one is \
named — either the proposition's own words or a DIRECT synonym ('mourning' names grief; \
'wife' names spouse; 'his little boy' names her young child); related-but-different states \
are NOT synonyms ('regret' does not name grief, 'worried' does not name relieved). Do NOT \
infer components: scene details, narrative fragments, or reconstructed passage text from \
which a reader could deduce a state do NOT count — only content that states it. Answer \
strictly yes or no."""

SCHEMA = schema_block("yn", {"answer": {"type": "string"}}, ["answer"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("items")
    ap.add_argument("gen_dir", type=Path)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--layers", default=None)
    ap.add_argument("--max-chars", type=int, default=24000)
    args = ap.parse_args()
    items = json.loads(Path(args.items).read_text())
    by_layer = load_readouts(args.gen_dir)
    layers = [int(x) for x in args.layers.split(",")] if args.layers else sorted(by_layer)

    agg: dict[str, str] = {}
    for layer in layers:
        for iid, txt in by_layer.get(layer, {}).items():
            prev = agg.get(iid, "")
            agg[iid] = (prev + f"\n[L{layer}] " + txt) if prev else f"[L{layer}] " + txt

    prompts, meta = [], []
    for it in items:
        r = agg.get(it["id"])
        if not r:
            continue
        r = r[: args.max_chars]
        r = r.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
        r = "".join(ch for ch in r if ch.isprintable() or ch in "\n\t ")
        for mode, label in (("gold", it["gold_label"]), ("flipped", it["contrast_label"])):
            prompts.append(
                (
                    SYSTEM,
                    f"READOUTS:\n{r}\n\nPROPOSITION: {label}\n\n"
                    f"Do the readouts express all components of the "
                    f"proposition? Answer yes or no.",
                )
            )
            meta.append((it["id"], mode))
    res = async_json(prompts, schema=SCHEMA, model=CLAUDE_JUDGE)

    out: dict[str, dict[str, str]] = {}
    for (iid, mode), r in zip(meta, res):
        ans = ((r or {}).get("answer") or "").strip().lower()
        out.setdefault(iid, {})[mode] = "yes" if ans.startswith("y") else "no"
    n = len(out)
    yes_gold = sum(v.get("gold") == "yes" for v in out.values())
    no_flip = sum(v.get("flipped") == "no" for v in out.values())
    both = sum(v.get("gold") == "yes" and v.get("flipped") == "no" for v in out.values())
    summary = {
        "tag": args.tag,
        "n": n,
        "yes_on_gold": yes_gold,
        "no_on_flipped": no_flip,
        "both": both,
        "per_item": out,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=1))
    print(f"{args.tag}: yes-on-gold {yes_gold}/{n}  no-on-flipped {no_flip}/{n}  both {both}/{n}")


if __name__ == "__main__":
    main()
