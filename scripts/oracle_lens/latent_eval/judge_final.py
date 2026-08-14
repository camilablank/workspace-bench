# ruff: noqa  (final instrument; typed at release)
"""FINAL readout instrument (Camila, 2026-08-05) — single uniform judge, per component.

For every item and every scored axis, ONE question: is this compositional item present in
the readouts, as the direct word(s) or an EXACT synonym? yes/no + verbatim quote. The same
question is asked for the flipped sibling's differing component(s). The script verifies each
quote is a verbatim substring of the readout; an unverifiable quote voids the yes.

Item passes iff: every gold component -> verified yes, AND every flipped differing
component -> no. Per-axis presence rates are reported alongside.

Usage: uv run --no-sync python scripts/oracle_lens/latent_eval/judge_final.py \
    <items.json> <gen_dir> --tag <name> [--out results.json]
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_lens_readouts import load_readouts  # noqa: E402

from global_workspace.judges.llm_client import CLAUDE_JUDGE, async_json, schema_block  # noqa: E402

SYSTEM = """You check readouts from an interpretability lens: text decoded from a language \
model's internal activations while it read a passage you cannot see (possibly labeled by \
layer; may contain noise, fragments, or other languages). You are given ONE compositional \
item. Answer yes ONLY if it is present in the readouts as the direct word(s) or an EXACT \
synonym — a word or phrase naming precisely the same thing ("mourning" names grief; "wife" \
names spouse; "his little boy" names her young child). Do NOT count related-but-different \
states ("regret" is not grief; "worried" is not relieved), generic words too broad to name \
it ("sad", "upset"), or evidence, causes, scene details, or narrative from which it could \
merely be inferred ("small sneakers" is not "her young child"). Answer yes or no; if yes, \
give the exact quote VERBATIM from the readouts."""

SCHEMA = schema_block(
    "presence", {"answer": {"type": "string"}, "quote": {"type": "string"}}, ["answer", "quote"]
)


def norm(t: str) -> str:
    return " ".join(t.lower().split())


DEFAULT_ITEMS = Path(__file__).resolve().parents[3] / "hillclimbing_evals/compositional_association/items.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("items", nargs="?", default=str(DEFAULT_ITEMS), help="frozen items.json")
    ap.add_argument("gen_dir", type=Path)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--resume", action="store_true", help="skip items already present in --out")
    args = ap.parse_args()
    items = json.loads(Path(args.items).read_text())
    by_id = {it["id"]: it for it in items}

    by_layer = load_readouts(args.gen_dir)
    layers = sorted(by_layer)
    agg = {}
    for l in layers:
        for iid, txt in by_layer[l].items():
            agg[iid] = agg.get(iid, "") + f"\n[L{l}] " + txt
    agg = {
        k: "".join(ch for ch in v[:24000] if ch.isprintable() or ch in "\n\t ")
        for k, v in agg.items()
        if all(k in by_layer[l] for l in layers)
    }

    # flipped differing components: the contrast partner's concept on axes where it differs
    label_to_item = {it["gold_label"]: it for it in items}
    done: dict = {}
    if args.resume and args.out and args.out.exists():
        done = json.loads(args.out.read_text()).get("per_item", {})
    prompts, meta = [], []
    for it in items:
        r = agg.get(it["id"])
        if not r or it["id"] in done:
            continue
        partner = label_to_item.get(it["contrast_label"])
        for axis in it["scored_axes"]:
            prompts.append(
                (
                    SYSTEM,
                    f"COMPOSITIONAL ITEM: {it['axis_concepts'][axis]}\n\n"
                    f"READOUTS:\n{r}\n\nIs this item present? Answer yes or "
                    f"no; if yes, quote it verbatim.",
                )
            )
            meta.append((it["id"], "gold", axis))
            if partner and partner["axis_concepts"][axis] != it["axis_concepts"][axis]:
                prompts.append(
                    (
                        SYSTEM,
                        f"COMPOSITIONAL ITEM: "
                        f"{partner['axis_concepts'][axis]}\n\nREADOUTS:\n{r}"
                        f"\n\nIs this item present? Answer yes or no; if "
                        f"yes, quote it verbatim.",
                    )
                )
                meta.append((it["id"], "flipped", axis))

    res = async_json(prompts, schema=SCHEMA, model=CLAUDE_JUDGE, concurrency=256)
    verdict: dict[str, dict[str, dict[str, bool]]] = defaultdict(lambda: defaultdict(dict))
    for iid, sides in done.items():  # carry over resumed verdicts
        for side, axes in sides.items():
            verdict[iid][side].update(axes)
    for (iid, side, axis), r in zip(meta, res):
        yes = bool(r and (r.get("answer") or "").strip().lower().startswith("y"))
        if yes:
            q = (r.get("quote") or "").strip()
            if not q or norm(q) not in norm(agg[iid]):
                yes = False  # quote verification failed -> void
        verdict[iid][side][axis] = yes

    n = n_pass = n_gold_all = n_flip_clean = 0
    per_axis = defaultdict(lambda: [0, 0])
    for iid, v in verdict.items():
        it = by_id[iid]
        n += 1
        gold_ok = all(v["gold"].get(a, False) for a in it["scored_axes"])
        flip_clean = not any(v.get("flipped", {}).values())
        n_gold_all += gold_ok
        n_flip_clean += flip_clean
        n_pass += gold_ok and flip_clean
        for a in it["scored_axes"]:
            per_axis[a][1] += 1
            per_axis[a][0] += v["gold"].get(a, False)
    print(
        f"{args.tag}: all-gold-present {n_gold_all}/{n}   flipped-absent {n_flip_clean}/{n}"
        f"   PASS(both) {n_pass}/{n}"
    )
    print("per-axis present:", {a: f"{ok}/{tot}" for a, (ok, tot) in sorted(per_axis.items())})
    if args.out:
        args.out.write_text(
            json.dumps(
                {
                    "tag": args.tag,
                    "n": n,
                    "gold_all": n_gold_all,
                    "flip_clean": n_flip_clean,
                    "pass": n_pass,
                    "per_axis": {a: v for a, v in per_axis.items()},
                    "per_item": {k: dict(v) for k, v in verdict.items()},
                },
                indent=1,
            )
        )


if __name__ == "__main__":
    main()
