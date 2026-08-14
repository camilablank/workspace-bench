# ruff: noqa  (analysis; typed at release)
"""Score lens readouts (teacher free-text or J-lens top-k tokens) on the latent eval.

Reads an olens_sglang gen dir (per-label L0NN.jsonl unit rows with `samples`), builds one
readout string per (item, layer), and scores it with the primary per-axis metric.

Judge policy (cost control): the per-layer profile is LEXICAL-ONLY; the single best layer
then gets the full judge-equivalence pass, which is the number to quote.

Usage:
  uv run --no-sync python scripts/oracle_lens/latent_eval/score_lens_readouts.py \
      <items.json> <gen_dir> [--dump N] [--judge-layer L | --no-judge-pass]
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_readout import axis_lexical, score_items  # noqa: E402
from lemma_scan import words  # noqa: E402

BPE_JUNK = re.compile(r"[ĠĊ▁]")


def norm_token(t: str) -> str:
    t = BPE_JUNK.sub(" ", t)
    return t.replace("_", " ").strip()


def load_readouts(gen_dir: Path) -> dict[int, dict[str, str]]:
    """{layer: {item_id: readout text}}"""
    out: dict[int, dict[str, str]] = defaultdict(dict)
    for label_dir in sorted(gen_dir.iterdir()):
        if not label_dir.is_dir():
            continue
        for lf in sorted(label_dir.glob("L*.jsonl")):
            for line in lf.read_text().splitlines():
                row = json.loads(line)
                text = " | ".join(norm_token(s) for s in row["samples"])
                prev = out[row["layer"]].get(row["label"], "")
                out[row["layer"]][row["label"]] = (prev + " | " + text) if prev else text
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("items")
    ap.add_argument("gen_dir", type=Path)
    ap.add_argument("--dump", type=int, default=0, help="print N example readouts per layer")
    ap.add_argument("--judge-layer", type=int, default=None)
    ap.add_argument("--no-judge-pass", action="store_true")
    ap.add_argument(
        "--agg-all",
        action="store_true",
        help="also score the ALL-layers x ALL-positions aggregate per item",
    )
    args = ap.parse_args()
    items = json.loads(Path(args.items).read_text())
    by_layer = load_readouts(args.gen_dir)
    if not by_layer:
        raise SystemExit(f"no unit rows found under {args.gen_dir}")

    print(
        f"layers: {sorted(by_layer)}; items with readouts: {max(len(v) for v in by_layer.values())}"
    )

    best_layer, best_axis_rate = None, -1.0
    for layer in sorted(by_layer):
        readouts = by_layer[layer]
        n_ax = n_ax_ok = n_comp = n = 0
        for it in items:
            r = readouts.get(it["id"])
            if r is None:
                continue
            n += 1
            w = set(words(r))
            vs = [axis_lexical(w, it, a) for a in it["scored_axes"]]
            n_ax += len(vs)
            n_ax_ok += sum(v == "correct" for v in vs)
            n_comp += all(v == "correct" for v in vs)
        rate = n_ax_ok / max(n_ax, 1)
        print(
            f"L{layer:02d}  axis-correct {n_ax_ok}/{n_ax} ({rate:.0%})  "
            f"composition {n_comp}/{n}  [lexical only]"
        )
        if rate > best_axis_rate:
            best_axis_rate, best_layer = rate, layer
        if args.dump:
            for it in items[: args.dump]:
                print(f"    {it['id']}: {readouts.get(it['id'], '')[:180]!r}")

    if args.agg_all:
        agg: dict[str, str] = {}
        for layer in sorted(by_layer):
            for iid, txt in by_layer[layer].items():
                prev = agg.get(iid, "")
                agg[iid] = (prev + f"\n[L{layer}] " + txt) if prev else f"[L{layer}] " + txt
        print("\n== ALL layers x ALL positions aggregate (judge pass) ==")
        v = score_items(items, agg, use_judge=not args.no_judge_pass, verbose=False)
        n_comp = sum(all(x == "correct" for x in vv.values()) for vv in v.values())
        per_axis = defaultdict(lambda: [0, 0])
        outcome = defaultdict(int)
        for it in items:
            if it["id"] not in v:
                continue
            for a, x in v[it["id"]].items():
                per_axis[a][1] += 1
                per_axis[a][0] += x == "correct"
                outcome[x] += 1
        for a, (ok, tot) in sorted(per_axis.items()):
            print(f"  axis {a:10s} {ok}/{tot}")
        print(f"  outcomes: {dict(outcome)}")
        print(f"  composition-correct: {n_comp}/{len(v)}")
        return

    jl = args.judge_layer if args.judge_layer is not None else best_layer
    if not args.no_judge_pass:
        print(f"\n== full judge-equivalence pass at L{jl} ==")
        v = score_items(items, by_layer[jl], use_judge=True, verbose=False)
        n_comp = sum(all(x == "correct" for x in vv.values()) for vv in v.values())
        per_axis: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for it in items:
            if it["id"] not in v:
                continue
            for a, x in v[it["id"]].items():
                per_axis[a][1] += 1
                per_axis[a][0] += x == "correct"
        for a, (ok, tot) in per_axis.items():
            print(f"  axis {a:10s} {ok}/{tot}")
        print(f"  composition-correct: {n_comp}/{len(v)}")


if __name__ == "__main__":
    main()
