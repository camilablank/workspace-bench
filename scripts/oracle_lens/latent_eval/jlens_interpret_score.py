# ruff: noqa  (analysis; typed at release)
"""J-lens scoring via blind interpretation (Camila's protocol, 2026-08-05).

Per (item, layer): the top-10 J-lens tokens go to an interpreter LM — "what are these 10
outputs trying to say?" — with NO access to the stimulus, options, or gold label. The
interpretation is then scored by the same per-axis judge as every other readout (exact
credit variant or direct semantic equivalent; composition = all scored axes correct).

Usage:
  uv run --no-sync python scripts/oracle_lens/latent_eval/jlens_interpret_score.py \
      hillclimbing_evals/compositional_association/items.json \
      outputs/oracle_lens_evals/olens_sglang/gen-latent-eval-jlens \
      --out outputs/oracle_latent_eval/jlens_interpretations.json [--layers 20,44] [--dump 3]
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_lens_readouts import load_readouts  # noqa: E402
from score_readout import score_items  # noqa: E402

from global_workspace.judges.llm_client import CLAUDE_JUDGE, async_json, schema_block  # noqa: E402

INTERP_SYSTEM = """You are shown the top-10 token readouts from an interpretability lens at one \
position inside a language model that was reading a passage you cannot see. Tokens may include \
noise, fragments, other languages (translate them), or byte artifacts. In one or two sentences, \
state what these outputs are collectively trying to say — the situation or mental content they \
point to. Commit to the most specific reading the tokens support; do not just say they are noisy."""

SCHEMA = schema_block("interp", {"interpretation": {"type": "string"}}, ["interpretation"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("items")
    ap.add_argument("gen_dir", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--layers", default=None, help="csv subset; default all found")
    ap.add_argument("--dump", type=int, default=0)
    args = ap.parse_args()
    items = json.loads(Path(args.items).read_text())
    by_layer = load_readouts(args.gen_dir)
    layers = [int(x) for x in args.layers.split(",")] if args.layers else sorted(by_layer)

    # cache interpretations across runs
    cache: dict[str, str] = {}
    if args.out.exists():
        cache = json.loads(args.out.read_text())

    pending, keys = [], []
    for layer in layers:
        for iid, toks in by_layer[layer].items():
            key = f"L{layer}:{iid}"
            if key not in cache:
                pending.append(
                    (
                        INTERP_SYSTEM,
                        f"TOP-10 READOUTS:\n{toks}\n\nWhat are these "
                        f"outputs collectively trying to say?",
                    )
                )
                keys.append(key)
    if pending:
        print(f"interpreting {len(pending)} grid points ...")
        res = async_json(pending, schema=SCHEMA, model=CLAUDE_JUDGE)
        n_failed = 0
        for key, r in zip(keys, res):
            interp = (r or {}).get("interpretation", "")
            if not interp:
                # a failed call must NOT be cached: an empty interpretation is a guaranteed
                # judged miss forever after (and only the J-lens arm runs this stage, so the
                # bias would be directional). Leave the key absent; the next run retries it.
                n_failed += 1
                continue
            cache[key] = interp
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(cache, indent=1, ensure_ascii=False))
        if n_failed:
            print(f"WARNING: {n_failed} interpretation calls failed (not cached; rerun to retry)")

    for layer in layers:
        readouts = {
            iid: cache[f"L{layer}:{iid}"]
            for iid in by_layer[layer]
            if f"L{layer}:{iid}" in cache
        }
        n_missing = len(by_layer[layer]) - len(readouts)
        if n_missing:
            print(f"L{layer:02d}: {n_missing} items have no interpretation — SKIPPED, not failed")
        v = score_items(items, readouts, use_judge=True, verbose=False)
        n_comp = sum(all(x == "correct" for x in vv.values()) for vv in v.values())
        per_axis: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for it in items:
            if it["id"] not in v:
                continue
            for a, x in v[it["id"]].items():
                per_axis[a][1] += 1
                per_axis[a][0] += x == "correct"
        ax = "  ".join(f"{a}:{ok}/{tot}" for a, (ok, tot) in sorted(per_axis.items()))
        print(f"L{layer:02d}  composition {n_comp}/{len(v)}   {ax}")
        if args.dump:
            for it in items[: args.dump]:
                print(f"    {it['id']} toks: {by_layer[layer][it['id']][:110]!r}")
                print(f"          interp: {readouts[it['id']][:150]!r}")


if __name__ == "__main__":
    main()
