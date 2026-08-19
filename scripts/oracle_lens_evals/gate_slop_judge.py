"""Gate the slop judge on REAL lens readouts before wiring it into any eval.

The acceptance criterion (Camila 2026-08-19): on hallucinatory NLA outputs that assert extra
incorrect details on the subject alongside the correct answer, the judge must score HIGH; on
readouts that are correct with just a little extra elaboration, it must score LOW — good
readouts must not get thrown away. This script builds that evidence: it takes a gen dir, keeps
the grid rows whose readout mechanically HITS the item's target (the only rows the precision
condition can ever affect), rates them with the slop judge, and writes a review artifact with
the full distribution plus the highest- and lowest-scored readouts printed for qualitative
inspection. Pick the threshold by reading that artifact, not by trusting the mean.

    source /workspace-vast/camilablank/.secrets/anthropic.env   # or scripts/cluster/env.sh
    uv run --no-sync python scripts/oracle_lens_evals/gate_slop_judge.py \
        --gen-dir <...>/gen-nla.rl.iter400-audit --tag nla \
        [--families basic-readout,multihop,...] [--layers 36,44,52] [--per-family 15]

Free-text readouts only — raw J-lens token bags are not "asserting" anything (see
slop_judge.py's scope note); summarize them first if you need them rated.
"""

import argparse
import json
import random
from pathlib import Path
from typing import Any

from global_workspace.judges.oracle_lens_judge import label_for, load_readouts
from global_workspace.judges.slop_judge import judge_slop, mechanical_targets, summarize_slop
from global_workspace.olens_suite.bank.loader import DEFAULT_ROOT, load_bank
from global_workspace.olens_suite.bank.matching import hit_any

REPO = Path(__file__).resolve().parents[2]
DEFAULT_FAMILIES = ("basic-readout", "multihop", "multilingual", "typo", "association", "poetry")
item_targets = mechanical_targets  # shared with the pipeline driver (judge_slop.py)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gen-dir", required=True, type=Path)
    p.add_argument("--tag", required=True, help="artifact label (e.g. nla / ao)")
    p.add_argument("--banks-dir", default=str(DEFAULT_ROOT))
    p.add_argument("--families", default=",".join(DEFAULT_FAMILIES))
    p.add_argument("--layers", default="", help="comma-separated subset (default: all present)")
    p.add_argument("--per-family", type=int, default=15, help="judged HIT rows per family")
    p.add_argument("--include-misses", type=int, default=0, help="also judge N miss rows/family")
    p.add_argument("--model", default="", help="override judge model")
    p.add_argument("--concurrency", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--show", type=int, default=8, help="print N highest + N lowest readouts")
    p.add_argument("--out", default="", help="default: outputs/slop_gate/<tag>.json")
    args = p.parse_args()

    layers = [int(x) for x in args.layers.split(",") if x.strip()] or None
    rng = random.Random(args.seed)
    pairs: list[tuple[dict[str, Any], str]] = []
    meta: list[dict[str, Any]] = []
    for family in [f.strip() for f in args.families.split(",") if f.strip()]:
        items = load_bank(family, args.banks_dir)
        hits: list[tuple[dict[str, Any], dict[str, Any]]] = []
        misses: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for item in items:
            targets = item_targets(item)
            if not targets:
                continue
            for row in load_readouts(args.gen_dir, label_for(item["name"]), layers, "all"):
                bucket = hits if hit_any([row["readout"]], targets) else misses
                bucket.append((item, row))
        rng.shuffle(hits)
        rng.shuffle(misses)
        take = hits[: args.per_family] + misses[: args.include_misses]
        print(
            f"[gate] {family}: {len(hits)} hit rows / {len(misses)} miss rows -> judging "
            f"{len(take)}"
        )
        for item, row in take:
            pairs.append((item, row["readout"]))
            meta.append(
                {
                    "family": family,
                    "layer": row["layer"],
                    "pos": row["pos"],
                    "sample_idx": row["sample_idx"],
                    "mech_hit": hit_any([row["readout"]], item_targets(item)),
                    "readout": row["readout"],
                }
            )
    if not pairs:
        raise SystemExit(f"no judgeable rows found under {args.gen_dir}")

    kwargs: dict[str, Any] = {"concurrency": args.concurrency}
    if args.model:
        kwargs["model"] = args.model
    judged = judge_slop(pairs, **kwargs)
    rows = [{**m, **j} for m, j in zip(meta, judged, strict=True)]
    rows.sort(key=lambda r: (r.get("slop") is None, -(r.get("slop") or 0.0)))
    summary = summarize_slop(rows)

    out = Path(args.out) if args.out else REPO / "outputs" / "slop_gate" / f"{args.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "tag": args.tag,
                "gen_dir": str(args.gen_dir),
                "config": {k: str(v) for k, v in vars(args).items()},
                "summary": summary,
                "rows": rows,
            },
            ensure_ascii=False,
            indent=1,
        )
    )
    print(f"[gate] -> {out}")
    print(json.dumps(summary, indent=1))

    if args.show <= 0:
        return
    scored = [r for r in rows if r.get("slop") is not None]
    for title, chunk in (
        (f"HIGHEST slop ({args.show})", scored[: args.show]),
        (f"LOWEST slop ({args.show})", scored[len(scored) - args.show :][::-1]),
    ):
        print(f"\n===== {title} =====")
        for r in chunk:
            print(
                f"\n--- slop={r['slop']} {r['family']}/{r['name']} L{r['layer']} "
                f"target={r['target']!r} present={r.get('target_present')}"
            )
            print(f"    extra: {r.get('extra_claims')}")
            print("    " + r["readout"][:400].replace("\n", "\n    "))


if __name__ == "__main__":
    main()
