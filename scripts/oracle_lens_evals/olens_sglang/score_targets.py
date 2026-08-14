"""Stage 3 (CPU, local): score generated grids against each prompt's target strings.

Pass@k per grid point = any of the k samples contains any target (case-insensitive substring,
the eval_prompts.json convention). Aggregates, per prompt and per family:
- hit matrix summaries (which layers / positions surface the intermediate),
- earliest hitting layer, and hits-by-layer grids.

``--baseline-dir`` scores a second gen dir on the identical grid (e.g. the J-lens baseline
from jlens_eval.py — same unit-file schema) and prints a side-by-side per-family table.

This is the cheap mechanical pass; LLM-judge scoring of open-family items layers on top later.

    uv run --no-sync python scripts/oracle_lens_evals/olens_sglang/score_targets.py \
        --acts-dir outputs/oracle_lens_evals/olens_sglang/acts \
        --gen-dir  outputs/oracle_lens_evals/olens_sglang/gen-<name> \
        --baseline-dir outputs/oracle_lens_evals/olens_sglang/gen-jlens-n1000
"""

import argparse
import json
import sys
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ActsManifest, Progress, atomic_write_text, load_unit_rows, present_layers

# Hit semantics live in ONE place now (matching.py) so the headline scorer, the digests,
# and the ablation report cannot drift apart; these names stay importable from here.
from global_workspace.olens_suite.bank.matching import (
    content_targets,
    exact_targets,
    lenient_matcher,
    sample_equals_number,
    word_matcher,
)


def score_dir(
    manifest: ActsManifest,
    gen_dir: Path,
    match_mode: str = "word",
    exact: bool = True,
    restrict_layers: set[int] | None = None,
) -> dict[str, Any]:
    """Score one gen dir -> {per_family_pass_rate, prompts, units_*, dropped_contaminated}.

    Accounting runs over the layers the run actually generated (``layers_scored``), not all
    manifest layers — a 17-of-64-layer run is complete, not "47/64 missing". Rows come
    through ``common.load_unit_rows`` (eval-position mask + extract_phrase), and the
    worker's contamination drops are summed instead of silently deflating pass rates.

    ``restrict_layers`` intersects the scored layers with a given set — used to hold the
    J-lens baseline to the SAME layers the lens under test generated, so pass@(any layer)
    is never inflated by the baseline covering more layers (e.g. 63 J-lens layers vs 12 AO).
    """
    layers = present_layers(gen_dir, manifest.layers)
    if restrict_layers is not None:
        layers = [layer for layer in layers if layer in restrict_layers]
    tgt_of = {
        e.label: content_targets(exact_targets(e.targets) if exact else list(e.targets))
        for e in manifest.prompts
    }
    targets = {label: [t.lower() for t in ts] for label, ts in tgt_of.items()}
    make = lenient_matcher if match_mode == "lenient" else word_matcher
    matchers: dict[str, list[Callable[[str], bool]]] = {
        label: [make(t) for t in ts] for label, ts in tgt_of.items()
    }
    num_of = {label: [t for t in ts if t.strip().isdigit()] for label, ts in targets.items()}
    per_prompt: dict[str, dict[str, object]] = {}
    fam_hits: dict[str, list[float]] = defaultdict(list)
    n_units = n_missing = n_dropped = 0
    bar = Progress(len(manifest.prompts), f"score {gen_dir.name}", unit="prompt")
    for entry in manifest.prompts:
        tgt = targets[entry.label]
        fns = matchers[entry.label]
        by_layer, missing = load_unit_rows(gen_dir, entry, layers)
        n_units += len(layers)
        n_missing += len(missing)
        layer_hits: dict[int, list[int]] = {}
        for layer, rows in by_layer.items():
            hits = []
            for row in rows:
                n_dropped += int(row.get("dropped_contaminated", 0) or 0)
                # pass@k = ANY single sample contains a target. Match each sample on its own,
                # never a " \n "-join of all k samples: joining let adjacent J-lens top-k
                # tokens ("New" | "Delhi") fuse into a fake multi-word match that no single
                # token can hold — which silently inflated the J-lens -mt baseline.
                samples = [s.lower() for s in row["samples"]]
                if match_mode in ("word", "lenient"):
                    matched = bool(fns) and any(fn(s) for fn in fns for s in samples)
                    if not matched and match_mode == "word":
                        matched = any(
                            sample_equals_number(s, t)
                            for t in num_of[entry.label]
                            for s in row["samples"]
                        )
                    if matched:
                        hits.append(row["pos"])
                elif tgt and any(t in s for t in tgt for s in samples):
                    hits.append(row["pos"])
            layer_hits[layer] = hits
        bar.update(1)
        if not layer_hits or not tgt:
            continue
        hitting_layers = sorted(layer for layer, h in layer_hits.items() if h)
        fam_hits[entry.family].append(float(bool(hitting_layers)))
        per_prompt[entry.label] = {
            "family": entry.family,
            "hit": bool(hitting_layers),
            "earliest_layer": hitting_layers[0] if hitting_layers else None,
            "n_hitting_layers": len(hitting_layers),
            "hits_by_layer": {str(layer): h for layer, h in layer_hits.items() if h},
        }
    bar.close()
    return {
        "units_expected": n_units,
        "units_missing": n_missing,
        "layers_scored": layers,
        "dropped_contaminated": n_dropped,
        "per_family_pass_rate": {
            fam: round(sum(v) / len(v), 4) for fam, v in sorted(fam_hits.items())
        },
        "prompts": per_prompt,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--acts-dir", required=True, type=Path)
    p.add_argument("--gen-dir", required=True, type=Path)
    p.add_argument(
        "--baseline-dir",
        type=Path,
        default=None,
        help="second gen dir (same schema/grid) scored for a side-by-side table",
    )
    p.add_argument("--out", type=Path, default=None, help="default: <gen-dir>/scores.json")
    p.add_argument(
        "--match",
        default="word",
        choices=["substring", "word", "lenient"],
        help="word (default) = boundary-anchored match — the headline metric everywhere; "
        "substring = the legacy alias-inclusive mode",
    )
    p.add_argument(
        "--exact",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="drop component aliases (targets word-contained in a longer sibling target); "
        "on by default — every published number is word+exact. --no-exact for legacy runs",
    )
    p.add_argument(
        "--match-baseline-layers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="score the baseline (J-lens) only on the layers the lens under test generated, "
        "so pass@(any layer) is a like-for-like comparison — on by default. --no-match-"
        "baseline-layers to score each dir on all of its own layers (diagnostic only)",
    )
    args = p.parse_args()

    manifest = ActsManifest.load(args.acts_dir)
    summary = score_dir(manifest, args.gen_dir, match_mode=args.match, exact=args.exact)
    summary["match_mode"] = args.match
    summary["exact_targets"] = args.exact
    n_scored = len(summary["prompts"])
    print(
        f"[score] {args.gen_dir.name}: {n_scored} prompts (match={args.match} "
        f"exact={args.exact}), layers={summary['layers_scored']}, "
        f"{summary['units_missing']}/{summary['units_expected']} units missing"
    )
    if summary["dropped_contaminated"]:
        print(
            f"[score] WARNING: {summary['dropped_contaminated']} samples were dropped as "
            "bank-contaminated by the worker — affected rows scored on their remaining "
            "samples (a row with ALL samples dropped scores as a miss)"
        )

    if args.baseline_dir is not None:
        restrict = set(summary["layers_scored"]) if args.match_baseline_layers else None
        base = score_dir(
            manifest, args.baseline_dir, match_mode=args.match, exact=args.exact,
            restrict_layers=restrict,
        )
        if args.match_baseline_layers:
            print(
                f"[score] baseline scored on the lens-under-test's {len(summary['layers_scored'])} "
                f"layers only (layer-matched); baseline layers actually scored: "
                f"{base['layers_scored']}"
            )
        summary["baseline_dir"] = str(args.baseline_dir)
        summary["baseline_layer_matched"] = args.match_baseline_layers
        summary["baseline"] = {
            "per_family_pass_rate": base["per_family_pass_rate"],
            "units_missing": base["units_missing"],
            "layers_scored": base["layers_scored"],
        }
        fams = sorted(set(summary["per_family_pass_rate"]) | set(base["per_family_pass_rate"]))
        width = max((len(f) for f in fams), default=6)
        print(
            f"  {'family':>{width}}  {args.gen_dir.name[:20]:>20}  "
            f"{args.baseline_dir.name[:20]:>20}"
        )
        for fam in fams:
            a = summary["per_family_pass_rate"].get(fam)
            b = base["per_family_pass_rate"].get(fam)
            print(
                f"  {fam:>{width}}  {a if a is not None else '—':>20}  "
                f"{b if b is not None else '—':>20}"
            )
        # per-prompt earliness side-by-side for prompts both lenses hit
        earliness = {}
        for label, row in summary["prompts"].items():
            brow = base["prompts"].get(label)
            if brow:
                earliness[label] = {
                    "lens": row["earliest_layer"],
                    "baseline": brow["earliest_layer"],
                }
        summary["earliest_layer_side_by_side"] = earliness
    else:
        for fam, rate in summary["per_family_pass_rate"].items():
            print(f"  {fam:>16}: pass@k(any layer/pos) = {rate}")

    out = args.out or (args.gen_dir / "scores.json")
    atomic_write_text(out, json.dumps(summary, indent=1))
    print(f"[score] -> {out}")


if __name__ == "__main__":
    main()
