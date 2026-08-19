"""Slop pass for a gen dir: rate every mechanically-HIT readout row, 1.0-10.0 (CPU, local).

The precision condition (Camila 2026-08-19): the mechanical matcher is pure recall — a readout
that buries the correct answer under hallucinated detail (the NLA failure mode) scores the same
as one that delivers it cleanly. This driver rates each hit row with the slop judge
(``global_workspace.judges.slop_judge``; rubric + gate evidence in
``scripts/oracle_lens_evals/gate_slop_judge.py`` / ``outputs/slop_gate/``) and writes
``<gen-dir>/judge/slop.json``: per-row scores plus, per family, the mechanical pass rate next
to the slop-gated pass rate at candidate thresholds. Only hits are judged — a miss cannot lose
credit it never had.

Rows load through the pipeline's single read path (``common.load_unit_rows``: eval-position
mask + extract_phrase) and hits through the canonical matcher, so the "mechanical" numbers here
are bit-comparable with ``score_targets.py``. The judged context/target come from the bank item
(clean prompt text), falling back to the manifest tokens for labels missing from the bank.

The gate threshold is provisional until Camila fixes it qualitatively; every consumer reads it
from this artifact's ``config.threshold``, never hardcodes it.

    source scripts/cluster/env.sh   # ANTHROPIC_API_KEY
    uv run --no-sync python scripts/oracle_lens_evals/olens_sglang/judge_slop.py \
        --acts-dir outputs/oracle_lens_evals/olens_sglang/acts \
        --gen-dir  outputs/oracle_lens_evals/olens_sglang/gen-<name> [--threshold 5.0]

Free-text gen dirs only — raw J-lens token bags are not "asserting" anything (see the scope
note in slop_judge.py); a top-k token list's precision instrument is rank, not slop.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ActsManifest, PromptEntry, load_unit_rows, present_layers

from global_workspace.judges.oracle_lens_judge import label_for
from global_workspace.judges.slop_judge import (
    gated_hit,
    judge_slop,
    summarize_slop,
)
from global_workspace.olens_suite.bank.loader import DEFAULT_ROOT, load_bank
from global_workspace.olens_suite.bank.matching import content_targets, exact_targets, hit_any

THRESHOLD_CANDIDATES = (3.0, 4.0, 5.0, 6.0, 7.0)
# The deterministically-matched (mechanical) bank families: their headline is the substring
# matcher, so the slop gate is their ONLY junk control. LLM-judged families get slop inline
# from judge_readouts.py; add them here explicitly (--families) only for diagnostics.
MECHANICAL_FAMILIES = frozenset(
    {
        "basic-readout",
        "multihop",
        "multilingual",
        "poetry",
        "typo",
        "association",
        "basic-readout-mt",
        "multihop-mt",
        "multilingual-mt",
        "typo-mt",
    }
)


def bank_index(families: list[str], banks_dir: str) -> dict[str, dict[str, Any]]:
    """label -> bank item, for the judge's clean context/target strings."""
    out: dict[str, dict[str, Any]] = {}
    for family in families:
        try:
            items = load_bank(family, banks_dir)
        except FileNotFoundError:
            print(f"[slop] no bank file for {family} under {banks_dir}; using manifest fallback")
            continue
        for item in items:
            out[label_for(str(item["name"]))] = item
    return out


def fallback_item(entry: PromptEntry) -> dict[str, Any]:
    """Manifest-only stand-in when a label is missing from the bank (renamed/retired items)."""
    return {
        "name": entry.label,
        "prompt": "".join(entry.tokens),
        "units": [{"role": "readout", "headline": True, "match": list(entry.targets)}],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--acts-dir", required=True, type=Path)
    p.add_argument("--gen-dir", required=True, type=Path)
    p.add_argument("--banks-dir", default=str(DEFAULT_ROOT))
    p.add_argument(
        "--families",
        default="",
        help="comma-separated (default: the manifest's MECHANICAL families; judged families "
        "may be added explicitly — their slop rows are diagnostics beside their own judge)",
    )
    p.add_argument("--layers", default="", help="comma-separated subset (default: all present)")
    p.add_argument("--limit", type=int, default=0, help="first N prompts (smoke tests)")
    p.add_argument("--model", default="", help="override judge model")
    p.add_argument("--concurrency", type=int, default=256)
    p.add_argument(
        "--threshold",
        type=float,
        default=5.0,
        help="PROVISIONAL headline gate: a hit counts only when slop < threshold. Consumers "
        "read it from the artifact; re-run with a new value once Camila fixes the cut.",
    )
    p.add_argument("--out", default="", help="default: <gen-dir>/judge/slop.json")
    args = p.parse_args()

    manifest = ActsManifest.load(args.acts_dir)
    layer_subset = {int(x) for x in args.layers.split(",") if x.strip()}
    layers = [
        layer
        for layer in present_layers(args.gen_dir, manifest.layers)
        if not layer_subset or layer in layer_subset
    ]
    manifest_families = sorted({p.family for p in manifest.prompts})
    families = [f.strip() for f in args.families.split(",") if f.strip()] or [
        f for f in manifest_families if f in MECHANICAL_FAMILIES
    ]
    bank = bank_index(families, args.banks_dir)

    entries = [p for p in manifest.prompts if p.family in families]
    if args.limit:
        entries = entries[: args.limit]
    print(f"[slop] {len(entries)} prompts x {len(layers)} layers, families={families}")

    pairs: list[tuple[dict[str, Any], str]] = []
    meta: list[dict[str, Any]] = []
    n_miss_rows = 0
    for entry in entries:
        targets = content_targets(exact_targets(list(entry.targets)))
        item = bank.get(entry.label) or fallback_item(entry)
        by_layer, _missing = load_unit_rows(args.gen_dir, entry, layers)
        for layer, rows in by_layer.items():
            for row in rows:
                for k, sample in enumerate(row.get("samples") or []):
                    if not sample.strip():
                        continue
                    if not hit_any([sample], targets):
                        n_miss_rows += 1
                        continue
                    pairs.append((item, sample))
                    meta.append(
                        {
                            "name": entry.label,
                            "family": entry.family,
                            "layer": layer,
                            "pos": int(row["pos"]),
                            "sample_idx": k,
                            "readout": sample,
                        }
                    )
    if not pairs:
        raise SystemExit(f"no hit rows to judge under {args.gen_dir} for families {families}")
    print(f"[slop] judging {len(pairs)} hit rows ({n_miss_rows} miss rows untouched)")

    kwargs: dict[str, Any] = {"concurrency": args.concurrency}
    if args.model:
        kwargs["model"] = args.model
    judged = judge_slop(pairs, **kwargs)
    rows = [{**m, **j} for m, j in zip(meta, judged, strict=True)]

    # per-family: item-level pass@any (over the judged grid) at each candidate threshold —
    # every judged row IS a hit, so mechanical item pass = "has any judged row".
    per_family: dict[str, Any] = {}
    for family in families:
        fam_rows = [r for r in rows if r["family"] == family]
        fam_labels = {p.label for p in entries if p.family == family}
        if not fam_labels:
            continue
        items_hit = {r["name"] for r in fam_rows}
        gated: dict[str, Any] = {}
        for t in sorted({*THRESHOLD_CANDIDATES, args.threshold}):
            keep = {r["name"] for r in fam_rows if gated_hit(True, r.get("slop"), t)}
            gated[str(t)] = round(len(keep) / len(fam_labels), 4)
        per_family[family] = {
            "n_items": len(fam_labels),
            "mechanical_pass_rate": round(len(items_hit) / len(fam_labels), 4),
            "gated_pass_rate": gated,
            "slop": summarize_slop(fam_rows),
        }

    payload = {
        "schema_version": 1,
        "gen_dir": str(args.gen_dir),
        "acts_dir": str(args.acts_dir),
        "config": {
            "model": kwargs.get("model", "default"),
            "threshold": args.threshold,
            "threshold_provisional": True,
            "layers": layers,
            "families": families,
        },
        "summary": summarize_slop(rows),
        "per_family": per_family,
        "n_miss_rows": n_miss_rows,
        "rows": rows,
    }
    out = Path(args.out) if args.out else args.gen_dir / "judge" / "slop.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1))
    print(f"[slop] -> {out}")
    print(json.dumps({"summary": payload["summary"], "per_family": per_family}, indent=1))


if __name__ == "__main__":
    main()
