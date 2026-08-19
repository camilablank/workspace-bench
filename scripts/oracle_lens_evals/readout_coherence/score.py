#!/usr/bin/env python3
"""readout_coherence scoring — fold the judge passes into the eval's metrics.

Metric 1 (standalone): summary-position ratio. flag = effective_level >= 2 per the frozen
sumtok gate; the headline is n_olens_flagged / n_jlens_flagged (ideally ~1: both lenses
should expose the same summary positions). Position-level Jaccard is reported beside it —
the count ratio can be 1.0 while the two lenses flag disjoint positions.

Metrics 2+3 (combined): overall coherence = 0.8 * mean_quality + 0.2 * 10*(1 - malformed
summary-position rate). Quality (pass 2, per position x layer: 1-10 + hallucination/junk/
padded flags) dominates; summary-start formatting (pass 3) is the remaining 2-point slice
because it only exists on the small flagged subset. Weights are constants below, frozen in
the eval README — change them there or not at all.

Metrics 4+5 (standalone, skip-if-missing): bullet relevance (per-bullet relation shares,
headline = unrelated rate, + hallucinated rate) and bullet diversity (mean n_distinct,
diverse_aspects rate, pairwise relation shares) from bullet_judges.py's relevance.jsonl /
diversity.jsonl. Reported beside — never inside — the frozen overall-coherence score.

Usage:
  uv run --no-sync python scripts/oracle_lens_evals/readout_coherence/score.py \
      --verdicts outputs/oracle_lens_evals/readout_coherence/verdicts \
      --out outputs/oracle_lens_evals/readout_coherence/results.json
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

W_QUALITY = 0.8
W_SUMFMT = 0.2
FLAG_LEVEL = 2          # sumtok gate: effective_level >= 2 = summary position
INCOHERENT_MAX = 3      # quality <= 3 counts as an incoherent readout cell


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def dedupe_by_key(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Resume appends: keep the LAST row per key (a retry's success supersedes its error row,
    and double-runs can't double-count in the denominators). Keyless rows pass through."""
    by_key: dict[str, dict[str, Any]] = {}
    passthrough: list[dict[str, Any]] = []
    for r in rows:
        key = r.get("key")
        if key is None:
            passthrough.append(r)
        else:
            by_key[key] = r
    return passthrough + list(by_key.values())


def source_of(label: str) -> str:
    return label.split("-")[1] if label.startswith("rc-") else "?"


def sumtok_metrics(ao: list[dict[str, Any]], jl: list[dict[str, Any]]) -> dict[str, Any]:
    def flags(rows: list[dict[str, Any]]) -> tuple[set[tuple[str, int]], int, int]:
        ok = [r for r in rows if not r.get("api_error")]
        flagged = {(r["label"], r["pos"]) for r in ok if r.get("effective_level", 0) >= FLAG_LEVEL}
        return flagged, len(ok), sum(1 for r in rows if r.get("api_error"))

    ao_f, ao_n, ao_err = flags(ao)
    jl_f, jl_n, jl_err = flags(jl)
    union = ao_f | jl_f
    per_source: dict[str, dict[str, Any]] = {}
    for src in sorted({source_of(lb) for lb, _ in union} | {"wildchat", "dailydialog"}):
        a = {k for k in ao_f if source_of(k[0]) == src}
        j = {k for k in jl_f if source_of(k[0]) == src}
        per_source[src] = {"olens_flagged": len(a), "jlens_flagged": len(j),
                           "ratio": round(len(a) / len(j), 4) if j else None}
    return {
        "olens": {"judged": ao_n, "flagged": len(ao_f), "api_errors": ao_err,
                  "flag_rate": round(len(ao_f) / ao_n, 4) if ao_n else None,
                  "summary_to_nonsummary": round(len(ao_f) / (ao_n - len(ao_f)), 4)
                  if ao_n > len(ao_f) else None},
        "jlens": {"judged": jl_n, "flagged": len(jl_f), "api_errors": jl_err,
                  "flag_rate": round(len(jl_f) / jl_n, 4) if jl_n else None,
                  "summary_to_nonsummary": round(len(jl_f) / (jl_n - len(jl_f)), 4)
                  if jl_n > len(jl_f) else None},
        "ratio_olens_to_jlens": round(len(ao_f) / len(jl_f), 4) if jl_f else None,
        "position_jaccard": round(len(ao_f & jl_f) / len(union), 4) if union else None,
        "per_source": per_source,
    }


def quality_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in rows if not r.get("api_error")]
    if not ok:
        return {"judged": 0, "api_errors": sum(1 for r in rows if r.get("api_error"))}
    by_layer: dict[int, list[int]] = defaultdict(list)
    by_source: dict[str, list[int]] = defaultdict(list)
    for r in ok:
        by_layer[r["layer"]].append(r["quality"])
        by_source[source_of(r["label"])].append(r["quality"])

    def mean(xs: list[int]) -> float:
        return round(sum(xs) / len(xs), 3)

    def rate(k: str) -> float:
        return round(sum(bool(r[k]) for r in ok) / len(ok), 4)

    return {
        "judged": len(ok),
        "api_errors": sum(1 for r in rows if r.get("api_error")),
        "mean_quality": mean([r["quality"] for r in ok]),
        "incoherent_rate": round(
            sum(r["quality"] <= INCOHERENT_MAX for r in ok) / len(ok), 4),
        "hallucination_rate": rate("hallucination"),
        "junk_rate": rate("junk"),
        "padded_rate": rate("padded"),
        "mean_quality_by_layer": {la: mean(v) for la, v in sorted(by_layer.items())},
        "mean_quality_by_source": {s: mean(v) for s, v in sorted(by_source.items())},
    }


def sumfmt_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in rows if not r.get("api_error")]
    if not ok:
        return {"judged": 0, "api_errors": sum(1 for r in rows if r.get("api_error"))}
    n_malformed_pos = sum(1 for r in ok if r["n_malformed_layers"] > 0)
    total_layers = sum(r["n_layers"] for r in ok)
    return {
        "judged": len(ok),
        "api_errors": sum(1 for r in rows if r.get("api_error")),
        "malformed_position_rate": round(n_malformed_pos / len(ok), 4),
        "malformed_layer_rate": round(
            sum(r["n_malformed_layers"] for r in ok) / total_layers, 4) if total_layers else None,
        "raw_vs_verified": {
            "raw_fragments": sum(len(r.get("fragments_raw", [])) for r in ok),
            "verified_fragments": sum(len(r.get("fragments", [])) for r in ok),
        },
    }


def relevance_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Pass 4 (bullet_judges.py relevance.jsonl): per-bullet relation + hallucination."""
    rows = dedupe_by_key(rows)
    ok = [r for r in rows if not r.get("api_error")]
    n_err = sum(1 for r in rows if r.get("api_error"))
    # api_errors is always reported: a fully-failed judge pass must be distinguishable from a
    # pass that was never run (both used to return the same bare {"judged": 0}).
    if not ok:
        return {"judged": 0, "api_errors": n_err}
    relations: dict[str, int] = defaultdict(int)
    n_bullets = 0
    n_halluc = 0
    by_layer: dict[int, list[int]] = defaultdict(list)  # one 0/1 (unrelated?) per bullet
    for r in ok:
        # strict=False: a misbehaving judge may return more/fewer sample verdicts than
        # samples — score what aligns, same tolerance as the bullet truncation below.
        for verdict_sample, bullet_texts in zip(
            r["verdict"]["samples"], r["bullets"], strict=False
        ):
            for b in verdict_sample["bullets"][: len(bullet_texts)]:
                relations[b["relation"]] += 1
                n_bullets += 1
                n_halluc += bool(b["hallucinated"])
                by_layer[r["layer"]].append(int(b["relation"] == "unrelated"))
    if not n_bullets:
        return {"judged": len(ok), "api_errors": n_err, "n_bullets": 0}
    return {
        "judged": len(ok),
        "api_errors": n_err,
        "n_bullets": n_bullets,
        "relation_shares": {k: round(v / n_bullets, 4) for k, v in sorted(relations.items())},
        "unrelated_rate": round(relations.get("unrelated", 0) / n_bullets, 4),
        "hallucinated_rate": round(n_halluc / n_bullets, 4),
        "unrelated_rate_by_layer": {
            la: round(sum(v) / len(v), 4) for la, v in sorted(by_layer.items())
        },
    }


def diversity_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Pass 5 (bullet_judges.py diversity.jsonl): pairwise topic relations + n_distinct."""
    rows = dedupe_by_key(rows)
    ok = [r for r in rows if not r.get("api_error")]
    n_err = sum(1 for r in rows if r.get("api_error"))
    if not ok:
        return {"judged": 0, "api_errors": n_err}
    pair_rel: dict[str, int] = defaultdict(int)
    n_distinct: list[int] = []
    n_diverse = 0
    n_samples = 0
    by_layer: dict[int, list[int]] = defaultdict(list)
    for r in ok:
        # align verdict samples to the judged bullets exactly as relevance_metrics does: a
        # verdict for a sample with no bullets (or a surplus verdict from a misbehaving judge)
        # must not enter the fold — an n_distinct=0 phantom sample would drag the headline.
        for s, bullet_texts in zip(r["verdict"]["samples"], r["bullets"], strict=False):
            if not bullet_texts:
                continue
            n_samples += 1
            nd = max(0, min(int(s["n_distinct"]), len(bullet_texts)))  # clamp to bullet count
            n_distinct.append(nd)
            n_diverse += bool(s["diverse_aspects"])
            by_layer[r["layer"]].append(nd)
            seen_pairs: set[tuple[int, int]] = set()
            for pr in s["pairs"]:
                i, j = pr.get("i"), pr.get("j")
                valid = (
                    isinstance(i, int) and isinstance(j, int)
                    and 0 <= i < j < len(bullet_texts) and (i, j) not in seen_pairs
                )
                if not valid:  # out-of-range/duplicate/reversed pair from a malformed verdict
                    continue
                seen_pairs.add((i, j))
                pair_rel[pr["relation"]] += 1
    n_pairs = sum(pair_rel.values())
    return {
        "judged": len(ok),
        "api_errors": n_err,
        "n_samples": n_samples,
        "mean_n_distinct": round(sum(n_distinct) / len(n_distinct), 3) if n_distinct else None,
        "diverse_aspects_rate": round(n_diverse / n_samples, 4) if n_samples else None,
        "pair_relation_shares": {
            k: round(v / n_pairs, 4) for k, v in sorted(pair_rel.items())
        } if n_pairs else {},
        "mean_n_distinct_by_layer": {
            la: round(sum(v) / len(v), 3) for la, v in sorted(by_layer.items())
        },
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--verdicts", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    m1 = sumtok_metrics(read_jsonl(args.verdicts / "sumtok_ao.jsonl"),
                        read_jsonl(args.verdicts / "sumtok_jlens.jsonl"))
    m2 = quality_metrics(read_jsonl(args.verdicts / "quality.jsonl"))
    m3 = sumfmt_metrics(read_jsonl(args.verdicts / "sumfmt.jsonl"))
    m4 = relevance_metrics(read_jsonl(args.verdicts / "relevance.jsonl"))
    m5 = diversity_metrics(read_jsonl(args.verdicts / "diversity.jsonl"))

    if m2.get("judged"):
        fmt_term = (1.0 - m3["malformed_position_rate"]) if m3.get("judged") else None
        if fmt_term is not None:
            overall = round(W_QUALITY * m2["mean_quality"] + W_SUMFMT * 10 * fmt_term, 3)
            note = None
        else:
            overall = round(m2["mean_quality"], 3)
            note = "no summary positions judged — overall = mean_quality alone"
    else:
        overall, note = None, "quality pass missing"

    results = {
        "summary_ratio": m1,
        "quality": m2,
        "summary_formatting": m3,
        "bullet_relevance": m4,
        "bullet_diversity": m5,
        "overall_coherence": {"score": overall, "weights": {"quality": W_QUALITY,
                              "summary_formatting": W_SUMFMT}, "note": note},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1))

    print(f"metric 1  olens/jlens summary-position ratio: {m1['ratio_olens_to_jlens']}"
          f"  (olens {m1['olens']['flagged']}/{m1['olens']['judged']},"
          f" jlens {m1['jlens']['flagged']}/{m1['jlens']['judged']},"
          f" jaccard {m1['position_jaccard']})")
    if m2.get("judged"):
        print(f"metric 2  mean quality {m2['mean_quality']}/10, incoherent {m2['incoherent_rate']},"
              f" halluc {m2['hallucination_rate']}, junk {m2['junk_rate']},"
              f" padded {m2['padded_rate']}  (n={m2['judged']})")
    if m3.get("judged"):
        print(f"metric 3  malformed summary-start: {m3['malformed_position_rate']} of"
              f" {m3['judged']} summary positions")
    if m4.get("n_bullets"):
        print(f"metric 4  bullet relevance: unrelated {m4['unrelated_rate']},"
              f" halluc {m4['hallucinated_rate']},"
              f" shares {m4['relation_shares']}  (n={m4['n_bullets']} bullets)")
    if m5.get("n_samples"):
        print(f"metric 5  bullet diversity: mean n_distinct {m5['mean_n_distinct']},"
              f" diverse_aspects {m5['diverse_aspects_rate']},"
              f" pairs {m5['pair_relation_shares']}  (n={m5['n_samples']} samples)")
    print(f"OVERALL coherence: {overall}/10" + (f"  [{note}]" if note else ""))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
