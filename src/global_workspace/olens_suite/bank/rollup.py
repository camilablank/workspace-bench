"""Deterministic rollups of a scored bank-eval run — the numbers an analyst must not hand-count.

Pure ``dict -> dict`` functions over the artifacts the pipeline already writes
(``scores*.json``, ``digest/summary.json``, ``gates_report.json``, the gen JSONL rows).
``report.py`` wires them to disk; the analyst agents receive the OUTPUT of these functions
and judge, rather than re-deriving medians from markdown.
"""


import random
from typing import Any

from global_workspace.olens_suite.bank.matching import hit_any

__all__ = [
    "coverage",
    "degeneracy",
    "earliest_quantiles",
    "family_layer_profile",
    "gate_verdict",
    "permutation_chance",
]

# Four equal-width bands over the 64-layer stack — coarse enough to survive layer subsets.
BANDS = (("L0-15", 0, 16), ("L16-31", 16, 32), ("L32-47", 32, 48), ("L48-63", 48, 64))


def earliest_quantiles(earliest_layers: list[int]) -> dict[str, Any]:
    """median/p25/p75/min/max of a family's earliest hitting layers (empty -> nulls)."""
    if not earliest_layers:
        return {"n": 0, "median": None, "p25": None, "p75": None, "min": None, "max": None}
    xs = sorted(earliest_layers)
    n = len(xs)
    return {
        "n": n,
        "median": xs[n // 2],
        "p25": xs[n // 4],
        "p75": xs[(3 * n) // 4],
        "min": xs[0],
        "max": xs[-1],
    }


def family_layer_profile(
    prompts: dict[str, dict[str, Any]], layers: list[int], family: str
) -> dict[str, Any]:
    """Where in the stack a family hits: per-layer hit rate, peak, late fade, persistence.

    ``prompts`` is ``scores.json["prompts"]`` — ``hits_by_layer`` and ``n_hitting_layers``
    were always computed there and then thrown away by every previous report.
    """
    fam = [row for row in prompts.values() if row.get("family") == family]
    n = len(fam)
    if n == 0:
        return {"n_items": 0}
    by_layer = {
        layer: sum(1 for row in fam if str(layer) in (row.get("hits_by_layer") or {})) / n
        for layer in layers
    }
    peak_layer = max(by_layer, key=lambda k: by_layer[k]) if by_layer else None
    bands = {
        name: round(
            max((rate for layer, rate in by_layer.items() if lo <= layer < hi), default=0.0), 4
        )
        for name, lo, hi in BANDS
    }
    late = [rate for layer, rate in by_layer.items() if layer >= 48]
    peak_rate = by_layer.get(peak_layer, 0.0) if peak_layer is not None else 0.0
    tail_rate = max(late, default=0.0)
    persistence = sorted(int(row.get("n_hitting_layers") or 0) for row in fam)
    return {
        "n_items": n,
        "hit_rate_by_layer": {str(k): round(v, 4) for k, v in by_layer.items()},
        "peak_layer": peak_layer,
        "layer_bands": bands,
        "late_fade": {
            "peak_rate": round(peak_rate, 4),
            "tail_rate": round(tail_rate, 4),
            "is_fading": bool(peak_rate > 0 and tail_rate < 0.5 * peak_rate),
        },
        "persistence": {
            "median_n_hitting_layers": persistence[len(persistence) // 2] if persistence else 0,
            "single_layer_hit_frac": round(
                sum(1 for v in persistence if v == 1) / n, 4
            ),
        },
    }


def permutation_chance(
    items: list[tuple[list[str], list[list[str]]]], *, permutations: int = 20, seed: int = 0
) -> dict[str, Any]:
    """The item-level false-alarm rate of the ACTUAL scoring procedure, for one family.

    ``items`` = per item: (its scored targets, its units — each the SAMPLE LIST of one
    (layer, pos) grid point, so the per-sample numeric rule mirrors the scorer). For item i
    we evaluate i's pass criterion against DONOR items' units: any hit at any donor grid
    point = a false alarm. Analytic chance is the
    wrong model here — the substring/word metric's noise depends on target length and
    commonness ("3", "led"), not vocabulary size. Seeded and reproducible.
    """
    rng = random.Random(seed)
    n = len(items)
    if n < 2:
        return {"model": "permutation-within-family", "permutations": 0, "rate": None}
    false_alarms = trials = 0
    for i, (targets, _) in enumerate(items):
        if not targets:
            continue
        for _ in range(permutations):
            j = rng.randrange(n - 1)
            j = j if j < i else j + 1  # donor != self
            donor_units = items[j][1]
            trials += 1
            if any(hit_any(u, targets) for u in donor_units):
                false_alarms += 1
    rate = false_alarms / trials if trials else None
    return {
        "model": "permutation-within-family",
        "permutations": permutations,
        "seed": seed,
        "trials": trials,
        "rate": round(rate, 4) if rate is not None else None,
    }


def gate_verdict(
    gates: dict[str, Any] | None, run_sampling: dict[str, Any] | None
) -> dict[str, Any]:
    """PASS/WARN/FAIL/MISSING over gates_report.json — nothing read this file before.

    MISSING blocks analysis outright; leak > 0 is FAIL; ``bank_in_top64_steps > 0`` is WARN
    (the worker's contamination guard covers the residual risk — observable noise, not
    silent corruption); a gate run at a different sampling config than the eval means the
    certificate does not cover the run.
    """
    if not gates:
        return {"verdict": "MISSING", "reason": "gates_report.json absent — run gates.py first"}
    ge = str(gates.get("greedy_equivalent", "0/0"))
    done, total = (int(x) for x in ge.split("/"))
    leak = int(gates.get("leak_tokens_eval_sampling", -1))
    top64 = int(gates.get("bank_in_top64_steps", 0))
    sampling_ok = True
    if run_sampling and gates.get("eval_sampling"):
        gs = gates["eval_sampling"]
        sampling_ok = all(
            abs(float(gs.get(k, -1)) - float(run_sampling.get(k, -2))) < 1e-9
            for k in ("temperature", "top_p", "top_k")
        )
    if done < total or leak != 0:
        verdict = "FAIL"
    elif top64 > 0 or not sampling_ok:
        verdict = "WARN"
    else:
        verdict = "PASS"
    return {
        "verdict": verdict,
        "greedy_equivalent": ge,
        "leak_tokens_eval_sampling": leak,
        "bank_in_top64_steps": top64,
        "sampling_matches_run": sampling_ok,
    }


def coverage(
    scores: dict[str, Any], missing_units: list[dict[str, Any]], n_items: int
) -> dict[str, Any]:
    """Run completeness, with structurally-missing layers separated from breakage.

    A layer absent for >=95% of items is STRUCTURAL (e.g. the J-lens artifact has no L63
    Jacobian) and must not read as an incomplete run; scattered gaps above 2%/10% of units
    WARN/FAIL.
    """
    expected = int(scores.get("units_expected", 0))
    missing = int(scores.get("units_missing", 0))
    by_layer: dict[int, int] = {}
    for row in missing_units:
        for layer in row.get("layers", []):
            by_layer[int(layer)] = by_layer.get(int(layer), 0) + 1
    structural = sorted(
        layer for layer, cnt in by_layer.items() if n_items and cnt >= 0.95 * n_items
    )
    scattered = missing - sum(by_layer[layer] for layer in structural)
    frac = scattered / expected if expected else 0.0
    verdict = "FAIL" if frac > 0.10 else ("WARN" if frac > 0.02 else "OK")
    return {
        "units_expected": expected,
        "units_missing": missing,
        "structural_missing_layers": structural,
        "scattered_missing": scattered,
        "scattered_frac": round(frac, 4),
        "verdict": verdict,
    }


def degeneracy(rows: list[dict[str, Any]], *, max_new: int | None = None) -> dict[str, Any]:
    """Data-availability tallies over raw unit rows: a '0' from an empty or degenerate
    rollout is not a real miss, and nothing surfaced these before."""
    n = len(rows)
    empty = truncated = repetitive = dropped = all_dropped = 0
    for row in rows:
        samples = row.get("samples") or []
        d = int(row.get("dropped_contaminated", 0) or 0)
        dropped += d
        if not any(s.strip() for s in samples):
            empty += 1
            if d:
                all_dropped += 1
        if max_new is not None and int(row.get("gen_tokens", 0) or 0) >= max_new:
            truncated += 1
        joined = " ".join(samples).split()
        if len(joined) >= 8:
            grams = [tuple(joined[i : i + 4]) for i in range(len(joined) - 3)]
            if len(set(grams)) <= len(grams) // 2:
                repetitive += 1
    return {
        "rows": n,
        "empty_sample_rows": empty,
        "rows_all_dropped": all_dropped,
        "dropped_samples": dropped,
        "truncated_rows": truncated if max_new is not None else None,
        "repetitive_rows": repetitive,
        "verdict": "WARN" if (n and (empty + repetitive) > 0.02 * n) or dropped else "OK",
    }
