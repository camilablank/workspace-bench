"""Conjunctive per-unit scoring: a layer passes an item only when every required unit hits."""

from __future__ import annotations

import random
from typing import Any

from wsbench.hard.contract import ScoredUnit
from wsbench.matching import hit_forms

LayerRows = dict[int, list[list[str]]]  # layer -> the samples of each row at that layer


def layer_unit_hits(rows: list[list[str]], units: list[ScoredUnit]) -> dict[str, list[str]]:
    """role -> languages in which the unit hit any single sample of any row at this layer."""
    samples = [s for r in rows for s in r]
    return {u.role: hit_forms(samples, u.forms, numeric_match=u.numeric_match) for u in units}


def layer_passes(hits: dict[str, list[str]], units: list[ScoredUnit]) -> bool:
    return all(bool(hits.get(u.role)) for u in units if u.required)


def item_result(
    by_layer: dict[int, dict[str, list[str]]], units: list[ScoredUnit]
) -> dict[str, Any]:
    passing = sorted(layer for layer, hits in by_layer.items() if layer_passes(hits, units))
    unit_hit = {u.role: any(bool(h.get(u.role)) for h in by_layer.values()) for u in units}
    unit_langs = {
        u.role: [
            lang for lang in u.forms if any(lang in h.get(u.role, []) for h in by_layer.values())
        ]
        for u in units
    }
    return {
        "pass": bool(passing),
        "earliest_layer": passing[0] if passing else None,
        "n_passing_layers": len(passing),
        "passing_layers": passing,
        "unit_hit": unit_hit,
        "unit_langs": unit_langs,
        "first_lang": {role: (langs[0] if langs else None) for role, langs in unit_langs.items()},
        "any_hit": any(unit_hit.values()),
    }


def family_columns(results: list[dict[str, Any]]) -> dict[str, float]:
    """Item-level rates: headline ``pass``, the any-unit ``any_hit``, one ``<role>_hit`` per unit
    role over the items that carry it."""
    n = len(results)
    if n == 0:
        return {}
    cols = {
        "pass": round(sum(1 for r in results if r["pass"]) / n, 4),
        "any_hit": round(sum(1 for r in results if r["any_hit"]) / n, 4),
    }
    roles: list[str] = []
    for r in results:
        for role in r["unit_hit"]:
            if role not in roles:
                roles.append(role)
    for role in roles:
        carrying = [r for r in results if role in r["unit_hit"]]
        cols[f"{role}_hit"] = round(
            sum(1 for r in carrying if r["unit_hit"][role]) / len(carrying), 4
        )
    return cols


def permutation_chance(
    items: list[tuple[list[ScoredUnit], LayerRows]], *, permutations: int = 20, seed: int = 0
) -> dict[str, Any]:
    """False-alarm rate of the conjunctive criterion: item i's units scored against a donor
    item's rows, layer by layer, over the whole bank; any donor layer passing is a false alarm."""
    rng = random.Random(seed)
    n = len(items)
    if n < 2:
        return {"model": "permutation-within-family-conjunctive", "permutations": 0, "rate": None}
    false_alarms = trials = 0
    for i, (units, _rows) in enumerate(items):
        if not units:
            continue
        for _ in range(permutations):
            j = rng.randrange(n - 1)
            j = j if j < i else j + 1
            donor = items[j][1]
            trials += 1
            if any(layer_passes(layer_unit_hits(rows, units), units) for rows in donor.values()):
                false_alarms += 1
    rate = false_alarms / trials if trials else None
    return {
        "model": "permutation-within-family-conjunctive",
        "permutations": permutations,
        "seed": seed,
        "trials": trials,
        "rate": round(rate, 4) if rate is not None else None,
    }
