"""Read sweep output, report the peak cell per family, and show what the choice rests on.

Pure CPU. Prints a table you paste into spec.py — deliberately NOT an auto-writer, because
"the sweep picked a cell" and "the cell is defensible" are different claims and the second one
needs a human looking at the margin and the null.

    python -m global_workspace.olens_suite.order_ops.pick_cells \\
        'sweeps=["results/sweep_dec16.json","results/sweep_negdec.json"]'

CLI CONFIG (pydra; pass `--show` to print the resolved config and exit): `sweeps` is one path or
a list of paths (`--list sweeps a.json b.json list--` also works); when empty it defaults to
every results/order_ops/sweep_*.json.

For each family it reports, at every (layer, position):
    value  the readout asserts this item's own target
    null   it asserts SOME OTHER item's target (same cell, same code)
    margin value / null  -- a peak with a high null is a peak in arithmetic-flavoured output,
           not in this item's content. That distinction is why the null is computed per cell.
"""

import json
from pathlib import Path
from statistics import mean
from typing import Any

import pydra

from global_workspace.olens_suite.order_ops.score import asserts
from global_workspace.olens_suite.order_ops.spec import (
    FAMILIES,
    holdout_names,
    is_degenerate,
    null_separation,
    step_targets,
)

MIN_MARGIN = 3.0     # below this the cell is not distinguishing items, whatever its value


def sweep_table(path: Path) -> dict[str, Any] | None:
    d = json.loads(path.read_text())
    if d.get("stage") != "sweep":
        print(f"  ! {path.name}: stage={d.get('stage')!r}, not a sweep — skipped")
        return None
    items = {i["name"]: i.get("meta", i) for i in d["items"]}
    ro = {r["name"]: r for r in d["readouts"]}
    fam = next(iter({i["variant"] for i in items.values()}))
    spec = FAMILIES[fam]
    tol, sep = spec["tol"], null_separation(spec["tol"])

    # Same exclusions as the scorer: degenerate items out, holdout withheld. Choosing a cell on
    # the holdout would leak it.
    usable = [n for n in items if not is_degenerate(fam, items[n])]
    held = holdout_names(usable)
    names = [n for n in usable if n not in held]
    tgt = {n: step_targets(fam, items[n])[0] for n in names}

    rels = ro[names[0]]["keep_rel"]
    layers = sorted(ro[names[0]]["olens"]["layers"], key=int)
    grid = {}
    for lyr in layers:
        for pi, p in enumerate(rels):
            v, nu = [], []
            for n in names:
                cell = ro[n]["olens"]["layers"][lyr][pi]
                v.append(mean(asserts(s, tgt[n], tol) for s in cell))
                others = [tgt[m] for m in names
                          if abs(tgt[m] - tgt[n]) > sep * max(abs(tgt[n]), 1e-9)
                          and tgt[m] != tgt[n]]
                nu.append(mean(any(asserts(s, o, tol) for o in others) for s in cell))
            grid[(int(lyr), p)] = (mean(v), mean(nu))
    return {"family": fam, "n": len(names), "n_held": len(held),
            "n_degenerate": len(items) - len(usable), "grid": grid,
            "registered": spec["cell"], "role": spec["role"], "tol": tol}


class Config(pydra.Config):  # type: ignore[misc]
    """See the module docstring; `--show` prints the resolved config and exits."""

    def __init__(self) -> None:
        super().__init__()
        # sweep JSON path(s); default: results/order_ops/sweep_*.json
        self.sweeps: list[str] | str = []

    def finalize(self) -> None:
        if isinstance(self.sweeps, str):
            self.sweeps = [self.sweeps]


@pydra.main(Config)  # type: ignore[untyped-decorator]
def main(config: Config) -> None:
    paths = [Path(p) for p in config.sweeps] or sorted(
        Path("results/order_ops").glob("sweep_*.json")
    )
    for path in paths:
        t = sweep_table(path)
        if not t:
            continue
        fam, grid = t["family"], t["grid"]
        ranked = sorted(grid.items(), key=lambda kv: -kv[1][0])
        (bl, bp), (bv, bn) = ranked[0]
        margin = bv / bn if bn else float("inf")

        print(f"\n=== {fam}  ({t['role']}, tol={t['tol']}, n={t['n']}, "
              f"{t['n_held']} held, {t['n_degenerate']} degenerate)")
        print(f"  PEAK   L{bl} rel{bp}   value {bv:.3f}  null {bn:.3f}  margin "
              + (f"{margin:.1f}x" if margin != float("inf") else "inf")
              + ("" if margin >= MIN_MARGIN else f"   <-- BELOW {MIN_MARGIN}x, not distinguishing"))
        reg = t["registered"]
        if reg is None:
            print(f"  spec has cell=None -> paste:  \"cell\": {{\"layer\": {bl}, \"pos\": {bp}}},")
        elif (reg["layer"], reg["pos"]) != (bl, bp):
            rv, _rn = grid[(reg["layer"], reg["pos"])]
            print(f"  MISMATCH registered L{reg['layer']} rel{reg['pos']} = {rv:.3f} "
                  f"(peak is {bv:.3f}, +{bv - rv:+.3f})")
        else:
            print("  registered cell IS the peak")

        print("  runners-up:  " + "   ".join(
            f"L{ly} rel{pp}:{v:.2f}/{n:.2f}" for (ly, pp), (v, n) in ranked[1:5]))
        # position profile at the peak layer, so a one-token cliff is visible
        prof = [(pp, grid[(bl, pp)][0]) for (ly, pp) in grid if ly == bl]
        prof.sort(key=lambda x: -x[0])
        print(f"  L{bl} by position:  "
              + (" ".join(f"{pp}:{v:.2f}" for pp, v in prof if v > 0.02) or "(all <=0.02)"))
        best_by_layer: dict[int, tuple[float, int, float]] = {}
        for (ly, pp), (v, n) in grid.items():
            if v > best_by_layer.get(ly, (0.0, 0, 0.0))[0]:
                best_by_layer[ly] = (v, pp, n)
        print("  best per layer:  " + " ".join(
            f"L{ly}:{v:.2f}@{pp}" for ly, (v, pp, _) in sorted(best_by_layer.items())))


if __name__ == "__main__":
    main()
