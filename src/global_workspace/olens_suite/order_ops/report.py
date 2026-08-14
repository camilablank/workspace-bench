"""Eval report: per-family metrics with bootstrap CIs, token-lens comparison, controls.

Pure CPU, reads the stored read-stage output. Writes markdown.

    python -m global_workspace.olens_suite.order_ops.report \\
        'readouts=["results/read_a.json","results/read_b.json"]' out=REPORT.md

CLI CONFIG (pydra; pass `--show` to print the resolved config and exit): `readouts` is one path
or a list of paths (`--list readouts a.json b.json list--` also works); `holdout` releases the
withheld 20%; `out` is the markdown destination (default REPORT.md).

CI DESIGN — the unit of resampling is the ITEM, never the sample. The k=10 samples of a cell all
come from one activation, so they are one draw of "what does the lens say about this vector",
not ten independent draws; bootstrapping samples would shrink the interval by ~sqrt(k) and
overstate certainty. Percentile bootstrap, B=10_000, seeded, reported as [lo, hi] at 95%.

TOKEN-LENS SCORING — J-lens and logit lens emit top-k TOKENS, not text, so `asserts` does not
apply. A token-lens hit at a cell = any top-k token, parsed as a number, matching the target
within the family tolerance ... with the same sign-awareness as the text matcher. Also reported:
integer-part hit (the number-word route, e.g. token "十八" for 18.4) — separately, never summed.
"""

import json
import random
import re
from collections.abc import Callable
from pathlib import Path
from statistics import mean
from typing import Any

import pydra

from global_workspace.olens_suite.order_ops.score import asserts, quantities, score_family
from global_workspace.olens_suite.order_ops.spec import (
    FAMILIES,
    GATE,
    SAMPLING,
    holdout_names,
    is_degenerate,
    step_targets,
    tolerance_ok,
)

B = 10_000
SEED = 20260803


def boot_ci(vals: list[float], b: int = B, seed: int = SEED) -> tuple[float, float]:
    """95% percentile bootstrap over items."""
    if len(vals) < 2:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(vals)
    stats = sorted(mean(vals[rng.randrange(n)] for _ in range(n)) for _ in range(b))
    return (stats[int(0.025 * b)], stats[int(0.975 * b)])


def token_number(tok_str: str) -> float | None:
    """A top-k token parsed as a signed number, if it is one."""
    s = tok_str.strip()
    if re.fullmatch(r"-?\d+(\.\d+)?", s):
        return float(s)
    return None


def token_lens_rates(names: list[str], ro: dict[str, Any], tgt: dict[str, float],
                     which: str, layer: str, pos_idx: int, tol: str) -> dict[str, Any]:
    """Per-item hit rates for a token lens at one cell. Value hit and integer-part hit are
    separate columns — integer-part is the number-word route and must never be summed in."""
    full, ipart, avail = [], [], 0
    for n in names:
        tops = ro[n].get(which, {}).get(layer)
        if not tops:                                   # absent layer (J at 63) or not stored
            continue
        avail += 1
        toks = [token_number(t) for t in tops[pos_idx]]
        nums = [x for x in toks if x is not None]
        t = tgt[n]
        full.append(float(any(tolerance_ok(x, t, tol) for x in nums)))
        ipart.append(float(any(x == float(int(t)) for x in nums)))
    if not avail:
        return {"available": False}
    return {"available": True, "n": avail,
            "value": mean(full), "value_ci": boot_ci(full),
            "int_part": mean(ipart), "int_part_ci": boot_ci(ipart)}


def item_rates(names: list[str], ro: dict[str, Any], tgt: dict[str, float],
               layer: str, pos: int, tol: str,
               cell_of: Callable[[dict[str, Any]], list[str]]) -> list[float]:
    return [mean(asserts(s, tgt[n], tol) for s in cell_of(ro[n])) for n in names]


def fam_report(fam: str, items: dict[str, Any], ro: dict[str, Any],
               include_holdout: bool) -> dict[str, Any]:
    spec = FAMILIES[fam]
    layer, pos, tol = str(spec["cell"]["layer"]), spec["cell"]["pos"], spec["tol"]
    scored = score_family(fam, items, ro, include_holdout=include_holdout)

    degen = {n for n in items if is_degenerate(fam, items[n])}
    held = holdout_names(list(items))
    names = [n for n in items if n not in degen and (include_holdout or n not in held)]
    tgt = {n: step_targets(fam, items[n])[0] for n in names}

    def olens_cell(r: dict[str, Any]) -> list[str]:
        blk = r["olens"]["layers"][layer]
        if blk and isinstance(blk[0], list):
            picked: list[str] = blk[r["keep_rel"].index(pos)]
            return picked
        return list(blk)

    # ---- CIs on the headline step, bootstrap over items -----------------------------------
    per_item = item_rates(names, ro, tgt, layer, pos, tol, olens_cell)
    scored["value_ci"] = boot_ci(per_item)

    signed_item = []
    for n in names:
        t = tgt[n]
        vals = []
        for s in olens_cell(ro[n]):
            q = quantities(s)
            own = any(tolerance_ok(x, t, tol) for x in q)
            hp = any(tolerance_ok(x, abs(t), tol) for x in q)
            hn = any(tolerance_ok(x, -abs(t), tol) for x in q)
            vals.append(float(own) - float(hp and hn) - float(not own and (hp or hn)))
        signed_item.append(mean(vals))
    scored["signed_ci"] = boot_ci(signed_item)
    scored["signed_mean"] = mean(signed_item)

    # ---- repeat-seed stability: scored block (seed 0) vs an INDEPENDENT seed ---------------
    # Generation is deterministic given (activation, seed), so "repeat 0" in older runs is a
    # byte-identical copy of the scored block — never use it as an arm.
    reps = []
    for n in names:
        rp = ro[n]["olens"].get("repeat", {})
        indep = next((v for k, v in rp.items() if k != "0"), None)
        if indep:
            reps.append((mean(asserts(s, tgt[n], tol) for s in olens_cell(ro[n])),
                         mean(asserts(s, tgt[n], tol) for s in indep)))
    if len(reps) >= 3:
        xa, xb = [r[0] for r in reps], [r[1] for r in reps]
        ma, mb = mean(xa), mean(xb)
        cov = mean((a - ma) * (b - mb) for a, b in reps)
        va = mean((a - ma) ** 2 for a in xa) or 1e-12
        vb = mean((b - mb) ** 2 for b in xb) or 1e-12
        scored["repeat"] = {"r": cov / (va * vb) ** 0.5, "mean_a": ma, "mean_b": mb,
                            "n": len(reps)}

    # ---- token lenses at the IDENTICAL cell ------------------------------------------------
    pos_abs = {n: ro[n]["n_pos"] + pos for n in names}
    scored["token_lenses"] = {}
    for which in ("jlens_top", "logit_top"):
        # token-lens arrays are stored per absolute position
        rates: dict[str, Any] = {"available": False}
        if any(which in ro[n] and ro[n][which] for n in names):
            full, ipart = [], []
            for n in names:
                tops = ro[n].get(which, {}).get(layer)
                if not tops:
                    continue
                nums = [x for x in (token_number(t) for t in tops[pos_abs[n]]) if x is not None]
                t = tgt[n]
                full.append(float(any(tolerance_ok(x, t, tol) for x in nums)))
                ipart.append(float(any(x == float(int(t)) for x in nums)))
            if full:
                rates = {"available": True, "n": len(full),
                         "value": mean(full), "value_ci": boot_ci(full),
                         "int_part": mean(ipart), "int_part_ci": boot_ci(ipart)}
        scored["token_lenses"][which.removesuffix("_top")] = rates
    return scored


def fmt_ci(ci: tuple[float, float]) -> str:
    return f"[{ci[0]:.2f},{ci[1]:.2f}]" if ci == ci else "-"       # NaN-safe


class Config(pydra.Config):  # type: ignore[misc]
    """See the module docstring; `--show` prints the resolved config and exits."""

    def __init__(self) -> None:
        super().__init__()
        self.readouts: list[str] | str = []   # REQUIRED: read-stage output JSON path(s)
        self.holdout = False                  # release the withheld 20%
        self.out = "REPORT.md"                # markdown destination

    def finalize(self) -> None:
        if isinstance(self.readouts, str):
            self.readouts = [self.readouts]


@pydra.main(Config)  # type: ignore[untyped-decorator]
def main(config: Config) -> None:
    if not config.readouts:
        raise SystemExit("readouts= is required (one read-stage JSON path or a list of them)")
    items: dict[str, Any] = {}
    ro: dict[str, Any] = {}
    cfg = None
    for p in config.readouts:
        d = json.loads(Path(p).read_text())
        cfg = cfg or d.get("config")
        for i in d["items"]:
            items[i["name"]] = i.get("meta", i)
            # token-lens tops travel on the item record
            ro.setdefault(i["name"], {})["jlens_top"] = i.get("jlens_top", {})
            ro[i["name"]]["logit_top"] = i.get("logit_top", {})
            ro[i["name"]]["n_pos"] = i.get("n_pos", 0)
        for r in d["readouts"]:
            ro[r["name"]] |= r

    byfam: dict[str, dict[str, Any]] = {}
    for n, m in items.items():
        byfam.setdefault(m["variant"], {})[n] = m

    lines: list[str] = []
    lines.append("# order-ops eval report\n")
    if cfg:
        lines.append(f"- model `{cfg['model']}` · olens `{cfg['olens']}` · scale {cfg['scale']}")
        lines.append(f"- sampling k={SAMPLING['k']} T={SAMPLING['temperature']} · "
                 f"gate >={GATE['min_correct']:.0%} correct, "
                 f"{GATE['max_leak']:.0%} leak (k={GATE['k']})")
        lines.append(f"- CIs: 95% percentile bootstrap over ITEMS (B={B}), never over samples — "
                 f"a cell's k samples share one activation and are not independent")
        lines.append(f"- holdout {'RELEASED' if config.holdout else 'withheld'}\n")

    rows = []
    for fam in FAMILIES:
        if fam not in byfam or FAMILIES[fam]["cell"] is None:
            continue
        r = fam_report(fam, byfam[fam], ro, config.holdout)
        rows.append(r)

    lines.append("## Headline: does the lens say the intermediate?\n")
    lines.append("| family | role | n | cell | value [95% CI] | signed [95% CI] | net | comp-ev | "
             "noise | donor | d-own | J-lens | logit |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        s0 = next(iter(r["steps"].values()))
        j, g = r["token_lenses"]["jlens"], r["token_lenses"]["logit"]
        jl = f"{j['value']:.2f}" if j["available"] else "n/a"
        gl = f"{g['value']:.2f}" if g["available"] else "n/a"
        lines.append(
            f"| {r['family']} | {r['role']} | {r['n']} "
            f"| L{r['cell']['layer']} p{r['cell']['pos']} "
            f"| {s0['value']:.3f} {fmt_ci(r['value_ci'])} "
            f"| {r['signed_mean']:+.3f} {fmt_ci(r['signed_ci'])} "
            f"| {s0['net']:.3f} | {s0['comp_evidence']:.3f} "
            f"| {r['noise'] if r['noise'] is not None else float('nan'):.2f} "
            f"| {r['donor'] if r['donor'] is not None else float('nan'):.2f} "
            f"| {(r['donor_recovers_own'] if r['donor_recovers_own'] is not None
                   else float('nan')):.2f} "
            f"| {jl} | {gl} |")

    lines.append("\n`signed = value - anti - both` (confidently-wrong output penalised). "
             "`d-own` = donor recovers its own target (must be high or the donor is not a "
             "control). **Token-lens columns cover the DIGIT-TOKEN route only** (a top-10 token "
             "parsed as a numeral). The number-word route — single tokens like 十八/XVIII that "
             "denote a 2-digit integer, previously measured at ~0.83 integer-part for J-lens on "
             "2-digit targets via an LLM judge — is NOT in these columns; treat them as a lower "
             "bound on token-lens performance, pending the judge pass.\n")

    lines.append("## Per-family detail\n")
    for r in rows:
        lines.append(f"### {r['family']}  — {r['shape']}  ({r['role']}, tol {r['tolerance']})\n")
        if r.get("n_degenerate"):
            lines.append(f"- {r['n_degenerate']} degenerate items dropped "
                     f"(step == answer/operand); see banks/REMOVED.json")
        lines.append("| step | value | cross | net | comp-ev | incid | junk | mag "
                     "| anti | both | signed |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for lbl, s in r["steps"].items():
            lines.append(f"| {lbl} | {s['value']:.3f} | {s['cross']:.3f} | {s['net']:.3f} "
                     f"| {s['comp_evidence']:.3f} | {s['incidental']:.3f} "
                     f"| {s['junk'] if s['junk'] is not None else float('nan'):.2f} "
                     + (f"| {s['magnitude']:.3f} | {s['anti']:.3f} | {s['both']:.3f} "
                        f"| {s['signed']:+.3f} |" if "anti" in s else "| - | - | - | - |"))
        lines.append("\ntolerance curve (value/null): "
                 + "  ".join(f"{t.replace('rel', '').replace('pct', '%')}: "
                             f"{v['value']:.2f}/{v['null']:.2f}"
                             for t, v in r["tolerance_curve"].items()))
        for which in ("jlens", "logit"):
            tl = r["token_lenses"][which]
            if tl["available"]:
                lines.append(f"\n{which} @ same cell: value {tl['value']:.3f} "
                         f"{fmt_ci(tl['value_ci'])}, integer-part {tl['int_part']:.3f} "
                         f"{fmt_ci(tl['int_part_ci'])} (n={tl['n']})")
            else:
                lines.append(f"\n{which} @ same cell: n/a "
                         + ("(no Jacobian at this layer)" if which == "jlens" else "(not stored)"))
        if "repeat" in r:
            rp = r["repeat"]
            lines.append(f"\nrepeat-seed stability: r={rp['r']:.2f} "
                     f"(means {rp['mean_a']:.2f} vs {rp['mean_b']:.2f}, n={rp['n']}) — "
                     f"per-item rates carry ±~15% at k=10; treat sub-0.05 family gaps as noise")
        lines.append("")

    ctrl = [r for r in rows if r["role"] == "control"]
    if ctrl:
        c = ctrl[0]
        st = [r for r in rows if r["role"] == "structural"]
        sv = mean(next(iter(r["steps"].values()))["value"] for r in st) if st else float("nan")
        cv = next(iter(c["steps"].values()))["value"]
        lines.append("## Control check\n")
        lines.append(f"`{c['family']}` (intermediate IS a prompt token) reads {cv:.3f}; structural "
                 f"families average {sv:.3f}. "
                 + ("**Control reads like the real families — presence alone is uninformative "
                    "here; lean on comp-ev, signed, and the token-lens gap.**"
                    if abs(cv - sv) < 0.15 else
                    "The gap between them is what licenses reading presence as computation, "
                    "not operand echo."))

    out = Path(config.out)
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out} ({len(rows)} families)")


if __name__ == "__main__":
    main()
