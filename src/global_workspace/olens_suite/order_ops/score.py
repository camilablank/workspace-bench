"""Pure-CPU scorer. Reads stored readouts, writes metrics. No model, no GPU, no network.

The GPU stage stores raw text only, so every metric is a recompute — which matters because this
metric has been revised ten times, and each revision would otherwise have cost a GPU pass.

    python -m global_workspace.olens_suite.order_ops.score \\
        readouts=results/order_ops/read_dec16.json
    python -m ...order_ops.score readouts=<read.json> holdout=True   # release the 20%
    python -m ...order_ops.score 'readouts=["results/read_a.json","results/read_b.json"]' \\
        json_out=out.json

CLI CONFIG (pydra; pass `--show` to print the resolved config and exit): `readouts` is one path
or a list of paths (`--list readouts a.json b.json list--` also works); `holdout` releases the
withheld 20%; `json_out` writes the per-family results JSON.

METRICS (frozen 2026-08-03)

  value       asserts the SIGNED target within the family's tolerance
  cross       asserts a DIFFERENT item's target, and not its own
  net         value - cross.  0 for a coin flip, NEGATIVE for an anti-correlated reader
  comp-ev     value present AND the surrounding numbers are consistent with this computation
  incid       value present but the computation around it is irrelevant ('2.64575 = 1.64575')
  junk        share of asserted numbers matching no operand, step, or answer

  sign block, wherever the magnitude is ever seen:
  |v|         asserts +|t| or -|t| (magnitude, either sign)
  anti        asserts -sign(t)*|t| and NOT the target -- CONFIDENTLY WRONG, worse than silence
  both        asserts both +|t| and -|t| -- a hedge, not a hit
  commit      value / |v|  -- given the magnitude, how often the sign is right and unhedged
  signed      value - anti - both  -- the commitment-penalised score. 1.0 = always right,
              0 = as often wrong-signed as right, negative = anti-correlated.

  Sign metrics are computed for EVERY family, not only negative-target ones: a positive-target
  family that emits the negative is making the same error.

REMOVED: `op_only`. `produced_by` held bare symbols ("/", "-"), so it matched any slash or
hyphen -- 97.8% of dec16's 0.309 and 88.6% of negdec step 2's 0.886 were single punctuation
characters. Word forms give <=0.026 everywhere. The metric measured punctuation; it is deleted,
not repaired.
"""

import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import pydra

from global_workspace.olens_suite.order_ops.spec import (
    FAMILIES,
    TOL_LADDER,
    holdout_names,
    is_degenerate,
    null_separation,
    single_token_reachable,
    step_targets,
    tolerance_ok,
)

NUMERAL = re.compile(r"-?\d[\d,]*\.?\d*")


def quantities(text: str) -> list[float]:
    """Numerals a readout ASSERTS as values.

    List markers ("1. foo") and step labels ("Step 2") are structure, not quantities; counting
    them inflates the junk denominator. But a list marker is only stripped when PROSE follows:
    "29." alone on a line is a VALUE statement, and the original digit-dot-space rule deleted
    it — marking sqrt's correct "29." readouts as misses (caught by reading rollouts,
    2026-08-03).
    """
    t = re.sub(r"</?explanation>", " ", text)
    t = re.sub(r"(?m)^[ \t]*\d+\.[ \t]+(?=[A-Za-z*_#`(])", " ", t)  # same line only
    t = re.sub(r"(?i)step\s*\d+", " ", t)
    out: list[float] = []
    for m in NUMERAL.finditer(t):
        s = m.group(0).rstrip(".").replace(",", "")
        if s in {"", "-"}:
            continue
        try:
            out.append(float(s))
        except ValueError:
            continue
    return out


def asserts(text: str, target: float, tol: str) -> bool:
    return any(tolerance_ok(q, target, tol) for q in quantities(text))


# number words for the fraction families' pool (spec FRACTION_POOL plus computed numerators are
# arbitrary ints, so words only cover the denominator side and small numerators)
_NUM_WORDS = {3: "three", 7: "seven", 9: "nine", 11: "eleven", 13: "thirteen",
              17: "seventeen", 19: "nineteen", 23: "twenty-three"}
_DEN_WORDS = {3: "thirds?", 7: "sevenths?", 9: "ninths?", 11: "elevenths?",
              13: "thirteenths?", 17: "seventeenths?", 19: "nineteenths?",
              23: "twenty-thirds?"}


def fraction_asserts(text: str, num: int, den: int) -> bool:
    """Does the text state the fraction AS A FRACTION (num/den, 'num over den', word form)?

    Quarantined from `value` on purpose: for `frac` items the fraction string is operand echo
    (both digits sit in the prompt), for `fraccomp` it is computed content — the report labels
    the same column differently per family. Digit-boundary guards keep '17/3' from matching
    '7/3' and '73' from matching anything.
    """
    n, d = re.escape(str(num)), re.escape(str(den))
    pats = [rf"(?<![\d.]){n}\s*[/\u2044]\s*{d}(?![\d.])",  # solidus or fraction slash
            rf"(?<![\d.]){n}\s+over\s+{d}(?![\d.])"]
    if num in _NUM_WORDS and den in _DEN_WORDS:
        pats.append(rf"\b{_NUM_WORDS[num]}\s+{_DEN_WORDS[den]}\b")
    return any(re.search(p, text, re.IGNORECASE) for p in pats)


def computation_relevant(text: str, target: float, legitimate: list[float]) -> bool:
    """Is the value embedded in a computation consistent with the expression?

    Every OTHER number asserted must be a legitimate part of this computation. A PROXY: it cannot
    see a true-looking equation that describes the WRONG computation. That needs a judge.
    """
    t = re.sub(r"</?explanation>", " ", text)
    others = [q for q in quantities(t) if abs(q - target) > 0.02 * max(abs(target), 1e-9)]
    if not others:
        return True
    return all(any(abs(q - v) <= 0.02 * max(abs(v), 1e-9) for v in legitimate) for q in others)


def _legit(item: dict[str, Any], fam: str) -> list[float]:
    ops = [float(x) for x in item["operands"]]
    return ops + step_targets(fam, item) + [float(item["answer"])]


def _cell(readout: dict[str, Any], layer: str, pos: int) -> list[str]:
    block = readout["olens"]["layers"][layer]
    if block and isinstance(block[0], list):              # sweep-shaped [pos][k]
        picked: list[str] = block[readout.get("keep_rel", [pos]).index(pos)]
        return picked
    return list(block)


def score_family(fam: str, items: dict[str, Any], readouts: dict[str, Any],
                 *, include_holdout: bool) -> dict[str, Any]:
    spec = FAMILIES[fam]
    if spec["cell"] is None:
        raise ValueError(f"{fam}: cell is None — the Stage 1 sweep has not run. Refusing to "
                         f"score at a guessed cell (see sqrt: 0.992 at rel-7 vs 0.458 at rel-8).")
    layer, pos, tol = str(spec["cell"]["layer"]), spec["cell"]["pos"], spec["tol"]
    held = holdout_names(list(items))
    # Degenerate items are dropped BEFORE the holdout so the split stays deterministic in the
    # items that are actually evaluable.
    degen = {n: r for n in items if (r := is_degenerate(fam, items[n]))}
    names = [n for n in items if n not in degen and (include_holdout or n not in held)]
    if not names:
        return {}

    out: dict[str, Any] = {
        "family": fam, "n": len(names), "n_holdout": len(held),
        "holdout_released": include_holdout, "cell": {"layer": int(layer), "pos": pos},
        "role": spec["role"], "single_token_reachable": spec["reachable"],
        "shape": spec["shape"], "tolerance": tol, "steps": {}, "tolerance_curve": {},
        "n_degenerate": len(degen), "degenerate": degen,
    }

    all_targets = {n: step_targets(fam, items[n]) for n in names}

    for si, (label, _fn) in enumerate(spec["steps"]):
        tgt = {n: all_targets[n][si] for n in names}
        sep = null_separation(tol)
        hit = miss = tot = clean = incid = 0
        junk: list[float] = []
        mag = anti = both = 0
        for n in names:
            t = tgt[n]
            others = [tgt[m] for m in names
                      if abs(tgt[m] - t) > sep * max(abs(t), 1e-9) and tgt[m] != t]
            lg = _legit(items[n], fam)
            for s in _cell(readouts[n], layer, pos):
                tot += 1
                q = quantities(s)
                own = asserts(s, t, tol)
                hit += own
                miss += (any(asserts(s, o, tol) for o in others) and not own)
                if own:
                    if computation_relevant(s, t, lg):
                        clean += 1
                    else:
                        incid += 1
                    if q:
                        junk.append(sum(1 for x in q if all(
                            abs(x - v) > 0.02 * max(abs(v), 1e-9) for v in lg)) / len(q))
                # sign block: tolerance-consistent, computed for every family
                if t != 0:
                    hp = any(tolerance_ok(x, abs(t), tol) for x in q)
                    hn = any(tolerance_ok(x, -abs(t), tol) for x in q)
                    mag += (hp or hn)
                    if hp and hn:
                        both += 1
                    elif not own and (hp or hn):
                        anti += 1

        blk = {"label": label, "value": hit / tot, "cross": miss / tot, "net": (hit - miss) / tot,
               "comp_evidence": clean / tot, "incidental": incid / tot,
               "junk": mean(junk) if junk else None, "n_trials": tot,
               "reachable": all(single_token_reachable(tgt[n]) for n in names)}
        if mag:
            blk |= {"magnitude": mag / tot, "anti": anti / tot, "both": both / tot,
                    "commit": hit / mag, "signed": (hit - anti - both) / tot}
        out["steps"][label] = blk

        if si == 0:                                       # ladder on the headline step only
            for tl in TOL_LADDER:
                v = mean(mean(asserts(s, tgt[n], tl) for s in _cell(readouts[n], layer, pos))
                         for n in names)
                lsep = null_separation(tl)
                nu = mean(mean(any(asserts(s, tgt[m], tl) for m in names
                                   if abs(tgt[m] - tgt[n]) > lsep * max(abs(tgt[n]), 1e-9)
                                   and tgt[m] != tgt[n])
                               for s in _cell(readouts[n], layer, pos)) for n in names)
                out["tolerance_curve"][tl] = {"value": v, "null": nu}

    # ---- discrete-concept-family extensions (all additive; absent keys change nothing) ------
    frac_of = spec.get("fraction")
    if frac_of is not None:
        fr = []
        for n in names:
            num, den = (int(x) for x in frac_of([float(v) for v in items[n]["operands"]]))
            fr.append(mean(fraction_asserts(s, num, den)
                           for s in _cell(readouts[n], layer, pos)))
        out["frac_form"] = mean(fr)
    if len(spec["steps"]) >= 2:
        per = dict.fromkeys(("both", "either", "only_first", "only_second"), 0)
        tot2 = 0
        for n in names:
            for s in _cell(readouts[n], layer, pos):
                hits = [asserts(s, all_targets[n][si], tol) for si in (0, 1)]
                tot2 += 1
                per["both"] += all(hits)
                per["either"] += any(hits)
                per["only_first"] += hits[0] and not hits[1]
                per["only_second"] += hits[1] and not hits[0]
        out["multi_step"] = {k: v / tot2 for k, v in per.items()} | {"n_trials": tot2}
    if any("band" in items[n] for n in names):
        out["bands"] = {}
        for band in sorted({items[n].get("band", "-") for n in names}):
            bn = [n for n in names if items[n].get("band", "-") == band]
            out["bands"][band] = {
                "n": len(bn),
                "value": mean(mean(asserts(s, all_targets[n][0], tol)
                                   for s in _cell(readouts[n], layer, pos)) for n in bn)}
    cont = [mean(asserts(s, all_targets[n][0], tol)
                 for s in readouts[n]["continue"][str(pos)])
            for n in names
            if str(pos) in (readouts[n].get("continue") or {})]
    if cont:
        out["continue"] = mean(cont)

    # ---- controls, from the identical cell -------------------------------------------------
    t0 = {n: all_targets[n][0] for n in names}
    for key in ("noise", "donor"):
        vals = [mean(asserts(s, t0[n], tol) for s in readouts[n]["olens"][key])
                for n in names if key in readouts[n]["olens"]]
        out[key] = mean(vals) if vals else None
    dv = []
    for n in names:
        r = readouts[n]["olens"]
        dn = r.get("donor_from")
        if dn in items and "donor" in r:
            d = step_targets(fam, items[dn])[0]
            dv.append(mean(asserts(s, d, tol) for s in r["donor"]))
    out["donor_recovers_own"] = mean(dv) if dv else None
    return out


ROLE_MARK = {"structural": "STRUCT", "comparison": "compar", "control": "CONTROL",
             "advisory": "advis."}


class Config(pydra.Config):  # type: ignore[misc]
    """See the module docstring; `--show` prints the resolved config and exits."""

    def __init__(self) -> None:
        super().__init__()
        self.readouts: list[str] | str = []   # REQUIRED: read-stage output JSON path(s)
        self.holdout = False                  # release the withheld 20%
        self.json_out = ""                    # also write the per-family results JSON here

    def finalize(self) -> None:
        if isinstance(self.readouts, str):
            self.readouts = [self.readouts]


@pydra.main(Config)  # type: ignore[untyped-decorator]
def main(config: Config) -> None:
    if not config.readouts:
        raise SystemExit("readouts= is required (one read-stage JSON path or a list of them)")
    items: dict[str, Any] = {}
    ro: dict[str, Any] = {}
    for p in config.readouts:
        d = json.loads(Path(p).read_text())
        for i in d["items"]:
            items[i["name"]] = i["meta"]
            if i.get("continue"):        # the continue reader rides on the caps record
                ro[i["name"]] = {**ro.get(i["name"], {}), "continue": i["continue"]}
        for r in d["readouts"]:
            ro[r["name"]] = {**ro.get(r["name"], {}), **r}
    byfam: dict[str, dict[str, Any]] = defaultdict(dict)
    for n, m in items.items():
        byfam[m["variant"]][n] = m

    cols = (("value", "value"), ("net", "net"), ("comp_evidence", "comp-ev"),
            ("incidental", "incid"), ("junk", "junk"))
    print(f"{'family':<9}{'role':<8}{'n':>4}{'cell':>10}  {'step':<12}"
          + "".join(f"{h:>8}" for _, h in cols)
          + f"{'noise':>7}{'donor':>7}{'d-own':>7}  reach")
    print("-" * 104)
    results, skipped = {}, []
    for fam in FAMILIES:
        if fam not in byfam:
            if FAMILIES[fam]["cell"] is None:
                skipped.append((fam, "cell pending Stage 1 sweep"))
            continue
        if FAMILIES[fam]["cell"] is None:
            skipped.append((fam, "readouts present but cell pending — NOT scored"))
            continue
        r = score_family(fam, byfam[fam], ro, include_holdout=config.holdout)
        if not r:
            continue
        results[fam] = r
        for i, (lbl, s) in enumerate(r["steps"].items()):
            pre = (f"{fam:<9}{ROLE_MARK[r['role']]:<8}{r['n']:>4}"
                   f"{'L' + str(r['cell']['layer']) + ' p' + str(r['cell']['pos']):>10}  "
                   if i == 0 else " " * 33)
            print(pre + f"{lbl:<12}"
                  + "".join(f"{s[k]:>8.3f}" if s.get(k) is not None else f"{'-':>8}"
                            for k, _ in cols)
                  + (f"{r['noise']:>7.3f}{r['donor']:>7.3f}{r['donor_recovers_own']:>7.3f}"
                     f"  {'YES' if r['single_token_reachable'] else 'no'}" if i == 0 else ""))
            if "anti" in s:
                print(f"{'':<33}{'  sign:':<12}|v| {s['magnitude']:.3f}  commit {s['commit']:.3f}"
                      f"  anti {s['anti']:.3f}  both {s['both']:.3f}  "
                      f"SIGNED {s['signed']:+.3f}")
    print()
    print("tolerance curves (headline step; value / matched null):")
    for fam, r in results.items():
        print(f"  {fam:<9}" + "  ".join(
            f"{t.replace('rel', '').replace('pct', '%'):>6}:{v['value']:.2f}/{v['null']:.2f}"
            for t, v in r["tolerance_curve"].items()))
    if skipped:
        print("\nNOT SCORED:")
        for fam, why in skipped:
            print(f"  {fam:<10} {why}")
    dg = {f: r["degenerate"] for f, r in results.items() if r["n_degenerate"]}
    if dg:
        print("\nDROPPED as unevaluable (a scored step coincides with the answer or an operand):")
        for f, d in dg.items():
            ex = next(iter(d.items()))
            print(f"  {f:<10} {len(d)} items   e.g. {ex[0]}: {ex[1]}")
    held = sum(r["n_holdout"] for r in results.values())
    print(f"\nholdout: {held} items withheld"
          + (" — RELEASED in this run" if config.holdout else " (use holdout=True to release)"))
    if config.json_out:
        Path(config.json_out).write_text(json.dumps(results, indent=1))
        print(f"wrote {config.json_out}")


if __name__ == "__main__":
    main()
