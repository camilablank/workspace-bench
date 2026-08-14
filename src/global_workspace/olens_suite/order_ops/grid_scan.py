"""Where does each step's value surface for a TOKEN lens? Pure-CPU scan of the grid stage.

    python -m global_workspace.olens_suite.order_ops.grid_scan \\
        grids='["results/order_ops/grid_mulmul.json"]'

Reads grid_<family>.json (J-lens + logit-lens top-k token IDS at every layer x position, plus
the per-item vocab decode table) and reports, per family x lens x step, the hit rate by
(layer, position) — aligned two ways, because operand digit counts shift raw indices:

  by REL position   pos - n_pos (the boundary region the sweep covers, rel -1..-20)
  by ANCHOR role    anchors.find_anchor_roles (op1/op2/close/dot..., the expression region)

Channels (NEVER summed — Qwen numbers are digit-tokenized with no exceptions
(number_tokenization.md), so a >=2-digit value can reach a SINGLE token only as a number WORD:
'十五', ' fifteen'. The raw grids show exactly that. A channel-sum would sell digit fragments
as value reads):
  full       the token parses as a number — ASCII digits, fullwidth digits, CJK numerals, or
             English number words — matching the SIGNED step target at the family tolerance
  magnitude  same, matching |target| — the informative row for negative-target families, since
             number words are unsigned
  prefix2    a digit-run token of >=2 chars that is a PREFIX of the target's digit string
  litter     any numeric token at all — the honesty row every other rate is read against
  null       full's permutation null: ANY null_separation-separated OTHER item's target matches
             at the same cell (score.py's convention). A cell is only a finding at value >>
             null — negdiv8 step2 'hit' 0.98 at L62 rel-1 sat on litter 1.0 / null 0.71, i.e.
             eight of rel-1's top-10 are bare digits and ANY 1-digit target 'matches' there.
  magnull    magnitude's permutation null, same construction

Output: results/order_ops/grid_scan_<family>.json + a printed heatmap per family/lens/step
(layers x rel positions, and the anchor-role table). This is the funnel's narrowing evidence
and the "where to point OLens" input — OLens itself is never run on the grid.
"""

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import pydra

from global_workspace.olens_suite.order_ops.anchors import find_anchor_roles
from global_workspace.olens_suite.order_ops.spec import (
    FAMILIES,
    null_separation,
    step_targets,
    tolerance_ok,
)

REL_WINDOW = 20                       # boundary rels reported: -1 .. -REL_WINDOW
DIGITS = re.compile(r"^\d+$")
NUMBER = re.compile(r"^-?\d+\.?\d*$")
FULLWIDTH = str.maketrans("０１２３４５６７８９．", "0123456789.")  # noqa: RUF001

_CJK_DIGIT = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6,
              "七": 7, "八": 8, "九": 9}
_CJK_UNIT = {"十": 10, "百": 100, "千": 1000}
_EN_UNIT = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
            "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
            "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
            "eighteen": 18, "nineteen": 19}
_EN_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
            "seventy": 70, "eighty": 80, "ninety": 90}


def _parse_cjk(s: str) -> float | None:
    """Standard CJK numerals up to 9999 (十五=15, 四十九=49, 三百=300)."""
    if not s or any(c not in _CJK_DIGIT and c not in _CJK_UNIT for c in s):
        return None
    total, current = 0, 0
    for c in s:
        if c in _CJK_DIGIT:
            current = _CJK_DIGIT[c]
        else:
            total += (current or 1) * _CJK_UNIT[c]
            current = 0
    return float(total + current)


def _parse_english(s: str) -> float | None:
    """English number words up to 999 ('fifteen', 'forty-nine', 'three hundred')."""
    words = s.lower().replace("-", " ").split()
    if not words:
        return None
    total, current = 0, 0
    for w in words:
        if w in _EN_UNIT:
            current += _EN_UNIT[w]
        elif w in _EN_TENS:
            current += _EN_TENS[w]
        elif w == "hundred" and current:
            current *= 100
        elif w == "and" and current:
            continue
        else:
            return None
    return float(total + current)


def token_value(token: str) -> float | None:
    """A single top-k token parsed as a number, via any of the routes a single Qwen token can
    carry one: ASCII/fullwidth digits, CJK numerals, English number words."""
    t = token.strip().translate(FULLWIDTH)
    if NUMBER.match(t):
        try:
            return float(t)
        except ValueError:
            return None
    return _parse_cjk(t) if t and t[0] in (_CJK_DIGIT | _CJK_UNIT) else _parse_english(t)


def target_digits(v: float) -> str:
    s = f"{abs(v):.6f}".rstrip("0").rstrip(".")
    return s.replace(".", "")


def classify(token: str, target: float, tol: str) -> set[str]:
    """Channels this top-k token hits for this step target."""
    t = token.strip().translate(FULLWIDTH)
    out: set[str] = set()
    v = token_value(token)
    if v is not None:
        out.add("litter")
        if tolerance_ok(v, target, tol):
            out.add("full")
        if tolerance_ok(v, abs(target), tol):
            out.add("magnitude")
    if DIGITS.match(t) and len(t) >= 2 and target_digits(target).startswith(t):
        out.add("prefix2")
    return out


def scan_family(payload: dict[str, Any]) -> dict[str, Any]:
    items = payload["items"]
    fam = items[0]["meta"]["variant"]
    tol = FAMILIES[fam]["tol"]
    mode = FAMILIES[fam].get("anchors", "single")
    steps = [label for label, _ in FAMILIES[fam]["steps"]]
    # rate[lens][step][layer]["rel:-8" | "role:op1"] = (hits, trials)
    acc: dict[str, dict[str, dict[str, dict[str, list[int]]]]] = {
        lens: {s: defaultdict(lambda: defaultdict(lambda: [0, 0])) for s in steps}
        for lens in ("jlens", "logit")}
    sepn = null_separation(tol)
    all_step_targets = {it["name"]: step_targets(fam, it["meta"]) for it in items}
    for it in items:
        n = it["n_pos"]
        vocab = it["vocab"]
        tgts = dict(zip(steps, all_step_targets[it["name"]], strict=True))
        # the permutation null per step: every separated other item's target (twin excluded —
        # same rule as the banks' null_set)
        nulls = {s: [all_step_targets[o["name"]][si] for o in items
                     if o["name"] != it["name"] and o["name"] != it["meta"].get("pair")
                     and abs(all_step_targets[o["name"]][si] - tgts[s])
                     > sepn * max(abs(tgts[s]), 1e-9)]
                 for si, s in enumerate(steps)}
        roles = find_anchor_roles(it["tokens"], mode)
        pos_labels: dict[int, list[str]] = defaultdict(list)
        for p in range(n):
            if p - n >= -REL_WINDOW:
                pos_labels[p].append(f"rel:{p - n}")
        for role, p in roles.items():
            pos_labels[p].append(f"role:{role}")
        for lens, key in (("jlens", "jlens_top"), ("logit", "logit_top")):
            for layer, rows in it[key].items():
                if not rows:
                    continue          # J-lens absent layer
                for p, labels in pos_labels.items():
                    vals = [v for v in (token_value(vocab[str(t)]) for t in rows[p])
                            if v is not None]
                    toks = [vocab[str(t)] for t in rows[p]]
                    for s in steps:
                        chans = set()
                        for tk in toks:
                            chans |= classify(tk, tgts[s], tol)
                        if any(tolerance_ok(v, o, tol) for v in vals for o in nulls[s]):
                            chans.add("null")
                        if any(tolerance_ok(v, abs(o), tol)
                               for v in vals for o in nulls[s]):
                            chans.add("magnull")
                        cell = acc[lens][s][layer]
                        for lab in labels:
                            cell[lab][0] += bool(chans & {"full"})
                            cell[lab][1] += 1
                            # side channels ride in parallel keys
                            for ch in ("magnitude", "prefix2", "litter", "null", "magnull"):
                                c2 = acc[lens][s][layer][f"{ch}|{lab}"]
                                c2[0] += ch in chans
                                c2[1] += 1
    out: dict[str, Any] = {"family": fam, "n_items": len(items), "tolerance": tol,
                           "steps": steps, "rates": {}}
    for lens in acc:
        out["rates"][lens] = {}
        for s in steps:
            out["rates"][lens][s] = {
                layer: {lab: h / t for lab, (h, t) in cells.items() if t}
                for layer, cells in acc[lens][s].items()}
    return out


def heat(rates: dict[str, dict[str, float]], labels: list[str]) -> list[str]:
    """One text row per layer, one column per label; '.'<1% then 1-9 deciles then '#'>=95%."""
    lines = []
    for layer in sorted(rates, key=int):
        row = ""
        for lab in labels:
            v = rates[layer].get(lab)
            row += ("." if v is None or v < 0.01 else
                    "#" if v >= 0.95 else str(min(9, int(v * 10))))
        lines.append(f"  L{int(layer):02d} {row}")
    return lines


class Config(pydra.Config):  # type: ignore[misc]
    def __init__(self) -> None:
        super().__init__()
        self.grids: list[str] | str = []      # grid_<family>.json path(s)
        self.json_out_dir = "results/order_ops"

    def finalize(self) -> None:
        if isinstance(self.grids, str):
            self.grids = [self.grids]


@pydra.main(Config)  # type: ignore[untyped-decorator]
def main(config: Config) -> None:
    if not config.grids:
        raise SystemExit("grids= is required (grid-stage JSON path(s))")
    for p in config.grids:
        payload = json.loads(Path(p).read_text())
        r = scan_family(payload)
        fam = r["family"]
        rel_labels = [f"rel:{-i}" for i in range(REL_WINDOW, 0, -1)]
        print(f"\n==== {fam} (n={r['n_items']}, tol={r['tolerance']}) "
              f"— columns rel:-{REL_WINDOW}..-1, digits = deciles ====")
        for lens in ("jlens", "logit"):
            for s in r["steps"]:
                rates = r["rates"][lens][s]
                best = max(((layer, lab, v) for layer, cells in rates.items()
                            for lab, v in cells.items() if "|" not in lab),
                           key=lambda x: x[2], default=None)
                mbest = max(((layer, lab.split("|")[1], v) for layer, cells in rates.items()
                             for lab, v in cells.items() if lab.startswith("magnitude|")),
                            key=lambda x: x[2], default=None)
                line = f"\n-- {lens} / step {s}"
                if best:
                    nul = rates[best[0]].get(f"null|{best[1]}", 0.0)
                    line += f"   peak {best[2]:.2f} (null {nul:.2f}) @ L{best[0]} {best[1]}"
                if mbest:
                    mn = rates[mbest[0]].get(f"magnull|{mbest[1]}", 0.0)
                    line += f"   |v|peak {mbest[2]:.2f} (null {mn:.2f}) @ L{mbest[0]} {mbest[1]}"
                print(line)
                for line in heat({la: {k: v for k, v in ce.items() if "|" not in k}
                                  for la, ce in rates.items()}, rel_labels):
                    print(line)
                roles = sorted({lab for cells in rates.values() for lab in cells
                                if lab.startswith("role:")})
                if roles:
                    hdr = "        " + "  ".join(f"{ro[5:]:>12}" for ro in roles)
                    print(hdr)
                    for layer in sorted(rates, key=int):
                        vals = "  ".join(f"{rates[layer].get(ro, 0.0):>12.3f}" for ro in roles)
                        print(f"  L{int(layer):02d} {vals}")
        outp = Path(config.json_out_dir) / f"grid_scan_{fam}.json"
        outp.write_text(json.dumps(r, indent=1))
        print(f"\nwrote {outp}")


if __name__ == "__main__":
    main()
