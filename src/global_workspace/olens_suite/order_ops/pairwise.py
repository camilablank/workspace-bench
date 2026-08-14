"""Pairwise compositionality: match two readout sets to two twin expressions. Chance = 0.5.

The twins share one operand bag — (7/3)*21 vs (3/7)*21 both show the judge {7, 3, 21} — so a
bag-of-digits reader (J-lens's structural ceiling here) carries no matching signal, while a
workspace read that BINDS numerator to denominator does. This is the frac headline and the
negdiv8<->negdiv8x contrast's qualitative twin.

Discipline inherited from buggy_code.pairwise, whose pooled variant was VOIDED when cell labels
leaked which set was which (LM(J-lens) scored 1.000 at chance-level content):
  - positional structure only — the payload is asserted free of bank names, layer strings,
    anchor names, and control-arm words before it can be emitted (_assert_clean);
  - both sets truncated to the same sample count (a count asymmetry is a label);
  - expression order = sha256 of the expression TEXT, set order = sha256 of (pair, rubric) —
    deterministic, unguessable from the item, re-randomised only by bumping RUBRIC_VERSION;
  - FOIL pairs: both sets from the SAME item (scored block seed 0 vs the independent repeat
    seed 1), shown with the same two expressions. Pick accuracy on foils must sit at ~0.5;
    an arm beating chance on its foils has a leaking payload, and its real rate is void.
    Foils exist only for the sampled arm (olens) — J-lens top-k is deterministic, one set only.
  - answer_driven is computed MECHANICALLY per pair (a set asserting its own item's final
    answer and not the twin's decides the match without any intermediate reading) and reported
    beside the pick rate, never subtracted silently.

The lm_jlens / lm_olens arms are not judged here at all: their expression-blind value
estimates (lens_interpret.py) pair mechanically — assign the two estimates to the two targets
by total absolute error. An LLM judge would add noise to what is already a number.

    # 1. emit payloads for audit + the judge inputs
    python -m ...order_ops.pairwise build=True reads='["results/order_ops/read_frac.json"]'
    # 2. judge (llm_client.async_json; fail-open — ungraded pairs stay ungraded)
    python -m ...order_ops.pairwise judge=True reads='[...]'
    # 3. score
    python -m ...order_ops.pairwise reads='[...]'
"""

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, NamedTuple

import pydra

from global_workspace.olens_suite.order_ops.score import asserts
from global_workspace.olens_suite.order_ops.spec import FAMILIES, GATE, step_targets

RUBRIC_VERSION = "orderops-pairwise-1"
SYSTEM = (
    "You see two arithmetic prompts and two sets of text read out from a language model's "
    "hidden activations, taken at identical positions while it silently computed each prompt. "
    "Exactly one set belongs to each prompt. Decide which prompt SET A belongs to. Reply in "
    "the given JSON schema; set_A_prompt is 1 or 2."
)
# words that must never appear in a payload: bank names, cells, control arms, anchor roles
FORBIDDEN = re.compile(
    r"(mulmul-|frac-|fracadd-|fraccomp-|negdiv8|subsub|addmul|halves-|"
    r"L\d\d@|\bseed\b|\bdonor\b|\bnoise\b|"
    r"\brepeat\b|\bfoil\b|op_in|op_out|after_close|bugline)", re.IGNORECASE)


def first_set_is_first_item(pair_key: str, rubric_version: str = RUBRIC_VERSION) -> bool:
    digest = hashlib.sha256(f"{rubric_version}\x00{pair_key}".encode()).digest()
    return digest[0] % 2 == 0


class Payload(NamedTuple):
    text: str            # what the judge sees (with SYSTEM)
    answer: int          # ground truth: which prompt (1|2) SET A belongs to
    pair_key: str


def _assert_clean(text: str, names: list[str]) -> None:
    hit = FORBIDDEN.search(text)
    if hit:
        raise ValueError(f"payload contains forbidden scaffolding {hit.group(0)!r}")
    for n in names:
        if n in text:
            raise ValueError(f"payload contains bank name {n!r}")


def build_payload(pair_key: str, expr_a: str, expr_b: str,
                  set_a: list[str], set_b: list[str], *, names: list[str],
                  rubric_version: str = RUBRIC_VERSION) -> Payload:
    """(expr_a, set_a) and (expr_b, set_b) belong together; ordering is re-derived here.

    Expressions are shown in sha256-of-text order (bank order encodes generation order, which
    correlates with direction); sets are shown in hash-of-(pair, rubric) order. Sample counts
    are equalized by truncation — and an empty set raises, because 'one side has fewer lines'
    is a label.
    """
    if not set_a or not set_b:
        raise ValueError(f"{pair_key}: empty readout set — cannot build a fair payload")
    k = min(len(set_a), len(set_b))
    set_a, set_b = set_a[:k], set_b[:k]
    exprs = sorted([expr_a, expr_b], key=lambda e: hashlib.sha256(e.encode()).hexdigest())
    items = [(expr_a, set_a), (expr_b, set_b)]
    if not first_set_is_first_item(pair_key, rubric_version):
        items = items[::-1]
    flat = [" ".join(s.split()) if s.strip() else "(empty)" for s in items[0][1]]
    flat_b = [" ".join(s.split()) if s.strip() else "(empty)" for s in items[1][1]]
    text = (f"PROMPT 1: {exprs[0]}\nPROMPT 2: {exprs[1]}\n\n"
            + "SET A\n" + "\n".join(f"  {s}" for s in flat)
            + "\n\nSET B\n" + "\n".join(f"  {s}" for s in flat_b))
    _assert_clean(text, names)
    answer = exprs.index(items[0][0]) + 1
    return Payload(text, answer, pair_key)


# ---- assembling pairs from read-stage output ---------------------------------------------------

def _pairs(items: dict[str, Any]) -> list[tuple[str, str]]:
    """Unordered twin pairs present in the read output (both sides must be there)."""
    done: set[str] = set()
    out: list[tuple[str, str]] = []
    for n, meta in items.items():
        tw = meta.get("pair")
        if tw and tw in items and n not in done and tw not in done:
            out.append((n, tw))
            done.update((n, tw))
    return out


def _olens_set(readouts: dict[str, Any], name: str, layer: int) -> list[str]:
    return list(readouts[name]["olens"]["layers"][str(layer)])


def _jlens_set(items: dict[str, Any], name: str, layer: int, pos: int) -> list[str]:
    grid = items[name]["jlens_top"][str(layer)]
    return [" ".join(t.strip() or "·" for t in grid[items[name]["n_pos"] + pos])]


def build_all(items: dict[str, Any], readouts: dict[str, Any],
              rubric_version: str = RUBRIC_VERSION) -> dict[str, Payload]:
    """All payloads: `<pairkey>|<arm>|real` plus `|foil` for the sampled arm."""
    names = sorted(items)
    out: dict[str, Payload] = {}
    for a, b in _pairs(items):
        fam = items[a]["variant"]
        cell = FAMILIES[fam]["cell"]
        pk = "+".join(sorted((a, b)))
        ea, eb = items[a]["expr"], items[b]["expr"]
        out[f"{pk}|olens|real"] = build_payload(
            f"{pk}|olens", ea, eb, _olens_set(readouts, a, cell["layer"]),
            _olens_set(readouts, b, cell["layer"]), names=names,
            rubric_version=rubric_version)
        out[f"{pk}|jlens|real"] = build_payload(
            f"{pk}|jlens", ea, eb,
            _jlens_set(items, a, cell["layer"], cell["pos"]),
            _jlens_set(items, b, cell["layer"], cell["pos"]), names=names,
            rubric_version=rubric_version)
        # foil: both sets from item `a` (seed-0 scored block vs seed-1 repeat), same prompts.
        # `answer` is which prompt the seed-0 set was assigned — contentless, so accuracy ~0.5.
        rep = readouts[a]["olens"].get("repeat", {}).get("1")
        if rep:
            out[f"{pk}|olens|foil"] = build_payload(
                f"{pk}|olens|foil", ea, eb, _olens_set(readouts, a, cell["layer"]),
                list(rep), names=names, rubric_version=rubric_version)
    return out


def answer_driven(items: dict[str, Any], a: str, b: str,
                  readouts: dict[str, Any]) -> bool:
    """Could the pair be matched on final ANSWERS alone? (own answer asserted, twin's not)"""
    fam = items[a]["variant"]
    layer = FAMILIES[fam]["cell"]["layer"]
    tol = str(GATE["answer_tol"])
    for x, y in ((a, b), (b, a)):
        txt = " ".join(_olens_set(readouts, x, layer))
        if asserts(txt, float(items[x]["answer"]), tol) and \
                not asserts(txt, float(items[y]["answer"]), tol):
            return True
    return False


def lead_score(samples: list[str], num: int, den: int) -> int:
    """How strongly a readout set leads with num/ over den/ — the judge-free direction channel
    for fraction twins (the LLM pairwise judge could under-read real binding; this cannot)."""
    n = len(re.findall(rf"(?<![\d.]){num}\s*/", " ".join(samples)))
    d = len(re.findall(rf"(?<![\d.]){den}\s*/", " ".join(samples)))
    return n - d


def intermediate_pick(set_a: list[str], set_b: list[str], ta: float, tb: float,
                      tol: str) -> bool | None:
    """Judge-free twin assignment by WHICH intermediate each set asserts (quad-separation at
    generation guarantees ta/tb never collide at the scoring tolerance). None on a tie."""
    own = sum(asserts(s, ta, tol) for s in set_a) + sum(asserts(s, tb, tol) for s in set_b)
    cross = sum(asserts(s, tb, tol) for s in set_a) + sum(asserts(s, ta, tol) for s in set_b)
    if own == cross:
        return None
    return own > cross


def mechanical_pick(est_a: float | None, est_b: float | None,
                    tgt_a: float, tgt_b: float) -> bool | None:
    """Pair two expression-blind value estimates with two targets by total absolute error.
    None (no verdict) when either estimate is missing or the assignment is a tie."""
    if est_a is None or est_b is None:
        return None
    right = abs(est_a - tgt_a) + abs(est_b - tgt_b)
    wrong = abs(est_a - tgt_b) + abs(est_b - tgt_a)
    if right == wrong:
        return None
    return right < wrong


# ---- CLI ---------------------------------------------------------------------------------------

class Config(pydra.Config):  # type: ignore[misc]
    def __init__(self) -> None:
        super().__init__()
        self.reads: list[str] | str = []       # read-stage JSON path(s) with twin families
        self.build = False                     # dump payloads (audit) and exit
        self.judge = False                     # call the LLM judge, write verdicts
        self.interpret = ""                    # lens_interpret output (adds lm_* mechanical rows)
        self.verdicts = "results/order_ops/pairwise_verdicts.json"
        self.model = ""                        # judge model override (default: llm_client)
        self.rubric_version = RUBRIC_VERSION

    def finalize(self) -> None:
        if isinstance(self.reads, str):
            self.reads = [self.reads]


def _load(paths: list[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    items: dict[str, Any] = {}
    ro: dict[str, Any] = {}
    for p in paths:
        d = json.loads(Path(p).read_text())
        for i in d["items"]:
            items[i["name"]] = {**i["meta"], "jlens_top": i.get("jlens_top", {}),
                                "n_pos": i["n_pos"]}
        for r in d["readouts"]:
            ro[r["name"]] = r
    return items, ro


@pydra.main(Config)  # type: ignore[untyped-decorator]
def main(config: Config) -> None:
    if not config.reads:
        raise SystemExit("reads= is required (read-stage JSON with twin families)")
    items, ro = _load(list(config.reads))
    payloads = build_all(items, ro, config.rubric_version)
    if config.build:
        for key, p in payloads.items():
            print(f"### {key}  (answer: prompt {p.answer})\n{p.text}\n")
        return

    if config.judge:
        from global_workspace.judges.llm_client import async_json, schema_block

        schema = schema_block("pairwise_pick", {
            "set_A_prompt": {"type": "integer", "enum": [1, 2]},
            "rationale": {"type": "string"},
        }, ["set_A_prompt", "rationale"])
        keys = sorted(payloads)
        rows = [(SYSTEM, payloads[k].text) for k in keys]
        kwargs: dict[str, Any] = {"schema": schema}
        if config.model:
            kwargs["model"] = config.model
        got = async_json(rows, **kwargs)
        verdicts: dict[str, dict[str, Any] | None] = {}
        for k, v in zip(keys, got, strict=True):
            verdicts[k] = None if v is None else {
                "set_A_prompt": int(v["set_A_prompt"]),
                "correct": int(v["set_A_prompt"]) == payloads[k].answer,
                "rationale": v.get("rationale", "")}
        Path(config.verdicts).write_text(json.dumps(verdicts, indent=1))
        done = sum(1 for v in verdicts.values() if v is not None)
        print(f"wrote {config.verdicts} ({done}/{len(verdicts)} graded)")
        return

    # ---- score ----------------------------------------------------------------------------
    verdicts = json.loads(Path(config.verdicts).read_text()) if \
        Path(config.verdicts).exists() else {}
    rows_by: dict[tuple[str, str], list[bool]] = defaultdict(list)
    ungraded = 0
    for key in payloads:
        _pk, arm, kind = key.rsplit("|", 2)
        v = verdicts.get(key)
        if v is None:
            ungraded += 1
            continue
        rows_by[(arm, kind)].append(bool(v["correct"]))
    ad = [answer_driven(items, a, b, ro) for a, b in _pairs(items)]
    if not ad:
        print("no twin pairs found")
        return
    print(f"{'arm':<10}{'kind':<6}{'n':>4}{'pick':>8}   (chance 0.5; "
          f"{ungraded} payloads ungraded; answer_driven share {mean(ad):.2f} of {len(ad)})")
    for (arm, kind), got_rows in sorted(rows_by.items()):
        print(f"{arm:<10}{kind:<6}{len(got_rows):>4}{mean(got_rows):>8.3f}")
    int_rows = []
    for a, b in _pairs(items):
        fam_a, fam_b = items[a]["variant"], items[b]["variant"]
        ta = step_targets(fam_a, items[a])[0]
        tb = step_targets(fam_b, items[b])[0]
        pick = intermediate_pick(
            _olens_set(ro, a, FAMILIES[fam_a]["cell"]["layer"]),
            _olens_set(ro, b, FAMILIES[fam_b]["cell"]["layer"]),
            ta, tb, FAMILIES[fam_a]["tol"])
        if pick is not None:
            int_rows.append(pick)
    if int_rows:
        print(f"{'olens':<10}{'mint':<6}{len(int_rows):>4}{mean(int_rows):>8.3f}"
              f"   (deterministic: which set asserts which intermediate)")
    lead_rows = []
    for a, b in _pairs(items):
        frac_of = FAMILIES[items[a]["variant"]].get("fraction")
        if frac_of is None:
            continue
        pa, qa = (int(x) for x in frac_of([float(v) for v in items[a]["operands"]]))
        layer = FAMILIES[items[a]["variant"]]["cell"]["layer"]
        la = lead_score(_olens_set(ro, a, layer), pa, qa)
        lb = lead_score(_olens_set(ro, b, layer), pa, qa)
        if la != lb:
            lead_rows.append(la > lb)
    if lead_rows:
        print(f"{'olens':<10}{'lead':<6}{len(lead_rows):>4}{mean(lead_rows):>8.3f}"
              f"   (deterministic numerator-lead, fraction twins only)")
    if config.interpret:
        interp = json.loads(Path(config.interpret).read_text())
        for arm in ("jlens", "olens"):
            picks = []
            for a, b in _pairs(items):
                ta = step_targets(items[a]["variant"], items[a])[0]
                tb = step_targets(items[b]["variant"], items[b])[0]
                pick = mechanical_pick(
                    interp.get(a, {}).get(arm, {}).get("value"),
                    interp.get(b, {}).get(arm, {}).get("value"), ta, tb)
                if pick is not None:
                    picks.append(pick)
            if picks:
                print(f"{'lm_' + arm:<10}{'mech':<6}{len(picks):>4}{mean(picks):>8.3f}")


if __name__ == "__main__":
    main()
