"""Frozen specification for the buggy-code oracle-lens eval. Single source of truth.

The question: give the model raw buggy code (nothing asked); does the lens surface what the
model noticed but never said? Every design choice below was measured, not chosen — see
docs/project/experiments/oracle_lens/buggy_code_eval.md for the evidence.

THE DUAL GATE (why a scored item is interpretable at all; the gates were lettered A-E in older
records — verbalization ban = A, consequence = B, j-lens hint = C, articulation = D,
output-leak = E):
  verbalization ban  never-verbalizes  k=10+greedy RAW continuations, judged (not regexed); any
     naming, fixing, mechanism-stating or error-simulating continuation disqualifies (strict) /
     <=2 tolerated with the rate stored (tolerant). Famous patterns fail 5-10/11 — the model
     recites the blog post — so the bank contains NO textbook bugs.
  consequence gate  computes-consequence  chat render, "what does this print / what happens when
     run"; >=4/5 must match the EXECUTED truth (numeric lines 2% sign-strict). An item the model
     cannot simulate has nothing to read (sql-null-not-equal predicted the INTENDED semantics —
     cut).
  articulation gate  articulable-on-demand  review question finds the bug 4/4, else the item is a
     capability ceiling, not a lens question (tax-before-discount, 2/4 — cut).
  j-lens hint tag  (advisory stratification only, NEVER a filter): anchor-restricted top-5 L>=48
     scan for trouble tokens not present in the item text. Any-position scans light up clean
     items too — that version is the op-only bug and is banned.

ITEM PURITY (lint-enforced): no trouble vocabulary and no negative literals in code text —
J-lens emitted 'negative' at a bare '-3'; identifiers must not name the failure
('returns_as_negative_qty' echoed at rank 0). Badness may only exist at runtime.
"""

from typing import Any

MODEL_ID = "Qwen/Qwen3.6-27B"
DEFAULT_OLENS = "agu18dec/qwen3.6-27b-mlayer-ao-checkpoints:ao.asst.alldata-v2cont.k2.g8"
OLENS_SCALE = 33.152003804062936
OLENS_PROMPT_KIND = "explain"
DATASET = "agu18dec/olens_eval_suite"          # banks live under buggy-code/ on the dataset

# Read cells: bare code, EOF anchor, layer per language group — each from its OWN
# 3-anchor x 6-layer sweep (inheriting python's cell was measurably wrong: non-python peaks
# at L56, python at L60). Bug-line and innocent-line anchors are DEAD at every layer
# (<=0.05 flavour) — the standing no-localization result, reported, never scored.
READ_CELLS: dict[str, dict[str, Any]] = {
    "python": {"anchor": "eof", "layer": 60},
    "nonpython": {"anchor": "eof", "layer": 56},
}
SWEEP_LAYERS = (44, 48, 52, 56, 60, 63)
SWEEP_ANCHORS = ("eof", "bugline", "innocent")
SAMPLING = {"k": 10, "max_new": 44, "temperature": 0.8, "top_p": 0.95}
CONTROL_SEEDS = {"readout": 0, "repeat": 1, "noise": 101, "donor": 102}

# Judge rubric (gpt-5.5; planted-case calibration 60/60 before any data; quote stored per
# verdict so every positive is spot-checkable — the 490-readout run had 1 bad verdict, 0.2%).
RUNGS = ("S0", "S1", "corrective", "S2", "S3", "S4")
# S2 includes test-assertion frames ('# Expected: -2') asserting the buggy outcome — the AO's
# native idiom here is assertions, not prose; a prose-only rubric scores ~everything S0.
# anti (asserts-runs-fine on broken code) is reported per stratum, never folded into a scalar.
HEADLINE = "net_S2"     # P(rung >= S2 | buggy) - P(rung >= S2 | clean twin), per stratum

# Strata: consequence_class x lang_group. Measured result (first scored run, this checkpoint):
#   categorical +0.17 (all from negative-outcome items) | crash +0.40 (hazard-condition reads)
#   value -0.00 (hard zero) | nonpython +0.03 (language drift). S3-mechanism 0.00 everywhere.
CONSEQUENCE_CLASSES = ("categorical", "value", "crash")

# The deterministic value-matcher (order-ops style) is a SPOT-CHECK only: it false-positives
# on small-integer truths (digit collision; clean twins read 0.108 under it vs 0.005 judged).
# Usable where truth values are distinctive (|v| >= 10, non-round); agreement there was good.
DETERMINISTIC_CHECK_MIN_ABS = 10.0

HOLDOUT_FRAC = 0.20
HOLDOUT_SEED = 20260804


def holdout_names(names: list[str]) -> set[str]:
    """Deterministic holdout: same names -> same split, order-independent."""
    import hashlib

    def key(n: str) -> str:
        return hashlib.sha256(f"{HOLDOUT_SEED}\x00{n}".encode()).hexdigest()

    ordered = sorted(names, key=key)
    return set(ordered[: max(1, round(len(ordered) * HOLDOUT_FRAC))])
