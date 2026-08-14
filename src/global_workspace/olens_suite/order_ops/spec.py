"""Frozen specification for the order-ops oracle-lens eval. Single source of truth.

Nothing here reads data or runs a model. Everything downstream — the Modal stages, the CPU
scorer, the report — imports from here, so a tolerance or a read cell can only be changed in
one place.

WHY EACH FIELD EXISTS (all learned by getting it wrong first):

CELLS       (layer, position), pre-registered from a sweep BEFORE scoring. `sqrt` reads 0.992 at
            rel-7 and 0.458 at rel-8 — one token apart. `cell: None` means the sweep has not run
            and the family is not scoreable yet; the scorer refuses it rather than guessing.
STEPS       Every step of an expression is scored. Scoring negdec's SECOND step alone gave 0.003
            and the reading "the lens cannot do negative decimals"; step 1 reads 0.664 and step 2
            is a hard 0.000. With all steps scored, "reads the first intermediate and not later
            ones" is an OUTPUT, not an assumption.
            The lambda is cross-checked against the bank's own `intermediates` field and a
            mismatch RAISES — that is what catches "scored the wrong step".
TOL_LADDER  Presence at five tolerances, never one. `muladd` is 0.12 exact and 0.75 at 2%; the
            SHAPE of that curve is the lens's numeric precision.
HOLDOUT     20% withheld until the metric is frozen. Re-froze 2026-08-03 when `op_only` was
            found to be a punctuation artifact; the holdout has never been released.
REACHABLE   A 1- or 2-digit integer target is reachable by a SINGLE vocabulary token (十八 /
            XVIII / eighteen all denote 18), measured at 0.833 for J-lens. Families with such
            targets are COMPARISONS, not structural claims, and are flagged not averaged in.
ROLE        What a number from this family is allowed to support. `control` families exist to
            invalidate the design if they score like the real ones.

REMOVED 2026-08-03 — `op_only` and its `produced_by` word lists. `produced_by` contained bare
symbols ("/", "-"), so `op_only` matched any slash or hyphen: 97.8% of dec16's 0.309 was a bare
"/", and 88.6% of negdec step 2's 0.886 was a bare "-". With word forms only it is <=0.026 in
every family, i.e. the lens does not name operations in words when it lacks the value. The metric
measured punctuation. Deleted rather than repaired, and the "the AO writes operations as symbols"
claim that came from it is withdrawn.
"""

import math
from collections.abc import Callable
from typing import Any

MODEL_ID = "Qwen/Qwen3.6-27B"
DEFAULT_OLENS = "agu18dec/qwen3.6-27b-mlayer-ao-checkpoints:ao.asst.alldata-v2cont.k2.g8"
DEFAULT_JLENS = ("neuronpedia/jacobian-lens:"
                 "qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt")
DATASET = "agu18dec/olens_eval_suite"

# The lens the DISCRETE-CONCEPT families (mulmul/frac/fraccomp/negdiv8/negdiv8x/halves) are
# designed against and whose sweep registers their cells (decision 2026-08-09): the iolens chat
# AO — the registry's best multilayer chat cell (`docs/project/checkpoints.md`, val CE 0.7019).
# Passed to the runners via --olens/--olens-scale; DEFAULT_OLENS above stays the lens behind the
# original families' published numbers (order_ops_eval.md). The scale is the run's own frozen
# injection scale — from its run.json, never guessed.
IOLENS_CHAT_AO = "agu18dec/oracle-lens-ao-checkpoints:ao.iolens.chat.k4.L20plus.s2/step3002"
IOLENS_CHAT_AO_SCALE = 64.559

# From the checkpoint's own run.json. Never guess: the injection is `scale * h`, and a wrong
# scale silently degrades every readout instead of failing.
OLENS_SCALE = 33.152003804062936
OLENS_PROMPT_KIND = "explain"      # NOT "explanation", a different and more verbose template

# J-lens has NO Jacobian for the final layer (fitted with target_layer=-1). Reading it there is
# an empty slice, which silently scores 0. Recorded as N/A instead.
JLENS_ABSENT_LAYERS: tuple[int, ...] = (63,)

SWEEP_LAYERS: tuple[int, ...] = (44, 52, 56, 60, 63)
SWEEP_POSITIONS: tuple[int, ...] = tuple(range(-1, -15, -1))

SAMPLING = {"k": 10, "max_new": 40, "temperature": 0.8, "top_p": 0.95}
SWEEP_SAMPLING = {"k": 5, "max_new": 40, "temperature": 0.8, "top_p": 0.95}

# The `continue` reader (READERS below): the BASE model (adapter disabled) sampled from just
# after the read position. If it states the intermediate, the target was next-token predictable
# at that cell and OLens presence there is not evidence. max_new=16: an intermediate is 2-6
# tokens; 16 is generous without paying for essays. Seed 103 — the control-seed ladder is
# scored=0, repeat=1, noise=101, donor=102 (each control needs its OWN seed: generation is
# deterministic given (activation, seed), see the donor note above REPEAT_SEEDS).
CONTINUE = {"k": 10, "max_new": 16, "temperature": 0.8, "top_p": 0.95, "seed": 103}

# The token-lens GRID stage: J-lens + logit-lens top-k recorded at EVERY layer x EVERY position
# (one matmul per layer — cheap, no adapter). This is the funnel's second gate: where (if
# anywhere) each step's value surfaces for a token lens, per layer x position, BEFORE any OLens
# GPU is spent. Layer 63 has no Jacobian (JLENS_ABSENT_LAYERS) — recorded absent, never null.
GRID_LAYERS: tuple[int, ...] = tuple(range(64))
GRID_TOPK = 10

# ---- gating -------------------------------------------------------------------------------
# An item is usable only if the subject model (a) answers it correctly and (b) never states the
# target in its own committed output. If (a) fails, a lens miss is uninterpretable. If (b) fails,
# reading the target is next-token prediction, which the logit lens does for free.
#
# The original gate used ONE greedy rollout per item, which cannot distinguish "reliably correct"
# from "correct that time". A single sample also cannot measure leak RATE. Both are now k-of-n at
# the same temperature the eval samples at, because that is the distribution the activations are
# actually drawn from.
GATE = {
    "k": 10, "max_new": 32, "temperature": 0.8, "top_p": 0.95,
    "min_correct": 0.8,      # >=8/10 correct. Below this the item is noise, not an eval.
    "max_leak": 0.0,         # ANY rollout stating the target disqualifies the item.
    "answer_tol": "rel0.5pct",
}
# Also recorded per item, never used as the gate: the greedy rollout, so "does it need sampling
# to be right" stays answerable.

# ---- readers compared ---------------------------------------------------------------------
# The point of a baseline here is to answer "could you have got this without the OLens?". Each of
# these can, in principle, produce the target; the ones that can't are the interesting negatives.
READERS = {
    "olens": "the lens under test: inject scale*h into the <concept> slot, sample k times",
    "jlens": "official J-lens top-k via cosine_readout with the jlens_token_norms denominator",
    "logit": "plain logit lens top-k — BEATS J-lens on this task (0.933 vs 0.833 integer parts), "
             "so omitting it would overstate what the Jacobian contributes",
    "continue": "base model continues from the read position, no lens. If this recovers the "
                "target then the target was simply next-token predictable. Sampled per the "
                "CONTINUE dict (seed 103), scored by the same tolerance machinery.",
    "lm_jlens": "a judge LM sees the anonymized top-k J-lens tokens at the cell — never the "
                "expression — and estimates the held value. What is INFERABLE from the token "
                "bag, as opposed to what literally appears in it.",
    "lm_olens": "the same judge treatment of the k OLens samples. Denoises raw OLens sampling "
                "and is directly comparable to lm_jlens; reported beside the raw value, never "
                "silently in place of it.",
    "prompt_only": "a judge sees ONLY the expression and guesses the intermediate. This is the "
                   "guessability floor; anything at or below it is not evidence. (The lm_* arms "
                   "are expression-blind precisely so they cannot collapse into this.)",
}
# Controls, all drawn from the IDENTICAL cell as the signal:
CONTROLS = {
    "noise": "matched-norm Gaussian in place of h — is the readout driven by the activation?",
    "donor": "another item's h from this item's null_set, randomised. Must MISS this item and "
             "RECOVER its own (two-sided), or it is not a control but an easier item.",
    "null": "the same test asked about other items' targets, same cell, same selection steps.",
}

TOL_LADDER: tuple[str, ...] = ("exact", "rel0.5pct", "rel1pct", "rel2pct", "rel5pct")

HOLDOUT_FRAC = 0.20
HOLDOUT_SEED = 20260803
# Seed 0 IS the scored block (generation is deterministic given activation+seed),
# so only independent seeds belong here. (0,1) originally stored a byte-identical
# copy of the scored block as "repeat 0" — found by the qualitative audit.
REPEAT_SEEDS: tuple[int, ...] = (1,)

# Two items are in each other's null only if their targets differ by MORE than this.
#
# It must EXCEED the scoring tolerance (or a correct hit also registers as a cross-hit and the
# null is contaminated) but not by more than necessary (or items are needlessly excluded from the
# null, which understates it and OVERSTATES net). Both errors were made:
#   0.02 flat  -> equal to the 2% tolerance, contaminated the null            (understated net)
#   0.08 flat  -> 8% on an EXACT-tolerance family excluded 30 of maxsel's 39
#                 adjacent pairs from the null for no reason                  (overstated net)
# Under an exact tolerance two distinct targets can never both be matched by one numeral, so any
# strict inequality suffices; a 4x margin over a relative tolerance is comfortable.
def null_separation(tol: str) -> float:
    pct = {"exact": 0.0, "rel0.5pct": 0.005, "rel1pct": 0.01,
           "rel2pct": 0.02, "rel5pct": 0.05}[tol]
    return 4.0 * pct

def null_separated(a: list[float], b: list[float], tol: str) -> bool:
    """May two items sit in each other's null pool? EVERY step pair must clear
    null_separation — one shared definition for the generator and the freeze step (they were
    hand-synced copies until 2026-08-11). score.py's one-sided per-step criterion is a
    DIFFERENT rule by design; do not unify it into this."""
    sep = null_separation(tol)
    return all(abs(x - y) > sep * max(abs(x), abs(y), 1e-9) for x, y in zip(a, b, strict=True))


Step = tuple[str, Callable[[list[float]], float]]

# role: structural = no single-token route, can support a claim against token lenses
#       comparison = target is single-token reachable, so this is lens-vs-lens only
#       control    = expected to be readable by anything; if it scores like the real families,
#                    presence is uninformative and the design is broken
#       advisory   = reported, never headline (see round2's note)
FAMILIES: dict[str, dict[str, Any]] = {
    "dec16": {
        "shape": "(a / 16) + c",
        "steps": [("a/16", lambda o: o[0] / 16.0)],
        "tol": "rel2pct", "cell": {"layer": 60, "pos": -8},
        "reachable": False, "role": "structural",
        "note": "multi-token decimal — no single-token route for a token lens. Position axis "
                "NOT swept (rel-8 inherited from the pilot); layers swept 17-deep, peak L60.",
    },
    "sign": {
        "shape": "(a - b) * c",
        "steps": [("a-b", lambda o: o[0] - o[1])],
        "tol": "exact", "cell": {"layer": 56, "pos": -8},
        "reachable": True, "role": "comparison",
        "note": "2-digit magnitude IS single-token reachable, so comparison only; `signpair` is "
                "the structural replacement. Position axis NOT swept. A 3-digit version failed "
                "gating (15% correct): 566x83 exceeds one-pass arithmetic.",
    },
    "negdec": {
        "shape": "((a - b) / 8) * c",
        "steps": [("a-b", lambda o: o[0] - o[1]),
                  ("(a-b)/8", lambda o: (o[0] - o[1]) / 8.0)],
        "tol": "rel2pct", "cell": {"layer": 56, "pos": -8},
        "reachable": True, "role": "comparison",
        "note": "step 1 = 0.664, step 2 = 0.000 — the cleanest form of the first-intermediate "
                "result. Highest anti-correct rate of any family (0.229).",
    },
    "muladd": {
        "shape": "(a * b) + c",
        "steps": [("a*b", lambda o: o[0] * o[1])],
        "tol": "rel2pct", "cell": {"layer": 60, "pos": -8},
        "reachable": False, "role": "structural",
        "note": "widest tolerance gap (0.12 exact -> 0.75 at 2%), so the ladder matters most here.",
    },
    "sqrt": {
        "shape": "sqrt(N) * c",
        "steps": [("sqrt(N)", lambda o: o[0] ** 0.5)],
        "tol": "exact", "cell": {"layer": 60, "pos": -7},
        "reachable": True, "role": "comparison",
        "note": "cell is rel-7 NOT rel-8: 0.992 vs 0.458 one token apart. 3-digit roots failed "
                "gating (16%).",
    },
    # ---- gated, cells pending Stage 1 sweep. cell=None => the scorer refuses to score. -------
    "signpair": {
        "shape": "(a - b) * c",
        "steps": [("a-b", lambda o: o[0] - o[1])],
        "tol": "rel2pct", "cell": {"layer": 56, "pos": -8},
        "reachable": False, "role": "structural",
        "note": "3-digit |a-b| with a single-digit multiplier — the structural replacement for "
                "`sign`. 22 items share operands with an `absval` twin. Cell from the "
                "2026-08-03 sweep (peak 0.137 at exact). Tolerance is rel2pct, NOT exact: the "
                "rollouts show the lens emitting the value at ~1% precision (-300 for -301, "
                "200 for -202) — exact scores the representation's quantisation, not its "
                "presence. Same treatment as muladd's large integers; exact stays in the ladder.",
    },
    "absval": {
        "shape": "abs(a - b) * c",
        "steps": [("|a-b|", lambda o: abs(o[0] - o[1])),
                  ("a-b pre-abs", lambda o: o[0] - o[1])],
        "tol": "rel2pct", "cell": {"layer": 56, "pos": -7},
        "reachable": False, "role": "structural",
        "note": "22 of 28 paired with a `signpair` twin on IDENTICAL operands: same magnitude, "
                "with and without the sign. The clean test of whether anti means the lens has "
                "the magnitude but not the sign. Cell from the 2026-08-03 sweep (0.273 at "
                "exact, null 0.000). Tolerance rel2pct, matched to signpair so the twin "
                "comparison is like-for-like. The PRE-ABS step (a-b, negative in every item) is "
                "scored second: the audit showed the 0.200 'anti' rate is mostly the lens "
                "reading the upstream signed difference — first-intermediate behaviour, not a "
                "sign error — and scoring the step makes that an output instead of a misread.",
    },
    "floordiv": {
        "shape": "floor(a / b) + c",
        "steps": [("floor(a/b)", lambda o: float(int(o[0] // o[1])))],
        "tol": "exact", "cell": {"layer": 60, "pos": -7},
        "reachable": False, "role": "structural",
        "note": "MUST use exact: the designed distractor (the true quotient) sits 0.08-0.7% from "
                "the floor, so a 2% tolerance would admit it. Cell from the 2026-08-03 sweep "
                "(0.245, margin 5.4x) — and the rollouts show exactly the designed confusion: "
                "'135.483', '135.714...', '135.000' for floor=135, i.e. the lens carries the "
                "unfloored quotient's neighbourhood.",
    },
    "maxsel": {
        "shape": "max(a, b) - c",
        "steps": [("max(a,b)", lambda o: max(o[0], o[1]))],
        "tol": "rel2pct", "cell": {"layer": 60, "pos": -7},
        "reachable": False, "role": "control",
        "note": "CONTROL, not a capability test: the intermediate IS a prompt token. Tolerance "
                "matched to the structural families so the comparison is fair. 2026-08-03 sweep "
                "RESULT: 0.069 vs null 0.075 (margin 0.9x) — the control does NOT read, so "
                "presence in the real families is not operand echo. Its rollouts emit the "
                "ANSWER or garbled numerals, never a clean copy of the prompt operand.",
    },
    # ---- discrete-concept families (2026-08-09) ----------------------------------------------
    # Designed against the iolens chat AO (IOLENS_CHAT_AO above); cells come from ITS sweep.
    # Premise under test: the workspace holds DISCRETE concepts — 13 or 13.5 plausibly has a
    # representation of its own, 6.125 does not. Every integer intermediate stays within 1-1000.
    # Extra keys (absent on the original families; consumers use .get):
    #   band      items carry a "band" field ("lo" = intermediates 1-100, "hi" = 100-1000) so
    #             surfacing-vs-magnitude is an output, not a confound
    #   fraction  operands -> (numerator, denominator) of the fraction step, for the
    #             fraction-form column and the reducible-fraction degeneracy check
    #   pairwise  items carry a `pair` twin and feed the pairwise compositionality judge
    #   anchors   "dual" = the probe stage anchors BOTH sub-expressions, not one
    "mulmul": {
        "shape": "(a * b) + (c * d)",
        "steps": [("a*b", lambda o: o[0] * o[1]),
                  ("c*d", lambda o: o[2] * o[3])],
        "tol": "rel2pct", "cell": {"layer": 56, "pos": -8},
        "reachable": False, "role": "structural",
        "anchors": "dual", "band": True,
        "note": "two INDEPENDENT integer intermediates in one expression — do both surface, in "
                "what order, and where? The generator keeps |a*b - c*d| beyond null_separation "
                "so a readout attributes to one step, never ambiguously to both. band=lo rows "
                "have single-token-reachable products; the band breakdown, not the family "
                "average, is the honest unit.",
    },
    "frac": {
        "shape": "(p / q) * (p*q*k)",
        "steps": [("p/q", lambda o: o[0] / o[1])],
        "tol": "rel2pct", "cell": {"layer": 60, "pos": -8},
        "reachable": False, "role": "structural",
        "fraction": lambda o: (o[0], o[1]), "pairwise": True,
        "note": "direction-binding twins: (7/3)*21=49 vs (3/7)*21=9 — the SAME operand bag "
                "{7,3,21}, so a bag-of-digits reader cannot tell them apart; a workspace read "
                "must bind numerator to denominator. p,q coprime from {3,7,9,11,13}: the decimal "
                "never terminates (no clean-decimal route a la 7/2=3.5), and p/q vs q/p always "
                "differ in integer part. The VALUE target is the decimal expansion — computed, "
                "never a prompt token, so scoring it is leak-safe; the fraction STRING ('7/3') "
                "is operand echo here and lives in its own frac_form column, never in value. "
                "2026-08-09 sweep: the decimal VALUE is ~0.01 at EVERY swept cell — the lens "
                "holds fraction fragments ('7/', frac{7}{, and sometimes the exact pair), never "
                "the expansion. The cell is registered for the pairwise/frac_form/lm reads.",
    },
    "fraccomp": {
        "shape": "((a + b) / c) * (c*e)",
        "steps": [("a+b", lambda o: o[0] + o[1]),
                  ("(a+b)/c", lambda o: (o[0] + o[1]) / o[2])],
        "tol": "rel2pct", "cell": {"layer": 56, "pos": -8},
        "reachable": False, "role": "structural",
        "fraction": lambda o: (o[0] + o[1], o[2]),
        "note": "the COMPUTED-fraction counterpart of `frac`: gcd(a+b, c) = 1 with c from "
                "{3,7,9,11,13}, so the fraction's numerator is itself computed — here a "
                "fraction-string readout ('85/3') is computed content, the mitigation for "
                "frac's direct-copy caveat. d = c*e keeps the final answer an integer, so the "
                "no-CoT gate stays passable.",
    },
    "negdiv8": {
        "shape": "((a - b) / 8) * c,  8 | (a-b)",
        "steps": [("a-b", lambda o: o[0] - o[1]),
                  ("(a-b)/8", lambda o: (o[0] - o[1]) / 8.0)],
        "tol": "rel2pct", "cell": {"layer": 60, "pos": -8},
        "reachable": True, "role": "comparison",
        "pairwise": True,
        "note": "negdec's shape with (a-b) DIVISIBLE by 8: step 2 is a small negative INTEGER "
                "instead of an x.75-style decimal. The direct test of whether negdec's step-2 "
                "0.000 was about representability. c != 8 (step1==answer, negdec's trap) and "
                "c != 1 (step2==answer); |a-b| in 16..72 so step2 is -2..-9. Twin: negdiv8x, "
                "same c, |a-b| off by one — the contrast at the SAME cell is the claim; the "
                "absolute value is comparison-grade (both steps single-token reachable).",
    },
    "negdiv8x": {
        "shape": "((a - b) / 8) * c,  (a-b) = -(8k±1)",
        "steps": [("a-b", lambda o: o[0] - o[1]),
                  ("(a-b)/8", lambda o: (o[0] - o[1]) / 8.0)],
        "tol": "rel2pct", "cell": {"layer": 60, "pos": -8},
        "reachable": True, "role": "comparison",
        "pairwise": True,
        "note": "the matched NON-divisible twin of negdiv8: step 2 = -(k±0.125). If negdiv8's "
                "step 2 reads and this stays at 0.000, discreteness — not depth-in-chain — is "
                "the axis. rel2pct on x.125 excludes the neighbouring integer (2% of 2.125 = "
                "0.043), so the contrast cannot blur at the scoring tolerance.",
    },
    # ---- BODMAS round 2 (2026-08-10): bracketing twins + the summed-numerator fraction -------
    # Same discrete-concept discipline; the twins push the compositionality question harder than
    # frac's direction twins: BOTH members share the exact same operand bag AND operators — only
    # the parentheses move, so the readable intermediate is determined by WHICH computation the
    # model ran. A bag-of-tokens reader is structurally blind to that.
    "fracadd": {
        "shape": "(a / b) + (c / b),  b odd, b | (a+c)",
        "steps": [("a+c", lambda o: o[0] + o[2])],
        "tol": "rel2pct", "cell": {"layer": 56, "pos": -8},
        "reachable": False, "role": "structural",
        "note": "the common-denominator route's intermediate is the summed NUMERATOR a+c — "
                "computed, never a prompt token, and an integer (the discrete zone). Both "
                "addend fractions are non-terminating (gcd(a,b)=gcd(c,b)=1, b in {3,7,9,11,13}) "
                "and b | (a+c) keeps the final answer an integer for the no-CoT gate. Closes "
                "frac's copy loophole from the numerator side: here nothing fraction-shaped in "
                "a readout can be prompt echo of the TARGET.",
    },
    "subsub": {
        "shape": "(a - b) - c",
        "steps": [("a-b", lambda o: o[0] - o[1])],
        "tol": "rel2pct", "cell": {"layer": 56, "pos": -8},
        "reachable": True, "role": "comparison",
        "pairwise": True,
        "note": "bracketing twin of subsubx on the IDENTICAL (a,b,c): '(52 - 19) - 8' vs "
                "'52 - (19 - 8)'. The two prompts differ only in paren placement; the "
                "intermediates (a-b vs b-c) and answers all sit >4x tolerance apart by "
                "construction. If the lens reads the intermediate the model actually computed, "
                "the anonymized twin judge should separate them — the cleanest order-of-"
                "operations binding test in the suite. 2-digit intermediates: comparison role.",
    },
    "subsubx": {
        "shape": "a - (b - c)",
        "steps": [("b-c", lambda o: o[1] - o[2])],
        "tol": "rel2pct", "cell": {"layer": 56, "pos": -8},
        "reachable": True, "role": "comparison",
        "pairwise": True,
        "note": "the a-(b-c) member of the subsub bracketing twins. 2026-08-10 sweep: b-c reads "
                "0.033 (vs subsub's a-b at 0.758) and the a-b+c rewrite's step reads only "
                "0.093 — the readouts babble. Cell twin-matched to subsub for the pairwise "
                "reads; the asymmetry IS the finding.",
    },
    "addmul": {
        "shape": "(a + b) * c",
        "steps": [("a+b", lambda o: o[0] + o[1])],
        "tol": "rel2pct", "cell": {"layer": 60, "pos": -8},
        "reachable": True, "role": "comparison",
        "pairwise": True,
        "note": "bracketing twin of addmulx on the IDENTICAL (a,b,c): '(17 + 24) * 6' vs "
                "'17 + (24 * 6)'. Same bag, same operators; intermediate a+b vs b*c. The "
                "addmulx member is muladd's shape with explicit parens, so its absolute level "
                "has an in-suite anchor.",
    },
    "addmulx": {
        "shape": "a + (b * c)",
        "steps": [("b*c", lambda o: o[1] * o[2])],
        "tol": "rel2pct", "cell": {"layer": 60, "pos": -8},
        "reachable": False, "role": "structural",
        "pairwise": True,
        "note": "the a+(b*c) member of the addmul bracketing twins; b*c reaches 3 digits. "
                "2026-08-10 sweep: b*c reads 0.007 — the readouts say the LEADING OPERAND with "
                "its pending operation ('24 +', '24:'), supporting the leading-element reading "
                "of the boundary workspace. Cell twin-matched to addmul.",
    },
    "fracint": {
        "shape": "(a / b) + (c / b),  b | a, b | c",
        "steps": [("a/b", lambda o: o[0] / o[1]),
                  ("c/b", lambda o: o[2] / o[1])],
        "tol": "rel2pct", "cell": {"layer": 56, "pos": -8},
        "reachable": True, "role": "comparison",
        "note": "fracadd's shape with EXACT divisions: each fraction resolves to a 2-digit "
                "integer, so the LEADING element (a/b) is a computed integer in the readable "
                "zone. The leading-element rule predicts a/b reads and c/b does not — the "
                "direct test of whether fracadd's 0.227 was about element position rather "
                "than the summed-numerator target.",
    },
    "fracsmall": {
        "shape": "(a / b) + (c / b),  a,c single-digit, b | (a+c)",
        "steps": [("a+c", lambda o: o[0] + o[2])],
        "tol": "rel2pct", "cell": {"layer": 56, "pos": -8},
        "reachable": True, "role": "comparison",
        "note": "the 4/7 + 3/7 variant: single-digit numerators, tiny integer answer (1 or 2). "
                "Same shape as fracadd but every quantity is small and the leading fraction is "
                "a canonical small object. Tests whether fracadd's failure was magnitude.",
    },
    "halflead": {
        "shape": "(a / 2) + (c / 2),  a even, c odd",
        "steps": [("a/2", lambda o: o[0] / o[1]),
                  ("c/2", lambda o: o[2] / o[1])],
        "tol": "rel2pct", "cell": {"layer": 60, "pos": -8},
        "reachable": True, "role": "comparison",
        "pairwise": True,
        "note": "position-vs-resolvability twins, lead-resolvable member: the LEADING fraction "
                "resolves to an integer, the trailing one is n.5. PRE-REGISTERED (2026-08-11): "
                "both the leading-element rule and the resolve-what-resolves rule predict a/2 "
                "reads here. The twin (halftrail) is where they part.",
    },
    "halftrail": {
        "shape": "(a / 2) + (c / 2),  a odd, c even",
        "steps": [("c/2", lambda o: o[2] / o[1]),
                  ("a/2", lambda o: o[0] / o[1])],
        "tol": "rel2pct", "cell": {"layer": 60, "pos": -8},
        "reachable": True, "role": "comparison",
        "pairwise": True,
        "note": "the trail-resolvable member: only the SECOND fraction resolves to an integer. "
                "PRE-REGISTERED: if c/2 reads here, resolvability beats position (the workspace "
                "holds whatever resolved); if nothing reads (subsubx-style silence), position "
                "wins. Headline step = c/2 (the resolvable one) so the two rules give opposite "
                "predictions on ONE number. VERDICT (sweep 2026-08-11): POSITION wins — the "
                "trailing resolved integer reads 0.010 while the LEADING unresolvable n.5 "
                "reads 0.420 @2% (0.307 exact): the slot is positional, resolvability only "
                "sets its fidelity. Cell twin-matched to halflead.",
    },
    "mulmid": {
        "shape": "(a * b) + (c * d),  products 60-100",
        "steps": [("a*b", lambda o: o[0] * o[1]),
                  ("c*d", lambda o: o[2] * o[3])],
        "tol": "rel2pct", "cell": {"layer": 56, "pos": -8},
        "reachable": True, "role": "comparison",
        "note": "the missing middle band between mulmul's lo (12-60, reads 0.96) and hi "
                "(100-250, reads 0.34): both products in 60-100. Fills in the "
                "magnitude-vs-readability curve.",
    },
    "halves": {
        "shape": "(a / 2) + c,  a odd",
        "steps": [("a/2", lambda o: o[0] / 2.0)],
        "tol": "rel2pct", "cell": {"layer": 60, "pos": -8},
        "reachable": False, "role": "structural",
        "note": "n.5 intermediates (13.5-style) — the middle rung of the representability "
                "ladder: integer (negdiv8/mulmul) > half (this) > eighth (negdiv8x/negdec) > "
                "sixteenth (dec16, k2.g8-era numbers only). Same shape as dec16 with divisor 2, "
                "so the ladder holds the expression form fixed and varies only the "
                "denominator's representational grain. rel2pct on n.5 excludes both "
                "neighbouring integers. 2026-08-09 sweep: peak 0.267 at margin 2.0x (below the "
                "3x bar) — reads are real but partial ('6.5' verbatim x3 for target 6.5; "
                "'44.'/'.5' fragments elsewhere); the k=10 full-bank read quantifies it.",
    },
}

EXCLUDED: dict[str, str] = {
    "fracmix": "(a/b) + (c/d) with DISTINCT denominators. 1/40 correct at numerators 3-19 and "
               "0.05 mean-correct even at (9/2)+(11/5) — no-CoT mixed-denominator addition is "
               "beyond the model, so no activation encodes a completed computation to read. "
               "(Also structural: lowest-terms fractions with distinct denominators can never "
               "sum to an integer, forcing decimal answers.) The shared-denominator schema "
               "finding (fracadd/fracsmall) therefore has no mixed-denominator comparison arm.",
    "round2": "round(a/b,2) + c. 0.000 at ALL 70 swept cells (5 layers x 14 positions, "
              "2026-08-03) — the lens never emits the rounded decimal exactly, and the family "
              "already could not test rounding (the model's own precision, median error 0.03, "
              "is coarser than the rounded-vs-unrounded gap of 0.001-0.005). n=14 was also "
              "below the useful floor.",
    "power": "k^n - c. Correct ONLY under the bare render (100%), where it states the "
             "intermediate 100% of the time; 0% under chat. Correctness and non-leakage are "
             "mutually exclusive — no configuration exists. Unevaluable.",
    "mod": "(a % b) + c. 0/80 correct under chat: a >=3-digit remainder forces real long "
           "division. Same structural bind as `power`.",
    "threeop": "(a*b)/d + c. 14% under chat vs 100% bare — needs to show work to be right. "
               "Retry only with easier numbers.",
    "sign3": "3-digit |a-b| with a 2-digit multiplier. 15% correct. Fixed in `signpair` by using "
             "a single-digit multiplier.",
    "sqrt3": "3-digit root. 16% correct — Qwen cannot take 3-digit square roots in one pass.",
    "irrat": "sqrt(k)*c with an irrational intermediate. DISQUALIFIED, not failed: the lens "
             "recites square-root TABLES, so a hit on 1.41421 comes from an enumeration of "
             "sqrt2/3/5/7 rather than the activation. Distinctiveness and low prior are opposite "
             "properties; famous constants are the worst possible target.",
}

RENDER = "Compute {expr}. Reply with only the final number, nothing else."
STOP_MARKERS = ("</output>", "\n", "```")

SCOREABLE = tuple(f for f, s in FAMILIES.items() if s["cell"] is not None)
PENDING = tuple(f for f, s in FAMILIES.items() if s["cell"] is None)


def tolerance_ok(got: float, target: float, tol: str) -> bool:
    """The single definition of a numeric match. Sign-aware at every tolerance."""
    if tol == "exact":
        return got == target
    pct = {"rel0.5pct": 0.005, "rel1pct": 0.01, "rel2pct": 0.02, "rel5pct": 0.05}[tol]
    return abs(got - target) <= pct * abs(target)


def single_token_reachable(value: float) -> bool:
    """Expressible as ONE vocabulary token? True for 1- and 2-digit integers, because
    number-word tokens (十八, XVIII, eighteen) bypass digit-by-digit tokenisation."""
    return float(value).is_integer() and abs(value) < 100


def step_targets(family: str, item: dict[str, Any]) -> list[float]:
    """Targets for every step, cross-checked against the bank's own `intermediates`.

    The invariant is CONTAINMENT, not position: every value the bank names as an intermediate
    must appear somewhere in the spec's computed steps. A bank may name fewer steps than the
    spec computes (negdec's bank names only `(a-b)/8`, while the spec also scores `a-b`), but it
    must never name a value the spec does not produce — that means spec and data disagree about
    what this family computes, and it RAISES rather than scoring a plausible wrong number.

    A positional check was tried first and was wrong: it compared negdec's `intermediates[0]`
    (the division result) against spec step 1 (the subtraction) and rejected a correct spec.
    """
    ops = [float(x) for x in item["operands"]]
    out = [fn(ops) for _, fn in FAMILIES[family]["steps"]]
    for stated in (float(x) for x in item.get("intermediates", [])):
        if not any(abs(v - stated) <= 1e-6 * max(abs(stated), 1.0) for v in out):
            raise ValueError(
                f"{family}/{item.get('name')}: the bank names intermediate {stated!r}, which no "
                f"spec step produces (steps give {out!r}). Fix one before scoring."
            )
    return out


# High-prior constants the lens is known to RECITE (the irrat exclusion showed hits on famous
# values come from enumeration, not the activation). A target within the family tolerance of one
# of these is credited for reciting, so the item cannot distinguish reading from reciting.
# Found live: dec16-004 (target 3.125) was credited on "3.14159..." recitations at 2% tolerance.
FAMOUS = {"sqrt2": 2 ** 0.5, "sqrt3": 3 ** 0.5, "sqrt5": 5 ** 0.5, "phi": (1 + 5 ** 0.5) / 2,
          "e": 2.718281828, "pi": 3.141592653, "2pi": 6.283185307, "ln2": 0.693147181}


def is_degenerate(family: str, item: dict[str, Any]) -> str | None:
    """Is this item unevaluable because a scored step coincides with something the model must
    state anyway? Returns the reason, or None.

    Found in negdec: 8 of 35 items had multiplier c=8, so ((a-b)/8)*8 == a-b and step 1 IS the
    answer. A correct answer therefore states the target, which is the one thing gating exists to
    forbid. Excluding them moved negdec 0.664 -> 0.675, i.e. they were not inflating the score —
    but they cannot support the claim, so the scorer drops them rather than trusting a human to
    re-notice.
    """
    ans = float(item["answer"])
    ops = [float(x) for x in item["operands"]]
    is_control = FAMILIES[family]["role"] == "control"
    for (label, _fn), v in zip(FAMILIES[family]["steps"], step_targets(family, item),
                           strict=True):
        if abs(v - ans) <= 1e-9:
            return f"step {label} == answer ({v:g})"
        # A step that IS an operand is normally unevaluable — but it is exactly what a `control`
        # family is built from (maxsel's max(a,b) is always one of its operands). Purging it
        # would delete the control and with it the check on whether presence means anything.
        if not is_control and any(abs(v - o) <= 1e-9 for o in ops):
            return f"step {label} == an operand ({v:g})"
        # famous-constant confusability, per SCORED step at the family tolerance (originally
        # headline-only; the fraction families put the confusable value in a later step, where
        # a step-0-only check would let 2.71875-vs-e slip through). Parity checked 2026-08-09:
        # zero change on all nine frozen banks.
        for cname, cv in FAMOUS.items():
            if tolerance_ok(cv, v, FAMILIES[family]["tol"]):
                return f"step {label} ({v:g}) within tolerance of {cname} ({cv:.5f}) — " \
                       f"reciting scores"
    # a reducible fraction has a second, smaller surface form (14/6 == 7/3): the fraction-form
    # column could not tell which one the lens read, so the item is unevaluable on that column
    frac_of = FAMILIES[family].get("fraction")
    if frac_of is not None:
        num, den = frac_of(ops)
        if not (float(num).is_integer() and float(den).is_integer()):
            return f"fraction {num:g}/{den:g} has non-integer parts"
        if math.gcd(int(num), int(den)) != 1:
            return f"reducible fraction {num:g}/{den:g}"
    return None


def holdout_names(names: list[str]) -> set[str]:
    """Deterministic holdout: same names -> same split, independent of ordering, so it survives
    a re-run. Withholding a slice is the only defence against fitting the metric to the data."""
    import hashlib

    def key(n: str) -> str:
        return hashlib.sha256(f"{HOLDOUT_SEED}\x00{n}".encode()).hexdigest()

    ordered = sorted(names, key=key)
    return set(ordered[: max(1, round(len(ordered) * HOLDOUT_FRAC))])
