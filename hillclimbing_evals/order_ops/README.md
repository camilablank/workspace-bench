# order_ops — deterministic numeric-matcher family

Does the lens surface a **never-emitted arithmetic intermediate** while the model silently
computes a multi-step expression? Each item is one expression (`(7 * 7) + (5 * 3)`) with a scored
step target (`49`) that the model must *not* write in its answer. **No LLM judge** — scoring is a
deterministic per-cell numeric matcher against a tolerance-matched cross-item null. Read on
**Qwen3.6-27B** with the chat template (`enable_thinking=False`) — order_ops is the one family
that uses the chat render for the eval, because a bare `Compute (7*3)+8 =` makes the model *write*
`21`, which would prove nothing.

## Banks

**25 committed per-family banks, 1102 items total** (one JSON list per family; `variant` is the
grouping key and the loader asserts `variant == family`). Item counts and the frozen read cell +
tolerance + role per family:

| family | n | cell | tol | role | reachable |
|---|---|---|---|---|---|
| dec16 | 50 | L60 p−8 | rel2pct | structural* | no |
| sign | 25 | L56 p−8 | exact | comparison | yes |
| negdec | 27 | L56 p−8 | rel2pct | comparison | yes |
| muladd | 35 | L60 p−8 | rel2pct | structural | no |
| sqrt | 26 | L60 p−7 | exact | comparison | yes |
| signpair | 23 | L56 p−8 | rel2pct | structural | no |
| absval | 25 | L56 p−7 | rel2pct | structural | no |
| floordiv | 21 | L60 p−7 | exact | structural* | no |
| maxsel | 39 | L60 p−7 | rel2pct | **control** | no |
| mulmul | 103 | L56 p−8 | rel2pct | structural | no (banded lo/hi) |
| frac | 116 | L60 p−8 | rel2pct | structural | no (twin) |
| fraccomp | 44 | L56 p−8 | rel2pct | structural | no |
| negdiv8 / negdiv8x | 45 / 45 | L60 p−8 | rel2pct | comparison / comparison | yes |
| fracadd | 63 | L56 p−8 | rel2pct | structural | no |
| subsub / subsubx | 60 / 60 | L56 p−8 | rel2pct | comparison | yes (twin) |
| addmul / addmulx | 43 / 43 | L60 p−8 | rel2pct | comparison / structural | yes/– (twin) |
| fracint | 32 | L56 p−8 | rel2pct | comparison | yes |
| fracsmall | 16 | L56 p−8 | rel2pct | comparison | yes |
| halflead / halftrail | 45 / 45 | L60 p−8 | rel2pct | comparison | yes (twin) |
| mulmid | 33 | L56 p−8 | rel2pct | comparison | yes (mid band) |
| halves | 38 | L60 p−8 | rel2pct | structural | no |

`role` bounds what a number may be quoted for: **structural** = target has no single-token
route, can support a claim about *kind* of representation; **comparison** = target is a 1–2-digit
integer reachable by one vocab token (`十八`/`eighteen`), so lens-vs-lens comparisons only;
**control** (`maxsel`) = should be readable by anything, invalidates the design if it scores like
the real families; **advisory** = reported, never headline. (*`dec16` and `floordiv` are
deprecated from headline reporting — their tables are historical; see caveats.)

Item schema (fields): `variant, expr, prompt, operands` (strings), `answer`, `first_op`,
`first_value`, `intermediates` (cross-checked against `spec.step_targets`), `single_token_reachable`,
`tolerance`, `name`, `pair` (twin), `null_set` (donor/cross-target names), `band` (mulmul/mulmid),
plus gate fields: original 9 families carry `committed/raw/ok/ok_strict/leak/leak_numeric/
leak_substring/bare/bare_ok/bare_leak/pair_survived`; the 16 discrete-concept families carry
`gate_correct` (0–10) and `gate_greedy_ok`.

`excluded/round2.json` (14 items) and `pending/` hold pre-purge staging copies; `REMOVED.json`
records per-item drop reasons — **but only for the 9 original families** (see caveats).

## Admissibility gate (before freeze)

`spec.GATE`: 10 sampled rollouts at the eval temperature (`k=10, max_new=32, temp=0.8, top_p=0.95`),
pass = **≥8/10 assert the answer** (at `rel0.5pct`) **AND 0 rollouts leak any scored step target**
(at the family tolerance). "If the scorer would credit it, the model said it, and reading it back
is next-token prediction." Twins survive jointly (a twin whose partner failed the gate is dropped).
Greedy is recorded, never the gate. Generation-time rejections: exact `eval()` re-verification of
each expression, `spec.is_degenerate` (below), and rejection when any step is within
`null_separation(tol)` of the answer.

## Scoring — the deterministic matcher

**A cell = one activation = (item, layer, position).** Each item is read at ONE pre-registered
`(layer, pos)`; `SAMPLING = {k:10, max_new:40, temp:0.8, top_p:0.95}` — all 10 samples come from
that single vector. The scorer receives raw text only and every metric is a pure-CPU recompute.

**Numeric match** (`tolerance_ok`): `exact` = float equality (no epsilon); otherwise `abs(got −
target) ≤ pct·abs(target)` with pct ∈ {rel0.5pct 0.005, rel1pct 0.01, rel2pct 0.02, rel5pct 0.05}.
**Sign-aware at every tolerance** (−128 never matches 128).

**Quantity extraction** (`score.quantities`): regex `-?\d[\d,]*\.?\d*` after stripping
`<explanation>` scaffolding, list markers (`^N. ` *only* when prose follows on the same line —
a bare `29.` on its own line is kept, this was a real scoring bug), and `step N` labels;
thousands separators removed, dangling `-` dropped. `asserts(text, target, tol)` = any extracted
quantity matches.

**Per-sample metrics** (aggregated sample → cell → item → family, equal weight per item, never
pooled across layers):
- `value` = fraction of samples asserting the signed target;
- `cross` = fraction asserting a **different item's** target (and not its own) — **the null**;
- **`net = value − cross`** — the headline. **`net = 0` is the coin flip; `net < 0` is an
  anti-correlated reader.** The control family `maxsel` reads `net −0.021`, which licenses the
  real families' claims.
- plus `signed` (commitment-penalised: `(value − anti − both)/tot`, where `anti` = confidently
  wrong sign, `both` = hedged both signs), `magnitude`, `commit`, `comp_evidence` (a **proxy** for
  computation-consistency — cannot catch a true-looking equation describing the wrong computation),
  `incidental`, `junk`, and (fraction families) `frac_form`.

**The `net` null, precisely.** `cross` is built in the same loop, at the same cell, with the same
code, asked about other items' targets that clear `null_separation(tol) = 4·pct` (exact 0.0,
rel0.5pct 0.02, rel1pct 0.04, rel2pct 0.08, rel5pct 0.20). The separation must exceed the scoring
tolerance (or a correct hit also registers as a cross-hit and contaminates the null) but no more
than necessary. Note: this is a **one-sided, per-step** rule and is deliberately *different* from
`spec.null_separated` (the every-step rule used for `null_set` membership) — do not unify them.

**Tolerance is a curve, never one number.** `value` + a rung-matched null are recomputed at each
of `("exact","rel0.5pct","rel1pct","rel2pct","rel5pct")`. Reporting one rung is a caveat-level
error (`muladd` 0.109 exact → 0.723 @2%; `maxsel` @1% has null 0.162 *overtaking* value 0.154).

**Degenerate purge** (`spec.is_degenerate`, applied before the holdout split): drops steps equal
to the answer or an operand (except the `control` family), steps within tolerance of a FAMOUS
constant (`sqrt2/sqrt3/sqrt5/phi/e/pi/2pi/ln2` — "reciting scores"), and reducible/non-integer
fractions.

**Aggregation & CIs:** 95% percentile bootstrap over **items** (B=10000, seed 20260803) — not
over samples (the 10 samples share one activation; bootstrapping them would falsely shrink the
interval). Holdout `HOLDOUT_FRAC=0.20`, seed 20260803, sha256-sorted; **already released for the
published runs** (scored-vs-full deltas ≤0.041, inside the CIs), so it cannot be reused.

**Twin pairwise judge** (committed, `pairwise: True` families only — frac, negdiv8/x, subsub/x,
addmul/x, halflead/trail). An LLM sees two prompts' readout sets and must pick which prompt SET A
belongs to; **chance 0.5**. `RUBRIC_VERSION="orderops-pairwise-1"`, judge model
`llm_client.DEFAULT_MODEL` (`gpt-5.5` unless overridden; `claude-*` routes to Anthropic,
`CLAUDE_JUDGE=claude-opus-5`), fail-open (API error → ungraded, never wrong). Payload hygiene:
`_assert_clean` raises if any bank name or forbidden token leaks; sample counts equalized;
expression/set order seeded by sha256. **FOIL pairs** (both sets from the same item, scored vs
repeat seed) must sit at ~0.5 — an arm beating chance on its foils has a leaking payload and its
real rate is void. `answer_driven` is measured mechanically and reported beside the pick rate,
never subtracted silently.

## Random baseline / null (multiple, layered)

1. `net = value − cross` (primary); `net=0` coin flip, `net<0` anti-correlated.
2. Matched-norm Gaussian noise at the same cell (seed 101) — reads ~0.000.
3. Two-sided donor (another item's activation, seed 102) — must miss this item **and** recover its
   own.
4. `continue` (base model, adapter off, seed 103) — 0.000 everywhere (if it recovers the target,
   the target was next-token predictable).
5. `prompt_only` judge (sees only the expression) — the guessability floor.
6. Grid permutation null + `litter` row (any numeric token at all — the honesty row).
7. Pairwise: chance 0.5, foil arm as the empirical check.

## Caveats (repeat wherever numbers are published)

- **Two different lenses / scales.** The 9 original families' published numbers are on
  `ao.asst.alldata-v2cont.k2.g8` (scale 33.152, prompt_kind `explain`). The 16 discrete-concept
  families are on `IOLENS_CHAT_AO` (`ao.iolens.chat.k4.L20plus.s2/step3002`, scale 64.559) at
  cells registered from *its* sweep — **NOT comparable** to the k2.g8 tables. `dec16`/`floordiv`
  are deprecated from headline reporting.
- **`maxsel` must not ship as a row** — it's the control; its `value 0.226` is a 2%-tolerance
  blend artifact and at 1% its null overtakes value. Its verdict (`net −0.021`) belongs in prose.
- **`role` gates the claim** (comparison families are single-token-reachable → lens-vs-lens only).
- **Report the tolerance curve, not one rung.** **`comp_evidence` is a proxy, not verification.**
- **`anti` here ≠ buggy_code's `anti`** (same word, unrelated definitions).
- **Withdrawn claim:** the "AO writes operations as symbols" (`op_only`) result was a punctuation
  artifact (bare `/`/`-` matched any slash/hyphen) — metric deleted, claim withdrawn.
- **`spec.py` is authoritative over the stale `scripts/olens_suite/README.md`** (which lists
  pre-purge n and wrong tolerances for signpair/absval).
- **Discrete-family drop provenance is missing from `REMOVED.json`** — `finalize_order_ops.py`
  wrote it to a now-nonexistent path (`evals/olens_suite/…`); only per-item `gate_correct`/
  `gate_greedy_ok` survives for those 16 families.
- k=10/cell ⇒ ~±15% per-item error, 95% CIs ~±0.12; a 0.66-vs-0.68 gap may be noise. ~3.9% of
  samples are empty (folding empties into misses *understates* the lens). The read cell is a
  template token — this measures what's held at the moment of answering, not mid-computation.

## Verbatim judge prompt — the twin pairwise judge

order_ops is otherwise deterministic; the ONE LLM judge is the twin pairwise judge for the
`pairwise: True` families (`src/global_workspace/olens_suite/order_ops/pairwise.py`, clean at git
HEAD). Chance = 0.5. `RUBRIC_VERSION = "orderops-pairwise-1"`.

**System** (`SYSTEM`):

```text
You see two arithmetic prompts and two sets of text read out from a language model's hidden activations, taken at identical positions while it silently computed each prompt. Exactly one set belongs to each prompt. Decide which prompt SET A belongs to. Reply in the given JSON schema; set_A_prompt is 1 or 2.
```

**User** (the payload, built by `build_payload` — cell labels stripped to positional lines, sample counts equalized, `_assert_clean` forbids any bank name/scaffolding leaking in):

```text
PROMPT 1: {expr}
PROMPT 2: {expr}

SET A
  {readout sample}
  {readout sample}
  ...

SET B
  {readout sample}
  {readout sample}
  ...
```

Schema: `{set_A_prompt: 1|2, rationale: str}`. Expression order = sha256 of the expression text; set order = sha256 of `(pair_key, rubric_version)`. FOIL pairs (both sets from the same item, scored vs repeat seed) must sit at ~0.5 — an arm beating chance on its foils has a leaking payload and its real rate is void.
