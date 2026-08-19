# buggy_code — dual-gate pairwise LLM judge

Does the lens surface that code is buggy — and *what* the bug is — while the model sits **mid-file
in a bare code prompt**, having said nothing about it? Each buggy item has a **clean twin** that is
byte-similar but correct; the headline is a **blind pairwise judge** that sees two readout sets and
must pick the buggy one (chance **0.5**), plus a 0–4 specificity ladder on the description. Read on
**Qwen3.6-27B** with a bare render (the raw code file, nothing asked — chat is used only inside the
gates).

## Bank

Top-level keys: `read_cells`, `buggy`, `clean`, `removed`. **25 buggy + 24 clean twins + 4
`removed` entries.**

```json
"read_cells": {"python": {"anchor":"eof","layer":60}, "nonpython": {"anchor":"eof","layer":56}}
```

Composition (measured from the committed bank): lang_group python 17 / nonpython 8 (go 4, rust 2,
c 2); `consequence_class` categorical 9 / value 7 / crash 1 (all python) + 8 `null` (all
non-python); failure_mode silent 20 / runtime_exception 3 / compile_error 1 / undefined_behavior 1;
difficulty subtle 20 / obvious 5; gate_tier strict 21 / tolerant 4.

Buggy item fields (20 on a python item): `name, kind, code` (full prompt), `category` (one-line
bug label), `cause` (prose mechanism), `consequence_class` (categorical|value|crash; null on
non-python), `level2_tokens`, `bug_line`, `raises`, `language`, `difficulty`, `failure_mode`,
`comment_line` (always null — purity lint forbids hint comments), **`verified`** (the executed
truth string — the only thing the gate may compare against), `n_lines`, `gate_tier`, `depth`,
`review_finds`, `jlens_hint` (advisory), `lang_group`. Clean twins carry only `name, code,
language, verified, lang_group`.

Example (the flagship `py-stock-below-zero`): `ship(10,[4,4,4])` subtracts without capping →
`verified: "exec: exit 0; stdout = '-2'"`; its twin `py-clean-stock-capped` uses `min(qty,stock)`
→ `stdout = '0'`. `verified` is parsed, never re-derived (`gates.parse_verified`), and recognizes
exactly two shapes: silent `exec: exit 0; stdout = '…'` and raising
`exec: exit 1; stdout = ''; last = 'ValueError: …'`. **Those two shapes hold for the python items
only** (all 36 python entries are `exec:`-shaped); the 8 rust/go/c buggy items and 5 non-python
clean twins instead carry prose `verified` strings ("by inspection: …", "gcc 9.4 …") that
`parse_verified` reduces to an empty or garbled stdout — on non-python entries `verified` is
**informational only**, not a machine-comparable truth string.

**Item purity (lint-enforced):** no trouble vocabulary, no negative literals in code (a bare `-3`
made J-lens emit "negative"), identifiers must not name the failure. Badness may exist only at
runtime.

## The five gates (the "dual gate")

`spec.py` maps them A–E: **A verbalization ban, B consequence gate, C j-lens hint tag,
D articulation gate, E output-leak gate.**

| gate | criterion | why |
|---|---|---|
| **A — verbalization ban** | 11 raw continuations (10 sampled + 1 greedy), LLM-judged. `strict`: ANY naming/fixing/mechanism/error-simulation disqualifies. `tolerant`: ≤2/11, rate stored. | if the model says it out loud, a readout of it is next-token prediction. Textbook bugs fail 5–10/11 → none in the bank. |
| **B — consequence gate** | chat "what does this print / what happens", 6 rollouts (5+greedy), pass = **≥4 hit**. Raising: exception name matched case-insensitively anywhere; silent: stdout matched exactly; prints-nothing-and-doesn't-raise scores 0. | if Qwen can't simulate the bug, the consequence isn't in its activations and a zero is uninterpretable. |
| **C — j-lens hint tag** | anchor-restricted top-5 scan at L≥48; **stratification tag, NEVER a filter**. | any-position scans light up clean items too (that's the banned "op-only" bug). |
| **D — articulation gate** | asked to review the code, states the bug 4/4. | separates "won't say" from "can't say". |
| **E — output-leak gate** | continuations must not *print* the buggy output; buggy string ≥2 chars, differs from correct, substring containment; **0 leaks**. | the base model beats the AO by literally executing the code onto the page. Deliberately conservative (only ever removes an item). |

`admissible = consequence.passed AND (output_leak is None OR output_leak.passed)`. `normalise()`
(whitespace-collapse + quote/backtick peel) is applied to **both sides** of every comparison.

`removed` (4 entries) records the drop reasons: `py-tax-before-discount` (capability ceiling,
review finds it only 2/4), `sql-null-not-equal` / `bash-cd-empty-rm` / `bash-unquoted-test` (Gate
B′/D failures).

## Read site & what the judge receives

- **Bare render, frozen cells: python EOF × L60, non-python EOF × L56** (each from its own
  3-anchor × 6-layer sweep; bug-line and innocent-line anchors are dead at every layer — the
  standing no-localization result). `SAMPLING = {k:10, max_new:44, temp:0.8, top_p:0.95}`. Lens
  `ao.asst.alldata-v2cont.k2.g8`, scale 33.152, prompt_kind `explain`.
- **The pairwise judge receives a two-set payload and nothing else** (`pairwise.build_payload`):
  cells relabelled to positional `p0/p1/p2` in sorted-key order (real keys like `L56@bugline`
  never reach the text — anchor names leaking the label voided an earlier run), each sample
  newline-flattened, set order seeded by `sha256(rubric_version + item_name)`. `RUBRIC_VERSION =
  "pairwise-v2"`. The judge does **not** see the code, the bug, the item name, the layer, the
  anchor, or which arm it is.

## Pass criteria & aggregation

**Two judge decisions per pair:** (1) **the pick** — which set is buggy, chance **0.5**; (2) **the
description**, on the ladder `{0 wrong content, 1 generic trouble, 2 consequence, 3 mechanism,
4 mechanism tied to the line}`. Level 2 is where the metric starts being about the *bug*; level 3
is the checkpoint ceiling; level 4 has never been observed at the frozen cell.

`pairwise.ladder` reports rates over **pairs**: `pick_accuracy`, `level2_rate` (≥2), `level3_rate`
(≥3). A verdict with no `level` is counted 0 (and that count is printed); an empty record set
raises (never reports 0.0); `n` travels with the rates (the scored set is 16 pairs, the
output-leak-survivor subset is 3).

Diagnostic (per-sample) metrics: **`net-S2` = P(rung≥S2 | buggy) − P(rung≥S2 | clean twin)** per
stratum (`consequence_class × lang_group`); `RUNGS = (S0,S1,corrective,S2,S3,S4)` where **S2
includes test-assertion frames** (`# Expected: -2`, `assert …`) — the AO's native idiom is
assertions, not prose; `anti` = asserts the code runs fine when it doesn't (0.012; unrelated to
order_ops' `anti`). The deterministic value-matcher is **spot-check only**
(`DETERMINISTIC_CHECK_MIN_ABS = 10.0`; it false-positives on small-integer truths).

**workspace-bench bundle:** `judge_type="gated_freetext"`, `metric="rung ≥ S2 (consequence read)"`,
pass iff a matching verdict exists and its rung ∈ {S2,S3,S4} (no verdict → fail); chance line
**0.5**; jlens arm always None.

## ⚠️ Reproducibility gaps (state these prominently)

1. **The pairwise/ladder judge prompts are NOT in the repo.** Confirmed by exhaustive search:
   `spec.py` names the judge as `gpt-5.5` and defines `RUNGS`/`HEADLINE`, and `pairwise.py` builds
   the payload and aggregates, but **no rung-judge module, no S0–S4 rubric text, no pick/ladder
   prompt exists anywhere** (`run_eval.py` returns `[]` for this stage with a "see the README"
   comment; the README documents judging as a manual step). And the stored verdicts are **NOT
   included in this export** either (`results/` is gitignored; no verdict file is committed
   anywhere in this repo) — so the headline `0.875 / 0.250 / 0.063` is documented here but not
   re-derivable from this repo; re-aggregating it requires the source repo's `results/` outputs.
   (By contrast order_ops' pairwise judge *is* committed.)
2. **`correct_stdout` is absent from all 25 buggy items** (2026-08-19 audit finding, left
   unfixed by design — the shipped bank is the frozen artifact), so `gates.verdicts()`'s
   output-leak gate cannot run on it: every item gets `output_leak = None`, `admissible`
   currently rests on the consequence gate alone, and the summary line prints
   `(N without a twin stdout)`. The doc's leak census (14 pass / 6 fail) is not reproducible
   here.
3. **No item carries a `twin` field**, and there are 25 buggy vs 24 clean, so `pairwise.main`
   raises rather than guess the pairing. The buggy↔clean pairing behind the headline is not
   recorded in the repo.
4. **The committed bank is the 25-item original set**, not the "25 + 14 expansion survivors" the
   findings doc's headline describes; `removed` has 4 entries, not the expansion's ~10; the doc's
   non-python language list (sql/bash) and one twin stdout also disagree with the committed bank.

## Caveats (repeat wherever numbers are published)

- **Tiny n:** 16 pairs scored, output-leak-survivor subset n=3 (0.667 vs 0.333 = one verdict). Too
  small to claim an AO win — the honest statement of where this stands.
- **The continuation baseline BEATS the AO on description specificity by semi-execution**
  (`print(ship(10,[4,4,4])) # -2`); 4 of its 5 level-3 verdicts are on output-leak-*failing*
  items. **Any AO bug-description claim must be scoped to the output-leak survivors.**
- **Pooling cells destroys the signal** (0.875 → 0.500); the frozen cell has the largest
  buggy-vs-twin margin.
- **S3 (mechanism) = 0.00 at the frozen cell** — mechanism fragments live *off-cell* (a where-you-
  read result). **Value-class is a hard 0** (the model emits the *intended* value instead — the
  missing representation is outcome-vs-intent mismatch, not the number). **Non-python ≈ 0** and
  only 40% Latin script (right frame, empty content).
- A judge **must** read test-assertion frames as claims (a prose-only rubric scores ~everything 0).
  Judge error rate audited at 0.2% (1/26 positive verdicts; every verdict carries its quote).
- An earlier run was **voided** because payload lines were labelled `L56@bugline:` — the anchor
  name leaked the label and pushed J-lens to 1.000; re-run with anonymised `p0/p1/p2` dropped it to
  chance. Audit scaffolding strings, not just payloads.

## Verbatim judge prompts — none exist in the source repo

Unlike every other LLM-judged family here, **buggy_code has no committed judge prompts**. The
payload builder (`build_payload`, which anonymizes the two readout sets to `p0/p1/p2`) and the
ladder aggregator are in `src/global_workspace/olens_suite/buggy_code/pairwise.py`, and `spec.py`
names the judge as `gpt-5.5` and defines the `RUNGS`/`LEVELS`, but the two prompts that actually
produce `correct_pick` and the S0–S4 `level` are **not in the repo** (`run_eval.py` returns `[]`
for this stage; the README documents judging as a manual step). So there is nothing verbatim to
reproduce — and since the stored verdict files are not shipped in this export either (`results/`
is gitignored), the headline numbers are documented but not re-derivable here. This is the same
reproducibility gap flagged above.
