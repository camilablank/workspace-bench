# 0011 — jlens_concept_pr: precision against the J-lens top-50

Issue #36. Camila, 2026-09-18: "officially change the precision eval to be top 50".

## Why

A 60-item pilot (Gemini 3.8 Flash Stage P, s3d-RL600 at L44, NLA-RL at L42; source repo
`wt-jlens-topk`, 2026-09-18) re-judged Stage P against the J-lens top-10 / 20 / 50 on the
same concepts:

| arm | k=10 | k=20 | k=50 | paired Δ k50−k10 (95% CI) | foil P at k=50 |
|---|---|---|---|---|---|
| s3d L44 (60 cells) | 0.333 | 0.408 | 0.482 | +0.149 [+0.111, +0.185] | 0.043 |
| NLA L42 (54 cells) | 0.249 | 0.290 | 0.356 | +0.107 [+0.081, +0.132] | 0.047 |

The shuffled-partner foil stays near zero, and a deterministic word-overlap check (foil ≈0.005
per rank bucket) agrees: J-lens ranks 11–50 carry real support, so a top-10 reference
under-credits every arm. Score decay is smooth (rank 10 ≈ 0.83× rank 1, rank 50 ≈ 0.71×).

## Decisions

1. **Precision (Stage P, pfoil) reads the top-50 content tokens. Recall (Stage B, foil,
   recall@10, raw_recall, punct_frac) stays on the top-10** — the user asked for the precision
   eval only, and recall@m's denominator is "the tokens to cover": widening it to 50 is a
   different metric (and 5x the Stage B calls). `RECALL_K = 10`, `PRECISION_K = 50` are module
   constants in `judge.py`; the scorer's `M = 10` (hypergeometric m) is unchanged.
2. **Reference files carry the top-50 in place.** `evals/jlens_concept_pr/gen-jlens-pr-jlens/
   <label>/L###.jsonl` rows get `samples` / `scores` of length 50 (all 11 layers + `L042`),
   produced by the source repo's `jlens_topk_modal.py` (`jlens_eval.py k=50`) over the SAME
   captured activations (`wsbench-acts:/acts-jlens-pr`, `/acts-jlens-pr-L42`). Verified before
   the copy: on all 3,588 files every non-list field is identical and `samples[:10]` /
   `scores[:10]` equal the old top-10 byte-for-byte — so recall numbers cannot move. A test
   pins "every reference row has exactly 50 samples" and the prefix invariant is recorded in the
   README (the old top-10 is `samples[:10]`).
3. **Instrument version `jlens-pr-v3`** (results `config.prompt_version`, the `JudgeConfig`, the
   README judge table). Prompts are byte-identical; v3 marks the reference change. **Cache
   fingerprints:** Stages A / B / foil keep `prompt_version="jlens-pr-v2"` in `run_calls`
   (`AB_CACHE_VERSION`) because their system + user text is unchanged — re-judging an arm
   whose v2 `cells.jsonl` exists re-runs only Stage P (≈ 1 call per 60 concepts per cell), not
   the ≈ 30k A/B calls. Stage P / pfoil use `jlens-pr-v3` (their user text changed anyway).
   The retained invariant: a cache hit is only ever served for an identical (system, user,
   model, reasoning, temperature) tuple — the version string only narrows, never widens, reuse.
4. **NaN rule per axis.** Today a cell whose top-10 has no content token is `ok` with NaN on
   BOTH axes and issues no Stage P call. New rule: recall is NaN iff the top-10 has no content
   token (unchanged); precision is NaN iff the top-50 has no content token. A cell with no
   content in the top-10 but some in ranks 11–50 gets a Stage P call and a real precision
   (its recall stays NaN, its status `ok`). `support_expected` = the top-50 has content. The
   `missing_p` / `missing_a` / `incomplete_b` semantics are otherwise unchanged (recall is
   still kept for `missing_p`).
5. **Result fields.** `extras.precision_k = 50`, `extras.recall_k = 10`; per-cell rows gain
   `n_precision_content_tokens`. `chance_label` unchanged. Existing field names
   (`recall_at_10`, …) unchanged.
6. **No re-runs in this PR.** Existing results (e.g. `wsbench-runs/out/nla-rl-L42`) are v2 /
   top-10 numbers; the live benchmark keeps running from the untouched main checkout. Re-judging
   the prose arms is a separate, cheap step (Stage P only, thanks to decision 3) — ask Camila.

## Changes

- `src/wsbench/evals/jlens_concept_pr/judge.py`: constants; `reference_tokens` returns the whole
  row (≤ 50); Stage B / foil iterate `ref[:RECALL_K]`; Stage P / pfoil use the content tokens
  of `ref[:PRECISION_K]`; `AB_CACHE_VERSION` for A and B runs; `CellInput` gets
  `precision_tokens` (and `foil_precision_tokens`); `base_config` records `precision_k`,
  `recall_k`. Docstrings: "top-10" -> "top-50 (precision) / top-10 (recall)".
- `.../score.py`: `CellInput.tokens` = the top-10 (recall set, unchanged meaning), new
  `precision_tokens`; `score_one_cell(..., n_precision_content)` implements decision 4;
  extras + rows per decision 5; module docstring.
- `.../prompts.py`: `PROMPT_VERSION = "jlens-pr-v3"`, `AB_CACHE_VERSION = "jlens-pr-v2"`, the
  docstring's version history line.
- `.../__init__.py`: one-line doc.
- `evals/jlens_concept_pr/gen-jlens-pr-jlens/**`: replaced (decision 2).
- `evals/jlens_concept_pr/README.md`: reference = top-50 (prefix invariant, provenance, date);
  Instrument section (stages, NaN rule, v3, cache note, pilot table, Gemini caveat);
  Sources line.
- `README.md` (§J-lens concept precision + judge table + sources), `NOTICE.md` (reference
  tokens top-50), `CLAUDE.md` if it mentions the top-10 reference.
- `tests/test_jlens_concept_pr.py`: Stage B sees ≤ 10 tokens and Stage P ≤ 50; cache reuse
  (a v2 A/B cache entry is a hit, Stage P is a miss); NaN-rule cases (no content in top-10 but
  content in 11–50 -> precision real, recall NaN, status ok; no content in top-50 -> both NaN,
  no Stage P call); every shipped reference row has 50 samples; `extras.precision_k == 50`.
  Pre-existing tests are edited only where they assert the top-10 Stage P token set.

## Verify

`uv run pytest`, `uv run ruff check .`, the repo's CI command set; a `dry_run=True` judge on
`examples/readouts/jlens_concept_pr.jsonl` prints a Stage P prompt with > 10 tokens.

## Critique round 1 — resolutions (2026-09-18)

1. **NaN rule vs `score_item`.** `concept_pr.score_item` stays verbatim (it NaNs precision on an
   empty grid). `score_one_cell` handles the precision axis itself: when the top-10 has no
   content token but the top-50 does, precision = mean(Stage P support) if the cell has
   concepts and support, `0.0` for zero concepts / no text, NaN + `missing_p` if support is
   missing; recall / raw_recall stay NaN. Data: 18 of 3,588 cells have no top-10 content, 16
   of them have content in ranks 11–50. `ItemScore.n_content_tokens` / `n_tokens` stay
   **top-10** counts (so `punct_frac` is unchanged); the top-50 count lives only in the new
   row field `n_precision_content_tokens`. **`incomplete_b` keeps NaN on both axes on purpose**
   (a judge failure is excluded from every mean, as today; it is rare and decoupling it would
   change which cells enter the precision mean).
5. **Pre-existing tests sanctioned to change, exactly:** `test_jlens_concept_pr.py:77`
   `PROMPT_VERSION == "jlens-pr-v2"` → `"jlens-pr-v3"` (and assert `AB_CACHE_VERSION ==
   "jlens-pr-v2"`); `:182` `len(toks) == 10` → `len(toks) == 50` plus the same first-token /
   `日本語` checks; `:614` `[is_content_token(t) for t in toks] == [True] * 10` → the same over
   `toks[:10]`. No other pre-existing assertion may change; any other failure is a bug to fix
   in code.
6. **Verify** uses the fake-LLM test path (Stage P user text carries > 10 tokens, Stage B never
   sees a rank ≥ 10); the dry run still prints Stage A only (`test_dry_run_prints_stage_a_only`
   unchanged).
7. The Gemini over-credit caveat is labelled "measured against the top-10 hand labels; its
   size at top-50 is unmeasured" wherever it is restated.
8. **Mixing guard.** `chance_label` becomes "shuffled-partner foil precision (measured, see
   extras.foil); reference = J-lens top-50" so `wsbench report` rows show the reference;
   `scripts/judge_swap.py::_jlens_parity` refuses (returns `None` with a reason / raises the
   script's existing mismatch path) when the source summary's `precision_k` (default 10) differs
   from `PRECISION_K`. A new test pins the refusal.

**Amendment to decision 2 (runbook invariant "frozen banks are never edited in place"):** the
top-10 reference `gen-jlens-pr-jlens/` stays byte-for-byte as frozen; the top-50 ships as a
NEW sibling `evals/jlens_concept_pr/gen-jlens-pr-jlens-k50/` (same layout, 50 samples per row).
The family reads every reference token from the k50 dir: Stage B / recall use `[:10]`, which a
test pins equal to the frozen top-10 file for every (label, layer). The critique-round test
sanction for `:182` / `:614` applies to `reference_tokens` reading the k50 dir.
