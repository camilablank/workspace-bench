# relational — two-hop cross-domain relation (hillclimbing)

Renamed from `crossdom_2hop` and scaled 16 → 100 items on 2026-08-13. Full spec:
`docs/project/experiments/ola/relational_cloze_eval.md`.

## What it tests

Can a lens read an **un-lexicalized composed two-hop relation** from a possessive cloze? Each
item is `"{p1}'s {r1} is {p2}. {p2}'s {r2} is {p3}. In other words, {p3} is {p1}'s"`, read at
the second name + the following `'s`. One hop is a professional relation, one is kinship (never
kinship×kinship, never boss+spouse — those have single-token composites the J-lens could
shortcut). Gold = `{r1}'s {r2}` (= hop1's hop2).

- **100 items = 50 scenes × a permuted `-a`/`-b` pair** (hops swapped; identical token multiset,
  opposite gold). Any presence-only reader is at chance across a pair; headline = pair-consistency.
- Behavioral gate: greedy plain completion must correctly compose the relation; failing items dropped.

## Judge protocol — what the judge is given

`scripts/oracle_lens_evals/judge_relational.py` (Opus, concurrency 256). **Per (item, layer)**:
the judge sees the readout from **ONE read position (default `--pos 20`, the token at the blank —
where the composed relation must be resolved)** and answers **two forced-choice MC questions**
over it — "What is X (outer relation)?" and "What is Y (inner relation)?". Options per question =
gold + the other hop (mandatory, so a direction-flip fails) + 3 same-domain distractors sampled
across the bank + a "cannot tell" escape (6 lines). Item(layer) PASS = X and Y both correct; item
score = pass at any layer.

> ⚠️ **Audit 2026-08 (instrument fix):** the pre-audit instrument concatenated ALL read positions
> into one judge call. Readouts sitting ON the relation words echo them, so the pooled blob always
> contained both answers — 100/100 for every arm including raw J-lens token bags, zero
> discrimination. Position-by-position judging reads ~0 everywhere. `--pos all` remains only to
> reproduce the artifact and tags its output `bundled_DO_NOT_QUOTE` — never quote its numbers.

- **J-lens** gen dirs MUST run with `--interp`: the blind summary-LM stage turns the top-k token
  bag into prose first (item-blind), and the MC judge scores that — apples-to-apples with AO free
  text (which goes to the judge directly). In `--interp` mode, cells whose interpretation call
  failed are **skipped**, never judged as empty text (interpretations are cached; failed calls are
  retried on rerun).

**Judge output (2026-08-19 additions).** The summary payload now carries `n_cells_judged` /
`n_cells_api_failed` / `n_cells_interp_missing`: failed or uninterpreted cells are skipped and
counted rather than scored, so when either count is nonzero the reported item rate is a **lower
bound** — rerun to retry the dropped cells. `--pos all` output additionally embeds
`"bundled_DO_NOT_QUOTE": true` in the summary payload itself, not just the filename, so the
marker survives a copy-paste of the numbers.

## Random baseline

Per MC question: **1/6** (5 content options + "cannot tell"). Per (item, layer) cell,
both-correct (X and Y): **(1/6)² = 1/36 ≈ 2.8%**.

The **headline statistic is max-over-layers** (item passes if ANY of its 6 judged layers passes),
so its chance floor is **1 − (35/36)⁶ ≈ 15.5%**, not 1/36 — a report of the headline MUST show
the 15.5% floor. The assumption-free per-cell rate (vs the 1/36 line) is the secondary column.

## Files

- `cloze_items_final.json` — 100 items (`rel-sNN-{a,b}`), built by `gen_relational.py`
  (deterministic; generator script lives in the source repo, not shipped here).
- Prompt rows: `outputs/oracle_lens_evals/relational_eval/prompts_final.json` (family
  `relational_final`; source-repo output, not shipped).
- Readouts: `oa_eb_modal.py --families relational_final --out_suffix relationalf`
  → `gen-relationalf-{jlens,ao28500}` (source-repo outputs, not shipped). Judge →
  `outputs/oracle_lens_evals/relational_eval/verdicts_p20_*.json`
  (`--pos all` writes `verdicts_bundled_DO_NOT_QUOTE_*.json`).

## Verbatim judge prompts

From `scripts/oracle_lens_evals/judge_relational.py` (clean at git HEAD). Two forced-choice MC
calls per (item, layer) over the single-position (p20) readout; `{...}` are placeholders.

**MC judge system** (`JUDGE_SYSTEM`):

```text
You judge the READOUT of an interpretability lens — noisy text decoded from a language model's internal activations. Answer the multiple-choice question based ONLY on what the readout states or clearly implies (synonyms, gendered forms, or another language's word for a relation count); do not use world knowledge or plausibility to fill gaps. The readout describes a relation of the form "X's Y" (X is the outer/first relation word, Y the inner/second). If the readout does not support an answer, choose 'cannot tell from the readout'. Give a SHORT exact quote (a few words, VERBATIM) from the readout that supports your choice.
```

**MC user body** (built per question; asked twice — once with `{role}` = "X (the OUTER/first relation word)", once with "Y (the INNER/second relation word)"):

```text
READOUT:
---
{readout}
---
The readout describes a relation of the form "X's Y". What is {role}?
  1. {option}
  2. {option}
  ...
  N. cannot tell from the readout
```

Options per question = gold + the other hop (mandatory) + 3 same-domain distractors, seeded-shuffled, then the trailing "cannot tell". Schema: `{choice: int, quote: str}`. Item(layer) PASS = both X and Y correct.

**J-lens interpretation stage** (`INTERP_SYSTEM`, run on the position's top-k tokens before the MC judge):

```text
You are shown the top-10 token readouts from an interpretability lens at one position inside a language model that was reading a passage you cannot see. Tokens may include noise, fragments, other languages (translate them), or byte artifacts. In one or two sentences, state what these outputs are collectively trying to say — the situation or mental content they point to. Commit to the most specific reading the tokens support; do not just say they are noisy.
```

Schema: `{interpretation: str}`.
