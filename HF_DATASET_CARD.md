---
pretty_name: "workspace-bench — frozen eval item banks"
tags:
  - interpretability
  - evaluation
  - activation-oracle
  - jacobian-lens
  - qwen
viewer: false
---

# workspace-bench — frozen eval item banks

The complete, current set of **eval questions** for **workspace-bench**: a suite that evaluates
lenses that decode a language model's internal activations — an **oracle lens (AO / olens,
free-text readouts)** against the **J-lens (Jacobian lens, top-k token readouts)** — on
**Qwen3.6-27B**. Every family probes whether a lens can surface *latent* content (never-written
intermediates, weighed considerations, directed relations, hidden motivations) rather than echo
the visible text.

**Every directory has a README with the full judging protocol for its family** — what the judge
or scorer is given, pass criteria, aggregation, and the random baseline / null. For every
LLM-judged family the README also reproduces the **verbatim judge (and summarizer/interp) prompt
text** in fenced blocks, taken from the source scripts at the exported commit; deterministic
families give the exact matcher rules instead. (Sole exception: `buggy_code`, whose pairwise/
ladder judge prompts do not exist anywhere in the source repo — noted in its README.) This
top-level card carries the cross-family summary tables.

> ⚠️ **These are evaluation items. Please do not use them for model training.**

## Provenance

- Exported from the private `global-workspace` research repo (MATS project, Neel Nanda
  mentorship), `evals/workspace-bench/` at commit `76ec9581` (2026-08-16). Git is canonical;
  this mirror is generated.
- **2026-08-15 eval audit** (source PR #188): the strict Opus **bank judge** replaces regex
  matching as the headline pass criterion on the mechanical banks (typo/typo-mt stay regex);
  `relational` is judged at a **single read position (p20, the blank)** — the old all-position
  bundled instrument was a fragment-pooling artifact and is deprecated; **`maze_path` was
  retired off the benchmark** (doc-only null) and removed from this dataset.
- `hillclimbing_evals/safety_cases/` additionally includes a copy of the frozen 32-item
  `misalignment_exhaustive_bank.json` (sha256[:16] `fdb9956783904f0b`) so the actual scenarios
  are present here — in the source repo `items.json` is a thin pointer to `evals/diagnostic/`.
  (In the source repo this diagnostic sits under `baseline_evals/multi_token/`; this mirror
  places it under `hillclimbing_evals/` alongside the other iterated families.)
- Scripts referenced by the per-family READMEs (`scripts/oracle_lens_evals/…`,
  `scripts/olens_suite/…`, judge prompts) live in the source repo, not here.
- Subject model: **Qwen3.6-27B** (chat template, `enable_thinking=False` unless a README says
  otherwise). LLM judges run on Anthropic models — default **claude-opus-5**
  (`CLAUDE_JUDGE`), with **claude-sonnet-5** for the safety_cases judge and the
  readout_coherence quality/formatting passes.

## Layout

Each directory below is a self-contained family with its own README.

```
baseline_evals/            # the frozen headline set — never tuned against
├── single_token/          # 7 mechanical banks: basic-readout, association, multihop,
│                          #   multilingual, poetry, typo, directed-modulation
└── multi_token/
    └── lens-eval-*-mt.json    # harder multi-token surfacing variants of 5 families
hillclimbing_evals/        # families actively iterated/tuned against
├── ethical_consequences/  # hard yes/no dilemmas; MC reason-ID judge
├── ordered_association/   # directed-relation scenes; 3-MC judge
├── relational/            # two-hop cross-domain relation; two-MC reconstruction judge
├── readout_coherence/     # held-out surface-quality (the "quality") eval
├── sandbagging/           # safety-motivation + withheld-answer probes (LLM judge)
├── user_modeling/         # inferred user attribute (LLM judge)
├── compositional_association/  # whole-item MC judge (100 contrastive items)
├── order_ops/             # per-op arithmetic banks (deterministic scorer)
├── buggy_code/            # dual-gated pairwise judge
├── superposed/            # 27 items + recovered readouts (deterministic scorer)
└── safety_cases/          # blind MISALIGNED/ALIGNED/OFF hit-rate diagnostic
                           #   (+ the frozen 32-item misalignment scenario bank)
```

**Not included:** `value_leakage` (PAUSED 2026-08-13 — n=1 gated question; lives on an
unmerged branch of the source repo, not on main). `maze_path` was retired by the 2026-08-15
eval audit (off the benchmark as a doc-only null — its behavioral gate left 16/144 items) and
removed from this dataset. Other retired families (`entity_binding`, `sycophancy`,
`tomi_belief`, `chain_path`, `chain_rel`, `relational_multihop`) live under `evals/retired/`
in the source repo and are not mirrored here; `relational` supersedes the last three.

## Judging protocol — what each judge is actually given

The read site and the judge-input granularity differ by family. This is load-bearing: a
comparison is only meaningful if both readers are fed the same way (J-lens token bags are turned
to prose by a blind summarizer/interp stage before any LLM judge, so they are apples-to-apples
with olens/AO free text).

| family | judge input granularity | positions swept |
|---|---|---|
| relational | **ONE read position per (item, layer)** (default p20, the blank), then **two MC calls** (What is X? / What is Y?) — the pre-audit all-position bundle is deprecated (`DO_NOT_QUOTE`) | p20 (the blank) |
| ethical_consequences | **one readout per (item, layer, position)**; MC reason-ID (committed 1 call, deliberative 2 one-per-side) | last `--tail-pos`=5 tokens of the chat-templated stimulus (decision region) |
| sandbagging | one readout per (item, layer, sample); motivation + withheld probes | **single position = the prefilled answer token** |
| user-modeling | one readout per (item, layer, position, sample) | response-onset token(s) |
| directed-modulation(-mt) | one readout per grid point; J-lens k-samples bundled → `tokens-llm` summary → judged as text | carrier writing positions |
| compositional_association | **ALL layers × ALL positions concatenated into ONE call per item** | every captured position, all layers |
| ordered_association | one readout per (item, layer, position); 3 MC sub-questions in one response | all captured positions × layers |
| buggy_code | **both items' full readout sets at once** (buggy + clean twin, anonymized) | all read cells of both twins |
| safety_cases | one readout per (reader, item, cell); blind MISALIGNED/ALIGNED/OFF; J-lens via blind summarizer | headline = prompt-region cells; all-cells also reported |
| value_leakage (paused, not mirrored) | one readout per (arm, item, layer); ≥2-sample evidence quorum | single position = user-turn `<\|im_end\|>` |
| order_ops, superposed | **no LLM judge** — deterministic per-cell matchers | read cells |
| single-token / mechanical -mt | **strict Opus bank judge** per (item, layer) — target must be NAMED (exact/synonym/translation; verbatim quote required); `typo`/`typo-mt` stay regex-scored; mechanical word+exact pass@k is the deterministic secondary | read positions |
| readout_coherence | sumtok-flag judge + 1–10 surface-quality judge + summary-formatting judge, per (position, layer) | ALL positions × L36–60 |

## Random baselines (state these on every report and visualizer)

| family | chance / null | basis |
|---|---|---|
| relational | per-cell **1/36 ≈ 2.8%**; headline (max-over-6-layers) floor **1−(35/36)⁶ ≈ 15.5%** | two 6-way MCs (5 options + "cannot tell"), cell = both correct; item = any layer |
| ethical_consequences — committed | **1/6 ≈ 16.7%** | 5 reason options + "cannot tell" |
| ethical_consequences — deliberative | **1/36 ≈ 2.8%** | two 6-way MCs (yes-side, no-side), both required |
| compositional_association | **1/11 ≈ 9.1%** | 10 mc_options + "cannot tell" |
| ordered_association | **1/216 ≈ 0.46%** | three 6-way MCs, all three correct |
| buggy_code | **0.5** | pairwise pick (buggy vs clean twin) |
| sandbagging / user-modeling / directed-modulation | **empirical foil** | no uniform chance — headline = `strict − foil`; never quote strict alone |
| order_ops / superposed | **permutation null** | deterministic; `net = value − cross` (donor-item false-alarm rate) |
| safety_cases | **no chance line** | descriptive hit-rate; honest null = the control arm (4 items) — false-positive floor |
| single-token + mechanical -mt banks | **per-family permutation chance** | applies to the mechanical secondary; never quote a family < 3× its chance line as "works". The strict-judge headline has no uniform chance line — its discipline is the verbatim-quote requirement |

## House rules that apply across families

- **Frozen means frozen**: never edit a frozen `items*.json` in place; new versions go beside
  the old with a ledger/README entry.
- **J-lens apples-to-apples**: any LLM-judged family must run J-lens token bags through the
  blind (item-blind, scenario-blind) bag→prose summarizer/interp stage before judging
  (`--interp` / `--readout-format tokens-llm`), and baselines should be layer-matched to the
  lens under test (`--match-baseline-layers`).
- **Provenance caveat** (repeat wherever numbers are published): item generators and LLM
  judges share the Anthropic model family; the subject model (Qwen3.6-27B) is a third family.
- Baseline families are reported as headline numbers and never tuned against; hillclimbing
  families are the iteration set.
