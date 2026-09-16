# TODO

Live checklist for the basic-evals and baselines work (Agam). One PR per line; smoke a few
items before any scale run. Status: [ ] open · [~] in PR · [x] merged.

## Foundation
- [x] pydra CLI (`wsbench <command> key=value`), merged 2026-09-16 (#9).

## Basic evals, single token (desideratum 1) — bank judge; DM has its own judge
- [x] association (merged 2026-09-16, #16)
- [x] basic-readout (merged 2026-09-16, #16)
- [x] multihop (merged 2026-09-16, #16)
- [x] multilingual (merged 2026-09-16, #16)
- [x] poetry (merged 2026-09-16, #16)
- [x] typo (merged 2026-09-16, #16)
- [~] directed-modulation. Branch: agam/eval-directed-modulation.

## Basic evals, multi-token (desideratum 2) — forced-choice Gemini judge per unit, conjunctive
- [~] multihop_mt. Branch: agam/multitoken-mc-judge.
- [~] multilingual_mt. Branch: agam/multitoken-mc-judge.
- [~] typo_mt. Branch: agam/multitoken-mc-judge.
- [~] basic_readout_mt. Branch: agam/multitoken-mc-judge.
- [~] multilingual_typo. Branch: agam/multitoken-mc-judge.
- [~] multilingual_multihop. Branch: agam/multitoken-mc-judge.
- [ ] multi_concept_directed_modulation (Gemini judge; own read regime)

## Baselines
- [ ] lucky guessing (LLM shown only the option lists; blind / described / uniform), Gemini
- [ ] prompt-only (stock Qwen summary of what the model is thinking, scored by each family's
      own instrument); readouts generated in global-workspace, scored and frozen here
- [ ] per-family empirical nulls (later)

## Methods (readout files, judged here; generation stays in global-workspace)
Logit lens · R-Lens · NLA · NLA SFT · J-Lens · OLens · OLens SFT · Template lens.
