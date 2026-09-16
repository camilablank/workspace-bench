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
- [x] directed_modulation (#17)

## Basic evals, multi-token (desideratum 2) — forced-choice Gemini judge per unit, conjunctive
- [x] multihop_mt (#20)
- [x] multilingual_mt (#20)
- [x] typo_mt (#20)
- [x] basic_readout_mt (#20)
- [x] multilingual_typo (#20)
- [x] multilingual_multihop (#20)
- [x] multi_concept_directed_modulation (#21)

## Baselines
- [~] lucky guessing (judge shown only the option lists; blind / described / uniform), Gemini. Branch: agam/lucky-guessing.
- [ ] prompt-only (stock Qwen summary of what the model is thinking, scored by each family's
      own instrument); readouts generated in global-workspace, scored and frozen here
- [ ] per-family empirical nulls (later)

## Methods (readout files, judged here; generation stays in global-workspace)
Logit lens · R-Lens · NLA · NLA SFT · J-Lens · OLens · OLens SFT · Template lens.
