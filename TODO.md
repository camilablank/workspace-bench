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

## Basic evals, multi-token hard (desideratum 2) — conjunctive regex only, no judge
- [ ] multihop-hard
- [ ] multilingual-hard
- [ ] typo-hard
- [ ] basic-readout-hard
- [ ] multilingual-typo
- [ ] multilingual-multihop
- [ ] multi_concept_directed_modulation (deterministic scorer, own read regime)

## Baselines
- [ ] lucky guessing (LLM shown only the option lists; blind / described / uniform), Gemini
- [ ] prompt-only (stock Qwen summary of what the model is thinking, scored by each family's
      own instrument); readouts generated in global-workspace, scored and frozen here
- [ ] per-family empirical nulls (later)

## Methods (readout files, judged here; generation stays in global-workspace)
Logit lens · R-Lens · NLA · NLA SFT · J-Lens · OLens · OLens SFT · Template lens.
