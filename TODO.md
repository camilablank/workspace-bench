# TODO

Live checklist for the basic-evals and baselines work (Agam). One PR per line; smoke a few
items before any scale run. Status: [ ] open · [~] in PR · [x] merged.

## Foundation
- [~] pydra CLI (`wsbench <command> key=value`), replacing argparse; rebuilt on main after phases 1-7 merged. Branch: agam/pydra-cli.

## Basic evals, single token (desideratum 1) — regex + bank judge; DM has its own judge
- [ ] association
- [~] basic-readout. Branch: agam/eval-basic_readout (stacked).
- [~] multihop. Branch: agam/eval-multihop (stacked).
- [~] multilingual. Branch: agam/eval-multilingual (stacked).
- [~] poetry. Branch: agam/eval-poetry (stacked).
- [~] typo. Branch: agam/eval-typo (stacked).
- [ ] directed-modulation

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
