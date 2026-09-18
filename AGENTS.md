# Working in this repo

This repo holds the banks, the judges and the baselines of WorkspaceBench. It does **not** hold
inference code: producing readouts (capturing activations, running a lens) happens elsewhere and
arrives here as a readouts file. `CLAUDE.md` has the code conventions and how to add a family;
this file is about using the benchmark on a model other than the one it was built for, and about
the checks to run before you trust a number.

## What the banks assume

Every bank was written and gated against **Qwen3.6-27B**. A gate means the model actually does
the task: it answers correctly at least 8 times in 10, or copies the carrier sentence without
mentioning the held concept, or names the user attribute when asked outright. If the model under
test fails a bank's gate, its readouts on that bank cannot be interpreted: a lens that says
nothing about an intermediate the model never computed is not wrong.

Run the gate yourself before porting:

```
wsbench capable model=<openrouter-model> families=poetry,moral_rationale draws=10
wsbench capable model=<openrouter-model>                 # all 21 families that have a question
```

It asks each bank's own question, grades the answers with the repo judge, and reports
`gate` (items answered right in at least `threshold` of draws, default 8/10) and `accuracy`
(mean over draws). Results land in `outputs/capable/<model>/<family>/capable.json`, cached, so a
re-run resumes. `dry_run=True` prints the first question and makes no call.

## Which families port, and which need work

| family | its gate | on a new model |
|---|---|---|
| association, basic_readout, multihop, multilingual, typo | the model produces the answer | check with `capable`; usually fine, the tasks are easy |
| basic_readout_mt, multihop_mt, multilingual_mt, multilingual_multihop, multilingual_typo, typo_mt | the model produces the multi-token answer | check with `capable`; the option lists stay valid either way |
| **poetry** | the model commits to a specific rhyme word | **re-gate.** The scored latent is the word THIS model would write. Another model rhymes differently, and every item whose rhyme changes is scoring the wrong target |
| **moral_rationale** | the model commits to one side, and the item's reasons are written for that side | **re-gate.** `capable` reports which side the new model takes; an item answered the other way needs its YES/NO reasons rewritten, not just re-scored |
| chain_intermediates, brew_intermediates, arithmetic_intermediates | the model gets the answer right with no chain of thought | check with `capable`. A model that cannot do it silently has no intermediate to read; a model that writes reasoning breaks the no-CoT premise |
| buggy_code | the model predicts the executed consequence when asked, and never volunteers the bug | check with `capable`; the volunteering half needs a separate look at unprompted rollouts |
| user_modeling | the model names the attribute when asked (at least 8/10) | check with `capable` |
| conjunctive_association, relational_multihop, role_bound_association | the model can state the composed answer | check with `capable` |
| directed_modulation, multi_concept_directed_modulation | compliance: the sentence is copied, the concept never surfaces | no `capable` question. Re-run the compliance screen on the new model before reading anything |
| hallucination | none: the bank IS the model's own responses | **re-generate.** The responses must come from the model under test, or the judge is checking a readout against another model's text |
| jailbreak_recognition | none: verbatim WildChat conversations | portable, but the read sites are token positions — re-capture with the new tokenizer |
| agentic_misalignment | the rollout itself (the model did misbehave) | **re-run the scenarios.** A model that does not take the misaligned action has nothing to read |
| jlens_concept_pr | none: the items are captured activations | re-capture; the J-lens reference must come from the same model |

Two things are model-specific no matter what the gate says. Read positions are token indices, so
any re-capture needs the new model's tokenizer. And the prompt-only and lucky-guessing floors in
`evals/baselines/` were measured with Qwen3.6-27B and Gemini 3.8 Flash; re-measure them if you
change either.

## Sanity checks before you trust a number

1. `wsbench list` — families, item counts, calls per arm.
2. `wsbench judge family=<f> readouts=examples/readouts/<f>.jsonl out=... dry_run=True` — prints
   the first prompt and the grid. Read the prompt before spending.
3. Smoke `limit=3` live. Check the verdicts by eye against the readout text.
4. Missing cells are fatal by design. If a run demands `allow_missing=True`, find out why the
   cells are missing before setting it.
5. Check `n_unjudged_cells` and `n_empty_cells` in `results.json`. An all-zero family is usually
   a broken key or an empty gen dir, not a finding.
6. Read the family's floor beside its rate: `wsbench report dir=<out>` draws the frozen lucky
   guessing and prompt-only columns, and marks the floors that saturate.
7. Prefer the family's own null (permutation, decoy, role-swap, derangement foil) to any
   analytic chance line. Any-of-grid floors saturate and must not be quoted.
8. If the judge changed, the floors stamped with the old `prompt_version` stop being drawn. That
   is the intended behaviour, not a bug: re-measure them.

## Making the evals harder

The banks here are frozen. The generators that built them live in the source repo, so a harder
bank is authored, gated and then frozen into `evals/<family>/items.json`. What "harder" has meant
so far, and what a new item has to satisfy:

- **Multi-token targets.** A top-10 token bag can hold `iron` but not `Simula 67`, so the hard
  tier moved to multi-token bridges and the judge became a forced choice among five confusables.
- **No leak.** The target must not appear in the prompt, in any script or spelling. The
  arithmetic and chain banks check that no intermediate appears as a numeral in the prompt.
- **A real gate.** Greedy-correct, then at least 8 of 10 samples at temperature 0.7.
- **Distractors drawn from the same pool.** Same kind, same specificity, seeded per item so every
  arm and every subset sees the same list. Then measure the lucky-guessing floor: blind guessing
  above chance means the options leak (multihop_mt blind is 0.66 against a 0.15 uniform).
- **A null you can compute.** A permutation over other items' answers, a decoy set matched in
  magnitude, a role swap, a derangement foil. Families without one are the weakest here.
- **Controls in the bank, not just baselines.** Clean twins, don't-think twins, ab/ba direction
  pairs, off-trajectory colours: a control that shares the item's surface but not its answer is
  what separates reading from echoing.

## What this repo will not do for you

There is no capture code, no lens implementation and no model serving here. A readouts file is
the contract: `{"id", "layer", "pos", "samples"}` for a prose lens, `tokens` (+ `scores`) for a
token lens. `wsbench convert-gen-dir` and `wsbench convert-read-json` turn the two common
producer layouts into it.
