# Ordered-association eval (FROZEN v1, 2026-08-12)

**`items.json` is the frozen benchmark — 100 items, all gates green; do not
edit.** Amendments go through a new version + a ledger entry in
`docs/project/experiments/ola/ordered_association_entity_binding_evals.md`.
50 scenario pairs (25/tier), 39 congruent / 39 incongruent / 22 neutral; 12 items carry
`qa.continuation_sent1 = leak` (mask those positions in sweeps).
(No `items_pilot.json` ships here — removed 2026-08-19; see Pilot / full sizes.)

Does the lens state the DIRECTION of a relation ("the policeman chased the thief"), not just
its concepts? Presence-only judges cannot see this: a readout listing "anger, a mother, a son"
passes with the direction inverted. Every scenario is a **pair** of items differing only in
direction (`pair_id`); the flipped sibling's label is the natural foil, so item PASS = gold
directed composition stated AND flipped direction absent (compositional_association's
flipped-sibling logic, applied to argument order).

## Design

- **Tier A** — role labels blocked in the stimulus (conveyed by description: "badge and
  shoulder radio", never "police officer"); the action verb ALLOWED. Isolates pure
  argument-order binding.
- **Tier B** — role labels AND the action word blocked (design decision: action-blocking is a
  stratum flag, not a hard rule — compare tiers). Full latent composition + direction.
- **Stereotypicality**: where a scenario has a prior direction (police chase thieves), the
  reversed item is `incongruent` — the discriminative half: a lens decoding priors instead of
  context fails exactly there. Report congruent vs incongruent (vs neutral) separately.
- **Generation**: scenario grid (template half) + Claude Opus 5 scene-writing in
  `scripts/oracle_lens_evals/gen_ordered_association.py`; word-boundary blocklist lint with
  regeneration-on-violation (≤3 retries), lint failures land in `pending/`, never the bank.
- **Read position**: `end_of_stimulus`.

## Pilot / full sizes

Pilot: 10 scenarios per tier × 2 tiers × 2 directions = 40 items (20 tier A, 20 tier B). The true
pilot bank exists only in the source repo — the `items_pilot.json` formerly shipped here was a
mislabeled byte-identical copy of the frozen `items.json` and was removed (2026-08-19 audit).
Full bank (frozen as `items.json`): 25 scenarios per tier × 2 tiers × 2 directions = 100 —
extend `GRID`, keep directions paired and the congruent/incongruent split roughly balanced
within each tier.

## Gates before freeze (all green at freeze, 2026-08-12)

- **Opus passage gate** (`opus_passage_gate.py` pattern): blind model must recover the gold
  direction from the stimulus and reject the flip — this is the direction-unambiguity check.
- **Behavioral**: subject model answers `asked_question` matching `gold_label` ≥8/10
  (mc over `mc_options`), so the model demonstrably carries the direction before the lens is
  blamed for missing it.

## Judge protocol — the current instrument (`oa_eb_readout_judge.py`, MC judge)

> This section documents the judge as of 2026-08-13 (it supersedes the pre-freeze plan of a
> quote-verified presence probe in `oracle_lens_judge.py`).

`scripts/oracle_lens_evals/oa_eb_readout_judge.py` (vendored here), judge model
**claude-opus-5** (`--model`, default `CLAUDE_JUDGE`), structured output, `--resume` supported.

- **One judge call per (item, layer, position) readout** — the judge sees ONLY the readout
  text (no stimulus, no gold) — answering **three multiple-choice questions in one structured
  response** about the implied `{A} → {action} → {B}`:
  - **Q1** who is the AGENT (does the action)? — 5 person options + "cannot tell from the readout"
  - **Q2** what is the action/state? — 5 action options (gold + 4 cross-scenario distractors) + "cannot tell"
  - **Q3** who is the PATIENT (receives it)? — same person options + "cannot tell"
- Person options carry name AND role ("Marcus, the police officer"), and BOTH in-item people
  appear in every option list (plus 3 distractor people from other scenarios, drawn seeded,
  excluding the item's own `pair_id`) — so the option wording does not leak direction;
  readouts may use either handle. Options are seeded-shuffled
  (`sha256(f"{DATASET_SEED}:{key}")`, `DATASET_SEED=20260810`), with "cannot tell" always the
  trailing line.
- Judge system prompt: answer based ONLY on what the readout states or clearly implies; no
  world knowledge or plausibility fill-in; garbled readouts OK; **partial name matches
  ("Marc", "Bangk") count** as mentions of the full name.
- **PASS (per readout) = all three choices correct** (design decision, 2026-08-11) — this is what
  makes the metric *directed*: Q1/Q3 swap under a direction flip, so a lens decoding priors
  or an undirected concept bag fails the incongruent half.
- **J-lens**: token-bag readouts go through the blind interpretation stage first
  (`INTERP_SYSTEM`, wording verbatim from `latent_eval/jlens_interpret_score.py`): the
  interpreter sees ONLY the top-10 tokens — no stimulus, no options, no gold — and must commit
  to a specific reading; the MC judge then scores that prose. Co-occurring names+action force
  a ~50%-chance direction commitment which the counterbalanced pairs account for.
- Sweep: `--layers` (default = all layers; e.g. `--layers 20,28,36,44,52,60`), `--pos-stride`
  for position subsampling; item-level aggregation in reports is best-over-grid unless stated
  otherwise. Mask the 12 items with `qa.continuation_sent1 = leak` in position sweeps (see above).
- **Judge output (2026-08-19):** the judge reports `n_api_failed` (failed API calls are never
  persisted; `--resume` retries them) and `chance.any_of_grid_floor` — the mean over items of
  1−(1−(1/6)³)^n_sites, the honest baseline for the ANY-over-read-sites item pass
  (`item_pass_any`); the per-site (1/6)³ understates it. entity_binding is retired from the
  benchmark: when the retired EB bank is absent (as in this repo), the vendored judge runs the
  ordered_association half alone.

### Random baseline

Each MC is 6 lines (5 content + "cannot tell") → per-readout all-three-correct chance =
**(1/6)³ = 1/216 ≈ 0.46%**. Report congruent vs incongruent (vs neutral) separately — the
incongruent half is the discriminative one.

## Caveats

- Generator (Opus 5) and judge share a model family; subject model is a third family (same
  caveat as compositional_association — repeat wherever numbers are published).
- Within a pair the two scenes are independently written (lexical content is matched between
  `gold_label` and `contrast_label` within each item — that's where the discrimination
  happens — not across the two stimuli).

## Verbatim judge prompts

From `scripts/oracle_lens_evals/oa_eb_readout_judge.py` (clean at git HEAD).

**System** (`SYSTEM`):

```text
You judge the READOUT of an interpretability lens — noisy text (or a token list) decoded from a language model's internal activations. Answer the multiple-choice questions based ONLY on what the readout states or clearly implies. Do not use world knowledge or plausibility to fill gaps: if the readout does not support an answer, choose the 'cannot tell' option. Readouts may be garbled; partial name matches ('Marc', 'Bangk') count as mentions of the full name.
```

**User** (three MC sub-questions in one call): `READOUT:\n{readout}\n\n{block}`, where `{block}` is:

```text
The readout may describe two people; the sentence it implies has the form {person A} -> {action} -> {person B}. People may be referred to by name or description.

Q1. Who performs the action (the agent)?
  1. {option}
  ...
  N. cannot tell from the readout

Q2. What is the action or state?
  ...

Q3. Who is the action directed at (the patient)?
  ...
```

Person options carry name AND role and include both in-item people plus 3 cross-scenario distractors; actions = gold + 4 distractors; each MC seeded-shuffled with trailing "cannot tell from the readout". Schema: `{q1_choice: int, q2_choice: int, q3_choice: int, evidence: str}`. Per-readout PASS = all three correct.

**J-lens interpretation stage** (`INTERP_SYSTEM`, fed `TOKEN READOUTS:\n{txt}`):

```text
You are shown the top-10 token readouts from an interpretability lens at one position inside a language model that was reading a passage you cannot see. Tokens may include noise, fragments, other languages (translate them), or byte artifacts. In one or two sentences, state what these outputs are collectively trying to say — the situation or mental content they point to. Commit to the most specific reading the tokens support; do not just say they are noisy.
```
Schema: `{interpretation: str}`.
