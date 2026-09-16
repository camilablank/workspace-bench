# basic_readout_hard

The model's obvious next concept when that concept is a multi-token phrase (a dynasty, a compound, a named process); on the L2 factual items the language must be read too.

100 items (source bank `lens-eval-basic-readout-hard.json`, family key `basic-readout-hard`, parent
`basic-readout-mt`, frozen 2026-09-10). Plain render, one read position: the
final prompt token, immediately before the answer word. Every item and every bridge was gated on
Qwen3.6-27B (greedy-verified, then >= 8/10 at temperature 0.7).

Example: a factual prompt whose completion is the Aghlabid dynasty → required unit `readout` (forms: Aghlabid dynasty, Aghlabids, Banu al-Aghlab).

Units: `readout` (required) and, on `l2_factual` items, `language` (required).

## Scoring

**No LLM judge.** The headline is a deterministic conjunctive regex (`wsbench.hard`):

- Contract `{"multi_token": true, "include_target": false, "conjunctive_units": true}`: only multi-token forms may award credit
  (filtered with the token counts the bank recorded at freeze time, `probe_token_lens`; the
  `language` units are exempt, `multi_token: false`), and an item passes a layer only when
  EVERY required unit hits somewhere in that layer's readouts (any position, any single
  sample). Item pass = any layer.
- Matcher (`wsbench.matching.unicode_word_matcher`): both sides folded (diacritics, case,
  apostrophes), Unicode word boundaries; CJK and Hangul forms are substring; numeric forms use
  the answer-position rule. A phrase is never assembled across two samples.
- `extras.columns`: `pass`, `any_hit` (any unit hit anywhere, the parent-comparable number) and
  one `<role>_hit` per unit over the items carrying that role. Rows carry `unit_langs` /
  `first_lang`, the language-of-readout split.
- Chance (`chance`): the permutation null — each item's units scored against a donor item's
  readouts, layer by layer, 20 donors per item over the whole bank (`extras.permutation_null`).
  A family below three times its null is not a read.
- Token readouts (a J-lens) are summarized first by the shared summarizer (`docs/summarizer.md`,
  one aux call per cell, cached in `cells.jsonl`), because a top-10 token bag cannot hold a
  multi-token form and scores zero raw. A cell whose summary failed leaves its item undecided.
- Missing (item, layer) cells over the file's layer set are fatal (exit 2) unless
  `allow_missing=True`; an item with a missing layer and no passing layer is undecided and out
  of the denominator. The measured floor is the prompt-only baseline (later PR).
- Reference (source repo, 2026-09-10, same instrument): s3d RL600 0.71, NLA-RL iter400 0.87
  (NLA ungated by any precision condition).

## Readouts

`{"id": "<item name>", "layer": L, "pos": P, "samples": [...]}` rows; `tokens` rows for a token
lens. `examples/readouts/basic_readout_hard.jsonl` is a toy file.

```
wsbench judge family=basic_readout_hard readouts=<arm>.jsonl out=outputs/<arm>/basic_readout_hard
```

## Judge prompts

None. `prompt_version` (`conjunctive-regex-2026-09-16`) names the
scorer. The summarizer prompt used for token readouts is in `docs/summarizer.md`.
