# typo_hard

A sentence ending in a misspelled multi-token word or phrase; the lens must name the corrected form.

100 items (source bank `lens-eval-typo-hard.json`, family key `typo-hard`, parent
`typo-mt`, frozen 2026-09-10). Plain render, one read position: the
final prompt token, immediately before the answer word. Every item and every bridge was gated on
Qwen3.6-27B (greedy-verified, then >= 8/10 at temperature 0.7).

Example: a sentence ending in a misspelling → required unit `correction` (the corrected multi-token form).

Units: `correction` (required).

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
- Reference (source repo, 2026-09-10, same instrument): s3d RL600 0.83, NLA-RL iter400 0.88
  (NLA ungated by any precision condition).

## Readouts

`{"id": "<item name>", "layer": L, "pos": P, "samples": [...]}` rows; `tokens` rows for a token
lens. `examples/readouts/typo_hard.jsonl` is a toy file.

```
wsbench judge family=typo_hard readouts=<arm>.jsonl out=outputs/<arm>/typo_hard
```

## Judge prompts

None. `prompt_version` (`conjunctive-regex-2026-09-16`) names the
scorer. The summarizer prompt used for token readouts is in `docs/summarizer.md`.
