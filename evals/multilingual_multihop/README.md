# multilingual_multihop

A non-English two-hop prompt; the bridge concept must be read in English or Chinese (the native forms), with the L2 bridge form and the language as optional extra units.

98 items (source bank `lens-eval-multilingual-multihop.json`, family key `multilingual-multihop`, parent
`multihop-mt`, frozen 2026-09-10). Plain render, one read position: the
final prompt token, immediately before the answer word. Every item and every bridge was gated on
Qwen3.6-27B (greedy-verified, then >= 8/10 at temperature 0.7).

Example: a two-hop prompt in another language → required unit `bridge_native` (the bridge in English or Chinese); the L2 bridge and the language are reported, not required.

Units: `bridge_native` (required; en or zh forms), `bridge_l2` and `language` (optional).

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
  readouts, layer by layer, 20 donors per item over every bank item the readouts file carries,
  whatever subset is scored (`extras.permutation_null`).
  A family below three times its null is not a read.
- Token readouts (a J-lens) are summarized first by the shared summarizer (`docs/summarizer.md`,
  one call per cell on the resolved judge model, cached in `cells.jsonl`), because a top-10 token
  bag cannot hold a multi-token form and scores zero raw. A cell whose summary failed leaves its
  item undecided. Prose runs touch no model, so `judge_model=` overrides leave them pinned.
- Exactly one read position per (item, layer) is expected (the final prompt token); a file with
  more is refused. Missing (item, layer) cells over the file's layer set are fatal (exit 2) unless
  `allow_missing=True`; an item with a missing layer and no passing layer is undecided and out
  of the denominator. The measured floor is the prompt-only baseline (later PR).
- Reference (source repo `evals/workspace-bench/hillclimbing_evals/multi_token/HARD.md`, round 1,
  2026-09-10, same instrument): s3d RL600 0.40, NLA-RL iter400 0.64
  (NLA ungated by any precision condition).

## Readouts

`{"id": "<item name>", "layer": L, "pos": P, "samples": [...]}` rows; `tokens` rows for a token
lens. `examples/readouts/multilingual_multihop.jsonl` is a toy file.

```
wsbench judge family=multilingual_multihop readouts=<arm>.jsonl out=outputs/<arm>/multilingual_multihop
```

## Judge prompts

None. `prompt_version` (`conjunctive-regex-2026-09-16`) names the
scorer. The summarizer prompt used for token readouts is in `docs/summarizer.md`.
