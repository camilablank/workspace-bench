# multilingual_mt

A non-English prompt whose answer is a multi-token concept; the judge must pick both the concept's English name and the passage's language, each from five options, in one layer.

100 items (source bank `lens-eval-multilingual-hard.json`, parent `multilingual-mt`,
frozen 2026-09-10). Plain render, one read position: the final prompt token, immediately
before the answer word. Every item and every bridge was gated on Qwen3.6-27B (greedy-verified,
then >= 8/10 at temperature 0.7).

Example: a Polish sentence about a concept → the concept's English name among four same-kind confusables, and Polish among Czech, Slovak, Ukrainian, Croatian, both in one layer.

Judged units: `concept` (frozen `mc` block) and `language` (the fixed confusable set for the prompt's language).

The bank file is a frozen copy of the source repo's hard-tier bank: its `family` header and `contract` block describe the source's regex contract and are not read by this judge.

## Scoring

- **Headline: judged pass rate.** One forced-choice call per (item, layer, judged unit). The
  judge never sees the prompt: it reads the cell's readout (the samples at the read position,
  one per line) and five options, the unit's gold plus four confusables, with the escape
  "cannot tell" last. It must abstain when the readout names none or several of the options, so
  a hedged list of candidates fails. A pick counts only with a verbatim quote from the readout.
- **Conjunctive:** a layer passes an item only when EVERY judged unit is picked correctly at
  that layer; item pass = any layer. `extras.unit_any_layer` gives each unit's any-layer
  accuracy on its own, `extras.abstain_rate` the escape rate, `extras.kinds` the pick counts.
- **Options** are frozen per item (`tests/golden/multilingual_mt_options.json`): the bank's `mc` blocks
  where they exist (concept, bridge and readout units: four same-kind confusables written by
  the source repo's option pass), the source instrument's fixed confusable set per language, and
  for typo corrections four other items' corrections drawn with a seeded rule. Option order is a
  seeded shuffle keyed by item and unit, identical for every arm and subset.
- Token readouts (a J-lens) are summarized first by the shared summarizer (`docs/summarizer.md`),
  because a top-10 token bag cannot hold a multi-token form; a failed summary leaves its item
  undecided.
- An item with no passing layer and an unjudged unit, a failed summary, or a missing (item,
  layer) cell is undecided and left out of the denominator (`extras.n_items_undecided`); a cell
  whose samples are all blank is a negative. Exactly one read position per (item, layer) is
  expected; a file with more is refused. Missing cells are fatal (exit 2) unless
  `allow_missing=True`.
- No analytic floor for an any-layer conjunction (1/6 per unit call). The measured floors are
  the lucky-guessing baseline (an LLM shown only these option lists) and the prompt-only
  baseline (stock Qwen3.6-27B given the prompt text, no activation), both later PRs.

## Readouts

`{"id": "<item name>", "layer": L, "pos": P, "samples": [...]}` rows; `tokens` rows for a token
lens. `examples/readouts/multilingual_mt.jsonl` is a toy file for `dry_run=True`.

```
wsbench judge family=multilingual_mt readouts=examples/readouts/multilingual_mt.jsonl out=outputs/toy/multilingual_mt dry_run=True
wsbench judge family=multilingual_mt readouts=<arm>.jsonl out=outputs/<arm>/multilingual_mt limit=3
```

## Judge prompts

`prompt_version`: `mc-2026-09-16`. Judge: the repo default (`google/gemini-3.8-flash`),
temperature 0. Output schema: `{"choice": "A".."F", "quote": str}`. The user message is the
`USER` template with the readout, one question (`QUESTION_UNIT` with the unit's role, or
`QUESTION_LANGUAGE`) and the lettered options, `CANNOT` last.

`SYSTEM`:

```text
You judge the READOUT of an interpretability lens: text decoded from a language model's internal activations while it read a passage you cannot see. The readout may be noisy, verbose, in any language, and may speculate. You are given ONE multiple-choice question about what the readout NAMES. Pick an option ONLY if the readout clearly names that one option (the same thing, in any language or spelling). If the readout names NONE of the options, or names TWO OR MORE of them (a hedge, a list of candidates, 'X or Y'), choose the "cannot tell" option — a readout must commit to a single answer to be credited. Do not use your own knowledge to fill gaps or to decide which of several named candidates is right. Give a SHORT VERBATIM quote (a few words copied exactly from the readout) that supports your choice; leave it empty when you choose cannot-tell.
```

`USER`:

```text
READOUT:
"""
{readout}
"""

{question}

{options}

Answer with the letter.
```

`QUESTION_UNIT`:

```text
Which ONE of these does the readout name (unit: {role})?
```

`QUESTION_LANGUAGE`:

```text
In which LANGUAGE is the passage the readout describes written? Naming the language counts; so does the readout quoting the passage's words in that language's own script. English and Chinese are the readout's own working languages and are not evidence by themselves.
```

`CANNOT`:

```text
cannot tell: the readout names none of these, or several of them
```
