# multi_concept_directed_modulation

The model is told to hold one to three unrelated concepts in mind ("Think about the plumber's
blue ladder leaning against the mango tree.") and then to write a fixed dictated sentence; the
lens reads the activations at the positions where it is WRITING that sentence. Are the held
concepts there, how many, and does the binding survive ("Adam being angry at Betty" versus the
reverse)?

27 items (source bank `read_bank.json`, frozen 2026-08-05 in the source repo; items unchanged,
wrapped in a header): 8 dictation items (strata `novel`, `anchor`, `chain`, `pair`), 16 binding
items (`mix`, `same`, in ab/ba pairs) and 2 controls (`d-none`, `b-none`) with nothing dictated.
Bare render, layers 44/52/56/60, read cells = the last 20 tokens of the model's own completion
(offsets -1 .. -20).

## Scoring

- **Headline: pass rate.** An item passes when some in-sentence (cell, layer) readout names at
  least one of its dictated concepts (the source repo's bundle rule, `per_item_union > 0`).
- **One judge call per in-sentence cell.** The judge never sees the prompt: it reads the cell's
  readout and six candidates, and selects every candidate the readout names, each with a
  verbatim quote (a selection without a verbatim quote is dropped). Candidates: the item's
  dictated concepts, its binding partner's concepts (the ab/ba item or the other fox item), and
  seeded draws from the rest of the bank. Order is a seeded shuffle keyed by the item, so every
  arm and every subset sees the same lists (`tests/golden/multi_concept_directed_modulation_options.json`).
- **Regions.** After the dictated sentence the model usually opens a `<think>` block and then
  re-writes the prompt, where the held phrase may be reproduced for output rather than held
  off-site. Every cell is labelled from the window's tokens (in-sentence, in-think, post-think,
  self-added meta) and only in-sentence cells are judged. A window that never wrote the dictated
  sentence (no run of three consecutive target words; `d-petrichor` wrote its own sentence) is
  off-task and the item is excluded, not scored 0.
- **Denominator.** Controls, off-task items, items without readouts and items with no hit whose
  cells are unjudged or missing are undecided and leave the denominator (`rows[].reason`).
  Missing in-sentence cells are fatal (exit 2) unless `allow_missing=True`; a cell whose samples
  are all blank decodes nothing.
- **Extras.** `capacity` (means over decided items of the source scorer's four levels:
  per-activation max, per-cell union over layers, per-item union, per-activation mean),
  `false_pick_rate` (judged cells naming a non-dictated candidate), `control_pick_rate` (the
  controls' cells naming anything: the lexical-loading floor of this sentence),
  `partner_confusion_rate` (items whose readouts named the partner's concept), `per_stratum`.
- **Floors.** No analytic floor for a multi-select. The measured floors are the lucky-guessing
  baseline (an LLM shown only these candidate lists), the prompt-only baseline (stock
  Qwen3.6-27B given the prompt text) and the controls' pick rate, the first two in later PRs.

## Readouts

Rows are write cells: `{"id": "<item name>", "layer": L, "pos": -k, "token": "<token at the
cell>", "samples": [...]}`, `pos` the offset from the last completion token and `token`
required (the region rule reads the window's tokens). Token-lens rows (`tokens`) go through the
shared summarizer first. The source repo's `read.json` converts with:

```
wsbench convert-read-json read=results/<run>/read.json out=readouts/multi_concept_directed_modulation.jsonl
wsbench judge family=multi_concept_directed_modulation readouts=examples/readouts/multi_concept_directed_modulation.jsonl out=outputs/toy/mcdm dry_run=True
wsbench judge family=multi_concept_directed_modulation readouts=<arm>.jsonl out=outputs/<arm>/mcdm items=d-plumber,b-ange-mix-ab,d-fox-a
```

`examples/readouts/multi_concept_directed_modulation.jsonl` is a toy file: three items of the
s3d RL600 read with the samples truncated.

## Judge prompts

`prompt_version`: `mcdm-2026-09-16`. Judge: the repo default (`google/gemini-3.8-flash`),
temperature 0. Output schema: `{"picks": [{"choice": "A".."F", "quote": str}, ...]}`, an
empty list meaning none. The user message is the `USER` template with the readout, `QUESTION`
and the lettered candidates.

`SYSTEM`:

```text
You judge the READOUT of an interpretability lens: text decoded from a language model's internal activations while it was writing a sentence you cannot see. Before writing, the model may have been told to hold one or more unrelated concepts in mind. You are given the readout and a list of candidate concepts. Select EVERY candidate the readout clearly names: the same thing, in any language or wording. A paraphrase that unmistakably denotes one candidate counts; a single shared word that fits several candidates does not. Where two candidates differ only in who does what to whom, select one only if the readout states that direction. Select nothing when the readout names none of them, or only lists possibilities without committing. Do not use your own knowledge to guess what the model was probably told. For each selection give a SHORT VERBATIM quote (a few words copied exactly from the readout) that names it.
```

`QUESTION`:

```text
Which of these candidates does the readout name? Select every one that applies, or none.
```

`USER`:

```text
READOUT:
"""
{readout}
"""

{question}

{options}

Answer with the letters of your selections and a quote for each; an empty list means none.
```
