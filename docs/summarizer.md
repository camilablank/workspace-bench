# The J-lens summarizer

Every family accepts two readout kinds (see `CLAUDE.md` › Readout contract). Prose
("O-lens") samples are judged as they are. Top-k token ("J-lens") readouts first go through
**one shared blind-interpretation step** — `wsbench/summarizer.py` — that turns a token bag into
one or two sentences of prose; the judge then sees that interpretation text in place of the
readout. The interpreter sees only the tokens: no stimulus, no options, no gold.

- `SUMMARIZER_PROMPT_VERSION = "interp-v1"` — bump on any edit.
- Model: `aux_models["summarizer"]` of the family's `JudgeConfig` if set, else the judge model
  (all four MC families leave it at the default).
- Cache key `summ:<cell key>`, fingerprint `(SUMMARIZER_PROMPT_VERSION, model, bundle_text)`
  in the family's `<out>/cells.jsonl`. A failed summary leaves the cell unjudged
  (`n_unjudged_cells`); an empty interpretation counts as a failure — an empty string is never
  judged.
- `render_bag(tokens, scores)` renders one row: `tok (score)` best first, scores to 2 dp, joined
  by `" | "`; without scores the tokens joined by `" | "`.
- Per-family bundle text: moral_rationale / role_bound_association = `render_bag` of the row;
  relational_multihop = the `[position …]` bundle line; conjunctive_association = the per
  (layer, item) `" | "` blob. See each family README.
- `dry_run=True` on a `tokens` file prints only the summarizer prompt (no judge prompt exists
  before a summary).

## Prompt

Copied verbatim from the source repo's `scripts/oracle_lens_evals/oa_eb_readout_judge.py`
L52-58 (identical to `judge_relational_multihop.py` L88-93). System:

```text
You are shown the top-10 token readouts from an interpretability lens at one position inside a language model that was reading a passage you cannot see. Tokens may include noise, fragments, other languages (translate them), or byte artifacts. In one or two sentences, state what these outputs are collectively trying to say — the situation or mental content they point to. Commit to the most specific reading the tokens support; do not just say they are noisy.
```

User message (`{txt}` = the bundle text):

```text
TOKEN READOUTS:
{txt}
```

## Schema

`schema_block("interp", …)`:

```json
{
 "name": "interp",
 "strict": true,
 "schema": {
  "type": "object",
  "additionalProperties": false,
  "required": [
   "interpretation"
  ],
  "properties": {
   "interpretation": {
    "type": "string"
   }
  }
 }
}
```
