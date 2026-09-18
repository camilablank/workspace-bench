# J-lens concept precision (`jlens_concept_pr`)

How much of what an AO (prose) arm says at a position is supported by the J-lens top-50 read
at the same activation? An LLM judge decomposes the arm's text into concepts (Stage A), grades
each content token of the J-lens top-10 against the concepts (Stage B, the recall direction) and
grades each CONCEPT against the top-50 content-token set (Stage P, the **precision** headline),
on one in / partial / out scale. A seeded within-family derangement **foil** (the same concepts against another
item's tokens) is the measured floor. 299 fresh on-policy activations (150 chat + 149 pt), read
in the generated region only. Ported from the source repo's `scripts/oracle_lens_evals/jlens_pr/`
(judge of record `judge_openrouter.py`, scorer `score_pr.py`, numerics
`global_workspace/judges/concept_pr.py`) without changing the instrument. This is a precision
instrument: `chance = None` and it is excluded from the `pass_rate` macro.

## Bank (`evals/jlens_concept_pr/`)

Frozen 2026-09-12 from `evals/workspace-bench/jlens_pr/` in the source repo (copied verbatim):

| file | what |
|---|---|
| `manifest.json` | the acts manifest (`acts-jlens-pr/manifest.json`) minus the rollouts: 299 `prompts` rows with `label`, `family` (`chat` / `pt`), `n_pos` (sequence length) and `eval_positions` (exactly ONE read position per label), `layers` 20-60 step 4. `wsbench list` counts its `prompts`. The decoded `tokens` (seed prompt + Qwen rollout) were stripped on 2026-09-17; only the source repo's manifest carries them. |
| `manifest_L42.json` | `acts-jlens-pr-L42/manifest.json`: the same 299 items and read positions captured at layer 42 only (NLA-RL's training layer). Provenance only — a readouts layer is valid iff its reference file exists. |
| `gen-jlens-pr-jlens-k50/<label>/L###.jsonl` | the **reference** (`jlens-pr-v3`, added 2026-09-18): J-lens top-50 tokens (neuronpedia n1000 wikitext lens, cosine readout) at every (label, layer) for the 11 layers plus `L042.jsonl`; one row per file at the eval position, `samples` = 50 byte-level-BPE display strings in rank order (decoded with `bpe_display_to_text`), `scores` unused. Read by the source repo's `jlens_topk_modal.py` (`jlens_eval.py k=50`) from the SAME captured activations, so on every one of the 3,588 files `samples[:10]` / `scores[:10]` and every other field equal the frozen top-10 below byte-for-byte (a test pins it). |
| `gen-jlens-pr-jlens/<label>/L###.jsonl` | the frozen top-10 reference of `jlens-pr-v1` / `v2` (10 samples per row), kept unchanged as the record; v3 reads its recall set as `samples[:10]` of the k50 files, which equals it. |
| `items_manifest.json` | counts, rollout sampling (HF `generate`, T 1.0, top-p 1.0, top-k 0, max-new 512), seed screen |
| `precision_gold_L44.jsonl` / `recall_gold_L44.jsonl` / `recall_gold_L44_cells.jsonl` | the hand-labelled judge-audit gold sets (120 concepts / 60 tokens at L44) that calibrated `STAGE_P_SYSTEM`; provenance only |

Not shipped: the rollout text (the manifests' `tokens`, stripped 2026-09-17), the source's
`items.json` (792 KB capture rows with the exact `input_ids`) and `seeds.json` (the screened
seeds). They live in the source repo under `evals/workspace-bench/jlens_pr/` and are needed
only to recapture the activations (`wsbench-acts:/acts-jlens-pr` on Modal); judging needs only
the labels, families and read positions plus the reference files above.

## Instrument

- **Inputs:** an AO arm's readouts file in the contract, prose only (a `tokens` file is refused:
  the J-lens is the reference, not an arm). The cell is (label, layer, eval position); rows at
  any other position are skipped (`pos_not_selected`). `text = concat_samples(samples)` (the
  non-empty stripped samples joined by `\n---\n`).
- **Grid:** 299 labels x the selected layers (`layers=`, else every layer in the file) at
  `pos == eval_positions[0]`. Any selected layer without a reference file
  `gen-jlens-pr-jlens-k50/<label>/L{layer:03d}.jsonl` is fatal (exit 2) before any call, and so
  is a reference row with fewer than 50 samples (a top-10 file would silently grade precision
  against the top-10). An
  ABSENT row is a missing cell (fatal without `allow_missing=True`; the source scored it silently as
  zero); a PRESENT-but-empty row is scored as zero concepts ("lens silent"), carries
  `has_text = false` and counts in `n_empty_cells` — so the shared "empty cells <= 5% of
  expected" completeness rule applies here, which the source did not have.
- **Reference sets (`jlens-pr-v3`, 2026-09-18):** precision (Stage P / pfoil) reads the content
  tokens of the top-50 (`PRECISION_K`); recall (Stage B / foil, recall@10, `raw_recall`,
  `punct_frac`) reads the top-10 (`RECALL_K`), unchanged. Why: a 60-item pilot (Stage P on
  Gemini 3.8 Flash, the same concepts judged against the top-10 / 20 / 50) found ranks 11-50
  carry real support — precision rose s3d-RL600 L44 0.333 -> 0.408 -> 0.482 and NLA-RL L42
  0.249 -> 0.290 -> 0.356 (paired k50 - k10 +0.149 [+0.111, +0.185] and +0.107 [+0.081,
  +0.132]) while the shuffled-partner foil stayed at 0.03-0.05; the J-lens cosine decays
  smoothly (rank 10 ≈ 0.83x rank 1, rank 50 ≈ 0.71x). **v3 precision numbers are not comparable
  with v1 / v2 (top-10) numbers**; recall numbers are.
- **Models:** `JudgeConfig(model="google/gemini-3.8-flash", prompt_version="jlens-pr-v3",
  reasoning={"effort": "minimal"})`. v3 (2026-09-18) = precision against the top-50, prompts
  unchanged; Stages A / B / foil keep the cache fingerprint version `jlens-pr-v2`
  (`AB_CACHE_VERSION`) because their prompt text is byte-identical, so re-judging an arm whose
  v2 `cells.jsonl` exists re-runs only Stage P. All three stages run on the family judge. **Instrument
  change (2026-09-16, Camila):** the source judge of record ran Stage A (concept extraction) on
  `deepseek/deepseek-v4-flash` with reasoning off; this repo runs it on Gemini 3.8 Flash like
  Stages B and P, hence `jlens-pr-v2` (the prompt text is unchanged). Concept lists, and so
  precision/recall numbers, are not comparable to the source's DeepSeek-extracted ones until
  a re-run is done; the L44 hand-label gold files audited the DeepSeek lists. Stage A and
  Stage B run at the provider default temperature; Stage B at the default
  temperature; **Stage P at temperature 0.0**; `max_tokens = 16000` on every stage (a 60-concept
  answer echoes every concept back). The Anthropic route drops `temperature` with a warning.
- **Stages** (one `run_calls` batch each; keys carry no arm — `cell.key` is the contract key
  `<label>__L<layer>__p<pos>`):
  - **A** `<cell>:A` — one call per cell with text; the parsed concepts are deduplicated
    case/whitespace-insensitively.
  - **B** `<cell>:B<nn>` — one call per **content** token of the reference top-10 (`is_content_token`:
    WORD / CJK / emoji; punctuation and markup symbols are excluded and reported as
    `punct_frac`) per cell with concepts. **foil** `<cell>:F<nn>` — the same against the
    derangement partner's tokens (`foil_pairing`, seed 0 per family group over the whole
    manifest); runs by default.
  - **P** `<cell>:P<nnn>` — one call per chunk of at most `STAGE_P_CHUNK = 60` concepts, each
    chunk with the content tokens of the cell's reference top-50; the scorer concatenates a cell's chunks by
    `offset` and books `missing_p` unless they tile the concept list exactly. **pfoil**
    `<cell>:Q<nnn>` only with `opts=stage_p_foil=1`.
  - A parser `ValueError` (drifted, duplicate or missing concept, bad grade) is a **reject**:
    the answer is cached as a failure (`result: null`, raw answer in `meta.raw`) and exactly
    that key is re-queued by the next run, never partially scored.
- **Scoring** (`concept_pr.py`, unchanged): precision = mean Stage-P support over the cell's
  concepts (`in` 1 / `partial` 0.5 / `out` 0); recall@10 = mean over content tokens of the
  exact hypergeometric `expected_max_grade(m=10)`; `raw_recall` = mean over tokens of the max
  grade. Statuses per cell: `ok` / `missing_a` (text but no Stage A result) / `incomplete_b`
  (a content token without a grade row; NaN on both axes) / `missing_p` (precision NaN, recall
  kept). Recall is NaN when the top-10 has no content token, precision when the top-50 has none
  (dropped from that axis's mean): 18 of the 3,588 reference cells have no content token in the
  top-10, 16 of them have some in ranks 11-50 and so get a Stage P call and a real precision
  (status `ok`, recall NaN). `ItemScore`'s token counts and `punct_frac` stay top-10
  statistics; rows carry `n_precision_content_tokens`.
  Bootstrap: 1,000 resamples, seed 0 (`concept_pr.bootstrap_ci`).
- **Headline layer:** the one judged layer if there is exactly one, else L44. With several
  judged layers and no L44 the headline is `None`: `value = None`, `ci95 = None`,
  `n_items = 0`, `extras.headline_layer = None`, `complete = false`, with a printed warning.
- **FamilyResult:** `metric = "precision"`, `value` = mean cell precision at the headline
  layer, `ci95` = its bootstrap, `n_items` = cells at the headline layer with status `ok` and a
  non-NaN precision, `chance = None`, `chance_label = "shuffled-partner foil precision
  (measured, see extras.foil); reference = J-lens top-50"`; `extras = {headline_layer,
  precision_k (50), recall_k (10), recall_at_10, recall_at_10_ci,
  raw_recall, foil: {recall_at_10, precision} (precision null unless pfoil), by_layer: {L:
  block}, mean_n_concepts, punct_frac, reject_rate: {stage: {n_requests, n_missing, rate}},
  statuses}`. Per-cell `passed` (precision >= 0.5) in `rows` is display-only.
- **Complete** = pinned judge, no `items=` / `limit=` / `layers=` subset, zero missing
  cells, empty cells <= 5%, a headline layer, and **every stage's reject rate <= 5%**
  (`JLENS_PR_MAX_REJECT_RATE` of the source stage script) in place of the shared "zero unjudged
  cells" clause; `counts.n_unjudged_cells` still reports the cells with status `missing_a` /
  `incomplete_b` / `missing_p` at any judged layer.
- **Cost:** ≈ 35k calls per 11-layer arm (≈ 3.3k A + ≈ 30k B/foil + ≈ 3.5k P); OpenRouter's
  per-call `usage.cost` under-reports the billed spend by ~2.5x. Gemini over-credits precision
  against the hand labels (0.43 vs hand 0.33 at L44 for the s3d arm), quote the caveat with
  every number — **measured against the top-10**; its size at the top-50 is unmeasured (the
  flat foil rules out credit to unrelated tokens, not extra leniency on related ones).
- Not ported: `judge_pr.py` (legacy GPT route), `judge_modal.py`, `judge_bakeoff.py`,
  `validate_precision_judge.py`, `wsbench_stage.sh` (`wsbench run all=True` replaces it), the
  bundle adapter, `lexical_support` as a headline (it rides along in `concept_pr.py` unused).

## Example

```bash
# print the Stage A prompt for the first cell (no key, no calls; missing cells are reported)
uv run wsbench judge family=jlens_concept_pr readouts=examples/readouts/jlens_concept_pr.jsonl out=/tmp/j dry_run=True
# an in-house gen dir -> contract file, then a full arm
uv run wsbench convert-gen-dir gen_dir=outputs/gen-<run>-jlenspr out=readouts/jlens_concept_pr.jsonl kind=prose
OPENROUTER_API_KEY=sk-or-... uv run wsbench judge family=jlens_concept_pr readouts=readouts/jlens_concept_pr.jsonl out=outputs/<run>/jlens_concept_pr
```

`examples/readouts/jlens_concept_pr.jsonl` is hand-written toy data (three labels at L44 and
one empty cell at L48); a real run on it needs `allow_missing=True`.

## Judge prompts

Source: `scripts/oracle_lens_evals/jlens_pr/judge_prompts.py` — `STAGE_A_SYSTEM` (L17-30),
`STAGE_B_SYSTEM` (L32-47), `STAGE_P_SYSTEM` (L182-209; its worked-example guide is calibrated
against the hand-label audit and is copied byte-for-byte, never reflowed), the schemas
(L51-73, L211-228), `GRADE_VALUE` (L75), `STAGE_P_CHUNK` (L229-232) and the user bodies of
`stage_a_params` / `stage_b_params` / `stage_p_params`. Templates carry `{name}`
placeholders substituted in one pass: `{text}` = the cell's concatenated text,
`{token_repr}` = `repr(token)`, `{tok_list}` = `", ".join(repr(t))` over the content tokens,
`{listing}` = the concepts as `1. concept` lines.

### `STAGE_A_SYSTEM`

```text
You are given text produced by an interpretability lens that verbalizes a language model's internal state. Break the text into its UNIQUE concepts: short noun phrases naming each distinct thing, property, action or entity the text asserts. Merge duplicates and near-duplicates (inflections, restatements) into one concept. Ignore list bullets, markdown emphasis and boilerplate connectives, but NOT evaluative or affirmative markers: a checkmark or emoji (✅, ✔, 🎉), "Correct Answer", "Yes", "Final Answer" each name a concept ("correct / confirmed answer", "celebration") and must be listed when present. Do not add concepts the text does not contain.

Example. Text: "limited protein intake, potentially hindering wound healing. An elderly widow with inadequ" -> concepts: ["inadequate/insufficient amount", "protein intake", "wound healing", "elderly widow / older adult"].

Return JSON: {"concepts": [string, ...]}.
```

### `STAGE_A_USER`

```text
Text:
{text}
```

### `STAGE_B_SYSTEM`

```text
You grade whether ONE token read out of a language model's internal activations is supported by each of a list of concepts extracted from a lens's text. Grade every concept exactly once:

- "in": the concept has the token, modulo word inflection / casing / translation (e.g. tokens "accessing", "访问" match the concept "access pattern"). Emoji are words: an emoji token matches a concept that names the same thing in words ("✅" matches "correct answer"; "🎉" matches "celebration"), and emoji with the same meaning are the same token ("🎉" and "🥳" both match "celebration"; "✅", "✔" and "✓" all match "confirmed / correct").
- "partial": the concept has a close relative of the token (e.g. token "lacks" matches "inadequate intake"; token "EXIT" matches "return 0"; token "😊" is partial for "gratitude").
- "out": everything else, including words the concept does not contain (e.g. tokens "SUCCESS", "printf" do not match "return 0;").

Return JSON: {"grades": [{"concept": <exact concept string>, "grade": "in"|"partial"|"out"}, ...]} with one entry per listed concept, using the concept strings verbatim.
```

### `STAGE_B_USER`

```text
Token: {token_repr}

Concepts:
{listing}
```

### `STAGE_P_SYSTEM`

```text
You are given the SET of tokens read out of a language model's internal activations at one position (a lens's top-k readout, content tokens only), and a list of concepts extracted from another lens's text about that same position.

Stage B asks, of ONE token, which concepts have it. This is the same question asked the other way round: for each concept, grade the BEST correspondence between that concept and ANY token in the set. Grade every concept exactly once, on the same three-point scale:

- "in": the concept has one of the tokens, modulo word inflection / casing / translation (tokens "accessing", "访问" are in for the concept "access pattern"). A concept that says more than the token still counts as in — the token has to be in the concept, not equal to it: "working at Walt Disney" is in for the token "Disney", "no worry about overlap" is in for "overlap", "credit card approval rates" is in for "credit" or "rates", "objective lens" is in for "lens". Emoji are words: an emoji token matches a concept that names the same thing in words ("✅" is in for "correct answer"; "🎉" for "celebration"), and emoji with the same meaning are the same token ("🎉" and "🥳" both match "celebration"; "✅", "✔" and "✓" all match "confirmed / correct").
- "partial": the concept has a close relative of one of the tokens — the token names the concept's category, a sibling, or a part of it (token "lacks" is partial for "inadequate intake"; "technologies" for "filtration technologies"; "police, arrest" for "traffic stop"; "people" for "Anji people"; "Alumni" for "Yale University"; "loan, funding" for "SBA approval requirement"; "MRI, MR" for "MR-ME"; "plants, herbs" for "flowers"; "😊" for "gratitude").
- "out": everything else — no token is in the concept and none is a close relative of it (tokens "Azure, Haskell, PowerShell" are out for "lambda expression (fun x -> ...)"; "Supplier" is out for "based in Victoria, Australia"; "はありません" is out for "final token").

Return JSON: {"grades": [{"concept": <exact concept string>, "grade": "in"|"partial"|"out"}, ...]} with one entry per listed concept, using the concept strings verbatim.
```

### `STAGE_P_USER`

One call per chunk of at most `STAGE_P_CHUNK = 60` concepts (`stage_p_chunks`: `(offset,
slice)` pairs tiling the concept list in order); the cell's full content-token set goes into
every chunk.

```text
Tokens: {tok_list}

Concepts:
{listing}
```

### Grade values

`GRADE_VALUE = {"in": 1.0, "partial": 0.5, "out": 0.0}` — `in` 1.0, `partial` 0.5, `out` 0.0.

### Output schemas

`STAGE_A_SCHEMA`:

```json
{
 "name": "concepts",
 "strict": true,
 "schema": {
  "type": "object",
  "additionalProperties": false,
  "required": [
   "concepts"
  ],
  "properties": {
   "concepts": {
    "type": "array",
    "items": {
     "type": "string"
    }
   }
  }
 }
}
```

`STAGE_B_SCHEMA`:

```json
{
 "name": "grades",
 "strict": true,
 "schema": {
  "type": "object",
  "additionalProperties": false,
  "required": [
   "grades"
  ],
  "properties": {
   "grades": {
    "type": "array",
    "items": {
     "type": "object",
     "properties": {
      "concept": {
       "type": "string"
      },
      "grade": {
       "type": "string",
       "enum": [
        "in",
        "partial",
        "out"
       ]
      }
     },
     "required": [
      "concept",
      "grade"
     ],
     "additionalProperties": false
    }
   }
  }
 }
}
```

`STAGE_P_SCHEMA`:

```json
{
 "name": "support",
 "strict": true,
 "schema": {
  "type": "object",
  "additionalProperties": false,
  "required": [
   "grades"
  ],
  "properties": {
   "grades": {
    "type": "array",
    "items": {
     "type": "object",
     "properties": {
      "concept": {
       "type": "string"
      },
      "grade": {
       "type": "string",
       "enum": [
        "in",
        "partial",
        "out"
       ]
      }
     },
     "required": [
      "concept",
      "grade"
     ],
     "additionalProperties": false
    }
   }
  }
 }
}
```

## Sources

- Reference lens: *Verbalizable Representations Form a Global Workspace in Language Models*
  (Anthropic / Transformer Circuits, 2026, arXiv:2607.15495); code
  `anthropics/jacobian-lens` (Apache-2.0); the reference tokens come from the
  `neuronpedia/jacobian-lens` n1000 wikitext artifact (cosine readout, top-50; top-10 until
  `jlens-pr-v3`).
- Seed corpora of the 299 items (first user turns / 128-token prefixes, answered on-policy by
  Qwen/Qwen3.6-27B): [`lmsys/lmsys-chat-1m`](https://huggingface.co/datasets/lmsys/lmsys-chat-1m)
  (the LMSYS-Chat-1M Dataset License Agreement applies to those turns),
  [`ConvLab/dailydialog`](https://huggingface.co/datasets/ConvLab/dailydialog) (CC BY-NC-SA 4.0;
  Li et al., 2017, *DailyDialog: A Manually Labelled Multi-turn Dialogue Dataset* — the same
  data terms as the hallucination family), [`NeelNanda/pile-10k`](https://huggingface.co/datasets/NeelNanda/pile-10k),
  [`HuggingFaceFW/fineweb-edu`](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
  (sample-10BT).
