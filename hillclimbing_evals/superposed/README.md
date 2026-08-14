# superposed — deterministic word-matcher + region gate

How many dictated concepts can the lens surface **at once**, read from the model's own output
*while it writes an unrelated sentence*? Each prompt tells the model to "think about" some
concepts and then write one fixed sentence; the lens is read over the model's completion, and a
**deterministic word matcher** (no LLM judge) counts how many dictated concepts surface. Read on
**Qwen3.6-27B**, bare render.

> **Recovered artifacts.** The bank and readouts were never pushed to the HF dataset and existed
> only as untracked files from the 2026-08-05 run; they were recovered 2026-08-07 and committed
> here — the readouts under `readouts/` are the **only surviving copy** (git is canonical). This
> is why run artifacts are committed for this family, against the usual rule.

## Bank — 27 items

A bare JSON list; `d-*` = dictation (10), `b-*` = binding (17).
- **Dictation (10):** 1 control (`d-none`) · 4 `novel` · 1 `anchor` (`d-petrichor`) · 2 `chain`
  · 2 `pair` (`d-fox-a/b`). Headline-usable = **8** (minus control and the excluded `d-petrichor`).
- **Binding (17):** 1 control (`b-none`) · 8 `mix` · 8 `same`. Headline-usable = **16**.

Item fields: `name`, `prompt` (built by `spec.render_prompt` = `'Think about %s. Now write this
sentence: "%s"'` with the think-clause **first**; the fixed `TARGET_SENTENCE` is *"The committee
will meet on Thursday to review the annual budget."*), `concepts` (`[]` for the two controls),
`stratum`, and optionally `order_gt` (chains), `binding`/`emotion`/`order` (binding items).

Committed alongside: gate artifacts (`readouts/binding_gate_items.json`,
`binding_gate_out.json`) and per-reader readout files (`read.json`, `dictation_ao.json`,
`binding_ao.json`, `dictation_jlens.json`, `binding_jlens.json`).

## Read site & what the scorer receives

- Lens `ao.asst.alldata-v2cont.k2.g8`, scale 33.152, prompt_kind `explain`. **`LAYERS =
  (44,52,56,60)`** (4 only). **`READ_CELLS_REL = range(-1,-21,-1)`** — the **last 20 tokens of the
  model's own completion** (not prompt positions). `SAMPLING = {k:6, max_new:44, temp:0.8,
  top_p:0.95}` ⇒ 20 cells × 4 layers × 6 samples = 480 AO readouts per item.
- The write-position regime is deliberate: because the written sentence (committee/Thursday/
  budget) shares **no content word** with any dictated concept, an echo of a concept at a read
  position is structurally impossible — this replaced prompt-position reading after four
  echo-filter failures.
- The scorer receives the 6 raw sample strings per (cell, layer) plus the item's `concepts` mapped
  through `spec.content_words`. No judge; pure CPU word matcher. Comparison readers at the same
  activations: AO free text (k=6), J-lens top-10, logit-lens top-10.

## Scoring

**Word matcher** (`score.matches_word`): `re.search(r"\b" + re.escape(word), text, IGNORECASE)`.
The **leading `\b` is load-bearing** (it rejects `cross` inside "across"); the **missing trailing
boundary is deliberate** (so `float` still matches "floating", `librar` "librarian" — the actual
paraphrase idiom). Case-insensitive, stem-tolerant, no other normalization. A concept counts as
decoded if ANY of its content words matches ANY of the cell's samples.

**Concept-word rule** (`spec.content_words`): keep alpha words ≥4 letters not in a fixed
stopword set. This deliberately **drops `fox`, `egg`, `key`, `jar`, `pan`** — so "silver fox" rides
on `silver` alone; the rule was fixed before counting and applied uniformly.

**Region labelling** (`regions.classify_regions`) — the load-bearing provenance rule. One forward
pass labels each read cell `IN_SENTENCE / IN_THINK / POST_THINK / SELF_ADDED_META / OFF_TASK`,
keyed on `<think>`/`</think>` tokens and the paragraph-break blank line. **Only `IN_SENTENCE`
cells carry headline claims.** `POST_THINK` is excluded on the same footing as `IN_THINK` because
a verbalization-ban census (176 continuations) found the dominant leak (47/176 restate the
dictated phrase) lands *after* `</think>` in 29 of 47 cases — and that's exactly where read cells
rel −1..−4 land for 14 of the 17 binding items. (The blank-line cell itself is labelled on the
contaminated side; dropping it leaves the headline numbers unchanged.)

**Write-compliance gate** (`regions.write_compliance`, `min_run=3`): the longest common contiguous
run between the target sentence and the cell tokens must be ≥3 words, else the item is all
`OFF_TASK` → **no `IN_SENTENCE` cells, cannot reach a headline number, dropped from the aggregate.**
This is what excludes **`d-petrichor`** (best run 1 < 3): the model never wrote the committee
sentence (it wrote about petrichor), so all 20 read cells sit in its own text and the lens is
reading words literally present. Including it inflated the typical-activation mean 0.16 → 0.27 — a
retracted number. Region labelling can't catch this (no think block to key on), which is why it's
a **gate**.

**Capacity** (`score.capacity`, over `IN_SENTENCE` activations only):
- `per_activation_max` — most concepts decoded by any single activation — **the headline
  "capacity ≈ 2"** (published 1.88 of 3);
- `per_item_union` (2.12), `per_activation_mean` (**0.16** — the typical activation; 88% of
  activations decode nothing).

**Specificity screen** (mandatory for any per-word claim): a word passes iff **≥3 hits AND ≥3×
its base rate** in items that did NOT dictate it (same template, same lens). This killed the
"blue ladder read back as RED" drift claim (`blue` base 13/1000 — "blue ladder" rests on `ladder`)
and caveated `d-chain-key` (`key` is loaded at 25/1000 → rests on `unlock`).

**workspace-bench bundle:** family `superposed`, `judge_type="deterministic"`, `metric="≥1 concept
surfaced"`, chance None. Oracle arm: a cell hits iff `IN_SENTENCE` and any sample matches any
concept; question passes iff any cell/layer hits (= `per_item_union > 0`; off-task items never
pass). J-lens arm: same rule over top-k tokens at every (layer, position), **no region gate**
(J-lens reads the prompt, which names the concept). Controls have empty `concepts` → both arms
`pass=False` by construction.

## Random baseline / null (no permutation null; layered)

1. **Control items** (`d-none`/`b-none`) — 0 hits over their in-sentence readouts (the `key`
   10/52 and `owe/debt` 6/52 control cells *are* the lexical-loading measurement, not failures).
2. **Base-rate arm of the specificity screen** (≥3 hits AND ≥3× base).
3. **J-lens (3.9%/4.0%) and logit-lens (0 in-sentence cells)** at the identical activations.
4. **Twin flip as internal null:** `silver` 16/312 in fox-a, 0/312 in fox-b; `paper` 18 in fox-b,
   0 in fox-a. Zero crossovers.
5. **J-lens bag Jaccard bands** (direction flip 0.81–0.86 vs cross-relation 0.61–0.79) — direction
   is unreadable from the bag.
6. **Binding-answerability gate** (sampled 33/80 correct direction, 0/80 reversed).

## Caveats (repeat wherever numbers are published)

- **`d-petrichor` is excluded for write-non-compliance and must be named** (its 401/480 is an order
  of magnitude above every other item; it needs an output-compliance gate + re-run).
- **"Capacity ≈ 2" is the ceiling of the best read, not a property of an activation** — quote it
  beside the mean (0.16) and the 88%-decode-nothing figure.
- **Do NOT quote per-cell union = 2.00** — it does not reproduce (`score.capacity()` gives 1.88;
  no item gains a concept from layer pooling).
- The specificity thresholds (≥3 hits, ≥3× base) were chosen after seeing the base-rate
  distribution, not pre-registered. `key` and `owe/debt` are lexically loaded; the red-ladder
  colour-substitution claim is **not** reportable (`blue` base 13/1000).
- Only **4 layers** sampled; the frame→lexical crossover between L44 and L52 is unresolved. k=6
  per cell — only the direction-of-effect contrasts (16 vs 0) are large relative to sampling noise.
- The J-lens comparison is **top-10 bags** — "does not surface it" means "not in the top 10", not
  "not represented". Both banks use **one** target sentence, so the AO's framing language may be
  partly a template property. The `render:"chat"` field in the stored J-lens/logit metadata is
  **stale** — the rendering is bare.
- ⚠️ **Denominator note:** the committed labeller yields **12 `IN_SENTENCE` cells per item** (blank
  line excluded) → 8×48 = **384** dictation / 16×48 = **768** binding in-sentence activations,
  whereas the findings doc's `/412` and `/832` (and per-item `/312`) come from the 13-cell
  (blank-line-included) variant. **The numerators reproduce exactly** (verified: `d-fox-a` silver
  16, `d-fox-b` paper 18, `d-chain-egg` egg 26/crack 17, etc.). A report publishing `x/312` must
  say which cell set the denominator is from.
