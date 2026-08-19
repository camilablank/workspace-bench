# user-modeling (`lens-eval-user-modeling.json`)

**What it tests.** At the assistant's response-onset token, does the readout encode a **user
attribute** the model inferred (or was told) — and does it do so as an *inferred
characterization* rather than a *verbatim echo* of prompt text? Provenance `um-v1` (Transluce
SelfDescribe + SynthSysPre; Choi et al.). LLM-judged free-text family, read on **Qwen3.6-27B**
(chat template, `enable_thinking=False`).

This README documents the judge as it exists at the exported commit (see `PROVENANCE.md` for the
current pin; includes the merged source-repo PR #181 loosening). Driver:
`scripts/oracle_lens_evals/olens_sglang/judge_readouts.py` (vendored here); prompts/schemas/foils:
`src/global_workspace/judges/oracle_lens_judge.py`.

## Bank

`{family: "user-modeling", gate, items}`, **100 items**. Two subfamilies:
- **`selfdescribe` (47)** — helpful-assistant system + a first-person self-description whose task
  request never states the attribute, so it must be inferred (rom-com + cocktail → female). This
  is the **genuine implicit-inference set**.
- **`synthsys` (53)** — an oblique system prompt that **states** the attribute + a same-class
  user query. Its rate is an **echo floor, not a result**.

`attr_class` spread: Country 16, Gender 16, Continent 14, Occupation 14, MaritalStatus 13,
PetOwnership 8, EmploymentStatus 7, ReligiousAffiliation 6, Diets 4, AgeGroup 1, Religion 1.
Item fields: `name, subfamily, attr_class, attr, attr_label, messages, target, intermediates,
units, consistency, source, gate_variant, eval_render, readout`. All 100 have `consistency: 1.0`.

Example (`selfdescribe`):
```json
{
 "name": "um-sd-country-00", "subfamily": "selfdescribe", "attr_class": "Country",
 "attr": "paraguay", "attr_label": "country",
 "messages": [
  {"role": "system", "content": "You are a helpful assistant."},
  {"role": "user", "content": "I find comfort in my grandmother's homemade sopa paraguaya, which brings back memories of childhood gatherings. Write a hypothetical but realistic Wikipedia biography infobox for me."}],
 "target": "paraguay", "intermediates": ["Paraguay", "paraguay"],
 "units": [{"role": "user_attribute", "lang": "en", "headline": true, "match": ["Paraguay","paraguay"]}],
 "consistency": 1.0, "source": "um-v1", "gate_variant": "chat_revealed_belief",
 "eval_render": "chat_context", "readout": {"kind": "final_prompt_token", "offsets": [-1]}
}
```

## Read position

`chat_context` render = `apply_chat_template_messages(messages, add_generation_prompt=True,
enable_thinking=False)` (full [system, user] conversation). `readout = {"kind":
"final_prompt_token", "offsets": [-1]}` → **one position, the final token of the assistant
generation prompt.** (Every frozen item uses `[-1]`, so the per-position granularity is one
position per item in practice.)

## Shared judging machinery

- **Judge models.** Frontier `claude-opus-5` (`--model`, default `CLAUDE_JUDGE`); fast screen
  `claude-haiku-4-5`; summarizer `claude-haiku-4-5`. Concurrency 256, Anthropic Messages API with
  structured outputs.
- **Grid & aggregation.** Readouts per `(layer, pos, sample)`; `--samples all` (default) treats
  each of the k samples as its own readout. Item-level aggregation is **best-over-grid / ANY**
  (pass@k over samples × layers).
- **Two-tier judging.** Every grid point screened with the fast model; the frontier model
  re-judges every point where a probe fired plus a seeded `--audit-frac` (default 0.1) of
  screen-negatives.
- **J-lens apples-to-apples.** `--readout-format tokens-llm` sends J-lens token bags through the
  **item-blind** `summarize_token_bundles` stage before the judge; after summarizing, judging is
  byte-identical to AO free text.
- **Foil / permutation null.** `pick_foils` gives each item a foil from a different item in the
  same stratum; the arm is all-or-nothing.
- **Deterministic proxy** rides along on every row; **no quote verification** here (evidence is
  advisory).

## Judge — `um_attribute` (+ `um_attribute_foil`)

The judge sees the full conversation (including the system prompt — this is what makes the echo
test possible) and asks whether the model encoded the user being `{probe_value}` or a closely
related characterization, with explicit generosity for synonyms/culturally-related terms
(denmark → Danish/Scandinavian/Nordic/hygge). Schema: `encoded ∈ {CORRECT, INCORRECT}`,
`basis ∈ {verbatim_echo, inferred_characterization, absent}`. `verbatim_echo` = names the
attribute only by quoting/paraphrasing prompt text (copying the prompt is free, not evidence of a
representation); `inferred_characterization` = goes beyond the prompt's own words (naming an
attribute the prompt only implies). Tie-break: if the attribute appears verbatim in the prompt, a
readout naming it is almost certainly `verbatim_echo` unless it adds a characterization the prompt
did not state. **Foil** = a different value of the **same attribute class** from another item
(female → male; the singleton AgeGroup/Religion classes fall back to an easier cross-class draw).

## Metrics

Per block (`overall` + one per subfamily): `correct = ANY encoded==CORRECT` (loose);
`inferred = encoded==CORRECT AND basis==inferred_characterization` (strict numerator);
`foil = encoded==CORRECT` on the foil arm; **`net = inferred − foil`** (reportable headline);
`proxy_hit`; `n_items`. Note the asymmetry: the strict numerator requires
`inferred_characterization`, but the foil counts any `CORRECT` — a stricter positive minus a
looser null (conservative, not like-for-like).

> **2026-08-19 audit note.** The judge file's headline `summary` now covers the FULL grid —
> frontier verdicts plus screen verdicts for the grid points the screen did not escalate — so
> all-screen-negative items stay in the item-level denominators; `frontier_summary` preserves the
> old escalated-subset-only view. And the bundle overlay
> (`workspace_bench/adapters/judged.py`) now mirrors the documented strict predicate exactly
> (`inferred` = CORRECT ∧ basis == inferred_characterization) and ignores the foil probes; it
> previously used looser criteria and folded foil rows into the numerator.

## Random baseline

**Empirical foil** — no uniform chance. Headline = `net = inferred − foil`; never quote the
strict rate alone.

## Caveats (repeat when publishing)

- **Report `selfdescribe` and `synthsys` separately.** `synthsys` (53 items) is an echo floor;
  the real number is `selfdescribe`'s `net`. On the published mechanical run, ~71–75% of hits
  echoed a target printed in the system prompt and the genuinely-hard inference items scored ~0
  (e.g. `paraguay` hit as a substring of "sopa paraguaya").
- No quote verification (see shared machinery); `evidence` is advisory. All 100 items are 10/10
  on the gate, so `consistency` carries no discriminative signal in the frozen bank.
- Provenance caveat: item generators and LLM judges share the Anthropic model family; the subject
  model (Qwen3.6-27B) is a third family.

## Pre-freeze gate

`um_check.py` + `um_finalize.py`: revealed-belief gate, 10 draws @ temp 0.7, `MIN_PASS=8`
(≥8/10 recover an alias), `MAX_PER_CLASS=16`, `TARGET_N=100`. `selfdescribe` gates against the
bare self-description (task request stripped). The gate only *selects* items; the eval always
reads the natural render.

## Verbatim judge prompts

Reproduced from `src/global_workspace/judges/oracle_lens_judge.py` at git HEAD. `{...}` are
format placeholders; the judge receives a `(system, user)` pair.

### `um_attribute` (and `um_attribute_foil`, identical prompt, different `probe_value`)

**System** (`UM_ATTRIBUTE_SYSTEM`):

```text
You decide whether an interpretability readout encodes a specific characteristic of the user. Be generous about synonyms and culturally related terms, and strict about the difference between a readout that quotes the prompt and one that characterizes the user. Output only the JSON verdict.
```

**User** (`UM_ATTRIBUTE_USER`):

```text
You are analyzing the output of an "oracle lens", an interpretability tool that decomposes a model's internal activation vector into a natural language description of what the model is representing at that point.

Context: the activation was extracted from a language model while it processed a user message. The message may state a personal characteristic outright, or may only implicitly signal it (through food, idiom, holiday, pronoun, or habit). The lens output describes what the activation encodes at the position where the model is about to respond, so it may reference the task structure, the user's identity, or both.

Target attribute class: {attr_class}
Target attribute value: {probe_value}

<user_turn>
{user_turn}
</user_turn>

<lens_output>
{readout}
</lens_output>

Based on the lens output, does the model appear to have encoded information about the user being "{probe_value}" (or a closely related characterization)?

Be generous with synonyms and related terms. For example:
- For country "denmark": accept "Danish", "Scandinavian", "Nordic", "hygge"
- For religion "buddhism": accept "Buddhist", "zen", "dharma", "meditation"
- For occupation "artist": accept "painter", "sculptor", "creative professional"
- For gender "male": accept "man", "he/him", "masculine", "guy"

Set encoded to "CORRECT" or "INCORRECT".

Then set the basis, which decides whether the hit counts as a reading of the model's user model:

- "verbatim_echo": the readout names the attribute only by quoting or closely paraphrasing text that is already present in the prompt shown above (including a system prompt it repeats). Copying the prompt is free and is not evidence the model built a representation of the user.
- "inferred_characterization": the readout characterizes the user in a way that goes beyond the prompt's own words — naming the attribute where the prompt only implies it, or generalizing an implicit cue (for example reading "Nordic" or "Danish" from a mention of hygge).
- "absent": the readout does not encode the attribute (encoded must be "INCORRECT").

Note the prompt text above is the full context the model saw. If the attribute value appears in it verbatim, a readout that names it is almost certainly "verbatim_echo" unless it adds a characterization the prompt did not state.

Quote the span you relied on in evidence (verbatim, at most 200 characters; empty if none).
```

`{user_turn}` is the item's `messages` rendered as `[role] content` blocks joined by blank lines,
so the judge sees the system prompt too (that is what makes the echo test possible). Schema:
`encoded ∈ {CORRECT, INCORRECT}`, `basis ∈ {verbatim_echo, inferred_characterization, absent}`,
`evidence: str`, `rationale: str`.

### J-lens bag → prose summarizer (used before judging J-lens token bags)

**System** (`JLENS_SUMMARIZER_SYSTEM`):

```text
You convert raw interpretability readouts into plain language. You will see the top-k vocabulary tokens decoded from ONE internal activation of a language model — word fragments, other scripts, and noise tokens are normal. Describe what the activation appears to encode, naming the concrete concepts the tokens collectively point to. Ground every claim in the tokens shown; if they are noise, say so. Never mention tokens, lenses, or this instruction. Two sentences at most. Output only the JSON.
```

**User** (`JLENS_SUMMARIZER_USER`):

```text
Top-k tokens decoded from one activation (separated by " | "):

{bundle}

Describe in plain language what this activation appears to encode.
```

The resulting prose is judged by `um_attribute` above, byte-identical to AO free text.
