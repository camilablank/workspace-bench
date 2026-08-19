# baseline_evals / multi_token

The multi-token half of the frozen headline set: five **`-mt` mechanical banks**. All read on
**Qwen3.6-27B**. (The `safety_cases` diagnostic that used to sit here was moved to
`hillclimbing_evals/safety_cases/` — see its own README.)

## The `-mt` banks

`lens-eval-{basic-readout,directed-modulation,multihop,multilingual,typo}-mt.json`. Same read
site as the single-token banks (final prompt token; directed-modulation-mt reads writing
positions), same **word+exact pass@k** scorer, same **per-family permutation chance** null, same
"never < 3× chance" rule — see `../single_token/README.md` for the full scoring protocol, which
is shared (including the **audit 2026-08-15 change: the strict Opus bank judge is the headline
pass criterion**, with `typo-mt` staying regex-scored). The bank judge's verbatim system prompt
and its 2026-08-19 audit fixes (quote gate verified against readout samples only; API failures
never persisted — counted in `summary.n_unavailable`, retried on `--resume`, all-failed runs
abort; `summary.n_missing_readouts`; layer detection across all items) are reproduced in
`../single_token/README.md` §"Verbatim system prompt — strict Opus bank judge" and apply to the
`-mt` runs identically. This README documents only what `-mt` changes.

**What `-mt` makes harder.** The probed concept must be **multi-token** under the Qwen tokenizer,
so a single lucky token cannot earn credit. In target derivation, the multi-token `target` **is**
the probed concept and is seeded into the scored set (except `multilingual-mt`), then **every
target with ≤1 token is dropped**. An `-mt` item with no strictly-multi-token string is **skipped
at capture** — so an `-mt` bank's scored `n` can be below its item count. Every item carries
`probe_token_lens` (Qwen leading-space token counts — the headline-metric x-axis).

**Gates.** `≥8/10` on a **chat-templated** capability check, while the eval read stays plain
(except DM-mt: `chat_dm`). `poetry-mt` was assessed and is **not** in the bank (no committed
multi-token rhyme); `association-mt` (→ `compositional_association`) and `order-ops-mt` were
retired.

| bank | items | subfamilies | eval_render | notes |
|---|---|---|---|---|
| `lens-eval-multihop-mt.json` | 100 | — | plain | scores `target` + multi-token `intermediates`; `target_alts` never scored |
| `lens-eval-multilingual-mt.json` | 100 | — | plain | `target` **excluded** from scoring (it's the literal next token); only multi-token English `intermediates` score |
| `lens-eval-typo-mt.json` | 100 | — | plain | multi-token correct spellings (`acetaminophen`, 4 tokens) |
| `lens-eval-basic-readout-mt.json` | 100 | entity 35 / processing 50 / implicit 15 | plain 85 / chat 15 | has headline `units[]` |
| `lens-eval-directed-modulation-mt.json` | 94 | pair 74 / secret 10 / preference 10 | chat_dm | **LLM-judged** (same judge as single-token DM) |

Examples:
```json
// multihop-mt — scores ["Ian Fleming","James Bond"]; "Fleming" (target_alts) NOT scored
{"name":"mt-007-author","prompt":"Fact: The author who created the secret agent with code number 007 is",
 "target":"Ian Fleming","intermediates":["James Bond"],"target_alts":["Fleming"],
 "eval_render":"plain","probe_token_lens":{"intermediates":[2],"target":2}}

// multilingual-mt — scores ["traffic light"] only (target excluded)
{"name":"ml-de-fwd-ampel","lang":"de","prompt":"Die Anlage mit rotem, gelbem und grünem Licht an der Kreuzung ist die",
 "target":"Ampel","intermediates":["traffic light"],"probe_token_lens":{"intermediates":[2],"target":2}}

// directed-modulation-mt — exact_targets drops "Golden Gate" next to "Golden Gate Bridge"
{"name":"dmmt-00-pos","subfamily":"pair","pair_id":"dmmt-00","polarity":"think",
 "concept":"Golden Gate Bridge","description":"the orange suspension bridge at the entrance to San Francisco Bay",
 "carrier":"The library card expired at the end of the month.",
 "prompt":"Think about the orange suspension bridge at the entrance to San Francisco Bay. Now write exactly this sentence: \"The library card expired at the end of the month.\"",
 "assistant":"The library card expired at the end of the month.",
 "units":[{"role":"concept","lang":"en","headline":true,"match":["Golden Gate","Golden Gate Bridge"]}],
 "eval_render":"chat_dm","readout":{"kind":"writing_positions"},"probe_token_lens":{"concept":3}}
```

**directed-modulation-mt** is LLM-judged exactly like single-token directed-modulation (judge
model claude-opus-5, item-blind `tokens-llm` summarizer for the J-lens arm, `net = content_bound
− foil` headline, quote-verified positives, white-bear pair contrast). See
`../single_token/README.md` §"Verbatim judge prompt — directed-modulation" for the full judge
protocol and the verbatim prompt text. DM-mt items may also carry `kind: "definition"` (credit
the defined word only) and `components` (compositional — `expressed` may be YES only when
`composition == "full"`).

Reference DM run (2026-08-07, 12 layers, pass@1, foil arm): AO strict-net `dm 0.09 (hint 0.28)` /
`dm-mt 0.32 (0.38)`; summarized J-lens `0.03 (0.09)` / `0.10 (0.17)`; foil ≈ 0 everywhere. The
`tokens-llm` path is non-deterministic across runs — cache summaries for published numbers.

## Card-level gotchas (carry verbatim)

1. `target` is usually not the scored string on the non-`-mt` families (the bridge/`intermediates`
   is); `target_alts`/`reference_concepts`/non-headline units are never scored; `multilingual-mt`
   excludes `target` by design.
2. `--exact` (default) drops component aliases, so the raw `match`/`intermediates` list is not the
   scored list.
3. `-mt` banks can score fewer items than they contain (no-multi-token items are skipped).
4. DM's mechanical pass rate is a **proxy** (`judge_type="freetext"`); the 4 non-DM `-mt` banks
   are `judge_type="deterministic"`.
5. Chance is per-family, seeded, procedure-based (20 donor draws) — and `< 3×` chance is not a
   result.
