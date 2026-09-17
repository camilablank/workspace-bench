# 0009 — every family on Gemini 3.8 Flash (2026-09-17)

Camila: "make sure all the evals have gemini as a judge too". Two families are pinned to
`claude-sonnet-5` today: `agentic_misalignment` (README §Judge: "no Gemini agreement data")
and `jailbreak_recognition` (pinned 2026-09-16: "Gemini refuses a share of jailbreak cells").
Both move to the repo default `google/gemini-3.8-flash`. The Anthropic route stays in `llm.py`
for `judge_model=` / `WSBENCH_JUDGE_MODEL` overrides; it is no longer any family's pin.

## 1. `llm.py`: the free-text primitive routes by model id

Today `stream_text_async` raises `JudgeConfigError` for any non-`claude-*` model, because
agentic's three stages are free text. Add an OpenRouter branch mirroring the Anthropic one:

- `_openrouter_text_once(client, user, *, model, reasoning, max_tokens, timeout, spend)`:
  `client.chat.completions.create(timeout, model, max_tokens=budget,
  messages=[{"role": "user", "content": user}], extra_body={"usage": {"include": True},
  "provider": {"require_parameters": True}, "reasoning": reasoning})` — no system block, no
  `response_format`. `spend.calls += 1`; no `choices` → `ValueError` (retried like the JSON
  path); `spend.usd += usage.cost` (same lookup as `_openrouter_once`); a non-empty
  `message.refusal` → `spend.refusals += 1`, print, return `None`; text =
  `(message.content or "").strip()`; **budget doubling** when the text is empty and
  `finish_reason == "length"` and `budget < _TEXT_BUDGET_CEILING` (same loop as
  `_anthropic_text_once`); otherwise return the text (`""` is a valid result).
- `_one_text` gains `route_: Route` and `reasoning: dict | None` and dispatches; the retry /
  fatal classification is shared verbatim (it is the same loop body).
- `stream_text_async`: drop the `claude-*` guard; `route_ = route(model)`; `client =
  _make_client(route_, api_key(model))`; on the OpenRouter route `reasoning =
  {"effort": "high"} if thinking else {"effort": "minimal"}` (Gemini cannot turn reasoning
  off; `minimal` is the repo's "off"); on Anthropic `thinking` still maps to
  `adaptive` / `disabled` exactly as today. Docstring updated. The `stream_text` wrapper is
  unchanged. Invariant retained: the Anthropic branch (streaming, adaptive thinking, refusal →
  `None`, doubling up to 64k) is byte-for-byte what it is now.

## 2. `agentic_misalignment`

- `__init__.py`: `JUDGE_MODEL = DEFAULT_JUDGE` (import from `judge_config`), comment "the only
  non-default pin" deleted; `reasoning=None` stays (resolves to `{"effort": "minimal"}` on the
  OpenRouter route; the family ignores `ResolvedJudge.reasoning` and passes per-stage
  `thinking` as before — retained).
- `prompts.py`: `PROMPT_VERSION = "am-narrative-v2"`. The prompt text is unchanged; the bump
  marks the judge change, exactly as `jlens-pr-v2` did on 2026-09-16 (a frozen baseline
  stamped `am-narrative-v1` no longer applies).
- README: §Stages "All calls are free-text on the Anthropic route" → routed by model id
  (Gemini via OpenRouter by default; thinking on/off = reasoning effort high/minimal there,
  adaptive/disabled on the Anthropic route); §Judge rewritten (Gemini of record from
  2026-09-17, `am-narrative-v2`; Sonnet 5 was the judge of record under `am-narrative-v1`
  and is reachable only by override, unpinned); `counts.spend_usd` is now OpenRouter's
  `usage.cost` (the "0 on the Anthropic route" clause becomes "0 under a `claude-*`
  override").
- Tests: `test_spec_and_list` (model, version, reasoning `{"effort": "minimal"}`, `wsbench
  list` line); `FakeText` becomes OpenRouter-shaped (`chat.completions.create`, user-only
  `messages`, returns `choices[0].message.content`, `finish_reason`, `usage.cost`); the
  `fake_text` fixture sets `OPENROUTER_API_KEY`; `_Stream` goes. Every end-to-end assertion
  (cache keys, fingerprints, stage sequencing, counts) is retained.

## 3. `jailbreak_recognition`

- `__init__.py`: `judge=JudgeConfig(model=DEFAULT_JUDGE, prompt_version=PROMPT_VERSION,
  reasoning=None)`; pin comment deleted.
- `prompts.py`: `PROMPT_VERSION = "jb-v2"` (prompt text unchanged, judge changed).
- README §Judge model and §Instrument change: default judge, `jb-v2`; the refusal caveat is
  kept as a caveat, not a pin: a refused / failed cell is `n_unjudged_cells` (`n_api_failed`),
  and because `complete` requires zero unjudged cells an arm with any refusal is reported but
  not `complete` (excluded from the macro) — say so explicitly. Sonnet 5 was judge of record
  2026-09-16 → 2026-09-17 under `jb-v1`.
- Tests: the two `jb-v1` literals → `jb-v2`; add `SPEC.judge.model == DEFAULT_JUDGE`.

## 4. Repo docs and tests

- `README.md` §Judges: paragraph → every family on `google/gemini-3.8-flash`, reasoning
  `{"effort": "minimal"}`; the two former pins named with their dates; `claude-*` only via
  override. Table rows for the two families updated (model + version). §Credits "Judge
  models" bullet and `NOTICE.md` L19: Gemini for every family; Claude Sonnet 5 reachable by
  override only.
- `CLAUDE.md` §Judge layer: drop the "Pins:" sentence; `stream_text[_async]` is no longer
  "Anthropic-only" — it routes by model id like `stream_json`.
- `tests/test_readme.py::test_judges_table_pins` → every family row is `| <name> |
  google/gemini-3.8-flash |`, no `claude-sonnet-5 |` row, "override" in the paragraph.
- `tests/test_llm_text.py`: `test_openrouter_model_raises` → OpenRouter tests: success shape
  (no `system`, no `response_format`, `reasoning` minimal / high, `usd` from `usage.cost`),
  budget doubling on empty + `length`, ceiling → `""`, `message.refusal` → `None` +
  `refusals == 1`, transient retry. Anthropic tests untouched.
- `scripts/judge_swap.py`, `plans/0007-judge-swap.md`, `docs/judge_swap/` are historical
  records of the Opus/Sonnet baselines and are not edited.

## Invariants restated

- `judge_config.resolve` precedence and `pinned_instrument` semantics unchanged; both families
  are now pinned to the default, so a default run is `pinned_instrument = true`.
- Agentic's per-stage thinking (A off, B/C on), `max_tokens` (400 / 16000 / 12000), cache
  keys and chained fingerprints are retained; only the model string inside `fp_A` changes.
- No prompt text changes anywhere; `tests/test_prompts_in_readme.py` must still pass.

## Consequences to surface to Camila

- Numbers of record for both families restart at `am-narrative-v2` / `jb-v2`; nothing from the
  Sonnet runs is comparable.
- Jailbreak: if Gemini refuses cells, those arms are never `complete` and drop out of the
  macro; the rate is reported so the size of the problem is visible after the first run.
- Agentic Stage A at `max_tokens = 400` on Gemini: minimal reasoning still spends tokens; the
  doubling loop covers an empty reply, so no behaviour change is expected, only cost.

## Verification

Full `pytest -q`, `ruff check`, `ruff format --check`. No live call (see memory: capabilities,
not runs); a `dry_run=True` on the agentic and jailbreak examples must print Gemini as the
model.

## Amendments after critique round 1

- **Fingerprints.** `fp_A = fingerprint("A", PROMPT_VERSION, model, user)` and `fp_B` embed
  both the prompt version and the model, so BOTH change (`am-narrative-v2`, Gemini); every
  chained fingerprint follows. The earlier "only the model string" sentence is withdrawn.
- **Token tallies.** The OpenRouter text branch adds `usage.prompt_tokens` /
  `usage.completion_tokens` to `spend.input_tokens` / `spend.output_tokens` (beside
  `usage.cost` → `spend.usd`), so `extras.usage` stays meaningful; README §Numbers "token
  counts are in `extras.usage`" holds, and the `spend_usd` sentence says OpenRouter's
  `usage.cost` (which under-reports the billed spend ~2.5x). The JSON branch is unchanged.
- **Refusals on OpenRouter.** Both `_openrouter_once` (JSON) and the new text branch treat
  `choices[0].finish_reason == "content_filter"` or a non-empty `message.refusal` as a refusal:
  `spend.refusals += 1`, print, return `None` — no retry (today an empty JSON body burns 12
  retries × backoff). Text branch: empty content + `finish_reason == "length"` → double the
  budget (up to the 64k ceiling); empty content otherwise → `""` (valid, as on Anthropic).
  This closes the "refusal cached as an uninformative Stage A note" hole for agentic and makes
  a jailbreak refusal one call, not thirteen.
- **`stream_text_async` signature is unchanged** (no `reasoning` kwarg; `tests/test_llm_text.py`
  asserts `"reasoning" not in` its parameters); the effort dict is derived from `thinking`
  inside.
- **Tests to touch, explicitly:** `tests/test_agentic_misalignment.py` — `FakeText` →
  OpenRouter shape (`chat.completions.create`; returns `choices=[{message: {content,
  refusal: None}, finish_reason}]`, `usage={prompt_tokens: 7, completion_tokens: 3, cost:
  0.0}`); `_responder` + `test_three_chunks_use_partials` assert
  `kw["extra_body"]["reasoning"] == {"effort": "minimal" | "high"}` instead of `kw["thinking"]`;
  `test_non_claude_judge_raises` deleted (replaced by a test that a `claude-*` override still
  builds the Anthropic client); `spend_usd == 0.0` stays (fake cost 0.0); `usage == {14, 70,
  28}` stays via the token tally; `am-narrative-v1` → `v2`. `tests/test_readme.py`
  `test_judges_table_pins`: every family row on Gemini, no `claude-sonnet-5 |` row, the word
  "override" in the paragraph (the `"refuses"` assertion goes; the refusal caveat moves to the
  jailbreak README). `tests/test_llm_text.py`: `test_openrouter_model_raises` → the OpenRouter
  tests listed in §4 plus `content_filter` → `None`.
- **More sentences that become false:** `README.md` L21 ("`ANTHROPIC_API_KEY` for the two
  Claude-pinned ones" → only for a `claude-*` override); `evals/agentic_misalignment/README.md`
  L69 "thinking off" → "thinking off (reasoning effort `minimal` on OpenRouter)"; the stale
  `concept_pr.py:12` docstring ("claude-sonnet-5" for Stage P) is corrected to Gemini while here.
- **Design risk kept on the record:** `max_tokens = 400` at Stage A may include Gemini's
  reasoning tokens on OpenRouter; the doubling loop covers an empty reply, a truncated
  non-empty note is returned as-is (same as on Anthropic). Only a live `limit=3` smoke settles
  it; not run in this PR.
