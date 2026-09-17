# workspace-bench — agent runbook

`wsbench` is a growing set of evals of whether an activation-reading lens (prose "O-lens" samples or
top-k "J-lens" tokens) surfaces what a model computes but never writes. Every family plugs
into the shared package in `src/wsbench/` (client, judge config, cache, readout contract,
results schema, registry, runner, CLI). Design: `plans/0000-bench-v2-design.md`; phase plans
`plans/000N-*.md`. Public docs: `README.md`, `NOTICE.md`, `CITATION.cff`.

## Readout contract (input to every family)
One JSONL file per (family, arm), one row per cell; a file is all-prose or all-tokens:
```
{"id": "<item id>", "layer": 36, "pos": 33, "samples": ["...", "..."]}
{"id": "<item id>", "layer": 36, "pos": 33, "tokens": ["Ġword", "..."], "scores": [10.8, 9.9]}
```
Optional `"token"` = the read-site token. `load_readouts` counts malformed lines (never fatal),
keeps the first duplicate `(id, layer, pos)`, counts empty readouts (a result, not a missing
cell). `wsbench convert-gen-dir gen_dir=GEN out=F.jsonl kind=prose|tokens` converts the in-house
`<gen_dir>/<label>/L###.jsonl` layout (id = directory name, layer = filename).

## Judge layer (`llm.py`, `judge_config.py`)
- Default judge `google/gemini-3.8-flash` via OpenRouter, reasoning `{"effort": "minimal"}`;
  every family runs on it (no pins since 2026-09-17; the two former Sonnet pins are noted in
  README §Judges). jlens runs all three stages on the default judge.
- Override precedence: `judge_model=` flag > `WSBENCH_JUDGE_MODEL` env > family pin.
  `pinned_instrument` is true only when the resolved model equals the pin; unpinned numbers
  are never numbers of record. Aux models are not overridden.
- Routes by model id: `claude-*` -> Anthropic SDK (`ANTHROPIC_API_KEY`, structured output,
  refusal -> `None`); else OpenRouter via the OpenAI SDK (`OPENROUTER_API_KEY`, `sk-or-…`).
  Shared: JSON-schema output, 12 jittered retries on 408/409/429/5xx/timeouts, process-wide
  thread-safe RPM pacer (`rpm=`, default 240), `preflight` fail-fast, `Spend` tally. A failed
  call returns `None` and never scores. `stream_json_async` is the primitive, `stream_json`
  wraps it (`temperature` dropped on the Anthropic route); `stream_text[_async]` is agentic's
  free-text primitive, routed the same way (`thinking` -> adaptive/disabled on Anthropic,
  reasoning effort high/minimal on OpenRouter; budget doubling on an empty cut-off reply). On
  OpenRouter a `content_filter` finish or a `message.refusal` is a refusal -> `None`, no retry.

## Family conventions
- `wsbench/family.py` holds the helpers the newer families share: `require_cells` (missing
  cells exit 2 unless `allow_missing=True`; a dry run only reports), `cell_text` (a token bag is
  judged scored but verified against bare tokens), `last_pos_rows`, `quote_in` (verbatim after
  `mc.fold`), `tri_state` (pass / fail / undecided), `rate` / `mean`, `pass_rate_result` (the
  pass-rate `FamilyResult` epilogue: decided-only value, bootstrap CI, completeness, counts).
  A pass-rate family's `run()` is: load bank -> `item_scope` -> `load_readouts` -> select cells ->
  `cell_text` -> `Call`s -> `run_calls` -> per-cell verdicts -> per-item rows -> `pass_rate_result`
  -> `with_readout_count`. `chain_intermediates/judge.py` is the shortest complete example.
- `mcjudge.py`: `Call` + `run_calls` (one schema per batch, fingerprint `(prompt_version, model,
  reasoning, temperature, system, user)`, cache `{result, meta}` in `<out>/cells.jsonl`,
  `validate` re-queues an invalid answer); `Preflighter` (one preflight per model per process);
  `item_scope` = bank ∩ `items=` then `limit=`; `base_config` / `base_counts`. `mc.py`: seeds,
  shuffles, `fold`, `letter_index` (a lone letter or a letter with its own option text).
  `summarizer.py`: token bag -> prose (prompt in `docs/summarizer.md`).
- Shared judges: `basic/` (the bank judge, `bank_family(name, title)`) and `multitoken/`
  (forced choice per unit, `mt_family(name, title, calls_per_arm=)`); their families are three-line
  packages under `evals/`. Option lists are pinned by goldens in `tests/golden/`.
- Baselines (`baselines/`): `lucky_guessing` (option lists from the judges' own builders, blind /
  described / uniform) and `prompt_only`; `wsbench freeze` stamps entries with the family's
  `prompt_version`, and `report` shows a floor only while the stamp matches.
- Per-family read regimes, pass rules and floors are documented in each `evals/<family>/README.md`
  (the source of truth); do not restate them here. Camila's original families (agentic, jailbreak,
  hallucination, jlens_concept_pr, moral, relational, role-bound, conjunctive, user_modeling) keep
  their own `judge.py` + `score.py`.
- `opts=key=value,k2=v2` fills `JudgeArgs.extra`; `aux_models` comes from the family `JudgeConfig`.
  `EvalSpec.calls_per_arm` / `.sources` feed `wsbench list`; every non-empty `sources` must
  appear verbatim in README §Credits and NOTICE.md (`tests/test_readme.py`).

## Driver (`runner.py`, `cli.py`)
- `run all=True|families=a,b readouts_root=DIR out=DIR [family_workers=3] [json=True]` reads
  `DIR/<family>.jsonl` (missing -> `skipped`). Order: parse `opts=` once (bad pair = one exit
  2), resolve every judge once, preflight each distinct judge model once (not aux; skipped
  under `dry_run=True`; failure = exit 3 before any thread), then judge in a `ThreadPoolExecutor`
  (`min(n, family_workers)`, 1 under `dry_run=True`) under the shared pacer. Fail-soft per
  family: `JudgeConfigError` -> failed/3, `SystemExit` -> failed/2 (text `e.code` or `exit N`),
  listed under "## skipped / failed" in `summary.md`; other exceptions propagate. Exit 130 on
  `KeyboardInterrupt` (unstarted families cancelled, in-flight ones finish their batch), else
  3 > 2 > 0. Writes `summary.md` + `run.json` = `{started, finished, families: {name: {status,
  error?, results_path, value, complete, pinned_instrument, spend_usd, n_calls?}},
  judge_overrides: {flag, env}, readouts_root}`.
- `report dir=DIR [json=True]`: table with `(lower is better)`, `n (k no readouts)`, macro row (in-row
  `excluded:` + "Not in macro:" footnotes); rewrites `summary.md`. Exit 2 = not a directory /
  unreadable results.

## Results contract (`results.py`)
`results.json` = `{schema_version, family, complete, pinned_instrument, config, n_items,
counts: {n_expected_cells, n_missing_cells, n_unjudged_cells, n_empty_cells, skipped_rows,
spend_usd}, numbers: {metric, value, ci95, chance, chance_label, higher_is_better, extras},
rows}`. `complete` = pinned judge, no subset, zero missing/unjudged cells, empty ≤ 5%. The
macro averages only complete `pass_rate` families and lists every exclusion with its reason.

## Invariants
- Every judge prompt lives in the family's `prompts.py` (`PROMPTS`; `{name}` placeholders
  rendered by `str.replace`, never `str.format` — hallucination's `JUDGE_USER` is the one
  exception) AND verbatim in a fenced block of the family README under "Judge prompts";
  `tests/test_prompts_in_readme.py` asserts equality. Bump `prompt_version` on any edit.
- Failures never score; they are counted. Frozen banks are never edited in place. `spec.run`
  returns a `FamilyResult` and writes nothing. `counts` has a fixed key set (other tallies go
  in `extras`); no `Spend` crosses the family boundary. Cache rows are append-only.
- README: each family entry is `**<spec.title>** — [link]` followed by exactly three lines
  `- *What it is:*` / `- *Example:*` / `- *Judged by:*`, grouped Basic (single token) · Basic
  (multi-token) · Computational · Safety · Association · Bag of words · Precision · Logical processing. Credits live in README §Credits and NOTICE.md, one
  bullet per external source; in-house families get no credit line.

## Adding a family
`src/wsbench/evals/<family>/{__init__,prompts,judge}.py` following the skeleton above (register an
`EvalSpec` in `__init__` with `calls_per_arm` and `sources`), `evals/<family>/{items.json,README.md}`
with every prompt verbatim, a toy `examples/readouts/<family>.jsonl` (a few rows), an offline
`tests/test_<family>.py` (monkeypatch the module's `run_calls`), the README entry and Judges row,
and a credit if the items are external. Smoke `limit=3` live before any full run.

## Workflow
- CLI (pydra, `wsbench <command> key=value ...`): `wsbench list` | `wsbench judge family=F
  readouts=F.jsonl out=DIR` | `wsbench run all=True readouts_root=DIR out=DIR` |
  `wsbench report dir=DIR` | `wsbench baseline` / `wsbench freeze` | `wsbench convert-gen-dir
  gen_dir=GEN out=F.jsonl kind=prose|tokens` | `wsbench convert-read-json read=R out=F.jsonl`.
  Shared judge keys: `judge_model=`, `layers=20,36`, `items=a,b`, `limit=N`, `allow_missing=True`,
  `concurrency=64`, `rpm=240`, `dry_run=True`, `opts=k=v,k2=v2`; `--show` prints the resolved
  config, `--help` a command's keys. Each command is a `pydra.Config` in `cli.py`: declare a
  field in `__init__`, normalise it in `finalize()`; `runner.py` reads the same attribute names.
- `uv sync --extra dev`; `uv run pytest -q`; `uv run ruff check .`; `uv run ruff format .`. Keys
in the environment, never committed; tests make no network calls. Work
in a git worktree (prefix commands with `PYTHONPATH=src` when it shares the main checkout's
`.venv`); PRs are gated by `.github/workflows/ci.yml` (ruff + pytest).
