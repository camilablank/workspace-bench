# Compositional-association eval (frozen v1, 2026-08-05)

Does a lens **state** a compositional latent (who × feeling × about-what × …) that the model
demonstrably carries? `items.json` is the frozen benchmark — 100 contrastive items = 10
clusters × 10 cells; **do not edit** (amendments go through a new version + a ledger entry).
This eval replaces the retired `association-mt` bank family as the multi-token/latent-content
eval of record.

Everything about the design and selection process (generating grid, decisions ledger, style
pilots) moved to `docs/project/experiments/ola/compositional_association_design.md`; protocol
+ final results of record: `docs/project/experiments/oracle_lens/latent_eval.md`. Run
pipeline: readouts via the olens_sglang pipeline (see the design doc §1 for the exact
commands; teacher readouts MUST use the embeds reference path), then judge as below.

## Item schema

Per item: `id`, `cluster`, `level` (L3/L4), `anchor`, `varied_axes`, `cell`, `stimulus` (the
ONLY text the subject model sees), `read_position`, `gold_label` (3rd-person composition,
doubles as MC option), `mc_options`, `contrast_label` (sibling flipped on one axis),
`scored_axes`, `axis_concepts` (what the judge is asked about, per axis), `credit` (per-axis
credit word lists = the authoring blocklists).

## Final judge (the instrument)

Multiple-choice, one forced choice per item (`judge_mc.py`; the default instrument as of
2026-08-13):

```bash
uv run --no-sync python scripts/oracle_lens/latent_eval/judge_mc.py \
    <gen_dir> --tag teacher --out outputs/oracle_latent_eval/verdicts_teacher.json
```

`items.json` defaults to this frozen set (pass a path only to score a different version).
The judge (**claude-opus-5** via `global_workspace.judges.llm_client`, structured output,
concurrency 256, `--resume` supported) asks ONE multiple-choice question per item: which of
the item's 10 `mc_options` (whole-composition labels — `gold_label` + near-miss distractors,
seeded-shuffled) the readout STATES, plus an 11th **"cannot tell from the readout"** escape.
The exact-synonym / presence-only rule is pinned in-prompt: pick an option ONLY if the
readout NAMES it as the direct word(s) or an EXACT synonym ("mourning" names grief / "wife"
names spouse); do NOT pick something merely INFERABLE from evidence, scene, or narrative, and
do not use world knowledge. The full system prompt lives in `judge_mc.py::SYSTEM` — it IS the
instrument; don't paraphrase it. The judge returns a supporting quote (free-text, for audit
— unlike the prior binary instrument it is NOT hard-verified as a verbatim substring).

**Item PASS = the judge picks `gold_label`** (chance ~1/11). The script prints PASS/n and a
pick breakdown (gold / contrast = the flipped sibling / other distractor / cannot-tell), and
writes per-item verdicts to `--out`. For J-lens gen dirs pass `--jlens-interp` (added
2026-08-13): the item-blind bag→prose summarizer runs on the top-k token readouts first and
the MC judge scores the resulting prose — apples-to-apples with AO free text. This mirrors the ordered_association MC judge
(`scripts/oracle_lens_evals/oa_eb_readout_judge.py`) — same seeded shuffle + "cannot tell"
escape — here whole-item rather than per-axis (Camila's 2026-08-13 choice).

Reference MC results (recorded 2026-08-13, all-position × L{20,28,36,44,52,60} readouts):
teacher **PASS 39/100** (cannot-tell 47, contrast 5, other 9); J-lens **PASS 38/100**
(cannot-tell 40, contrast 5, other 17); chance ~9/100. Verdicts:
`outputs/oracle_latent_eval/verdicts_mc_{teacher,jlens}.json`.

### Prior binary presence instrument (`judge_final.py`, retained)

Before 2026-08-13 the instrument asked, per (item, scored axis), a yes/no presence question
with scripted verbatim-quote verification; Item PASS = every gold component verified-present
AND every flipped component absent. Reference results (that instrument): teacher lens 19/100
all-gold-present, 73/100 flipped-absent, **PASS 10/100**; official J-lens 10/82/**9**. Those
numbers measure a different thing (free naming of every component) and are NOT comparable to
MC PASS above.

## Caveats (repeat wherever numbers are published)

- Generator (Claude) and judge (Opus 5) share a model family; subject model (Qwen3.6-27B)
  is a third family.
- Teacher readouts must use the embeds reference path (design doc §1, ledger #15).
- Presence inside a confabulated wrapper still counts if quote-verified (ledger #22).
- Sparse API failures (~1–3 calls/run) count conservatively against readouts.

## Verbatim judge prompt

From `scripts/oracle_lens/latent_eval/judge_mc.py` (the `SYSTEM` prompt is identical at git HEAD
and working tree). ONE forced-choice MC per item; the judge sees ALL layers × positions
concatenated.

**System** (`SYSTEM`):

```text
You judge the READOUT of an interpretability lens: text decoded from a language model's internal activations while it read a passage you cannot see (possibly labeled by layer; may contain noise, fragments, or other languages). You are given ONE multiple-choice question — which of the listed compositional descriptions the readout STATES. Pick an option ONLY if the readout names its content directly, as the direct word(s) or an EXACT synonym ("mourning" names grief; "wife" names spouse; "his little boy" names her young child). Do NOT pick an option you can merely INFER from evidence, causes, scene details, or narrative ("small sneakers" do not name "her young child"), do NOT count related-but-different states ("regret" is not grief; "worried" is not relieved) or generic words too broad to name it ("sad", "upset"), and do NOT use world knowledge or plausibility to fill gaps. If the readout does not state any listed option, choose the "cannot tell from the readout" option. Give a SHORT exact quote (a few words, VERBATIM) from the readouts that supports your choice.
```

**User**: `READOUTS:\n{r}\n\n{q}` where `{r}` is the per-layer-tagged concatenation (`[L20] … [L28] …`) and `{q}` is:

```text
Which of the following does the readout state?
  1. {mc_option}
  ...
  10. {mc_option}
  11. cannot tell from the readout
```

The 10 `mc_options` (whole-composition labels: gold + near-miss distractors) are seeded-shuffled; "cannot tell from the readout" is always appended last. Schema: `{choice: int, quote: str}`. Item PASS = judge picks `gold_label` (chance ~1/11). For J-lens, `--jlens-interp` runs the item-blind `summarize_token_bundles` on each per-layer bag first (same summarizer as the other families).
