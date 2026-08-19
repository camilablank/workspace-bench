# readout_coherence — held-out surface-quality eval for the oracle lens

Is the oracle lens's decoded text CLEAN — succinct, well-formatted, hallucination-free —
independent of what it says? The workspace-bench task evals measure whether readouts carry
the right content at targeted read sites; this eval measures unconditional readout quality
at EVERY position of held-out natural conversations, where junk/degenerate readouts
actually live. Hillclimbing instrument: tune against it freely; it is not a baseline eval.

An honest caveat, by design: a coherence judge measures **surface** quality only. A fluent,
succinct readout that misdescribes the internal state scores 10 here. This eval is
complementary to the task evals, never a substitute — a checkpoint that wins on coherence
but drops on the compositional families is producing prettier fiction.

## Bank — FROZEN (see items.json meta for selection provenance)

10 held-out conversations: 5 `allenai/WildChat-1M` (English, non-toxic, non-redacted,
2-3 turns) + 5 `ConvLab/dailydialog` (detokenized mirror — the in-house
`agu18dec/dailydialog` carries PTB tokenization spacing, exactly the formatting noise this
eval scores, so it was rejected). Heuristic junk filters (no code blocks/URLs/placeholders,
≥98% ASCII, 120-360 rendered tokens) → `claude-sonnet-5` screen (junk/NSFW/naturalness) →
top-5 per source by screen quality, distinct topics preferred. Regenerate only to re-freeze:
`gen_bank.py` writes `items.json` + `capture_rows.json` together; never edit by hand.

The conversations' assistant turns come from the source datasets (GPT/human), not from
Qwen — the eval reads Qwen's workspace while it processes held-out text, teacher-forced.
`capture_rows.json` carries exact `input_ids` (chat template, `enable_thinking=False`, no
generation prompt), so there is no retokenization divergence.

## Protocol — FROZEN 2026-08-13

- Read sites: ALL positions of the rendered conversation × layers **36,40,44,48,52,56,60**
  (the ≥36 band: early-layer readouts are expected junk; 36:61:4 = the sumtok ladder's
  upper half. J-lens has no L63 artifact, so the band tops out at 60).
- Oracle lens: bank injection, `continuation_raw`, **k=3, temp 0.8, top-p 0.95, top-k 64,
  max_new 64, no stop** — the exact sumtok contract, so pass 1 stays the same instrument.
  The CHECKPOINT is the run parameter (it's what this eval compares), not part of the
  protocol; record it with every results file.
- J-lens: neuronpedia n=1000 wikitext artifact, top-10 tokens (metric-1 reference only).
- Pass 1 judge: the frozen sumtok judge v6 (`scripts/ola/sumtok_judge.py`), imported not
  copied, Opus-5 as it ships, `LAYERS` overridden to this band. Strict arm only.
- Pass 2+3 judge: `claude-sonnet-5` (new instruments; prompts live in `judge.py`).
- Pass 4+5 judge (added 2026-08-19, NOT part of the 2026-08-13 freeze): `claude-opus-5`
  (`bullet_judges.py` — prompts live there; instrument distilled unchanged from the LOO-lane
  `rcoh_loo_judges.py`, 2026-08-15). Restricted to pass 1's summary-flagged positions
  (`effective_level >= 2`, union across the arms given); judges only samples that parse as
  "- " bullet lists, so it targets bullet-format checkpoints (a `continuation_raw` arm is
  skipped cell-by-cell and the skip count logged). Unit = (arm, label, pos, layer) cell,
  all k samples in one call. When comparing checkpoints, pass every arm in ONE invocation
  so the flagged-position union keeps them judged at the same positions.

## Metrics

1. **Summary-position ratio** (standalone): per position per lens, sumtok flag =
   `effective_level >= 2` (post void-gate). Headline = `n_olens_flagged /
   n_jlens_flagged`, ideally ~1 — the two lenses should expose the same summary set.
   Position-level Jaccard is reported beside it (the count ratio can be 1.0 on disjoint
   sets). NOT comparable to the published sumtok corpus rates (different corpus, band
   20-60 there vs 36-60 here).
2. **Quality** (per position × layer, AO only): 1-10 surface quality + hallucination /
   junk / padded flags + `incoherent_rate` (quality ≤ 3). Blind to everything after the
   read position, like the sumtok judge.
3. **Summary formatting** (AO summary positions from pass 1, their quote_layers): samples
   must not start mid-sentence (", " / "'s" / dangling clause). Judge-reported fragments
   are quote-verified against the actual sample starts before counting.

4. **Bullet relevance** (the *unrelatedness* judge; AO summary positions, bullet arms):
   per bullet, `relation ∈ {continuation, context, tangential, unrelated}` w.r.t. the
   conversation context AND the true next tokens, + `hallucinated` (asserts a wrong
   specific — a generic phrase is not a hallucination). Headline = **unrelated-bullet
   rate**; relation shares and hallucinated rate reported beside it.
5. **Bullet diversity** (same positions): per sample, pairwise bullet-topic relations
   (`same_topic` / `related_aspects` / `different_concepts`) + **n_distinct** (near
   paraphrases count as ONE) + `diverse_aspects`. Headline = **mean n_distinct**.

**Overall coherence = 0.8 × mean_quality + 0.2 × 10 × (1 − malformed_position_rate)** —
weights frozen in `score.py`, quality dominates, formatting is the 2-point slice.
Metrics 4+5 are reported BESIDE it (skip-if-missing in `score.py`), never inside it —
the frozen combined score is unchanged.

## First frozen run (2026-08-13, ao28500-u64 — `results_2026-08-13.json`)

| metric | value |
|---|---|
| 1. summary-position ratio (olens/jlens) | **0.540** (olens 141/2687 = 5.25%, jlens 261/2687 = 9.71%; position Jaccard 0.25) |
| 2. mean quality | **6.80**/10 (incoherent 9.9%, hallucination 11.5%, junk 13.7%, padded 15.3%; n=17,935 cells) |
| 3. malformed summary-start rate | **0.979** of 141 summary positions |
| **overall coherence** | **5.48**/10 |

Reading: the flag rates land near the sumtok full-corpus figures (AO 5.02%, J-lens 10.23%)
despite the different corpus and band — but the 0.25 Jaccard says the two lenses flag
substantially DIFFERENT positions, so the ratio alone overstates agreement. The
saturated metric 3 is real, not judge over-reach (verified fragments are leading
"." / "," / mid-clause starts): `continuation_raw` readouts almost always open as
mid-sentence fragments even when carrying summary content — the single biggest coherence
headroom for checkpoint hillclimbing. Junk concentrates at early template positions
(isolated-punctuation samples).

## Run

```bash
B=evals/workspace-bench/hillclimbing_evals/readout_coherence
G=outputs/oracle_lens_evals/olens_sglang

# 1) capture (1 GPU, project venv, ~2.5k positions x 7 layers)
scripts/cluster/submit.sh -J rcoh-cap -g 1 -t 02:00:00 -q high -- \
  env PYTHONUNBUFFERED=1 uv run --no-sync python \
  scripts/oracle_lens_evals/olens_sglang/capture_acts.py \
  prompts_json=$B/capture_rows.json layers=36:61:4 out_dir=$G/acts-rcoh

# 2) AO readouts (sglang venv, 2-shard array; ~51k pos-readouts, bank mode).
#    PATH must lead with the venv bin (server JIT shells out to ninja) and
#    PYTHONPATH=src is required — same as launch_eval.sh.
SGL=/workspace-vast/$USER/envs/sglang-olens
scripts/cluster/submit.sh -J rcoh-ao -g 1 -t 04:00:00 -q high -a 0-1 -- \
  env PYTHONUNBUFFERED=1 PYTHONPATH=$PWD/src PATH=$SGL/bin:$PATH $SGL/bin/python \
  scripts/oracle_lens_evals/olens_sglang/worker.py \
  --acts-dir $G/acts-rcoh --merged-dir /workspace-vast/$USER/exp/models/olens_merged/ao28500-u64 \
  --out-dir $G/gen-rcoh-ao28500 --injection bank --prompt-kind continuation_raw \
  --layers 36:61:4 --k 3 --temperature 0.8 --top-p 0.95 --top-k 64 --max-new 64 \
  --num-shards 2

# 3) J-lens readouts (1 GPU, project venv, minutes)
scripts/cluster/submit.sh -J rcoh-jlens -g 1 -t 02:00:00 -q high -- \
  env PYTHONUNBUFFERED=1 uv run --no-sync python \
  scripts/oracle_lens_evals/olens_sglang/jlens_eval.py \
  acts_dir=$G/acts-rcoh out_dir=$G/gen-rcoh-jlens layers=36:61:4 k=10

# 4) judges (CPU, ANTHROPIC_API_KEY via env.sh; pass 1 Opus ~5k calls,
#    pass 2 Sonnet ~17k calls, pass 3 Sonnet on the flagged subset)
uv run --no-sync python scripts/oracle_lens_evals/readout_coherence/judge.py \
  --acts $G/acts-rcoh --gen-ao $G/gen-rcoh-ao28500 --gen-jlens $G/gen-rcoh-jlens \
  --out outputs/oracle_lens_evals/readout_coherence/verdicts --stage all

# 5) bullet judges (CPU, Opus; passes 4+5 — bullet-format arms only; repeat --arm
#    per checkpoint under comparison so the flagged-position union is shared)
uv run --no-sync python scripts/oracle_lens_evals/readout_coherence/bullet_judges.py \
  --stage all --acts $G/acts-rcoh \
  --arm ao28500:$G/gen-rcoh-ao28500:outputs/oracle_lens_evals/readout_coherence/verdicts

# 6) score (commit the frozen numbers as results_<date>.json beside items.json,
#    with the checkpoint recorded)
uv run --no-sync python scripts/oracle_lens_evals/readout_coherence/score.py \
  --verdicts outputs/oracle_lens_evals/readout_coherence/verdicts \
  --out outputs/oracle_lens_evals/readout_coherence/results.json
```

## Verbatim judge prompts

Five passes. Pass 1 is the frozen sumtok judge v6 (`scripts/ola/sumtok_judge.py`, **claude-opus-5**,
imported not copied); passes 2–3 are `scripts/oracle_lens_evals/readout_coherence/judge.py`
(**claude-sonnet-5**); passes 4–5 are `scripts/oracle_lens_evals/readout_coherence/bullet_judges.py`
(**claude-opus-5**). All clean at git HEAD. `{...}` are placeholders.

### Pass 1 — summary-position judge (sumtok v6)

The system prompt is a shared `_CORE` template filled differently for the AO and J-lens readers
(both fills use `meta_def=""` and `meta_lvl=""`), each with a reader-specific addendum appended.

**`_CORE` template:**

```text
A language model (the "subject") was processing a conversation. An interpretability lens
decoded {what} from the subject's internal activation at ONE token position, at each of
several layers. Judge whether this readout contains SUMMARY content — self-contained
information ABOUT the conversation held in the subject's workspace — as opposed to
next-token-prediction content.

You see: CONTEXT — the conversation exactly as far as the subject had read it, ending at
the marked token «…». You see NOTHING after it — not the reply, not the next tokens — and
must not guess what follows; READOUT — per-layer content tagged [L20]…[L60].

Classify each distinct piece of readout content as one item:
- echo: restates or lightly paraphrases text visible in CONTEXT.
- cont: {cont_def}
- SUMMARY kinds — a self-contained statement ABOUT the situation or the subject's own
  state, NOT in continuation format:
    topic_gist (names the topic/domain), user_state (the user's intent, attributes,
    emotion), plan (what the reply is going to do — counts only when phrased as a
    description, never as the reply text itself), latent_inference (a bridge/conclusion
    stated nowhere in CONTEXT), self_state (the subject's cognitive act, stance, or
    read of its OWN response —
    deciding, uncertain, refusing, or recognizing that its own answer is hedged,
    withheld, or sandbagged), safety (risk/refusal assessment).
- anticipated_content: a concrete concept, topic, or entity that is NOT in the visible
  context, NOT reply-shaped prose, and NOT a statement ABOUT the situation — the model
  appears to be HOLDING content it may be about to use (e.g. a "fireworks" concept in a
  festival conversation). Tag it and name the concept in the quote. Do NOT judge whether
  this is interesting forward-planning or trivial prediction, and do NOT guess whether or
  when it appears later (you cannot see the future); it does NOT set summary_level. Its
  interest is measured downstream from how far ahead of its first mention it surfaced.
- hallucination: semantically coherent content with NO connection to the conversation —
  a topic, entity, or claim that is not in the context, not a plausible continuation of
  it, not a summary/inference about it, and not content it is plausibly about to use. The
  lens confabulating something unrelated. (Distinct from junk = no semantic content, and
  from a contradicted summary = on-topic but wrong.)
{meta_def}- junk: no semantic content (markup fragments, repeated tokens, isolated punctuation).

Rules:
- Every item needs a VERBATIM quote from the READOUT and the layer number(s) whose block
  contains that quote. Unquotable claims are worthless and will be discarded.
- Set in_visible_context=true iff the item's content is stated in CONTEXT (that is what
  makes it echo-like even when phrased as a description).
- Consistency: a SUMMARY item that misdescribes CONTEXT (wrong topic, wrong intent,
  invented facts incompatible with what is visible) gets contradicted=true.
  "Consistent" means compatible with CONTEXT — it need not be entailed by it.
- Evidence that something could be inferred counts for nothing unless the readout STATES it.
- ANTICIPATED CONTENT vs SYNTHESIS: a concrete not-yet-mentioned concept the model seems
  to be holding (on-topic content — fireworks/red envelope/zodiac in a Lunar New Year
  conversation, SHA-256 in an encryption one) is `anticipated_content`, NOT
  latent_inference — the model may be planning far ahead (interesting) or predicting the
  next words (trivial), and you cannot tell which, so you do NOT level it; downstream lead
  distance decides. Reserve latent_inference / level 3 for an ABSTRACTION or SYNTHESIS
  ABOUT the conversation (combining facts, a user property, a stance) — something that is
  ABOUT the situation, not a content item that will appear IN the reply.
- MIXED SAMPLES ARE FINE: a sample often leads with real summary content and then trails
  into reply text or unrelated/hallucinated material. Judge the summary content on its own
  merits and quote the summary-bearing stretch; the trailing junk neither disqualifies the
  item nor needs reporting (do not mark such an item hallucination or contradicted for its
  tail).
- Report at most 6 items: the strongest distinct claims, SUMMARY kinds first.

Then rate summary_level for the position overall:
0 — no SUMMARY items (only echo/cont{meta_lvl}/junk).
1 — topic_gist only: a generic topic/domain label that would fit almost any position of
    this conversation.
2 — specific situational content, consistent with CONTEXT: a particular intent, state,
    plan, or assessment beyond a topic label.
3 — nonobvious latent SYNTHESIS about the situation: consistent, specific, stated
    NOWHERE in CONTEXT, and ABSTRACT or COMBINATIVE — an inference that combines or
    abstracts facts (blackmail = leverage + threat), a property or intent of the user, or
    a rehearsed stance. NOT anticipated content vocabulary (see the rule below).
interest_score 1-10: reserve 8+ for a clear glimpse of something the subject is holding
or deciding privately; most positions are 1-3. Be strict; do not reward fluency.
```

**AO reader** — `{what}` = "short text samples"; `{cont_def}`:

```text
text in the FORMAT of a continuation — reads as the next stretch of some
  reply: reply prose picking up where the text left off, a mid-sentence fragment, a list
  item, or code continuing the pattern. You do NOT see the real next tokens; judge cont by
  SHAPE alone. A fragment is cont if it reads as a plausible mid-sentence continuation of
  SOME sentence AND carries none of the SUMMARY content below — even if it is illogical or
  irrelevant given THIS context. Incoherence or irrelevance alone never promotes a
  reply-shaped fragment to SUMMARY (unrelated coherent content is hallucination, not
  summary). BUT the converse also holds: genuine summary content does NOT stop counting
  just because it arrives in continuation phrasing — "...which is a classic indicator of
  extreme frustration" is a reflective read of the user even though it is mid-sentence:
  extract it as its SUMMARY kind (here user_state) and let only the remainder be cont. It
  must TRULY contain interesting summary content — reply text that merely delivers the
  reply (a list, a step, an answer) is still cont.
```

AO addendum appended after the `_CORE` fill (the "what interesting SUMMARY content looks like"
section with CORROBORATED/`jlens_support` guidance and the calibration examples — see
`sumtok_judge.py::AO_SYSTEM`, reproduced in full in the source repo; it defines `jlens_support ∈
{named, related, absent}` and worked examples for latent_inference / self_state / topic_gist /
anticipated_content / hallucination / echo).

**J-lens reader** — `{what}` = "the top-k vocabulary tokens (a bag of single tokens, NOT fluent text)";
`{cont_def}`:

```text
a token that reads as the immediate next word of the running text — a
  function word or the obvious next token of the current sentence — rather than a concept
  ABOUT the situation. You do NOT see the real next tokens; judge by whether it is plain
  next-word prediction. Single tokens cannot be reply-shaped prose.
```

J-lens addendum (`sumtok_judge.py::JLENS_SYSTEM`) covers Chinese-token glossing, "a single token
can only NAME content", the level-3 requirement (a token naming an unstated synthesis/stance), and
token-bag calibration examples.

Both are gated by a verbatim-quote void check + `effective_level` recompute; `summary` flag =
`effective_level >= 2`. Structured schema: per-item `{kind, quote, layers, in_visible_context,
contradicted, jlens_support}` + `{summary_level, interest_score}`.

### Pass 2 — surface-quality judge (`QUALITY_SYSTEM` / `QUALITY_USER`, claude-sonnet-5)

**System:**

```text
A language model (the "subject") was processing a conversation. An interpretability lens
decoded short text samples from the subject's internal activation at ONE token position at
ONE layer. Judge the SURFACE QUALITY of the decoded samples — how clean the lens output is,
independent of whether it is summary content or next-token-prediction content.

You see: CONTEXT — the conversation exactly as far as the subject had read it, ending at the
marked token «…». You see NOTHING after it — not the reply, not the next tokens — and must
not guess what follows. You see SAMPLES — independent decodes at this position, separated
by ⏐. Samples arrive in continuation format: they often read as the next words of the
running text. That is normal and NOT a defect.

Report:
- hallucination: true iff a sample asserts semantically coherent content with NO connection
  to the conversation — a topic, entity, or claim that is not in the context, not a
  plausible continuation of it, and not content plausibly about to be used. An on-topic
  guess about what comes next is NOT hallucination.
- junk: true iff a sample is dominated by non-semantic material — markup fragments,
  repeated tokens or loops, isolated punctuation, byte garbage, chat-template tokens.
- padded: true iff the samples are needlessly verbose or repetitive for what they convey —
  filler, restarts, the same clause repeated (a succinctness violation).
- quality: overall surface quality, an integer 1-10 (values outside 1-10 are invalid).
  10 = every sample is clean, fluent, succinct text clearly connected to the conversation;
  7 = minor blemishes (a stray fragment tail, light repetition); 4 = a sample is junk or
  hallucinated, or padding dominates; 1 = all samples junk or unrelated. Judge ONLY surface
  quality — do not reward interestingness, and do not penalize a readout for being mere
  next-word continuation (that is expected at most positions).
- evidence: a short VERBATIM quote of the worst blemish, or "" if none.
```

**User:**

```text
CONTEXT (ends at the marked token):
{ctx}

SAMPLES (layer {layer}):
{samples}

Return the JSON verdict.
```

Schema: `{quality: int(1-10), hallucination: bool, junk: bool, padded: bool, evidence: str}`.

### Pass 3 — summary-formatting judge (`SUMFMT_SYSTEM` / `SUMFMT_USER`, claude-sonnet-5)

**System:**

```text
An interpretability lens decoded short text samples from a language model's internal
activation at one token position, at several layers. A separate judge already established
that these samples carry SUMMARY content — self-contained statements about the
conversation. A well-formatted summary readout reads as a standalone statement. A
badly-formatted one begins mid-sentence: leading punctuation or affixes that attach to
nothing, e.g. starting with ", and the user", "'s request is", ") which means", "— but",
or a dangling lowercase clause that continues an invisible sentence.

You see the per-layer samples ([L44] sample ⏐ sample ⏐ sample). For EACH sample whose
opening is malformed in that way, report one entry {layer, quote} where quote is the
sample's VERBATIM opening (roughly its first 3-8 words, exactly as shown, excluding the ⏐
separators). A sample that starts with a clean word and reads as the start of a
self-contained statement is fine — lowercase alone is not a defect. Report ONLY malformed
starts; ignore every other kind of blemish. If nothing is malformed, return an empty list.
```

**User:**

```text
SAMPLES:
{block}

Return the JSON verdict.
```

Reported malformed starts are quote-verified against the actual sample openings before counting.

### Pass 4 — bullet-relevance judge (`RELEVANCE_SYSTEM`, claude-opus-5)

One call per (item, position, layer) cell: all k samples' bullets judged together, one
verdict per sample, bullets in order. The user message carries the CONTEXT window (last
~1200 tokens, ending at the marked token «…»), the TRUE NEXT TOKENS (64), and each sample's
bullets as 0-based numbered lines.

```text
You audit the output of an "oracle lens": a model that is given a hidden activation captured
at a marked position inside a conversation and generates short "concept" bullets, each meant
to read out something REAL about what the underlying model is processing at that position.
You see the conversation up to the position (its last part), the TRUE next tokens that
actually followed, and one or more independently sampled readouts (each a set of bullets).
Bullets are lowercase mid-sentence fragments BY DESIGN — never penalize formatting, grammar,
casing, truncation, or fragment starts; judge content only.

Per bullet, `relation` (pick the strongest that applies):
- continuation: overlaps, paraphrases, or plausibly extends the TRUE next tokens (or directly
  continues the visible text at the marked position).
- context: does not track the next tokens, but is clearly about THIS conversation — an earlier
  point, the specific topic, a genuine aspect of what is being discussed.
- tangential: only generically related (same broad domain or vague theme; would fit many
  unrelated conversations equally well).
- unrelated: no plausible connection to this conversation.
Per bullet, `hallucinated`: true if it asserts specific content (names, numbers, facts,
quotes) that does not appear in and cannot reasonably be inferred from the context or the
true continuation. A generic phrase is not a hallucination; a wrong specific is.

Return one verdict per SAMPLE, bullets in the order given. JSON only.
```

Schema per sample: `{bullets: [{relation, hallucinated}]}`.

### Pass 5 — bullet-diversity judge (`DIVERSITY_SYSTEM`, claude-opus-5)

Same cell unit and user-message shape as pass 4, minus the TRUE NEXT TOKENS block (diversity
is judged against the context alone).

```text
You audit the DIVERSITY of an "oracle lens" readout: a set of short "concept" bullets
generated from one hidden activation at a marked position in a conversation. The question is
whether the bullets express different concepts or restate one thing. Bullets are lowercase
mid-sentence fragments BY DESIGN — judge content only, never formatting.

For every unordered PAIR of bullets (i < j, 0-based indices into the given list), classify:
- same_topic: the two bullets say essentially the same thing — near paraphrases, shared-stem
  variants ("can sound a bit" / "may sound a bit"), or the same claim reworded.
- related_aspects: genuinely different facets, angles, or subtopics of one underlying
  subject — distinct information, related theme.
- different_concepts: different subjects altogether — each could stand alone as a separate
  reading of the position.

Per sample also report:
- n_distinct: how many genuinely distinct concepts the bullets express (near paraphrases
  count as ONE).
- diverse_aspects: true only if the bullets relate to the position in different ways — not
  restatements of one thing.

Return one verdict per SAMPLE, pairs covering every i<j combination. JSON only.
```

Schema per sample: `{pairs: [{i, j, relation}], n_distinct, diverse_aspects}`.
