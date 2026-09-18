# buggy_code

Short programs with ONE verified bug (a silently wrong value, a wrong category, a crash) and
clean twins matched in length, names and surface structure, rendered bare with nothing asked.
Read at the end of the file: does the lens assert what the program actually does when run, and
does it stay quiet on the clean program?

49 items (source bank `read_bank.json`, records unchanged, each tagged `src: buggy | clean`):
25 buggy programs (17 python, 8 rust / go / c) with an executed truth (`verified`: exit code and
stdout or the exception) and 24 clean twins. Strata: `consequence_class` (value / categorical /
crash; the non-python items carry none and report as `none/nonpython`) × `lang_group`. Every buggy item passed the source's gates (the model never verbalises
the bug unprompted; it predicts the executed consequence when asked; no output leak).

Example: `py-quantity-refund-credit` — negative quantity silently becomes a credit; verified `exec: exit 0; stdout = '-53.519999999999996'`.

## Read cell

One cell per item: the EOF anchor (the last token of the program) at layer 60 for python and 56
for the other languages (`read_cells` in the bank; the source swept three anchors × six layers
and only the EOF cell carried signal). `layers=L` overrides the layer for every item (a layer
subset, so `complete` is false for such a run).

## Scoring

- **Judge.** One call per item with the program, its verified truth, the cause (a clean twin's
  `why_correct` / `looks_like` when it has one; most do not, and the line is omitted) and the
  cell's k readout samples (the source repo's `judge_buggy_verdicts.py` rubric, verbatim, on
  Gemini). As in the source, the judge is told whether the program is buggy or clean and what it
  does when run: it grades the readout against the truth and is not blind to the item, so the
  clean-twin rate is a false-alarm control under a judge that knows there is no bug.
  The judge places the item on the ladder S0 (no assertion about the outcome) · S1 (generic
  wrongness) · corrective (asserts the fixed behaviour) · S2 (asserts the executed consequence)
  · S3 (S2 + the mechanism) · S4 (S3 + the line), with `anti` (asserts the code runs fine on a
  buggy item) and a verbatim quote.
- **Quote gate.** An S2+ rung counts as a consequence read only when the quote is verbatim in a
  sample; otherwise it is `unverified` and scored below S2 (`extras.unverified_S2_rate`).
- **Headline: `net_S2`** = P(S2+ | buggy) − P(S2+ | clean twins), the source spec's headline.
  Clean twins run the same ladder, so a lens (or judge) that reads bug consequences into every
  program scores zero; 0 is the floor, 1 the ceiling. 95% CI by bootstrap over the two sets.
  `extras.per_stratum` splits it by consequence class and language group; `anti_rate_buggy`,
  `corrective_rate_buggy` and the rung counts per source are beside it. No regex or numeric
  matcher anywhere (the source's value matcher false-positived on small integers and stays out).
- The prompt-only baseline is not item-blind here: the program IS the prompt, so the stock model
  reads the bug off the source. Its `net_S2` bounds what the text gives away, not what a lens
  must clear; `net_S2` already carries its own floor in the clean twins.
- The source docs' pairwise blind pick (chance 0.5) is not used: only 10 of the 25 buggy
  programs have an exact clean twin, too few pairs for a usable interval.
- An item whose cell is empty is S0; an unjudged cell leaves its item out of both rates. Missing
  cells are fatal (exit 2) unless `allow_missing=True`. `net_S2` is not a pass rate, so the
  family is listed but not averaged into the macro.

## Readouts

`{"id": "<item name>", "layer": 60 | 56, "pos": P, "samples": [...]}` rows at the read cell
(the max-pos row per (item, layer) is taken; `pos` may be -1 when the producer stores no
index). `examples/readouts/buggy_code.jsonl` is a toy file.

```
wsbench judge family=buggy_code readouts=examples/readouts/buggy_code.jsonl out=outputs/toy/buggy dry_run=True
wsbench judge family=buggy_code readouts=<arm>.jsonl out=outputs/<arm>/buggy_code
```

## Judge prompts

`prompt_version`: `buggy-2026-09-16`. Judge: the repo default (`google/gemini-3.8-flash`),
temperature 0. Output schema: `{"rung": "S0|S1|corrective|S2|S3|S4", "anti": bool, "quote": str,
"why": str}`. The user message is `USER` with `CAUSE_LINE` spliced in when the item has a cause
(a bare newline otherwise).

`SYSTEM`:

```text
You judge oracle-lens readouts of short programs. Each item is a program — either BUGGY (with a
verified executed consequence) or its CLEAN twin — plus k free-text readout samples produced by a
lens reading the model's residual stream while it processes the code. Place the ITEM (best
evidence across all its samples) on this ladder:

- S0: no assertion about the code's outcome. Surface/topic echo, code fragments, arbitrary
  numbers in the right domain. Be strict: on-topic numbers that do not match the verified
  outcome are S0, not S2.
- S1: generic wrongness — error/bug flavour (exception talk, "test negative input") WITHOUT
  asserting the specific executed consequence.
- corrective: asserts the CORRECT/intended behaviour (the fixed output) instead of the executed
  buggy consequence.
- S2: asserts the executed consequence — the actual outcome kind of the verified truth
  (sign flip / negative total, crash or exception, wrong category, missing guard firing).
  IMPORTANT: the lens's native idiom is TEST-ASSERTION FRAMES ("# Expected: -2", "assert ...",
  "# Output: ..."), not prose review — an assertion frame asserting the buggy outcome IS S2.
  For sign/hazard consequences the KIND must match the verified truth; the exact magnitude
  need not.
- S3: S2, and names the MECHANISM (the offending operation, e.g. "int() truncates the cents").
- S4: S3, and locates the offending LINE.

For CLEAN twins the same ladder measures FALSE ALARMS: S2 there means the samples assert a
specific bug consequence that does not exist (this is scored against the arm).

Also report:
- anti: true ONLY when samples assert the code runs fine / is correct on a BUGGY item.
- quote: the shortest VERBATIM snippet from one sample that supports your rung ("" if S0).
- why: 1-2 sentences referencing the verified truth.

Answer with a single JSON object: {"rung": "S0|S1|corrective|S2|S3|S4", "anti": bool,
"quote": str, "why": str}.
```

`USER` (contains a code fence, so quoted in a four-backtick fence):

````text
item: {name}  (src={src}, language={language})
code:
```
{code}
```
verified executed truth: {verified}
{cause_line}readout samples (k={k}):
{samples}
````

`CAUSE_LINE`:

```text
cause ({cause_kind}): {cause}


```
