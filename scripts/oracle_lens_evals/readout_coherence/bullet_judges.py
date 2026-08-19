"""readout_coherence passes 4+5 — bullet-relevance + bullet-diversity judges (Opus).

Two extra judge passes over an arm's GENERATED readouts, run AFTER judge.py pass 1
(they consume its ``sumtok_ao.jsonl``) and restricted to the union of summary-flagged
positions (``effective_level >= 2``) across the given arms:

- ``relevance`` (the *unrelatedness* judge): per bullet,
  ``relation in {continuation, context, tangential, unrelated}`` + ``hallucinated``
  w.r.t. the conversation context and the TRUE next tokens. Taxonomy verbatim from the
  LOO-lane ``ao_dict_judge`` (2026-08-15), applied to generated readouts instead of picks.
- ``diversity``: PAIRWISE topic relation between the bullets of one generated sample
  (``same_topic`` / ``related_aspects`` / ``different_concepts``) + ``n_distinct`` and
  ``diverse_aspects`` — do the bullets express different concepts or restate one thing?

Unit = one (arm, label, pos, layer) cell; all k samples of the cell are judged in ONE call
(per-sample verdicts). Only samples that parse as "- " bullet lists are judged — these
passes target bullet-format checkpoints; cells with no parseable bullets are skipped (and
counted in the log). Verdicts land in each arm's verdict dir as ``relevance.jsonl`` /
``diversity.jsonl``; score.py folds them (skip-if-missing) WITHOUT touching the frozen
overall-coherence weights. Resume-by-key; plain semaphore (default 256) with throttle retry.

Pass ``--arm`` once per checkpoint under comparison — the flagged-position UNION keeps every
arm judged at the same positions (single-arm runs are fine too):

    env PYTHONUNBUFFERED=1 uv run --no-sync python \
        scripts/oracle_lens_evals/readout_coherence/bullet_judges.py --stage all \
        --arm finals0:outputs/.../gen-rcoh-finals0:outputs/.../verdicts-finals0 \
        --arm loow30k:...:... \
        --acts outputs/oracle_lens_evals/olens_sglang/acts-rcoh
"""

import argparse
import asyncio
import json
import random
from pathlib import Path
from typing import Any

RELEVANCE_MODEL = "claude-opus-5"
DIVERSITY_MODEL = "claude-opus-5"
CTX_TOKENS = 1200
TRUE_NEXT = 64
FLAG_LEVEL = 2
PAIR_RELATIONS = ("same_topic", "related_aspects", "different_concepts")
RELATIONS = ("continuation", "context", "tangential", "unrelated")

RELEVANCE_SYSTEM = """\
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
"""

DIVERSITY_SYSTEM = """\
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
"""


def relevance_schema() -> dict[str, Any]:
    # no minItems/minimum — the structured-output endpoint 400s on some constraint keywords
    return {
        "type": "json_schema",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "samples": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "bullets": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "properties": {
                                        "relation": {"type": "string", "enum": list(RELATIONS)},
                                        "hallucinated": {"type": "boolean"},
                                    },
                                    "required": ["relation", "hallucinated"],
                                },
                            }
                        },
                        "required": ["bullets"],
                    },
                }
            },
            "required": ["samples"],
        },
    }


def diversity_schema() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "samples": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "pairs": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "properties": {
                                        "i": {"type": "integer"},
                                        "j": {"type": "integer"},
                                        "relation": {
                                            "type": "string",
                                            "enum": list(PAIR_RELATIONS),
                                        },
                                    },
                                    "required": ["i", "j", "relation"],
                                },
                            },
                            "n_distinct": {"type": "integer"},
                            "diverse_aspects": {"type": "boolean"},
                        },
                        "required": ["pairs", "n_distinct", "diverse_aspects"],
                    },
                }
            },
            "required": ["samples"],
        },
    }


def parse_bullets(text: str) -> list[str]:
    out: list[str] = []
    cur: str | None = None
    for ln in text.split("\n"):
        if ln.startswith("- "):
            if cur is not None:
                out.append(cur)
            cur = ln[2:]
        elif cur is not None:
            cur += "\n" + ln
    if cur is not None:
        out.append(cur)
    return [b.strip() for b in out if b.strip()]


def flagged_positions(verdict_dirs: list[Path]) -> dict[str, set[int]]:
    """Union of summary-flagged (label, pos) across the arms' sumtok_ao verdicts."""
    flags: dict[str, set[int]] = {}
    for vd in verdict_dirs:
        p = vd / "sumtok_ao.jsonl"
        if not p.exists():
            raise SystemExit(f"missing {p} — run the rcoh judge first")
        for ln in p.read_text().splitlines():
            try:
                row = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if not row.get("api_error") and row.get("effective_level", 0) >= FLAG_LEVEL:
                flags.setdefault(row["label"], set()).add(int(row["pos"]))
    return flags


def gen_cells(gen_dir: Path, label: str, positions: set[int]) -> dict[tuple[int, int], list[str]]:
    """{(layer, pos): samples} for flagged positions, from gen-rcoh L%03d.jsonl files."""
    cells: dict[tuple[int, int], list[str]] = {}
    for fp in sorted((gen_dir / label).glob("L[0-9][0-9][0-9].jsonl")):
        for ln in fp.read_text().splitlines():
            try:
                row = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if int(row["pos"]) in positions:
                cells[(int(row["layer"]), int(row["pos"]))] = row.get("samples", [])
    return cells


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--arm",
        action="append",
        required=True,
        help="name:gen_dir:verdict_dir (dirs relative to CWD or absolute); repeat",
    )
    ap.add_argument("--acts", required=True, help="acts bank dir (manifest.json + safetensors)")
    ap.add_argument("--stage", default="all", choices=["relevance", "diversity", "all"])
    ap.add_argument("--concurrency", type=int, default=256)
    ap.add_argument(
        "--limit-positions", type=int, default=0, help="cap flagged positions per label (0 = all)"
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from safetensors import safe_open
    from transformers import AutoTokenizer

    arms: list[tuple[str, Path, Path]] = []
    for spec in args.arm:
        try:
            name, gd, vd = spec.split(":")
        except ValueError:
            raise SystemExit(f"--arm must be name:gen_dir:verdict_dir, got {spec!r}") from None
        arms.append((name, Path(gd), Path(vd)))

    acts = Path(args.acts)
    manifest = json.loads((acts / "manifest.json").read_text())
    conv_file = {e["label"]: e["file"] for e in manifest["prompts"]}
    tok = AutoTokenizer.from_pretrained(manifest["model_id"])

    flags = flagged_positions([vd for _, _, vd in arms])
    if args.limit_positions:
        rng = random.Random(args.seed)
        flags = {
            lb: set(rng.sample(sorted(ps), min(args.limit_positions, len(ps))))
            for lb, ps in flags.items()
        }
    n_flag = sum(len(v) for v in flags.values())
    print(
        f"[bullet-judges] {n_flag} summary-flagged positions across {len(flags)} convs "
        f"(union over {len(arms)} arms)",
        flush=True,
    )

    ids_cache: dict[str, list[int]] = {}

    def conv_ids(label: str) -> list[int]:
        if label not in ids_cache:
            with safe_open(str(acts / conv_file[label]), framework="pt", device="cpu") as f:
                ids_cache[label] = f.get_tensor("input_ids").tolist()
        return ids_cache[label]

    def context_block(ids: list[int], p: int) -> str:
        start = max(0, p + 1 - CTX_TOKENS)
        head = "…[earlier context truncated]…\n" if start > 0 else ""
        body = tok.decode(ids[start:p], skip_special_tokens=False) if p > start else ""
        marked = tok.decode([ids[p]], skip_special_tokens=False)
        return f"{head}{body}«{marked}»"

    stages = ["relevance", "diversity"] if args.stage == "all" else [args.stage]
    for stage in stages:
        system = RELEVANCE_SYSTEM if stage == "relevance" else DIVERSITY_SYSTEM
        sch = relevance_schema() if stage == "relevance" else diversity_schema()
        model = RELEVANCE_MODEL if stage == "relevance" else DIVERSITY_MODEL
        for name, gen_dir, vd in arms:
            out_path = vd / f"{stage}.jsonl"
            done: set[str] = set()
            if out_path.exists():
                for ln in out_path.read_text().splitlines():
                    try:
                        row = json.loads(ln)
                    except json.JSONDecodeError:
                        continue
                    if not row.get("api_error"):
                        done.add(row["key"])
            jobs: list[dict[str, Any]] = []
            n_skipped = 0
            for label, positions in sorted(flags.items()):
                for (layer, pos), samples in sorted(gen_cells(gen_dir, label, positions).items()):
                    key = f"{name}|{label}|L{layer}|p{pos}"
                    if key in done:
                        continue
                    all_bullets = [parse_bullets(s) for s in samples]
                    # judge ONLY samples that parsed to bullets: an empty SAMPLE block still
                    # draws a schema-required verdict, and score.py's per-sample folds would
                    # count it (n_distinct=0 for a prose sample would drag the diversity
                    # headline). The stored `bullets` list holds exactly the judged samples,
                    # so verdict samples and bullets stay index-aligned.
                    bullets = [bl for bl in all_bullets if bl]
                    if not bullets:
                        n_skipped += 1  # non-bullet readout (e.g. continuation_raw arm)
                        continue
                    ids = conv_ids(label)
                    parts = [f"CONTEXT (ends at the marked token):\n{context_block(ids, pos)}"]
                    if stage == "relevance":
                        nxt = tok.decode(
                            ids[pos + 1 : pos + 1 + TRUE_NEXT], skip_special_tokens=False
                        )
                        parts.append(f"TRUE NEXT TOKENS:\n{nxt}")
                    for si, bl in enumerate(bullets):
                        joined = "\n".join(f"  {bi}. {b}" for bi, b in enumerate(bl))
                        parts.append(f"SAMPLE {si + 1} bullets (0-based indices):\n{joined}")
                    parts.append("Return the JSON verdict (one entry per sample, in order).")
                    jobs.append(
                        {
                            "key": key,
                            "arm": name,
                            "label": label,
                            "layer": layer,
                            "pos": pos,
                            "bullets": bullets,
                            "n_samples_skipped": len(all_bullets) - len(bullets),
                            "user": "\n\n".join(parts),
                        }
                    )
            print(
                f"[bullet-judges] {stage}/{name}: {len(jobs)} calls ({len(done)} done, "
                f"{n_skipped} cells skipped — no parseable bullets)",
                flush=True,
            )
            if jobs:
                asyncio.run(_run(jobs, out_path, system, sch, model, args.concurrency))


async def _run(
    jobs: list[dict[str, Any]],
    out_path: Path,
    system: str,
    sch: dict[str, Any],
    model: str,
    concurrency: int,
) -> None:
    from anthropic import AsyncAnthropic

    client = AsyncAnthropic(timeout=300.0, max_retries=1)
    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fh = out_path.open("a")
    n_done = 0

    async def one(job: dict[str, Any]) -> None:
        nonlocal n_done
        verdict = None
        err = None
        for attempt in range(6):
            async with sem:
                try:
                    resp = await client.messages.create(
                        model=model,
                        max_tokens=8000,
                        system=system,
                        messages=[{"role": "user", "content": job["user"]}],
                        output_config={"format": sch},
                    )
                    if resp.stop_reason in ("refusal", "max_tokens"):
                        err = f"stop:{resp.stop_reason}"
                        break
                    verdict = json.loads(next(b.text for b in resp.content if b.type == "text"))
                    break
                except Exception as e:  # throttle/timeout -> backoff retry
                    err = f"{type(e).__name__}"
                    low = str(e).lower()
                    if not any(t in low for t in ("429", "529", "rate", "overload", "timeout")):
                        break
            await asyncio.sleep(2**attempt)
        row = {
            k: job[k]
            for k in ("key", "arm", "label", "layer", "pos", "bullets", "n_samples_skipped")
        }
        if verdict is None:
            row.update({"api_error": True, "error": err})
        else:
            row.update({"api_error": False, "verdict": verdict})
        async with lock:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            n_done += 1
            if n_done % 100 == 0:
                print(f"[bullet-judges] {n_done}/{len(jobs)}", flush=True)

    await asyncio.gather(*(one(j) for j in jobs))
    fh.close()
    print(f"[bullet-judges] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
