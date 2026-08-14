#!/usr/bin/env python3
"""readout_coherence judge — three passes over the all-positions readouts.

Pass 1 ``sumtok``  : the EXACT frozen sumtok judge (v6 prompts, void gates, verdict-row
                     format) on both lenses, restricted to this eval's layer band. Imported
                     from scripts/ola/sumtok_judge.py — never copied — with ``LAYERS``
                     overridden to the band the readouts were generated at (36:61:4; the
                     judge renders "(no readout)" for absent layers, so the module default
                     of 20:61:4 would dilute every call with 4 empty lines).
Pass 2 ``quality`` : NEW instrument (Sonnet 5): per (position, layer) surface quality of
                     the AO samples — hallucination / junk / padding flags + 1-10 score.
Pass 3 ``sumfmt``  : NEW instrument (Sonnet 5): on positions pass 1 flagged as summary
                     (effective_level >= 2, AO), do the summary-bearing layers' samples
                     start as mid-sentence fragments (", ", "'s", dangling clause)?
                     Quote-verified like the sumtok void gate.

Passes 2 and 3 fold into the combined coherence score in score.py; pass 1 stands alone as
the summary-position ratio (olens flagged / jlens flagged).

Usage (CPU, project venv, ANTHROPIC_API_KEY via env.sh):
  uv run --no-sync python scripts/oracle_lens_evals/readout_coherence/judge.py \
      --acts outputs/oracle_lens_evals/olens_sglang/acts-rcoh \
      --gen-ao outputs/oracle_lens_evals/olens_sglang/gen-rcoh-ao28500 \
      --gen-jlens outputs/oracle_lens_evals/olens_sglang/gen-rcoh-jlens \
      --out outputs/oracle_lens_evals/readout_coherence/verdicts \
      --stage all
"""

import argparse
import asyncio
import importlib.util
import json
from pathlib import Path
from typing import Any, cast

from global_workspace.judges.llm_client import schema_block

REPO = Path(__file__).resolve().parents[3]
LAYERS = tuple(range(36, 61, 4))  # 36,40,44,48,52,56,60 — the >=36 band (early layers junky)

_spec = importlib.util.spec_from_file_location("sumtok_judge", REPO / "scripts/ola/sumtok_judge.py")
assert _spec is not None and _spec.loader is not None
sj: Any = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sj)
sj.LAYERS = LAYERS  # judge exactly the generated band (see module docstring)

QUALITY_MODEL = "claude-sonnet-5"

QUALITY_SYSTEM = """\
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
"""

QUALITY_USER = """\
CONTEXT (ends at the marked token):
{ctx}

SAMPLES (layer {layer}):
{samples}

Return the JSON verdict."""


def quality_schema() -> dict[str, Any]:
    # integer minimum/maximum unsupported by structured outputs — range lives in the
    # prompt, clamped on read (same workaround as sumtok_judge / gen_bank)
    return schema_block("rcoh_quality", {
        "quality": {"type": "integer", "description": "1-10 overall surface quality"},
        "hallucination": {"type": "boolean"},
        "junk": {"type": "boolean"},
        "padded": {"type": "boolean"},
        "evidence": {"type": "string"},
    }, ["quality", "hallucination", "junk", "padded", "evidence"])


SUMFMT_SYSTEM = """\
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
"""

SUMFMT_USER = """\
SAMPLES:
{block}

Return the JSON verdict."""


def sumfmt_schema() -> dict[str, Any]:
    return schema_block("rcoh_sumfmt", {
        "fragments": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "properties": {"layer": {"type": "integer"}, "quote": {"type": "string"}},
            "required": ["layer", "quote"],
        }},
        "rationale": {"type": "string"},
    }, ["fragments", "rationale"])


# ---------------------------------------------------------------------------
# shared IO

def load_manifest(acts_dir: Path) -> list[dict[str, Any]]:
    m = json.loads((acts_dir / "manifest.json").read_text())
    return list(m["prompts"])


def done_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    keys = set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not row.get("api_error"):
            keys.add(row["key"])
    return keys


async def call_with_retry(client: Any, system: str, user: str, sch: dict[str, Any],
                          model: str, attempts: int = 6) -> dict[str, Any] | None:
    for attempt in range(attempts):
        verdict, err = await sj.one_call(client, system, user, sch, model)
        if err is None:
            return cast(dict[str, Any] | None, verdict)
        if err == "throttle":
            await asyncio.sleep(min(60.0, 2.0 * 2**attempt))
        elif attempt >= 1:  # non-throttle errors get one retry, then give up
            return None
    return None


async def pump(jobs: list[tuple[str, str, str, dict[str, Any], str, dict[str, Any]]],
               out_path: Path, concurrency: int) -> None:
    """jobs: (key, system, user, schema, model, row_stub). Appends one JSON line per job."""
    if not jobs:
        return
    from anthropic import AsyncAnthropic  # heavy import, deferred

    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    n_done = 0
    async with AsyncAnthropic(timeout=240.0, max_retries=0) as client:
        async def one(job: tuple[str, str, str, dict[str, Any], str, dict[str, Any]]) -> None:
            nonlocal n_done
            key, system, user, sch, model, stub = job
            async with sem:
                verdict = await call_with_retry(client, system, user, sch, model)
            row = {"key": key, **stub}
            if verdict is None:
                row["api_error"] = True
            else:
                row.update(stub["_finish"](verdict))
                row["api_error"] = False
            row.pop("_finish", None)
            async with lock:
                with out_path.open("a") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_done += 1
                if n_done % 200 == 0 or n_done == len(jobs):
                    print(f"[{out_path.name}] {n_done}/{len(jobs)}", flush=True)

        await asyncio.gather(*(one(j) for j in jobs))


def _norm_start(text: str) -> str:
    return " ".join(text.replace("⏐", " ").split()).lower()


# ---------------------------------------------------------------------------
# passes

def build_sumtok_jobs(entries: list[dict[str, Any]], acts_dir: Path, gen_ao: Path,
                      gen_jlens: Path, out_dir: Path, tok: Any, model: str,
                      ) -> dict[Path, list[Any]]:
    jobs: dict[Path, list[Any]] = {out_dir / "sumtok_ao.jsonl": [],
                                   out_dir / "sumtok_jlens.jsonl": []}
    schemas = {"ao": sj.schema(True), "jlens": sj.schema(False)}
    systems = {"ao": sj.AO_SYSTEM, "jlens": sj.JLENS_SYSTEM}
    done = {p: done_keys(p) for p in jobs}
    for e in entries:
        label = e["label"]
        conv = sj.load_conv(acts_dir, label, e["file"])
        rows = {"ao": sj.gen_rows(gen_ao, label), "jlens": sj.gen_rows(gen_jlens, label)}
        if rows["ao"] is None or rows["jlens"] is None:
            raise SystemExit(f"{label}: gen files incomplete for the {LAYERS} band")
        for p in range(conv.n_pos):
            ctx = sj.context_block(tok, conv.ids, p)
            jref_block, _ = sj.readout_block(rows["jlens"].get(p, {}), "jlens", tok)
            for lens in ("ao", "jlens"):
                key = f"{label}|{p}|{lens}|strict"
                out_path = out_dir / f"sumtok_{lens}.jsonl"
                if key in done[out_path]:
                    continue
                block, layer_texts = sj.readout_block(rows[lens].get(p, {}), lens, tok)
                user = sj.user_msg(ctx, block, jref_block if lens == "ao" else None)

                def finish(verdict: dict[str, Any], _lt: dict[int, str] = layer_texts
                           ) -> dict[str, Any]:
                    return {"verdict": verdict, **sj.apply_void_gates(verdict, _lt)}

                stub = {"label": label, "pos": p, "lens": lens, "arm": "strict",
                        "token": e["tokens"][p], "_finish": finish}
                jobs[out_path].append((key, systems[lens], user, schemas[lens], model, stub))
    return jobs


def build_quality_jobs(entries: list[dict[str, Any]], acts_dir: Path, gen_ao: Path,
                       out_dir: Path, tok: Any, model: str) -> dict[Path, list[Any]]:
    out_path = out_dir / "quality.jsonl"
    done = done_keys(out_path)
    jobs: list[Any] = []
    sch = quality_schema()
    for e in entries:
        label = e["label"]
        conv = sj.load_conv(acts_dir, label, e["file"])
        rows = sj.gen_rows(gen_ao, label)
        if rows is None:
            raise SystemExit(f"{label}: AO gen files incomplete for the {LAYERS} band")
        for p in range(conv.n_pos):
            ctx = sj.context_block(tok, conv.ids, p)
            for layer in LAYERS:
                key = f"{label}|{p}|{layer}"
                if key in done:
                    continue
                row = rows.get(p, {}).get(layer)
                samples = [s for s in (row or {}).get("samples", []) if s and s.strip()]
                if not samples:
                    continue  # nothing generated at this cell — not a quality datum
                user = QUALITY_USER.format(ctx=ctx, layer=layer,
                                           samples=" ⏐ ".join(s.strip() for s in samples))

                def finish(verdict: dict[str, Any]) -> dict[str, Any]:
                    return {"quality": max(1, min(10, int(verdict["quality"]))),
                            "hallucination": bool(verdict["hallucination"]),
                            "junk": bool(verdict["junk"]),
                            "padded": bool(verdict["padded"]),
                            "evidence": verdict.get("evidence", "")}

                stub = {"label": label, "pos": p, "layer": layer, "token": e["tokens"][p],
                        "n_samples": len(samples), "_finish": finish}
                jobs.append((key, QUALITY_SYSTEM, user, sch, model, stub))
    return {out_path: jobs}


def build_sumfmt_jobs(entries: list[dict[str, Any]], gen_ao: Path, out_dir: Path,
                      tok: Any, model: str) -> dict[Path, list[Any]]:
    """Summary-flagged AO positions only — requires sumtok_ao.jsonl from pass 1."""
    src = out_dir / "sumtok_ao.jsonl"
    if not src.exists():
        raise SystemExit("sumfmt needs pass 1 first (sumtok_ao.jsonl missing)")
    flagged: list[dict[str, Any]] = []
    for line in src.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not row.get("api_error") and row.get("effective_level", 0) >= 2:
            flagged.append(row)
    out_path = out_dir / "sumfmt.jsonl"
    done = done_keys(out_path)
    jobs: list[Any] = []
    sch = sumfmt_schema()
    by_label: dict[str, dict[int, dict[int, dict[str, Any]]]] = {}
    for f in flagged:
        label, p = f["label"], f["pos"]
        key = f"{label}|{p}"
        if key in done:
            continue
        if label not in by_label:
            rows = sj.gen_rows(gen_ao, label)
            assert rows is not None
            by_label[label] = rows
        layers = [la for la in f["quote_layers"] if la in LAYERS] or list(LAYERS)
        rows_p = by_label[label].get(p, {})
        lines, sample_starts = [], {}
        for la in layers:
            samples = [s.strip() for s in (rows_p.get(la) or {}).get("samples", [])
                       if s and s.strip()]
            if samples:
                lines.append(f"[L{la}] " + " ⏐ ".join(samples))
                sample_starts[la] = [_norm_start(s)[:80] for s in samples]
        if not lines:
            continue
        user = SUMFMT_USER.format(block="\n".join(lines))

        def finish(verdict: dict[str, Any],
                   _starts: dict[int, list[str]] = sample_starts) -> dict[str, Any]:
            verified = []
            for frag in verdict.get("fragments", []):
                la, quote = frag.get("layer"), _norm_start(str(frag.get("quote", "")))
                if quote and any(s.startswith(quote[:40]) for s in _starts.get(la, [])):
                    verified.append({"layer": la, "quote": frag["quote"]})
            return {"fragments_raw": verdict.get("fragments", []),
                    "fragments": verified,
                    "n_layers": len(_starts),
                    "n_malformed_layers": len({f["layer"] for f in verified}),
                    "rationale": verdict.get("rationale", "")}

        stub = {"label": label, "pos": p, "quote_layers": layers, "_finish": finish}
        jobs.append((key, SUMFMT_SYSTEM, user, sch, model, stub))
    print(f"[sumfmt] {len(flagged)} summary positions, {len(jobs)} to judge")
    return {out_path: jobs}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--acts", required=True, type=Path)
    p.add_argument("--gen-ao", required=True, type=Path)
    p.add_argument("--gen-jlens", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--stage", default="all", choices=["sumtok", "quality", "sumfmt", "all"])
    p.add_argument("--sumtok-model", default=sj.MODEL,
                   help="pass-1 model; default = the frozen sumtok judge's (claude-opus-5)")
    p.add_argument("--model", default=QUALITY_MODEL, help="pass-2/3 model")
    p.add_argument("--concurrency", type=int, default=256)
    args = p.parse_args()

    from transformers import AutoTokenizer  # heavy import, deferred

    entries = load_manifest(args.acts)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    args.out.mkdir(parents=True, exist_ok=True)

    stages = ["sumtok", "quality", "sumfmt"] if args.stage == "all" else [args.stage]
    for stage in stages:
        if stage == "sumtok":
            jobs = build_sumtok_jobs(entries, args.acts, args.gen_ao, args.gen_jlens,
                                     args.out, tok, args.sumtok_model)
        elif stage == "quality":
            jobs = build_quality_jobs(entries, args.acts, args.gen_ao, args.out, tok,
                                      args.model)
        else:
            jobs = build_sumfmt_jobs(entries, args.gen_ao, args.out, tok, args.model)
        for out_path, job_list in jobs.items():
            print(f"[{stage}] {len(job_list)} calls -> {out_path}")
            asyncio.run(pump(job_list, out_path, args.concurrency))


if __name__ == "__main__":
    main()
