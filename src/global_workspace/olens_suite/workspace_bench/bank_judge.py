"""Opus judge for the baseline bank families — replaces regex substring matching as the
headline pass criterion (audit 2026-08-15, user decision: pure Opus, no screen tier).

One call per (item, layer): the judge sees the item's targets and EVERY position's readout
samples at that layer, and decides whether any target concept is NAMED — strict mode
(2026-08-15): exact word / true synonym / translation only; derived words, fragments, and
associative imagery do NOT count; every positive must carry a verbatim quote (unverifiable quotes
void the verdict; quotes are verified against the READOUT SAMPLES only, never the prompt's
target/position headers — audit 2026-08-19). Works identically for oracle-lens free text and
J-lens token bags (the prompt says which). Deterministic bundling order, resume-by-key, full
verdicts persisted. API outages are NEVER persisted as verdicts: failed cells are counted in
``summary.n_unavailable``, left out of the file so ``--resume`` retries them, and a run where
every call failed (e.g. no ANTHROPIC_API_KEY) aborts instead of writing a 0-pass score file.

    python -m global_workspace.olens_suite.workspace_bench.bank_judge \
        --acts outputs/oracle_lens_evals/olens_sglang/acts-final \
        --gen  outputs/oracle_lens_evals/olens_sglang/gen-rl-ddp600 \
        --kind text --out <gen>/bank_judge

Output: <out>/<family>.json = {"family", "kind", "model", "verdicts": {"<label>|L<layer>":
{"expressed", "target", "quote", "quote_ok"}}, "summary": {"<family>": item pass rate}}.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from global_workspace.judges.llm_client import CLAUDE_JUDGE, async_json, schema_block

SYSTEM = """You are judging readouts from an activation lens for an interpretability
benchmark. You get one eval item: its TARGET concept(s) and the lens's readout samples at one
layer (several token positions; for the J-lens the samples are top-10 vocabulary tokens, for
the oracle lens they are free-text generations).

Decide whether ANY target concept is EXPRESSED in the readout. STRICT rules:
- Expressed means the target concept is NAMED: the exact target word/phrase as a standalone
  word (case/inflection of the SAME word is fine: "cats" for "cat"), an established synonym or
  alias that unambiguously denotes the SAME concept ("Mandarin" for "Chinese"), or a faithful
  translation of the target into another language ("二月" for "February").
- NOT expressed by a DIFFERENT or DERIVED word that merely contains the target string:
  "blackout" does NOT express "black"; "childhood" does NOT express "child".
- NOT expressed by word fragments or partial tokens, even distinctive ones: "paragus" does NOT
  express "asparagus"; " acet" does NOT express "acetaminophen". Fragments of different tokens
  sitting near each other in a token list NEVER combine.
- NOT expressed by thematically associated imagery, descriptions, or scene content that evokes
  the concept without naming it: "the wedding dress that never left the closet" does NOT
  express "sadness". Hypernyms, hyponyms, and category neighbors do NOT count.
- Judge only what is actually written. If uncertain, say not expressed.
- Every YES must include a VERBATIM quote copied exactly from one sample (the smallest span
  that names the target). No quote, no pass.
Return JSON: {"expressed": bool, "target": "<which target, or empty>", "quote": "<verbatim>"}."""

SCHEMA = schema_block(
    "bank_verdict",
    {
        "expressed": {"type": "boolean"},
        "target": {"type": "string"},
        "quote": {"type": "string"},
    },
    ["expressed", "target", "quote"],
)


def load_rows(gen: Path, label: str, layer: int) -> list[dict[str, Any]]:
    f = gen / label / f"L{layer:03d}.jsonl"
    if not f.exists():
        return []
    return [json.loads(line) for line in f.read_text().splitlines() if line.strip()]


def quote_verified(quote: str, samples_text: str) -> bool:
    """A positive's quote must be a verbatim span of the readout SAMPLES themselves.

    Verifying against the full prompt would let a hallucinated 'quote' of the target word pass
    the quote gate (the targets line and position headers contain it verbatim). Exact substring
    first; a whitespace/case-normalized fallback keeps legit verdicts from being voided by
    whitespace mangling."""
    if not quote:
        return False
    if quote in samples_text:
        return True
    norm_q = " ".join(quote.lower().split())
    return bool(norm_q) and norm_q in " ".join(samples_text.lower().split())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--acts", required=True, type=Path)
    ap.add_argument("--gen", required=True, type=Path)
    ap.add_argument("--kind", choices=["text", "tokens"], default="text")
    ap.add_argument("--out", type=Path, default=None, help="default <gen>/bank_judge")
    ap.add_argument("--families", default="", help="csv filter; empty = all with targets")
    ap.add_argument("--model", default=CLAUDE_JUDGE)
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    out_dir = args.out or (args.gen / "bank_judge")
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((args.acts / "manifest.json").read_text())
    fams = set(args.families.split(",")) if args.families else None
    # available layers = union over ALL items (probing only item 0 mis-selects layers for
    # everyone whenever item 0's coverage is atypical)
    labels = [p["label"] for p in manifest["prompts"]]
    layers = [
        ly
        for ly in manifest["layers"]
        if any((args.gen / lb / f"L{ly:03d}.jsonl").exists() for lb in labels)
    ]
    layers = layers or manifest["layers"]

    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for p in manifest["prompts"]:
        if p.get("targets") and (fams is None or p["family"] in fams):
            by_family[p["family"]].append(p)

    for family, entries in sorted(by_family.items()):
        out_path = out_dir / f"{family}.json"
        done: dict[str, Any] = {}
        if args.resume and out_path.exists():
            prior = json.loads(out_path.read_text()).get("verdicts", {})
            # legacy files may hold judge="unavailable" placeholders from an outage — drop
            # them so the resume RE-JUDGES those cells instead of freezing them as misses
            done = {k: v for k, v in prior.items() if v.get("judge") != "unavailable"}
        cells: list[tuple[str, str, str]] = []  # (key, user_prompt, samples_text)
        n_missing = 0  # items with targets but no readout rows at any selected layer
        for e in entries:
            found_any = False
            for ly in layers:
                key = f"{e['label']}|L{ly}"
                rows = load_rows(args.gen, e["label"], int(ly))
                if not rows:
                    continue
                found_any = True
                if key in done:
                    continue
                blocks = []
                for r in rows:
                    samples = "\n".join(f"  {s}" for s in r.get("samples", []))
                    blocks.append(f"[position {r['pos']}, token {r.get('token', '')!r}]\n{samples}")
                kind_note = (
                    "The samples below are TOP-10 VOCABULARY TOKENS (one token per line)."
                    if args.kind == "tokens"
                    else "The samples below are free-text readouts."
                )
                user = (
                    f"targets: {json.dumps(e['targets'])}\n{kind_note}\n\n" + "\n\n".join(blocks)
                )
                samples_text = "\n".join(s for r in rows for s in r.get("samples", []))
                cells.append((key, user, samples_text))
            if not found_any:
                n_missing += 1
        n_unavailable = 0
        if cells:
            prompts = [(SYSTEM, u) for _, u, _ in cells]
            results = async_json(
                prompts, schema=SCHEMA, model=args.model, concurrency=args.concurrency
            )
            if all(r is None for r in results):
                raise SystemExit(
                    f"[bank_judge] {family}: ALL {len(results)} judge calls failed "
                    "(missing/invalid ANTHROPIC_API_KEY or a full outage) — refusing to write "
                    "a 0-pass score file. Fix the key/outage and rerun (with --resume to keep "
                    "any prior verdicts)."
                )
            for (key, _user, samples_text), r in zip(cells, results, strict=True):
                if r is None:
                    # do NOT persist: the key stays absent so --resume retries it
                    n_unavailable += 1
                    continue
                quote_ok = quote_verified(r.get("quote") or "", samples_text)
                if r.get("expressed") and not quote_ok:
                    r["expressed"] = False  # unverifiable positive -> void (quote discipline)
                done[key] = {"expressed": bool(r.get("expressed")), "target": r.get("target", ""),
                             "quote": r.get("quote", ""), "quote_ok": quote_ok}
        # item pass = any layer expressed
        item_pass: dict[str, bool] = defaultdict(bool)
        for key, v in done.items():
            label = key.rsplit("|L", 1)[0]
            item_pass[label] = item_pass[label] or v["expressed"]
        n = len({e["label"] for e in entries})
        n_pass = sum(1 for e in entries if item_pass.get(e["label"]))
        payload = {
            "family": family, "kind": args.kind, "model": args.model,
            "summary": {
                "n_items": n, "n_pass": n_pass, "pass_rate": n_pass / max(1, n),
                # unjudged cells this run (retryable via --resume) + items with no readouts at
                # all: surfaced so a partial run can never masquerade as a clean 0/N
                "n_unavailable": n_unavailable, "n_missing_readouts": n_missing,
            },
            "verdicts": done,
        }
        tmp = out_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1))
        tmp.replace(out_path)
        flag = f", {n_unavailable} UNJUDGED (rerun --resume)" if n_unavailable else ""
        miss = f", {n_missing} items without readouts" if n_missing else ""
        print(
            f"[bank_judge] {family}: {n_pass}/{n} ({len(done)} cells{flag}{miss}) -> {out_path}",
            flush=True,
        )


if __name__ == "__main__":
    main()
