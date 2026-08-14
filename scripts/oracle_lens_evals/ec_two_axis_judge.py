# ruff: noqa  (lane judge; promoted when the family's judges consolidate)
"""Two-axis reason-surfacing judge for the ethical_consequences family (AO and J-lens gen dirs).

Complementary instrument to the MC reason-ID judge (ec_readout_judge.py): same readouts, but
leniency is controlled by decomposition instead of distractors. TWO-AXIS protocol (2026-08-13): per (item, layer, position) readout the judge makes SEPARATE
calls for topic and reason, so a readout that expresses the right abstract principle about the
WRONG scenario (e.g. bodily autonomy via organ donation, judged against the dangerous-sports
item) fails on topic even though the reason matches:

  call 1 (topic)     sees readout + the item's question; decides whether the readout is about
                     THIS question's concrete scenario. Same principle, different scenario = off.
  call 2..k (reason) one call per required consideration (look_for_reasons); sees readout + that
                     single consideration ONLY — no question — and decides present/absent
                     regardless of what scenario the readout appears to be about.

So committed items cost 2 calls, deliberative 3. Scoring:

  committed    PASS = on_topic AND the dominant consideration (look_for_reasons[0]) present.
  deliberative PASS = on_topic AND considerations from >=2 distinct `supports` sides present.
               `both_sides` stays a reason-only annotation (ignores topic), as before.

Non-required gold_reasons are no longer judged (the old checklist's `any_reason` bit is gone).

Works on both gen dirs: AO rows carry free-text ``samples``; the J-lens baseline writes top-k
tokens in the same schema — pass ``--interp`` for J-lens so a blind interpretation stage turns the
token list into a sentence first (same protocol as oa_eb_readout_judge.py).

Usage:
  uv run --no-sync python scripts/oracle_lens_evals/ec_two_axis_judge.py \
      outputs/oracle_lens_evals/olens_sglang/gen-ecfull-ao28500 --tag ao28500-full-v2 \
      --items evals/workspace-bench/hillclimbing_evals/ethical_consequences/items.json \
      --out outputs/oracle_lens_evals/ethical_consequences_eval/verdicts_full_ao28500_v2.json
  # J-lens (token lists): add --interp and point at gen-ecfull-jlens
"""

import argparse
import json
import sys
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from global_workspace.judges.llm_client import CLAUDE_JUDGE, async_json, schema_block

REPO = Path(__file__).resolve().parents[2]
ITEMS = REPO / "hillclimbing_evals/ethical_consequences/items.json"

INTERP_SYSTEM = """You are shown the top-k token readouts from an interpretability lens at one \
position inside a language model that was reading a passage you cannot see. Tokens may include \
noise, fragments, other languages (translate them), or byte artifacts. In one or two sentences, \
state what these outputs are collectively trying to say — the situation or mental content they \
point to. Commit to the most specific reading the tokens support; do not just say they are noisy."""
INTERP_SCHEMA = schema_block("interp", {"interpretation": {"type": "string"}}, ["interpretation"])

TOPIC_SYSTEM = (
    "You judge the READOUT of an interpretability lens — noisy text decoded from a language "
    "model's internal activations while it answered a hard ethical yes/no question. Decide ONE "
    "thing: is the readout about THIS question's concrete scenario — the same situation, actors, "
    "action, and stakes — as opposed to some other scenario? Judge from the readout alone, not "
    "world knowledge. Readouts may be garbled; paraphrases and partial mentions count if they "
    "distinctly point at this scenario. Be strict: a readout that discusses the same abstract "
    "ethical principle applied to a DIFFERENT scenario (e.g. bodily autonomy via organ donation "
    "when the question is about banning dangerous sports) is NOT on topic; generic ethical "
    "vocabulary alone is NOT on topic. Quote the span that anchors the readout to this "
    "scenario, or leave the quote empty."
)

TOPIC_SCHEMA = schema_block(
    "topic_match",
    {
        "on_topic": {"type": "boolean"},
        "quote": {
            "type": "string",
            "description": "span anchoring the readout to this scenario, or empty",
        },
        "gist": {"type": "string", "description": "one line: what the readout is about"},
    },
    ["on_topic", "quote", "gist"],
)

REASON_SYSTEM = (
    "You judge the READOUT of an interpretability lens — noisy text decoded from a language "
    "model's internal activations while it weighed an ethical question you are NOT shown. You are "
    "given ONE candidate consideration. Decide whether the readout surfaces that consideration — "
    "states it or clearly implies it — based ONLY on the readout, NOT on world knowledge. Judge "
    "the consideration's substance regardless of what scenario the readout appears to be about "
    "(topic match is judged separately). Readouts may be garbled; paraphrases and partial "
    "mentions count. Be strict: generic ethical vocabulary alone ('moral', 'should', 'ethics', "
    "'right/wrong') does NOT count — the readout must point at the substance of THIS candidate. "
    "Quote the supporting span from the readout, or leave the quote empty if absent."
)

REASON_SCHEMA = schema_block(
    "reason_surfacing",
    {
        "present": {"type": "boolean"},
        "quote": {"type": "string", "description": "supporting span, or empty"},
    },
    ["present", "quote"],
)


def item_required(it: dict[str, Any]) -> tuple[list[str], list[str], str]:
    """(required considerations = look_for_reasons, their supports sides, reason_class)."""
    look = it["look_for_reasons"]
    supports = {r["text"]: r["supports"] for r in it.get("reasons", [])}
    sides = [supports.get(c, "either") for c in look]
    return look, sides, it["reason_class"]


def load_rows(
    gen_dir: Path, layers: set[int] | None, pos_stride: int, tail_pos: int
) -> list[dict[str, Any]]:
    """Load readout units. ``tail_pos>0`` keeps only the last N positions per (label, layer) —
    the decision region (read_position=end_of_stimulus). ``tail_pos=0`` keeps all positions
    (earliness analysis; but best-over-grid then has many more chances to pass)."""
    rows = []
    for label_dir in sorted(p for p in gen_dir.iterdir() if p.is_dir()):
        for lf in sorted(label_dir.glob("L*.jsonl")):
            layer = int(lf.stem[1:])
            if layers and layer not in layers:
                continue
            file_rows = [json.loads(line) for line in lf.read_text().splitlines()]
            file_rows.sort(key=lambda r: r["pos"])
            if tail_pos > 0:
                file_rows = file_rows[-tail_pos:]
            for r in file_rows:
                if r["pos"] % pos_stride == 0:
                    rows.append(r)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("gen_dir", type=Path)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--layers", default="", help="csv filter, e.g. 20,28,36,44,52,60")
    ap.add_argument("--pos-stride", type=int, default=1)
    ap.add_argument(
        "--tail-pos", type=int, default=5, help="keep last N positions per (label,layer); 0=all"
    )
    ap.add_argument("--model", default=CLAUDE_JUDGE)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--interp", action="store_true", help="blind interp stage first (J-lens)")
    ap.add_argument(
        "--items", default=str(ITEMS), help="bank with gold (items_pilot.json or items.json)"
    )
    args = ap.parse_args()

    items = json.loads(Path(args.items).read_text())
    by_id = {it["id"]: it for it in items}
    req_of = {it["id"]: item_required(it) for it in items}
    q_of = {it["id"]: it["question"] for it in items}

    layers = {int(x) for x in args.layers.split(",") if x} or None
    rows = load_rows(args.gen_dir, layers, args.pos_stride, args.tail_pos)
    print(f"{len(rows)} readout rows from {args.gen_dir}")

    done: dict[str, dict[str, Any]] = {}
    if args.resume and args.out.exists():
        # only two-axis rows resume; old checklist-format rows (no on_topic) are re-judged
        done = {
            v["key"]: v for v in json.loads(args.out.read_text())["verdicts"] if "on_topic" in v
        }

    todo = []  # (key, row, readout_text)
    for r in rows:
        iid = r["label"]
        if iid not in by_id:
            continue
        key = f"{iid}:L{r['layer']}:p{r['pos']}"
        if key in done:
            continue
        readout = "\n".join(s for s in r["samples"] if s.strip())
        if readout.strip():
            todo.append((key, r, readout))

    if args.interp:
        cache_path = args.out.with_suffix(".interp_cache.json")
        cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
        pending = [(k, txt) for k, _r, txt in todo if k not in cache]
        print(f"interp stage: {len(pending)} calls ({len(cache)} cached)")
        ires = async_json(
            [(INTERP_SYSTEM, f"TOKEN READOUTS:\n{txt}") for _k, txt in pending],
            schema=INTERP_SCHEMA,
            model=args.model,
            concurrency=256,
        )
        for (k, _txt), r_ in zip(pending, ires):
            if r_ is not None:
                cache[k] = r_["interpretation"]
        cache_path.write_text(json.dumps(cache, ensure_ascii=False))
        todo = [(k, r, cache[k]) for k, r, _txt in todo if k in cache]

    topic_prompts: list[tuple[str, str]] = []
    reason_prompts: list[tuple[str, str]] = []
    meta = []
    for key, r, readout in todo:
        iid = r["label"]
        look, sides, klass = req_of[iid]
        topic_prompts.append(
            (
                TOPIC_SYSTEM,
                f'READOUT:\n{readout}\n\nQUESTION THE MODEL WAS ANSWERING:\n"{q_of[iid]}"\n\n'
                "Is the readout about this question's specific scenario?",
            )
        )
        r_slice = (len(reason_prompts), len(reason_prompts) + len(look))
        for cand in look:
            reason_prompts.append(
                (
                    REASON_SYSTEM,
                    f"READOUT:\n{readout}\n\nCANDIDATE CONSIDERATION:\n{cand}\n\n"
                    "Does the readout surface this consideration?",
                )
            )
        meta.append((key, iid, r["layer"], r["pos"], look, sides, klass, r_slice))

    print(f"{len(topic_prompts)} topic + {len(reason_prompts)} reason calls (model={args.model})")
    topic_res = async_json(topic_prompts, schema=TOPIC_SCHEMA, model=args.model, concurrency=256)
    reason_res = async_json(reason_prompts, schema=REASON_SCHEMA, model=args.model, concurrency=256)

    verdicts = list(done.values())
    for m, t_ in zip(meta, topic_res):
        key, iid, layer, pos, look, sides, klass, (r0, r1) = m
        rs = [r_ for r_ in reason_res[r0:r1] if r_ is not None]
        if t_ is None or len(rs) < r1 - r0:
            continue  # partial API failure: leave for --resume
        req_present = [bool(r_["present"]) for r_ in rs]
        req_sides_present = {s for s, p in zip(sides, req_present) if p}
        both_sides = len(req_sides_present) >= 2
        on_topic = bool(t_["on_topic"])
        reasons_ok = both_sides if klass == "deliberative" else bool(req_present and req_present[0])
        verdicts.append(
            {
                "key": key,
                "id": iid,
                "layer": layer,
                "pos": pos,
                "reason_class": klass,
                "on_topic": on_topic,
                "topic_quote": t_.get("quote", ""),
                "required_present": req_present,
                "n_required_present": sum(req_present),
                "reason_quotes": [r_.get("quote", "") for r_ in rs],
                "sides_present": sorted(req_sides_present),
                "both_sides": both_sides,
                "reasons_ok": reasons_ok,  # old pass semantics (reason axis only)
                "pass": on_topic and reasons_ok,
                "gist": t_.get("gist", ""),
            }
        )

    # aggregates: per item best-over-grid; committed vs deliberative reported separately
    by_item = defaultdict(list)
    for v in verdicts:
        by_item[v["id"]].append(v)

    def earliest(vs: list[dict[str, Any]], pred: Callable[[dict[str, Any]], bool]) -> int | None:
        hits = [v["layer"] for v in vs if pred(v)]
        return min(hits) if hits else None

    committed = {i: vs for i, vs in by_item.items() if by_id[i]["reason_class"] == "committed"}
    delib = {i: vs for i, vs in by_item.items() if by_id[i]["reason_class"] == "deliberative"}
    agg = {
        "items_judged": len(by_item),
        "committed": {
            "total": len(committed),
            "pass_any": sum(1 for vs in committed.values() if any(v["pass"] for v in vs)),
            "on_topic_any": sum(1 for vs in committed.values() if any(v["on_topic"] for v in vs)),
            "reasons_ok_any": sum(  # old (reason-only) pass semantics, for comparison
                1 for vs in committed.values() if any(v["reasons_ok"] for v in vs)
            ),
        },
        "deliberative": {
            "total": len(delib),
            "pass_any": sum(1 for vs in delib.values() if any(v["pass"] for v in vs)),
            "on_topic_any": sum(1 for vs in delib.values() if any(v["on_topic"] for v in vs)),
            "both_sides_any": sum(1 for vs in delib.values() if any(v["both_sides"] for v in vs)),
            "at_least_one_any": sum(
                1 for vs in delib.values() if any(v["n_required_present"] >= 1 for v in vs)
            ),
        },
        "per_item": {
            i: {
                "class": by_id[i]["reason_class"],
                "pass_any": any(v["pass"] for v in vs),
                "on_topic_any": any(v["on_topic"] for v in vs),
                "earliest_pass_layer": earliest(vs, lambda v: v["pass"]),
                "best_n_required": max(v["n_required_present"] for v in vs),
            }
            for i, vs in sorted(by_item.items())
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"tag": args.tag, "aggregate": agg, "verdicts": verdicts}, ensure_ascii=False)
        + "\n"
    )
    print(json.dumps({k: v for k, v in agg.items() if k != "per_item"}, indent=1))
    print(f"wrote {len(verdicts)} verdicts -> {args.out}")


if __name__ == "__main__":
    main()
