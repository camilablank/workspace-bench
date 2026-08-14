# ruff: noqa  (pilot readout judge; promoted when the family freezes)
"""Multiple-choice reason-identification judge for the ethical_consequences family.

Forced-choice sibling of the old per-candidate surfacing instrument. Instead of listing the
item's OWN gold reasons and asking "did the readout surface each?", the judge now answers a
multiple-choice question whose options are the gold reason plus DISTRACTORS sampled from OTHER
items' gold reasons (a cross-item pool). This mirrors the house MC style of
``oracle_lens_evals/oa_eb_readout_judge.py`` / ``oracle_lens/latent_eval/judge_mc.py``:
seeded_shuffle + a trailing "cannot tell from the readout" escape + an integer ``choice`` and a
supporting ``quote``. Scoring honors the committed/deliberative split:

  committed    ONE MC call/readout. Options = look_for_reasons[0] (gold) + 4 cross-item
               distractors from the committed pool (every item's look_for_reasons[0]), shuffled,
               + "cannot tell" (6 lines). PASS = choice == gold position.
  deliberative TWO MC calls/readout, one per side. yes-MC gold = the yes-supporting look_for
               reason, distractors from the yes pool; no-MC analogous. PASS = BOTH sides correct.

Distractors are drawn with a fixed seed (sha256(f"{EC_SEED}:{item_id}:{side}")), excluding any
candidate that (a) shares this item's topic_id (the chat_tf/chat_yn twin would leak), (b) is
normalized-equal to the gold or an already-picked distractor, or (c) has token-Jaccard > 0.6 vs
the gold (near-paraphrase guard). Pools are large (committed 100 / yes 45 / no 72 distinct), so
the "fall back to fewer distractors, never below 2" path is essentially never hit; short draws
are still flagged (``short_pool``).

Aggregation is best-over-grid ANY over (layer, position): a committed item passes if ANY grid
point's MC is correct; a deliberative item passes if its yes-side is EVER correct at some grid
point AND its no-side is EVER correct at some grid point (sides aggregated independently — they
need not land at the same grid point). Random baseline: committed 1/6, deliberative (1/6)^2 = 1/36.

Works on both gen dirs: AO rows carry free-text ``samples``; the J-lens baseline writes top-k
tokens in the same schema — pass ``--interp`` for J-lens so a blind interpretation stage turns the
token list into a sentence first (same protocol as oa_eb_readout_judge.py). ``--tail-pos`` keeps
only the last N positions per (label, layer) — the end-of-stimulus decision region.

Usage:
  uv run --no-sync python scripts/oracle_lens_evals/ec_readout_judge.py \
      outputs/oracle_lens_evals/olens_sglang/gen-ec-ao28500 --tag ao28500 \
      --out outputs/oracle_lens_evals/ethical_consequences_eval/verdicts_ao28500.json
  # J-lens (token lists): add --interp and point at gen-ec-jlens
"""

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from global_workspace.judges.llm_client import CLAUDE_JUDGE, async_json, schema_block

REPO = Path(__file__).resolve().parents[2]
ITEMS = REPO / "hillclimbing_evals/ethical_consequences/items.json"

EC_SEED = 20260812  # ethical_consequences freeze date; fixes distractor draw + option order
CANNOT = "cannot tell from the readout"

INTERP_SYSTEM = """You are shown the top-k token readouts from an interpretability lens at one \
position inside a language model that was reading a passage you cannot see. Tokens may include \
noise, fragments, other languages (translate them), or byte artifacts. In one or two sentences, \
state what these outputs are collectively trying to say — the situation or mental content they \
point to. Commit to the most specific reading the tokens support; do not just say they are noisy."""
INTERP_SCHEMA = schema_block("interp", {"interpretation": {"type": "string"}}, ["interpretation"])

SYSTEM = (
    "You judge the READOUT of an interpretability lens — noisy text decoded from a language "
    "model's internal activations while it answered a hard ethical yes/no question. You are given "
    "ONE multiple-choice question: which listed consideration the readout shows the model actually "
    "weighing. Pick an option ONLY if the readout states or clearly paraphrases the SUBSTANCE of "
    "that consideration — not merely a word it shares. Judge ONLY from the readout, never from "
    "world knowledge or from which answer seems ethically correct. Generic ethical vocabulary "
    "alone ('moral', 'should', 'ethics', 'right/wrong') does NOT identify a specific "
    "consideration. If the readout states none of the listed considerations, choose the 'cannot "
    "tell from the readout' option. Give a SHORT exact quote (a few words, VERBATIM) from the "
    "readout that supports your choice."
)

SCHEMA = schema_block(
    "ec_reason_mc",
    {"choice": {"type": "integer"}, "quote": {"type": "string"}},
    ["choice", "quote"],
)


def norm(t: str) -> str:
    return " ".join(t.lower().split())


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def seed_int(key: str) -> int:
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")


def seeded_shuffle(key: str, opts: list[str]) -> list[str]:
    order = list(range(len(opts)))
    random.Random(seed_int(f"{EC_SEED}:{key}:order")).shuffle(order)
    return [opts[j] for j in order]


def build_pools(items: list[dict]) -> tuple[list[tuple[str, str]], list[tuple[str, str]], list[tuple[str, str]]]:
    """(committed_pool, yes_pool, no_pool) as (topic_id, reason_text) pairs.

    committed_pool = every item's look_for_reasons[0]; yes/no pools = every item's look_for
    reasons tagged supports=='yes'/'no' (side read from reasons[*]). Same-polarity pools stop a
    reader from shortcutting on yes/no valence.
    """
    comm_pool = [(it["topic_id"], it["look_for_reasons"][0]) for it in items]
    yes_pool: list[tuple[str, str]] = []
    no_pool: list[tuple[str, str]] = []
    for it in items:
        supm = {r["text"]: r["supports"] for r in it.get("reasons", [])}
        for t in it["look_for_reasons"]:
            s = supm.get(t, "either")
            if s == "yes":
                yes_pool.append((it["topic_id"], t))
            elif s == "no":
                no_pool.append((it["topic_id"], t))
    return comm_pool, yes_pool, no_pool


def sample_distractors(
    pool: list[tuple[str, str]], gold: str, topic_id: str, seed_key: str, n: int = 4
) -> tuple[list[str], bool]:
    """Draw up to ``n`` distractors from ``pool`` (list of (topic_id, text)).

    Excludes candidates that share ``topic_id`` (twin leak), are normalized-equal to the gold or an
    already-picked distractor, or have token-Jaccard > 0.6 with the gold. Falls back to fewer
    distractors when the pool is exhausted (never fewer than 2 for the real banks). Returns
    (distractors, short_pool) where short_pool flags a draw of < n.
    """
    gold_tok = set(norm(gold).split())
    seen = {norm(gold)}
    uniq: list[str] = []
    for src_topic, text in pool:  # pool order is fixed (bank order) -> deterministic
        if src_topic == topic_id:
            continue
        tn = norm(text)
        if tn in seen:
            continue
        if jaccard(gold_tok, set(tn.split())) > 0.6:
            continue
        seen.add(tn)
        uniq.append(text)
    random.Random(seed_int(f"{EC_SEED}:{seed_key}")).shuffle(uniq)
    picked = uniq[:n]
    return picked, len(picked) < n


def _listing(shown: list[str]) -> str:
    return "\n".join(f"  {i + 1}. {o}" for i, o in enumerate(shown))


def committed_question(question: str, shown: list[str]) -> str:
    return (
        f"Which consideration is the readout showing the model weigh in answering:\n"
        f'"{question}"?\n{_listing(shown)}'
    )


def deliberative_question(question: str, side: str, shown: list[str]) -> str:
    if side == "yes":
        ask = (
            "Which YES-supporting reason (a reason to answer the ethical question 'yes'/'true') "
            "does the readout show the model weigh?"
        )
    else:
        ask = (
            "Which NO-supporting reason (a reason to answer 'no'/'false') "
            "does the readout show the model weigh?"
        )
    return f'The model was answering: "{question}"\n\n{ask}\n{_listing(shown)}'


def build_mcs(
    it: dict,
    comm_pool: list[tuple[str, str]],
    yes_pool: list[tuple[str, str]],
    no_pool: list[tuple[str, str]],
) -> dict[str, tuple[str, int, list[str], int, bool]]:
    """Per-item MC(s), keyed by side. Value = (question, gold 1-based pos, shown options, n_distractors, short_pool).

    committed -> {"committed": ...}; deliberative -> {"yes": ..., "no": ...} (a side is omitted if
    the item lacks a look_for reason tagged for it — that item can then never pass both sides).
    """
    q = it["question"]
    out: dict[str, tuple[str, int, list[str], int, bool]] = {}
    if it["reason_class"] == "deliberative":
        supm = {r["text"]: r["supports"] for r in it.get("reasons", [])}
        for side, pool in (("yes", yes_pool), ("no", no_pool)):
            gold = next((t for t in it["look_for_reasons"] if supm.get(t) == side), None)
            if gold is None:
                continue
            dists, short = sample_distractors(pool, gold, it["topic_id"], f"{it['id']}:{side}")
            shown = seeded_shuffle(f"{it['id']}:{side}", [gold] + dists) + [CANNOT]
            out[side] = (deliberative_question(q, side, shown), shown.index(gold) + 1, shown, len(dists), short)
    else:
        gold = it["look_for_reasons"][0]
        dists, short = sample_distractors(comm_pool, gold, it["topic_id"], f"{it['id']}:committed")
        shown = seeded_shuffle(f"{it['id']}:committed", [gold] + dists) + [CANNOT]
        out["committed"] = (committed_question(q, shown), shown.index(gold) + 1, shown, len(dists), short)
    return out


def load_rows(gen_dir: Path, layers: set[int] | None, pos_stride: int, tail_pos: int) -> list[dict]:
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
    ap.add_argument("--tail-pos", type=int, default=5, help="keep last N positions per (label,layer); 0=all")
    ap.add_argument("--model", default=CLAUDE_JUDGE)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--interp", action="store_true", help="blind interp stage first (J-lens)")
    ap.add_argument("--items", default=str(ITEMS), help="bank with gold (items.json or items_pilot.json)")
    args = ap.parse_args()

    items = json.loads(Path(args.items).read_text())
    by_id = {it["id"]: it for it in items}
    comm_pool, yes_pool, no_pool = build_pools(items)
    mc_of = {it["id"]: build_mcs(it, comm_pool, yes_pool, no_pool) for it in items}

    layers = {int(x) for x in args.layers.split(",") if x} or None
    rows = load_rows(args.gen_dir, layers, args.pos_stride, args.tail_pos)
    print(f"{len(rows)} readout rows from {args.gen_dir}")

    done: dict[str, dict] = {}
    if args.resume and args.out.exists():
        done = {v["key"]: v for v in json.loads(args.out.read_text())["verdicts"]}

    todo = []  # (row_key, row, readout_text) — one per readout; MC side(s) fan out below
    for r in rows:
        iid = r["label"]
        if iid not in by_id:
            continue
        row_key = f"{iid}:L{r['layer']}:p{r['pos']}"
        readout = "\n".join(s for s in r["samples"] if s.strip())
        if readout.strip():
            todo.append((row_key, r, readout))

    if args.interp:
        cache_path = args.out.with_suffix(".interp_cache.json")
        cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
        pending = [(k, txt) for k, _r, txt in todo if k not in cache]
        print(f"interp stage: {len(pending)} calls ({len(cache)} cached)")
        ires = async_json(
            [(INTERP_SYSTEM, f"TOKEN READOUTS:\n{txt}") for _k, txt in pending],
            schema=INTERP_SCHEMA, model=args.model, concurrency=256,
        )
        for (k, _txt), r_ in zip(pending, ires):
            if r_ is not None:
                cache[k] = r_["interpretation"]
        cache_path.write_text(json.dumps(cache, ensure_ascii=False))
        todo = [(k, r, cache[k]) for k, r, _txt in todo if k in cache]

    prompts, meta = [], []
    for row_key, r, readout in todo:
        iid = r["label"]
        klass = by_id[iid]["reason_class"]
        for side, (q, gold_pos, shown, ndist, short) in mc_of[iid].items():
            vkey = f"{row_key}:{side}"
            if vkey in done:
                continue
            prompts.append((SYSTEM, f"READOUT:\n{readout}\n\n{q}"))
            meta.append((vkey, iid, r["layer"], r["pos"], klass, side, gold_pos, shown, ndist, short))

    print(f"{len(prompts)} judge calls (model={args.model})")
    res = async_json(prompts, schema=SCHEMA, model=args.model, concurrency=256)

    verdicts = list(done.values())
    for m, r_ in zip(meta, res):
        vkey, iid, layer, pos, klass, side, gold_pos, shown, ndist, short = m
        if r_ is None:
            continue
        choice = r_.get("choice")
        quote = (r_.get("quote") or "").strip()
        correct = isinstance(choice, int) and choice == gold_pos
        if not isinstance(choice, int) or not (1 <= choice <= len(shown)):
            pick = "invalid"
        elif shown[choice - 1] == CANNOT:
            pick = "cannot_tell"
        elif correct:
            pick = "gold"
        else:
            pick = "distractor"
        verdicts.append({
            "key": vkey, "id": iid, "layer": layer, "pos": pos, "reason_class": klass,
            "side": side, "choice": choice, "gold_pos": gold_pos, "n_options": len(shown),
            "n_distractors": ndist, "short_pool": short,
            "pick": pick, "correct": bool(correct), "quote": quote,
        })

    # aggregates: per item best-over-grid ANY; committed vs deliberative reported separately
    by_item = defaultdict(list)
    for v in verdicts:
        by_item[v["id"]].append(v)

    def earliest(vs: list[dict], pred) -> int | None:
        hits = [v["layer"] for v in vs if pred(v)]
        return min(hits) if hits else None

    def side_correct(vs: list[dict], side: str) -> bool:
        return any(v["correct"] for v in vs if v["side"] == side)

    committed = {i: vs for i, vs in by_item.items() if by_id[i]["reason_class"] == "committed"}
    delib = {i: vs for i, vs in by_item.items() if by_id[i]["reason_class"] == "deliberative"}

    def per_item_entry(i: str, vs: list[dict]) -> dict:
        if by_id[i]["reason_class"] == "committed":
            return {
                "class": "committed",
                "pass_any": any(v["correct"] for v in vs),
                "earliest_pass_layer": earliest(vs, lambda v: v["correct"]),
            }
        yes_hit = [v for v in vs if v["side"] == "yes" and v["correct"]]
        no_hit = [v for v in vs if v["side"] == "no" and v["correct"]]
        return {
            "class": "deliberative",
            "yes_any": bool(yes_hit),
            "no_any": bool(no_hit),
            "pass_any": bool(yes_hit) and bool(no_hit),
            "earliest_yes_layer": min((v["layer"] for v in yes_hit), default=None),
            "earliest_no_layer": min((v["layer"] for v in no_hit), default=None),
        }

    agg = {
        "items_judged": len(by_item),
        "committed": {
            "total": len(committed),
            "pass_any": sum(1 for vs in committed.values() if any(v["correct"] for v in vs)),
            "chance": "1/6",
        },
        "deliberative": {
            "total": len(delib),
            "both_sides_any": sum(
                1 for vs in delib.values() if side_correct(vs, "yes") and side_correct(vs, "no")
            ),
            "yes_side_any": sum(1 for vs in delib.values() if side_correct(vs, "yes")),
            "no_side_any": sum(1 for vs in delib.values() if side_correct(vs, "no")),
            "chance": "1/36",
        },
        "short_pool_items": sorted(
            i for i, vs in by_item.items() if any(v.get("short_pool") for v in vs)
        ),
        "per_item": {i: per_item_entry(i, vs) for i, vs in sorted(by_item.items())},
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
