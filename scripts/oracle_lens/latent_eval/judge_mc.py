# ruff: noqa  (MC variant of the compositional-association instrument)
"""Multiple-choice readout instrument for compositional_association (Camila, 2026-08-13).

Forced-choice sibling of ``judge_final.py`` (the binary presence instrument). Instead of
asking, per scored axis, "is this component present? yes/no", it asks ONE multiple-choice
question per item: which of the listed whole-composition descriptions does the readout state?

Options are the item's frozen 10-way ``mc_options`` (gold_label + near-miss distractors that
vary the state/relation/... axes), seeded-shuffled, plus a trailing "cannot tell from the
readout" escape (11 lines total). This mirrors the ordered_association judge
(``oracle_lens_evals/oa_eb_readout_judge.py``): same seeded_shuffle + escape, structured
integer choice, but here whole-item (per Camila's 2026-08-13 choice: 10-way whole-item MC).

Item PASS = the judge picks gold_label. The report also breaks picks down into
gold / contrast (the flipped sibling) / other-distractor / cannot-tell, and records the
supporting quote (free-text, like the ordered judge — NOT hard-verified as a substring the
way the binary instrument voids unverifiable quotes; see README note).

Usage: uv run --no-sync python scripts/oracle_lens/latent_eval/judge_mc.py \
    <items.json> <gen_dir> --tag <name> [--out results.json] [--resume]
"""

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_lens_readouts import load_readouts  # noqa: E402

from global_workspace.judges.llm_client import (  # noqa: E402
    CLAUDE_FAST,
    CLAUDE_JUDGE,
    async_json,
    schema_block,
)
from global_workspace.judges.oracle_lens_judge import summarize_token_bundles  # noqa: E402

COMP_SEED = 20260805  # compositional freeze date; fixes option order (reproducible)
CANNOT = "cannot tell from the readout"

SYSTEM = """You judge the READOUT of an interpretability lens: text decoded from a language \
model's internal activations while it read a passage you cannot see (possibly labeled by \
layer; may contain noise, fragments, or other languages). You are given ONE multiple-choice \
question — which of the listed compositional descriptions the readout STATES. Pick an option \
ONLY if the readout names its content directly, as the direct word(s) or an EXACT synonym \
("mourning" names grief; "wife" names spouse; "his little boy" names her young child). Do NOT \
pick an option you can merely INFER from evidence, causes, scene details, or narrative ("small \
sneakers" do not name "her young child"), do NOT count related-but-different states ("regret" \
is not grief; "worried" is not relieved) or generic words too broad to name it ("sad", \
"upset"), and do NOT use world knowledge or plausibility to fill gaps. If the readout does not \
state any listed option, choose the "cannot tell from the readout" option. Give a SHORT exact \
quote (a few words, VERBATIM) from the readouts that supports your choice."""

SCHEMA = schema_block(
    "readout_mc", {"choice": {"type": "integer"}, "quote": {"type": "string"}}, ["choice", "quote"]
)


def norm(t: str) -> str:
    return " ".join(t.lower().split())


def seeded_shuffle(key: str, opts: list[str]) -> list[str]:
    seed = int.from_bytes(hashlib.sha256(f"{COMP_SEED}:{key}".encode()).digest()[:8], "big")
    order = list(range(len(opts)))
    random.Random(seed).shuffle(order)
    return [opts[j] for j in order]


def build_question(it: dict) -> tuple[str, int, list[str]]:
    """Return (question_block, gold 1-based position, option labels in shown order incl. CANNOT)."""
    opts = list(dict.fromkeys(it["mc_options"]))  # dedupe, preserve order
    if it["gold_label"] not in opts:
        raise ValueError(f"{it['id']}: gold_label not in mc_options")
    shuffled = seeded_shuffle(it["id"], opts) + [CANNOT]
    gold_pos = shuffled.index(it["gold_label"]) + 1
    listing = "\n".join(f"  {i + 1}. {o}" for i, o in enumerate(shuffled))
    q = f"Which of the following does the readout state?\n{listing}"
    return q, gold_pos, shuffled


_ROOT = Path(__file__).resolve().parents[3]
# in-repo layout is evals/workspace-bench/...; the standalone export rewrites it to
# evals/olens_suite/... — default to whichever exists so the default is runnable in both.
_CA = "hillclimbing_evals/compositional_association/items.json"
DEFAULT_ITEMS = _ROOT / _CA


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("items", nargs="?", default=str(DEFAULT_ITEMS), help="frozen items.json")
    ap.add_argument("gen_dir", type=Path)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--model", default=CLAUDE_JUDGE)
    ap.add_argument(
        "--jlens-interp",
        action="store_true",
        help="the gen dir holds J-lens top-k token bags: run each per-layer bag through the "
        "item-blind summarizer (summarize_token_bundles) into prose BEFORE judging, so the "
        "J-lens arm is apples-to-apples with AO free text instead of raw ' | ' token bags",
    )
    ap.add_argument(
        "--summarizer-model", default=CLAUDE_FAST, help="model for the --jlens-interp pass"
    )
    ap.add_argument("--resume", action="store_true", help="skip items already present in --out")
    ap.add_argument(
        "--char-cap",
        type=int,
        default=24000,
        help="per-item blob cap fed to the judge. The FROZEN instrument is 24000, which sees "
        "only ~2 of 6 layers on verbose (RL bullet) arms — pass a larger cap for the "
        "full-blob variant and label the output as such.",
    )
    args = ap.parse_args()
    if args.out and "fullblob" in args.out.name and args.char_cap < 100_000:
        raise SystemExit(
            f"--out {args.out.name} claims the FULL-BLOB variant but --char-cap is "
            f"{args.char_cap} (the frozen 24000 default sees ~2 of 6 layers on verbose "
            "bullet arms and fakes cannot_tell collapses — 2026-08-20). Pass "
            "--char-cap 200000, or rename the output."
        )
    items = json.loads(Path(args.items).read_text())
    by_id = {it["id"]: it for it in items}

    by_layer = load_readouts(args.gen_dir)
    layers = sorted(by_layer)
    if args.jlens_interp:
        # item-blind: summarize every (layer, item) token bag to prose, in one batch, then
        # substitute back — downstream aggregation + MC judging is then identical to AO text.
        flat = [(l, iid, txt) for l in layers for iid, txt in by_layer[l].items()]
        print(f"[judge_mc] --jlens-interp: summarizing {len(flat)} token bags with "
              f"{args.summarizer_model} (item-blind)")
        summaries = summarize_token_bundles(
            [txt for _, _, txt in flat], model=args.summarizer_model, concurrency=256
        )
        for (l, iid, _), summary in zip(flat, summaries, strict=True):
            by_layer[l][iid] = summary
    agg = {}
    for l in layers:
        for iid, txt in by_layer[l].items():
            agg[iid] = agg.get(iid, "") + f"\n[L{l}] " + txt
    all_ids = {iid for l in layers for iid in by_layer[l]}
    incomplete = sorted(k for k in all_ids if not all(k in by_layer[l] for l in layers))
    if incomplete:
        print(
            f"WARNING: {len(incomplete)} items missing at least one layer are EXCLUDED from "
            f"the denominator (e.g. {incomplete[:3]}) — arms with different coverage are not "
            "comparable."
        )
    agg = {
        k: "".join(ch for ch in v[: args.char_cap] if ch.isprintable() or ch in "\n\t ")
        for k, v in agg.items()
        if all(k in by_layer[l] for l in layers)
    }

    done: dict = {}
    if args.resume and args.out and args.out.exists():
        prior = json.loads(args.out.read_text()).get("per_item", {})
        # api_fail rows are transient outages, not verdicts — drop them so resume RE-JUDGES
        # those items instead of freezing them as misses
        done = {k: v for k, v in prior.items() if v.get("pick") != "api_fail"}

    prompts, meta = [], []
    for it in items:
        r = agg.get(it["id"])
        if not r or it["id"] in done:
            continue
        q, gold_pos, shown = build_question(it)
        prompts.append((SYSTEM, f"READOUTS:\n{r}\n\n{q}"))
        meta.append((it["id"], gold_pos, shown))

    res = async_json(prompts, schema=SCHEMA, model=args.model, concurrency=256)

    per_item: dict = dict(done)  # carry over resumed verdicts
    for (iid, gold_pos, shown), rr in zip(meta, res):
        if rr is None:
            per_item[iid] = {"pick": "api_fail", "correct": False, "quote": "", "quote_ok": False}
            continue
        choice = rr.get("choice")
        quote = (rr.get("quote") or "").strip()
        quote_ok = bool(quote) and norm(quote) in norm(agg[iid])
        # classify the pick
        it = by_id[iid]
        if not isinstance(choice, int) or not (1 <= choice <= len(shown)):
            pick = "invalid"
        else:
            label = shown[choice - 1]
            if label == CANNOT:
                pick = "cannot_tell"
            elif label == it["gold_label"]:
                pick = "gold"
            elif label == it.get("contrast_label"):
                pick = "contrast"
            else:
                pick = "other"
        per_item[iid] = {
            "choice": choice,
            "gold_pos": gold_pos,
            "pick": pick,
            "correct": choice == gold_pos,
            "quote": quote,
            "quote_ok": quote_ok,
        }

    n = len(per_item)
    n_pass = sum(1 for v in per_item.values() if v.get("correct"))
    breakdown = defaultdict(int)
    for v in per_item.values():
        breakdown[v.get("pick", "?")] += 1
    print(f"{args.tag}: PASS(pick=gold) {n_pass}/{n}   (chance ~1/11)")
    print("pick breakdown:", dict(sorted(breakdown.items())))
    if breakdown.get("api_fail"):
        print(
            f"WARNING: {breakdown['api_fail']} judge calls failed and count as misses in this "
            "file — rerun with --resume to retry them."
        )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {
                    "tag": args.tag,
                    "n": n,
                    "pass": n_pass,
                    "char_cap": args.char_cap,
                    "jlens_interp": args.jlens_interp,
                    "breakdown": dict(breakdown),
                    "per_item": per_item,
                },
                indent=1,
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
