# ruff: noqa  (one-off pilot gate; not linted)
"""Gate 4 (continuation gate), CPU side: check that Qwen's own continuations do not surface
the composition.

Check 1 (all continuations, both positions): blocklist scan — the model explicitly
verbalizing a label token (e.g. "...she was still grieving") is an instant fail.
Check 2 (end-position continuations, MC items): each continuation is fed to the Opus 5 judge
as a fake readout under the identical 10-way forced choice. Fail if the judge picks the gold
cell from any of the 4 end-position continuations (greedy + 3 sampled).

Usage: uv run --no-sync python scripts/oracle_lens/latent_eval/continuation_gate.py \
    outputs/oracle_latent_eval/pilots/style_a_items.json \
    outputs/oracle_latent_eval/pilots/gpu_gates.json
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from contamination_check import SCHEMA, SYSTEM, build_prompt, option_order  # noqa: E402
from lemma_scan import words  # noqa: E402

from global_workspace.judges.llm_client import CLAUDE_JUDGE, async_json  # noqa: E402


def main() -> None:
    items = {it["id"]: it for it in json.loads(Path(sys.argv[1]).read_text())}
    gpu = json.loads(Path(sys.argv[2]).read_text())

    # ---- check 1: blocklist scan over every continuation ----
    print("== check 1: blocklist scan on continuations ==")
    scan_fail: dict[str, list[str]] = {}
    for rec in gpu:
        it = items[rec["id"]]
        block = set(it["blocklist"]["block"])
        for pos, gens in rec["continuations"].items():
            for k, g in enumerate(gens):
                hits = sorted(set(words(g)) & block)
                if hits:
                    scan_fail.setdefault(rec["id"], []).append(f"{pos}[{k}]: {hits}")
    for iid, fails in scan_fail.items():
        print(f"{iid}: LEAK {fails}")
    print(f"{len(gpu) - len(scan_fail)}/{len(gpu)} items clean on check 1")

    # ---- check 2: judge end-position continuations as fake readouts (MC items only) ----
    print("\n== check 2: judge on end-position continuations ==")
    prompts: list[tuple[str, str]] = []
    meta: list[tuple[str, int, int]] = []  # (item_id, gen_idx, gold_pos)
    for rec in gpu:
        it = items[rec["id"]]
        if not it.get("mc_options"):
            continue
        opts = it["mc_options"]
        order = option_order(it["id"], len(opts))
        gold_pos = order.index(opts.index(it["gold_label"])) + 1
        for k, g in enumerate(rec["continuations"]["end"]):
            prompts.append((SYSTEM, build_prompt(g.strip(), opts, order)))
            meta.append((rec["id"], k, gold_pos))

    results = async_json(prompts, schema=SCHEMA, model=CLAUDE_JUDGE)
    hits_by_item: dict[str, list[int]] = {}
    for (iid, k, gold_pos), res in zip(meta, results):
        choice = res["choice"] if res else -1
        if choice == gold_pos:
            hits_by_item.setdefault(iid, []).append(k)

    n_mc = len({iid for iid, _, _ in meta})
    n_correct = sum(1 for (iid, k, gp), r in zip(meta, results) if r and r["choice"] == gp)
    for rec in gpu:
        it = items[rec["id"]]
        if not it.get("mc_options"):
            continue
        leak = rec["id"] in scan_fail
        judged = rec["id"] in hits_by_item
        status = "FAIL" if (leak or judged) else "pass"
        it.setdefault("qa", {})["continuation"] = status
        extra = f"  judge picked gold on gens {hits_by_item[rec['id']]}" if judged else ""
        print(f"{rec['id']:8s} {status}{extra}")
    print(
        f"\njudge correct-pick rate on continuations: {n_correct}/{len(meta)} "
        f"({n_correct / max(len(meta), 1):.0%}, chance = 10%)"
    )
    Path(sys.argv[1]).write_text(
        json.dumps(list(items.values()), indent=2, ensure_ascii=False) + "\n"
    )


if __name__ == "__main__":
    main()
