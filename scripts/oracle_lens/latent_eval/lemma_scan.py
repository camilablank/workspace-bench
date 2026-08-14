# ruff: noqa  (one-off pilot gate; not linted)
"""Gate 1: blocklist scan + gold-label-overlap check for the latent-eval items.

Checks, per item:
  1. BLOCK  (hard fail): any blocklist.block word appears in the stimulus.
  2. WARN   (review):    any blocklist.warn word (sibling-relation terms) appears.
  3. OVERLAP (hard fail): any content word of the gold label appears in the stimulus
     (stopwords/pronouns whitelisted) — the D3 zero-overlap rule.

Blocklists enumerate inflections explicitly (built in build_pilot_items.py), so this is a
whole-word matcher, stdlib-only — no spaCy needed on the shared venv.

Usage: uv run --no-sync python scripts/oracle_lens/latent_eval/lemma_scan.py <items.json>
Exit code 1 if any hard failure.
"""

import json
import re
import sys
from pathlib import Path

STOPWORDS = set(
    """a an the of at over to in on for with while and or but that this those these her his
    hers their its she he him they them it whose own was is are were be been being about into still already
    from by as feels feel felt joined keep kept one ones s t""".split()
)

WORD = re.compile(r"[a-z][a-z'-]*")


def words(text: str) -> list[str]:
    toks = WORD.findall(text.lower().replace("’", "'"))
    out = []
    for t in toks:
        out.append(t)
        if "'" in t:
            out.append(t.split("'")[0])  # colleague's -> colleague
    return out


def main() -> None:
    path = Path(sys.argv[1])
    items = json.loads(path.read_text())
    n_fail = 0
    for it in items:
        stim_words = set(words(it["stimulus"]))
        block_hits = sorted(stim_words & set(it["blocklist"]["block"]))
        warn_hits = sorted(stim_words & set(it["blocklist"]["warn"]))
        label_terms = [w for w in words(it["gold_label"]) if w not in STOPWORDS]
        overlap_hits = sorted(stim_words & set(label_terms))
        status = "FAIL" if (block_hits or overlap_hits) else "pass"
        if status == "FAIL":
            n_fail += 1
        it.setdefault("qa", {})["lemma_scan"] = status
        line = f"{it['id']:8s} {status}"
        if block_hits:
            line += f"  BLOCK: {block_hits}"
        if overlap_hits:
            line += f"  OVERLAP: {overlap_hits}"
        if warn_hits:
            line += f"  warn: {warn_hits}"
        print(line)
    path.write_text(json.dumps(items, indent=2, ensure_ascii=False) + "\n")
    print(f"\n{len(items) - n_fail}/{len(items)} pass gate 1")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
