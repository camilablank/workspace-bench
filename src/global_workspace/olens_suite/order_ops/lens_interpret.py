"""The LM(J-lens) / LM(O-Lens) arms: an expression-blind judge turns raw readouts into one
numeric estimate per item, which the deterministic machinery then scores like any other reader.

lm_jlens answers "what is INFERABLE from the token bag at the cell" (as opposed to what
literally appears in it); lm_olens is the same treatment of the k OLens samples — a denoiser,
reported beside the raw value and never silently in place of it (spec.READERS).

The judge NEVER sees the expression, the family, the cell, or the item name — otherwise it
could simply compute the intermediate and the arm collapses into spec's `prompt_only` floor.
Fail-open per llm_client: an outage stores null and the coverage count says so.

    python -m global_workspace.olens_suite.order_ops.lens_interpret \\
        reads='["results/order_ops/read_frac.json"]' out=results/order_ops/lens_interpret.json
"""

import json
from pathlib import Path
from typing import Any

import pydra

from global_workspace.judges.llm_client import CLAUDE_FAST, async_json, schema_block
from global_workspace.olens_suite.order_ops.spec import FAMILIES

JLENS_SYSTEM = (
    "You see the top-10 vocabulary tokens read from ONE hidden activation of a language model "
    "that is silently doing arithmetic. Tokens are shown one per line, most salient first, "
    "quoted so whitespace is visible. Infer the single numeric value the activation most "
    "plausibly encodes. If the tokens suggest a fraction, also give it as 'p/q'. If nothing "
    "numeric is inferable, use null."
)
OLENS_SYSTEM = (
    "You see several text samples read out from ONE hidden activation of a language model that "
    "is silently doing arithmetic. The samples are noisy paraphrases of the same underlying "
    "state. Infer the single numeric value the state most plausibly encodes. If the samples "
    "suggest a fraction, also give it as 'p/q'. If nothing numeric is inferable, use null."
)


def payload_rows(items: dict[str, Any], readouts: dict[str, Any]
                 ) -> tuple[list[str], list[tuple[str, str]]]:
    """(keys, (system, user) rows) for every item x arm. Keys are `<name>|<arm>`."""
    keys: list[str] = []
    rows: list[tuple[str, str]] = []
    for name, meta in sorted(items.items()):
        cell = FAMILIES[meta["variant"]]["cell"]
        grid = meta["jlens_top"].get(str(cell["layer"]))
        if grid:
            toks = grid[meta["n_pos"] + cell["pos"]]
            keys.append(f"{name}|jlens")
            rows.append((JLENS_SYSTEM, "\n".join(repr(t) for t in toks)))
        samples = readouts[name]["olens"]["layers"][str(cell["layer"])]
        keys.append(f"{name}|olens")
        rows.append((OLENS_SYSTEM,
                     "\n".join(f"- {' '.join(s.split())}" for s in samples if s.strip())
                     or "- (all samples empty)"))
    return keys, rows


class Config(pydra.Config):  # type: ignore[misc]
    def __init__(self) -> None:
        super().__init__()
        self.reads: list[str] | str = []
        self.out = "results/order_ops/lens_interpret.json"
        self.model = CLAUDE_FAST      # value extraction is nano-class work

    def finalize(self) -> None:
        if isinstance(self.reads, str):
            self.reads = [self.reads]


@pydra.main(Config)  # type: ignore[untyped-decorator]
def main(config: Config) -> None:
    if not config.reads:
        raise SystemExit("reads= is required (read-stage JSON path(s))")
    items: dict[str, Any] = {}
    ro: dict[str, Any] = {}
    for p in config.reads:
        d = json.loads(Path(p).read_text())
        for i in d["items"]:
            items[i["name"]] = {**i["meta"], "jlens_top": i.get("jlens_top", {}),
                                "n_pos": i["n_pos"]}
        for r in d["readouts"]:
            ro[r["name"]] = r
    keys, rows = payload_rows(items, ro)
    schema = schema_block("lens_value", {
        "value": {"type": ["number", "null"]},
        "fraction": {"type": ["string", "null"]},
        "confidence": {"type": "number"},
    }, ["value", "fraction", "confidence"])
    got = async_json(rows, schema=schema, model=config.model)
    out: dict[str, Any] = {"_config": {"model": config.model, "n_rows": len(rows)}}
    graded = 0
    for k, v in zip(keys, got, strict=True):
        name, arm = k.split("|")
        out.setdefault(name, {})[arm] = v
        graded += v is not None
    Path(config.out).write_text(json.dumps(out, indent=1))
    print(f"wrote {config.out} — {graded}/{len(keys)} graded on {config.model}")


if __name__ == "__main__":
    main()
