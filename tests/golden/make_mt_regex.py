"""Goldens for the multi-token regex scorer, computed by the SOURCE repo's own code
(``global_workspace.olens_suite.bank.{matching,contract,conjunctive}``), so the port in
``wsbench.multitoken.regex`` is pinned to the instrument it replaces the MC judge with:

- ``mt_regex_units.json``: per family, per item, the scored units (role, required, forms) from
  the source ``scored_units`` with the token counts read from the bank's ``probe_token_lens``
  stamp (the strict Qwen3.6-27B count capture used; verified equal to the capture manifests).
- ``mt_regex_matching.json``: ``unicode_word_matcher`` / ``hit_forms`` verdicts on a fixed set
  of (form, sample) pairs covering every rule branch, plus ``item_result`` on a scripted grid.

Run: cd <this repo> && uv run --no-sync python tests/golden/make_mt_regex.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _source import HERE, R

sys.path.insert(0, str(R / "src"))
from global_workspace.olens_suite.bank.conjunctive import item_result, layer_unit_hits
from global_workspace.olens_suite.bank.contract import (
    ScoredUnit,
    contract_for,
    scored_units,
)
from global_workspace.olens_suite.bank.matching import hit_forms, unicode_word_matcher

REPO = HERE.parents[1]
FAMILIES = [
    "basic_readout_mt",
    "multihop_mt",
    "multilingual_mt",
    "typo_mt",
    "multilingual_multihop",
    "multilingual_typo",
]

PAIRS: list[tuple[str, str]] = [
    # wordy forms: boundary, hyphen/space, fold, plural is NOT the word
    ("Bohr magneton", "the value is the **Bohr-magneton** here"),
    ("Bohr magneton", "bohr magnetons everywhere"),
    ("Simula 67", "designed Simula 67 with Nygaard"),
    ("Simula 67", "Simula 670 is not it"),
    ("kente", "kente cloth"),
    ("kente", "kentes"),
    ("Mexico", "México lindo"),
    ("Glauber's salt", "known as Glauber\u2019s salt"),
    ("hourglass", "an hourglass-shaped thing"),
    ("hourglass", "hourglasses"),
    ("the Irrawaddy", "called the Irrawaddy River"),
    ("Ayn Jalut", "Ain Jalut"),
    # a single top-k token can never hold a two-word form; a phrase token can
    ("Aghlabid dynasty", " Aghlabid"),
    ("Aghlabid dynasty", "Aghlabid dynasty"),
    ("Aghlabid dynasty", " dynasty"),
    # CJK / Hangul / Arabic / Hebrew / Cyrillic forms: substring or folded boundary
    ("阿因贾鲁特战役", "答案：阿因贾鲁特战役。"),  # noqa: RUF001
    ("沙漏", "这是沙漏的例子"),
    ("沙漏", "沙 漏"),
    ("손기정", "마라톤 영웅 손기정입니다"),
    ("보로부두르", "보로부두르는 자바섬에"),
    ("عين جالوت", "معركة عين جالوت سنة 1260"),
    ("عين جالوت", "عَين جالوت"),
    ("Байконур", "космодром Байконур в Казахстане"),
    ("Байконур", "Байконуре"),
    ("טריטון", "הירח טריטון"),
    ("Arabic", "written in Arabic"),
    ("Arabic", "Arabica beans"),
    # numeric forms: answer position only
    ("435", "answer: 435"),
    ("435", "= 435"),
    ("435", "435"),
    ("435", "**435**"),
    ("435", "the 435 people"),
    ("435", "4350"),
    ("435", "= 4351"),
    ("435", "答案是435"),
    ("435", "结果：四百三十五"),  # noqa: RUF001
    ("435", "435 + 1 = 436"),
    ("435", "Answer: 435"),
    ("58", "is 58"),
    ("58", "五十八"),
    ("58", "五十八个"),
    # scaffolding is stripped by the caller, not here: a raw tag stays a miss
    ("interpretability", "<|im_start|>assistant interpretability"),
]

FORMS = {"en": ["Battle of Ain Jalut", "Ayn Jalut"], "zh": ["阿因贾鲁特战役", "艾因贾卢特"]}
LANG_FORMS = {"en": ["Arabic"], "zh": ["阿拉伯语"], "ar": ["عين جالوت", "موقعة عين جالوت"]}
GRID_SAMPLES = {
    20: ["nothing here"],
    24: ["Ayn Jalut was in 1260"],  # readout hits, language does not
    28: ["battle of ain jalut, in Arabic"],  # both hit
    32: ["阿因贾鲁特战役 عين جالوت"],  # both hit, in zh / ar
}


def ntok_from_probe(item: dict) -> "callable":
    """The bank's stamp as the ``ntok`` callable the source expects (string -> count)."""
    table: dict[str, int] = {}
    ptl = item["probe_token_lens"]
    for u in item["units"]:
        for lang, fs in u["forms"].items():
            for f, n in zip(fs, ptl["units"][u["role"]][lang], strict=True):
                table[str(f)] = int(n)
    table[str(item["target"])] = int(ptl["target"])
    for a, n in zip(item.get("target_alts", []), ptl.get("target_alts", []), strict=False):
        table[str(a)] = int(n)

    def ntok(s: str) -> int:
        return table[s]

    return ntok


def main() -> None:
    units_golden: dict[str, dict[str, list[dict]]] = {}
    for fam in FAMILIES:
        d = json.loads((REPO / "evals" / fam / "items.json").read_text(encoding="utf-8"))
        contract = contract_for(d, d["family"])
        per: dict[str, list[dict]] = {}
        for it in d["items"]:
            units = scored_units(
                it, contract, ntok_from_probe(it) if contract.multi_token else None
            )
            per[it["name"]] = [u.to_json() for u in units]
        units_golden[fam] = per
    matching = {
        "pairs": [
            {"form": f, "sample": s, "hit": bool(unicode_word_matcher(f)(s))} for f, s in PAIRS
        ],
        "hit_forms": {
            s: hit_forms([s], FORMS)
            for s in ("Ayn Jalut said", "阿因贾鲁特战役", "nothing", "battle of ain jalut")
        },
        "grid": item_result(
            {
                layer: layer_unit_hits(
                    [{"samples": ss}],
                    [
                        ScoredUnit("readout", True, FORMS),
                        ScoredUnit("language", True, LANG_FORMS),
                    ],
                )
                for layer, ss in GRID_SAMPLES.items()
            },
            [ScoredUnit("readout", True, FORMS), ScoredUnit("language", True, LANG_FORMS)],
        ),
    }
    (HERE / "mt_regex_units.json").write_text(
        json.dumps(units_golden, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    (HERE / "mt_regex_matching.json").write_text(
        json.dumps(matching, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    n_items = sum(len(v) for v in units_golden.values())
    print(f"mt_regex_units.json: {n_items} items; mt_regex_matching.json: {len(PAIRS)} pairs")


if __name__ == "__main__":
    main()
