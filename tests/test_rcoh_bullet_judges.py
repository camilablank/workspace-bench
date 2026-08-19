"""CPU tests for readout_coherence passes 4+5 (bullet_judges.py) + their score.py folds."""

import importlib.util
import json
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]


def _load(name: str, rel: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bj = _load("rcoh_bullet_judges", "scripts/oracle_lens_evals/readout_coherence/bullet_judges.py")
sc = _load("rcoh_score", "scripts/oracle_lens_evals/readout_coherence/score.py")


# ------------------------------------------------------------------ bullet parsing


def test_parse_bullets_multiline_and_plain() -> None:
    text = "- first point\ncontinues here\n- second point\n"
    assert bj.parse_bullets(text) == ["first point\ncontinues here", "second point"]
    # non-bullet (continuation_raw style) readouts parse to nothing -> cell skipped
    assert bj.parse_bullets("just a plain continuation of the sentence") == []
    assert bj.parse_bullets("") == []


# ------------------------------------------------------------------ flagged positions


def test_flagged_positions_union_and_gates(tmp_path: Path) -> None:
    def write(d: Path, rows: list[dict[str, Any]]) -> Path:
        d.mkdir(parents=True, exist_ok=True)
        (d / "sumtok_ao.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        return d

    a = write(
        tmp_path / "a",
        [
            {"label": "rc-wildchat-0", "pos": 5, "effective_level": 2, "api_error": False},
            {"label": "rc-wildchat-0", "pos": 6, "effective_level": 1, "api_error": False},  # below
            {
                "label": "rc-wildchat-0",
                "pos": 7,
                "effective_level": 3,
                "api_error": True,
            },  # errored
        ],
    )
    b = write(
        tmp_path / "b",
        [
            {"label": "rc-wildchat-0", "pos": 9, "effective_level": 2, "api_error": False},
            {"label": "rc-dailydialog-1", "pos": 2, "effective_level": 4, "api_error": False},
        ],
    )
    flags = bj.flagged_positions([a, b])
    assert flags == {"rc-wildchat-0": {5, 9}, "rc-dailydialog-1": {2}}


# ------------------------------------------------------------------ score.py folds


def _rel_row(layer: int, relations: list[str], halluc: list[bool]) -> dict[str, Any]:
    bullets = [[f"b{i}" for i in range(len(relations))]]
    verdict = {
        "samples": [
            {
                "bullets": [
                    {"relation": r, "hallucinated": h}
                    for r, h in zip(relations, halluc, strict=True)
                ]
            }
        ]
    }
    return {
        "key": f"k{layer}",
        "label": "rc-wildchat-0",
        "layer": layer,
        "pos": 1,
        "bullets": bullets,
        "api_error": False,
        "verdict": verdict,
    }


def test_relevance_metrics_fold() -> None:
    rows = [
        _rel_row(
            36, ["continuation", "unrelated", "context", "tangential"], [False, True, False, False]
        ),
        _rel_row(40, ["unrelated", "unrelated", "context", "context"], [False, False, True, True]),
        {"key": "err", "api_error": True},
    ]
    m = sc.relevance_metrics(rows)
    assert m["judged"] == 2 and m["api_errors"] == 1 and m["n_bullets"] == 8
    assert m["unrelated_rate"] == round(3 / 8, 4)
    assert m["hallucinated_rate"] == round(3 / 8, 4)
    assert m["relation_shares"]["context"] == round(3 / 8, 4)
    assert m["unrelated_rate_by_layer"] == {36: 0.25, 40: 0.5}


def test_relevance_metrics_truncates_overlong_verdicts() -> None:
    # the judge returned 3 bullet verdicts for a 2-bullet sample -> extra one ignored
    row = _rel_row(36, ["unrelated", "unrelated", "unrelated"], [True, True, True])
    row["bullets"] = [["only", "two"]]
    m = sc.relevance_metrics([row])
    assert m["n_bullets"] == 2 and m["unrelated_rate"] == 1.0


def test_relevance_metrics_empty() -> None:
    assert sc.relevance_metrics([]) == {"judged": 0}
    assert sc.relevance_metrics([{"api_error": True}]) == {"judged": 0}


def test_diversity_metrics_fold() -> None:
    rows = [
        {
            "key": "k1",
            "label": "rc-wildchat-0",
            "layer": 44,
            "pos": 3,
            "api_error": False,
            "verdict": {
                "samples": [
                    {
                        "pairs": [
                            {"i": 0, "j": 1, "relation": "same_topic"},
                            {"i": 0, "j": 2, "relation": "different_concepts"},
                            {"i": 1, "j": 2, "relation": "related_aspects"},
                        ],
                        "n_distinct": 2,
                        "diverse_aspects": True,
                    },
                    {
                        "pairs": [{"i": 0, "j": 1, "relation": "same_topic"}],
                        "n_distinct": 1,
                        "diverse_aspects": False,
                    },
                ]
            },
        }
    ]
    m = sc.diversity_metrics(rows)
    assert m["judged"] == 1 and m["n_samples"] == 2
    assert m["mean_n_distinct"] == 1.5
    assert m["diverse_aspects_rate"] == 0.5
    assert m["pair_relation_shares"]["same_topic"] == 0.5
    assert m["mean_n_distinct_by_layer"] == {44: 1.5}


def test_diversity_metrics_empty() -> None:
    assert sc.diversity_metrics([]) == {"judged": 0}
