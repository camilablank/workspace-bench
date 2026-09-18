"""The capability runner: the questions come from the banks, and the gate is drawn from graded
answers with identical answers graded once."""

import json

from wsbench.capable import questions as capq
from wsbench.capable import run as caprun
from wsbench.cli import main
from wsbench.judge_config import JudgeConfig, resolve
from wsbench.mcjudge import load_bank


def test_every_family_builds_answerable_questions():
    for family in sorted(capq.BUILDERS):
        qs = capq.build(family)
        assert qs, family
        assert len({q.id for q in qs}) == len(qs), family
        for q in qs:
            assert q.ask.strip() and all(g.strip() for g in q.golds), (family, q.id)


def test_question_ids_and_golds_come_from_the_bank():
    bank = {it["id"]: it for it in load_bank("relational_multihop")}
    for q in capq.build("relational_multihop"):
        assert q.golds[0] == bank[q.id]["gold"]
        assert bank[q.id]["stimulus"] in q.ask
    moral = {it["id"]: it for it in load_bank("moral_rationale")}
    for q in capq.build("moral_rationale"):
        # not a right answer: the side the bank recorded the model committing to
        it = moral[q.id]
        words = capq._DIRECTION.get(str(it["answer_format"]), capq._DIRECTION["yes_no"])
        assert q.golds == [words[it["commit_direction"]]]
        assert q.meta["direction"] == moral[q.id]["commit_direction"]


def test_no_question_families_say_what_their_gate_is():
    from wsbench import registry

    registry.load_all()
    covered = set(capq.BUILDERS) | set(capq.NO_QUESTION)
    assert set(registry.FAMILIES) <= covered
    for family, why in capq.NO_QUESTION.items():
        assert why.strip()
        try:
            capq.build(family)
        except KeyError as e:
            assert why in str(e)
        else:
            raise AssertionError(f"{family} should have no capability question")


def test_scripted_run_grades_each_distinct_answer_once(tmp_path, monkeypatch):
    calls_seen = []

    def fake_run_calls(calls, **kw):
        calls_seen.append(calls)
        out = {}
        for c in calls:
            if c.system == caprun.ANSWER_SYSTEM:
                draw = c.meta["draw"]
                # item 0 answers right 3 of 4 draws, item 1 always wrong, one call fails
                if draw == 3 and c.meta["item"].endswith("0"):
                    out[c.key] = None
                else:
                    out[c.key] = {"answer": "right" if draw < 3 else "wrong"}
            else:
                out[c.key] = {"correct": c.meta["answer"] == "right", "why": "test"}
        return out

    monkeypatch.setattr(caprun, "run_calls", fake_run_calls)
    monkeypatch.setattr(
        caprun,
        "build",
        lambda family: [
            capq.Question("item-0", "ask 0", ["gold"]),
            capq.Question("item-1", "ask 1", ["gold"]),
        ],
    )
    judge = resolve(JudgeConfig(prompt_version=caprun.PROMPT_VERSION))
    r = caprun.run_family(
        "poetry", model="test/model", judge=judge, out=tmp_path, draws=4, threshold=0.8
    )

    answer_calls, grade_calls = calls_seen
    assert len(answer_calls) == 8  # 2 items x 4 draws
    # item 0: "right" x3 (one call failed); item 1: "right" x3 + "wrong" -> 3 distinct answers
    assert len(grade_calls) == 3
    rows = {row["id"]: row for row in r["rows"]}
    assert rows["item-0"]["rate"] == 1.0 and rows["item-0"]["passes_gate"]
    assert rows["item-1"]["rate"] == 0.75 and not rows["item-1"]["passes_gate"]
    a = r["aggregate"]
    assert a["gate_rate"] == 0.5 and a["mean_accuracy"] == 0.875
    assert a["n_answers_failed"] == 1 and a["n_items_decided"] == 2
    assert "poetry on test/model: gate=0.500" in caprun.report_line(r)


def test_unjudged_item_leaves_the_denominator(tmp_path, monkeypatch):
    def fake_run_calls(calls, **kw):
        return {
            c.key: ({"answer": "x"} if c.system == caprun.ANSWER_SYSTEM else None) for c in calls
        }

    monkeypatch.setattr(caprun, "run_calls", fake_run_calls)
    monkeypatch.setattr(caprun, "build", lambda family: [capq.Question("i", "ask", ["gold"])])
    judge = resolve(JudgeConfig(prompt_version=caprun.PROMPT_VERSION))
    r = caprun.run_family("poetry", model="m", judge=judge, out=tmp_path, draws=2)
    assert r["rows"][0]["rate"] is None and r["rows"][0]["passes_gate"] is None
    assert r["aggregate"]["gate_rate"] is None and r["aggregate"]["n_items_decided"] == 0


def test_cli_dry_run_and_unknown_family(tmp_path, capsys):
    assert main(["capable", "model=m", "families=poetry", "limit=2", "dry_run=True"]) == 0
    assert "2 questions, no calls made" in capsys.readouterr().out
    assert main(["capable", "model=m", "families=hallucination"]) == 2
    assert "no capability question" in capsys.readouterr().err


def test_cli_writes_one_file_per_family(tmp_path, monkeypatch):
    monkeypatch.setattr(
        caprun,
        "run_calls",
        lambda calls, **kw: {
            c.key: (
                {"answer": "a"}
                if c.system == caprun.ANSWER_SYSTEM
                else {"correct": True, "why": ""}
            )
            for c in calls
        },
    )
    assert (
        main(["capable", "model=m", "families=poetry", "limit=2", "draws=1", f"out={tmp_path}"])
        == 0
    )
    written = json.loads((tmp_path / "poetry" / "capable.json").read_text())
    assert written["model"] == "m" and written["aggregate"]["gate_rate"] == 1.0
