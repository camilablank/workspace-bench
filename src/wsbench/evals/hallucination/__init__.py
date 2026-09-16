"""Hallucination (chat): does the lens make things up about the conversation it is reading."""

from pathlib import Path

from wsbench.judge_config import DEFAULT_JUDGE, JudgeConfig
from wsbench.registry import EvalSpec, register

from . import judge
from .prompts import PROMPT_VERSION

SPEC = register(
    EvalSpec(
        name="hallucination",
        title="Hallucination (chat)",
        group="precision",
        bank=Path("evals/hallucination/items.json"),
        judge=JudgeConfig(
            model=DEFAULT_JUDGE, prompt_version=PROMPT_VERSION, reasoning={"effort": "minimal"}
        ),
        metric="hallucination_rate",
        higher_is_better=False,
        run=judge.run,
        calls_per_arm="≈ 5.6k",
        sources="LMSYS-Chat-1M",
    )
)
