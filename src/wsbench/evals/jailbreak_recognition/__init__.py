"""Jailbreak recognition: does the lens show Qwen realizing the last user turn is a jailbreak."""

from pathlib import Path

from wsbench.judge_config import JudgeConfig
from wsbench.registry import EvalSpec, register

from . import judge
from .prompts import PROMPT_VERSION

SPEC = register(
    EvalSpec(
        name="jailbreak_recognition",
        title="Jailbreak recognition",
        group="safety",
        bank=Path("evals/jailbreak_recognition/items.json"),
        judge=JudgeConfig(prompt_version=PROMPT_VERSION),
        metric="pass_rate",
        higher_is_better=True,
        run=judge.run,
    )
)
