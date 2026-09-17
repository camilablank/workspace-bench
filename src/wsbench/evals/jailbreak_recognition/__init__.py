"""Jailbreak recognition: does the lens show Qwen realizing the last user turn is a jailbreak."""

from pathlib import Path

from wsbench.judge_config import DEFAULT_JUDGE, JudgeConfig
from wsbench.registry import EvalSpec, register

from . import judge
from .prompts import PROMPT_VERSION

SPEC = register(
    EvalSpec(
        name="jailbreak_recognition",
        title="Jailbreak recognition",
        group="safety",
        bank=Path("evals/jailbreak_recognition/items.json"),
        # The repo default since 2026-09-17 (jb-v2); jb-v1 was the same prompt on claude-sonnet-5.
        # A refused cell is counted (n_unjudged_cells), never scored — see README §Judge model.
        judge=JudgeConfig(model=DEFAULT_JUDGE, prompt_version=PROMPT_VERSION, reasoning=None),
        metric="pass_rate",
        higher_is_better=True,
        run=judge.run,
        calls_per_arm="≈ 7.5k",
        sources="Zhao et al. 2024",
    )
)
