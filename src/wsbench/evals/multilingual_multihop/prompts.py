"""No judge prompt: this family is scored by regex. Token readouts go through the shared
summarizer (``wsbench.summarizer``, prompt in ``docs/summarizer.md``)."""

from wsbench.hard.family import SCORER_VERSION as PROMPT_VERSION

PROMPTS: dict[str, str] = {}

__all__ = ["PROMPTS", "PROMPT_VERSION"]
