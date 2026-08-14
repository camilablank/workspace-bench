"""Prompt rendering for the bank evals — the ``eval_render`` contract.

The load-bearing contract: **the lens eval feeds the prompt as PLAIN text with NO chat
template.** Two families are chat exceptions because the behavior they probe only exists in the
assistant persona: **directed-modulation** (``chat_dm``) and **basic-readout implicit
prediction** (``chat``, the model's own favorite). The behavioral gate that built the banks was
allowed to use the chat template regardless (it only checked capability); for plain families
that never affects the eval read. Use :func:`render_for_eval` so this cannot be gotten wrong.
"""

from typing import Any


def render_for_eval(item: dict[str, Any], backend: Any) -> str:
    """The exact string to feed the model for the lens read.

    - ``plain`` (every family except the two below): return ``prompt`` verbatim — NO chat template.
    - ``chat_dm`` (directed-modulation): chat-template the instruction and append the
      teacher-forced ``assistant`` carrier (the DM writing-position read).
    - ``chat`` (basic-readout implicit prediction): chat-template the "…? Answer in one word."
      question and append ``prompts.ANSWER_STEM`` so the model's favorite is the next
      leading-space token; the lens reads that final position.

    ``backend`` only needs ``apply_chat_template`` and is untouched for plain items.
    """
    mode = item.get("eval_render", "plain")
    if mode == "plain":
        return str(item["prompt"])
    if mode == "chat_dm":
        prefix = str(backend.apply_chat_template(str(item["prompt"]), enable_thinking=False))
        return prefix + str(item.get("assistant", ""))
    if mode == "chat":
        from global_workspace.prompts import ANSWER_STEM

        rendered = backend.apply_chat_template(str(item["prompt"]), enable_thinking=False)
        return str(rendered) + ANSWER_STEM
    if mode == "chat_context":
        # user modeling: render the full [system, user] conversation; read at the assistant
        # generation position (the model is about to respond, having inferred the user attribute).
        return str(
            backend.apply_chat_template_messages(
                item["messages"], add_generation_prompt=True, enable_thinking=False
            )
        )
    if mode == "chat_sandbag":
        # sandbagging: chat prompt + teacher-forced sandbagged answer "(<wrong_letter>"; the lens
        # reads at that final answer-letter token (the deliberately-wrong choice the model gave).
        prefix = str(backend.apply_chat_template(str(item["prompt"]), enable_thinking=False))
        return prefix + str(item["answer_prefill"])
    raise ValueError(f"unknown eval_render {mode!r} for item {item.get('name')!r}")
