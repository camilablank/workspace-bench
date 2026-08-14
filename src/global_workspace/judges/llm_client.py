"""Shared async LLM plumbing for the J-lens judge tools (interest judge + grid interpreter).

One structured call: :func:`async_json` takes a batch of ``(system, user)`` prompt pairs plus a
JSON-schema, runs them concurrently, and returns the parsed object per item — or ``None`` on any API
error, so callers fall back to a deterministic proxy (a judge outage must never crash a run). This
mirrors the pattern in :mod:`global_workspace.judges.judge`; the provider SDKs are imported lazily
so the package import (and the proxy-mode tests) never need them installed.

Two backends, routed by model id (Camila 2026-08-04: judging moved to Claude — no rate limit /
free usage on the Anthropic key):
- ``claude-*`` -> Anthropic Messages API with ``output_config.format`` structured outputs
  (needs ``ANTHROPIC_API_KEY``; the SDK auto-retries 429/5xx with backoff).
- anything else -> the original OpenAI Chat Completions path (needs ``OPENAI_API_KEY``).
"""

import asyncio
import json
from collections.abc import Sequence
from typing import Any

DEFAULT_MODEL = "gpt-5.5"  # stronger judge (cleaner positive/negative separation than 5.4-nano)
# Claude judge tiers (r2sf-dp round): opus = the frontier judge (CONT/DESC/JUNK band labels,
# posthoc), haiku = the nano-class bulk work (F2 dedup keep/drop, m3 selection).
CLAUDE_JUDGE = "claude-opus-5"
CLAUDE_FAST = "claude-haiku-4-5"


def schema_block(name: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    """A strict ``json_schema`` response-format block (the Chat Completions structured shape)."""
    return {
        "name": name,
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": required,
            "properties": properties,
        },
    }


async def _one(
    client: Any, system: str, user: str, schema: dict[str, Any], model: str
) -> dict[str, Any] | None:
    """One structured completion; returns the parsed dict, or ``None`` on any error."""
    try:
        resp = await client.chat.completions.create(
            timeout=90,  # a hung call must not stall the whole gather (the 2h-judge failure mode)
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_schema", "json_schema": schema},
        )
        data: dict[str, Any] = json.loads(resp.choices[0].message.content)
        return data
    except Exception as e:  # any API/parse error degrades to the proxy
        print(f"  llm error: {type(e).__name__}: {e}")
        return None


async def _one_claude(
    client: Any, system: str, user: str, schema: dict[str, Any], model: str
) -> dict[str, Any] | None:
    """One Anthropic structured completion; returns the parsed dict, or ``None`` on any error."""
    try:
        resp = await client.messages.create(
            model=model,
            max_tokens=16000,  # cap on thinking + text; Claude Opus 5 thinks by default — 8000
            # truncated ~5% of candidate-heavy label rows (stop_reason=max_tokens, 2026-08-05)
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"format": {"type": "json_schema", "schema": schema["schema"]}},
        )
        if resp.stop_reason in ("refusal", "max_tokens"):  # fail open like any other error
            print(f"  llm error: claude stop_reason={resp.stop_reason}")
            return None
        text = next(b.text for b in resp.content if b.type == "text")
        data: dict[str, Any] = json.loads(text)
        return data
    except Exception as e:  # any API/parse error degrades to the proxy
        print(f"  llm error: {type(e).__name__}: {e}")
        return None


CONCURRENCY = 32  # bound in-flight requests; firing all N at once exhausts the pool and stalls


async def _batch(
    items: list[tuple[str, str]],
    schema: dict[str, Any],
    model: str,
    concurrency: int,
    api_keys: Sequence[str] | None = None,
) -> list[dict[str, Any] | None]:
    # One client (+ own in-flight cap) per key; items round-robin across them. Rate limits are
    # per-org, so two keys from different orgs double the ceiling — ``concurrency`` is PER KEY.
    keys: Sequence[str | None] = api_keys if api_keys else [None]  # None -> env provider key
    claude = model.startswith("claude")
    clients: list[Any]
    if claude:
        from anthropic import AsyncAnthropic  # lazy: only the live path needs the dep

        # timeout > the OpenAI path's 90s: judge calls legitimately think for minutes on
        # Claude Opus 5; max_retries=4 rides out burst 429s (SDK honors retry-after).
        clients = [AsyncAnthropic(api_key=k, timeout=240.0, max_retries=4) for k in keys]
    else:
        from openai import AsyncOpenAI  # lazy: only the live path needs the dep

        clients = [AsyncOpenAI(api_key=k) for k in keys]
    sems = [asyncio.Semaphore(concurrency) for _ in clients]
    one = _one_claude if claude else _one

    async def _guarded(i: int, system: str, user: str) -> dict[str, Any] | None:
        j = i % len(clients)
        async with sems[j]:  # cap concurrent calls — unbounded gather is what hung the 2h judge
            return await one(clients[j], system, user, schema, model)

    try:
        return await asyncio.gather(*(_guarded(i, s, u) for i, (s, u) in enumerate(items)))
    finally:
        for client in clients:
            await client.close()  # close inside the loop (no "event loop closed" warning)


def async_json(
    items: list[tuple[str, str]],
    *,
    schema: dict[str, Any],
    model: str = DEFAULT_MODEL,
    concurrency: int = CONCURRENCY,
    api_keys: Sequence[str] | None = None,
) -> list[dict[str, Any] | None]:
    """Run ``(system, user)`` prompt pairs concurrently (at most ``concurrency`` in flight); each
    result is the parsed object or None.

    ``api_keys`` fans items out round-robin over one client per key (``concurrency`` applies per
    key); default is the single env-key client.

    Returns all-``None`` (never raises) if the client/key is unavailable, so the caller's proxy
    fallback always has something to fall back from.
    """
    if not items:
        return []
    try:
        return asyncio.run(_batch(items, schema, model, concurrency, api_keys))
    except Exception as e:  # missing key / no event loop / import error
        print(f"llm unavailable ({type(e).__name__}: {e}); caller falls back to proxy")
        return [None] * len(items)
