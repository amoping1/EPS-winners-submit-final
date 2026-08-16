"""runner.py — run an agent and get a validated Pydantic instance back.

`run_agent` never raises on bad model output. agent-spec.md §6.3: a parse
failure returns a minimal valid instance so a batch can continue. At 17:50 on
deadline day this is the only thing between a malformed tool call and a dead
run.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
import sys
from typing import Any, Sequence, TypeVar

from agents import Runner
from pydantic import BaseModel

from .agent import create_agent
from .config import settings
from .spec import AgentSpec
from .truncate import truncate_head

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# USD per 1M tokens, (input, output). Cached input reads are ~90% cheaper.
_RATES: dict[str, tuple[float, float]] = {
    "gpt-5.6-sol": (5.00, 30.00),
    "gpt-5.6-terra": (2.00, 12.00),
    "gpt-5.6-luna": (0.20, 1.20),
}
_CACHE_DISCOUNT = 0.10

# Usage from the most recent run_agent call *in this task*. A ContextVar, not a
# global, so the concurrent fan-out in run_agents does not cross-contaminate.
_last_usage: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
    "agent_core_last_usage", default=None
)


# Tool calls made during the most recent run_agent call *in this task*. Same
# ContextVar discipline as usage, and for the same reason: four analysts share
# one event loop. Kept here rather than pushed into the run store so agent_core
# stays free of any dependency on runstore — the caller reads it and records it.
_last_tool_calls: contextvars.ContextVar[list[dict] | None] = contextvars.ContextVar(
    "agent_core_last_tool_calls", default=None
)


def get_last_usage() -> dict | None:
    """Token usage and estimated cost of the last run_agent call in this task."""
    return _last_usage.get()


def get_last_tool_calls() -> list[dict]:
    """Tool calls from the last run_agent call in this task, in order.

    Each entry: {"tool", "arguments", "output_chars"}. Without this, "did the
    agent thrash or was it thorough?" cannot be answered from the run store —
    `events` recorded durations but never a single tool invocation.
    """
    return list(_last_tool_calls.get() or [])


def estimate_cost(model: str, prompt: int, cached: int, completion: int) -> float | None:
    """Cost in USD. None when the model's rate is unknown -- better than a wrong number."""
    rate = next((v for k, v in _RATES.items() if k in model), None)
    if rate is None:
        return None
    inp, out = rate
    fresh = max(0, prompt - cached)
    return (
        fresh * inp / 1e6
        + cached * inp * _CACHE_DISCOUNT / 1e6
        + completion * out / 1e6
    )


def _capture_usage(result: Any, model: str) -> dict | None:
    """Pull token usage off a RunResult. Never raises — usage is telemetry."""
    try:
        ctx = getattr(result, "context_wrapper", None)
        usage = getattr(ctx, "usage", None) if ctx else None
        if usage is None:
            return None
        prompt = int(getattr(usage, "input_tokens", 0) or 0)
        completion = int(getattr(usage, "output_tokens", 0) or 0)
        details = getattr(usage, "input_tokens_details", None)
        cached = int(getattr(details, "cached_tokens", 0) or 0) if details else 0
        return {
            "model": model,
            "prompt_tokens": prompt,
            "cached_tokens": cached,
            "completion_tokens": completion,
            "requests": int(getattr(usage, "requests", 0) or 0),
            "cache_hit_rate": round(cached / prompt, 4) if prompt else None,
            "cost_usd": estimate_cost(model, prompt, cached, completion),
        }
    except Exception as e:  # pragma: no cover - telemetry must never break a run
        logger.debug("Could not capture usage: %s", e)
        return None


_ARG_CHARS = 400  # enough to see WHICH document/section, not the whole payload


def _capture_tool_calls(result: Any) -> list[dict]:
    """Read every tool invocation off a RunResult. Never raises — telemetry.

    The SDK exposes the run as a list of items on `new_items`; tool calls appear
    as ToolCallItem (with the request on `raw_item`) and their results as
    ToolCallOutputItem. Both are matched positionally in call order, which is
    how the SDK emits them.
    """
    calls: list[dict] = []
    try:
        pending: list[dict] = []
        for item in getattr(result, "new_items", []) or []:
            kind = type(item).__name__
            raw = getattr(item, "raw_item", None)
            if kind == "ToolCallItem":
                name = getattr(raw, "name", None) or (
                    raw.get("name") if isinstance(raw, dict) else None
                )
                args = getattr(raw, "arguments", None) or (
                    raw.get("arguments") if isinstance(raw, dict) else None
                )
                entry = {
                    "tool": str(name or "?"),
                    "arguments": truncate_head(str(args or ""), _ARG_CHARS),
                    "output_chars": None,
                }
                calls.append(entry)
                pending.append(entry)
            elif kind == "ToolCallOutputItem" and pending:
                out = getattr(item, "output", None)
                pending.pop(0)["output_chars"] = len(str(out)) if out is not None else 0
    except Exception as e:  # pragma: no cover - telemetry must never break a run
        logger.debug("Could not capture tool calls: %s", e)
    return calls


def use_selector_event_loop() -> None:
    """Install the selector loop on Windows.

    The proactor (IOCP) loop segfaults at interpreter exit while tearing down
    the async HTTP client. Call before asyncio.run() for pure-HTTP workloads.
    Do NOT call it if you also drive Playwright — that needs the proactor loop.
    """
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


async def run_agent_raw(spec: AgentSpec, user_input: str) -> str:
    """Run an agent, return the terminal tool's raw output (JSON string).

    Records token usage where the SDK exposes it; read it with get_last_usage().
    """
    agent = create_agent(spec)
    _last_usage.set(None)
    _last_tool_calls.set(None)
    result = await Runner.run(
        starting_agent=agent,
        input=user_input,
        max_turns=spec.resolved_max_turns,
    )
    tool_calls = _capture_tool_calls(result)
    _last_tool_calls.set(tool_calls)
    if tool_calls:
        counts: dict[str, int] = {}
        for c in tool_calls:
            counts[c["tool"]] = counts.get(c["tool"], 0) + 1
        logger.info(
            "%s: %d tool calls — %s",
            spec.name,
            len(tool_calls),
            ", ".join(f"{k}x{v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1])),
        )
    usage = _capture_usage(result, settings.model_for(spec.resolved_profile))
    if usage:
        _last_usage.set(usage)
        logger.info(
            "%s: %d prompt (%d cached, %s hit) + %d completion over %d requests, ~$%.4f",
            spec.name,
            usage["prompt_tokens"],
            usage["cached_tokens"],
            f"{usage['cache_hit_rate']:.0%}" if usage["cache_hit_rate"] is not None else "n/a",
            usage["completion_tokens"],
            usage["requests"],
            usage["cost_usd"] or 0.0,
        )
    return str(result.final_output)


def _fallback(spec: AgentSpec, raw: str, error: str) -> BaseModel:
    """A valid instance to return when the agent's output won't parse."""
    if spec.fallback is not None:
        return spec.fallback
    try:
        # Best effort: many models have every field optional or defaulted.
        return spec.result_model()
    except Exception:
        logger.error(
            "No fallback available for %s and it cannot be default-constructed. "
            "Pass AgentSpec(fallback=...) to make failures survivable. "
            "Parse error was: %s",
            spec.result_model.__name__,
            error,
        )
        raise


async def run_agent(spec: AgentSpec, user_input: str) -> BaseModel:
    """Run an agent and return a validated `spec.result_model` instance.

    Returns the fallback instance on any failure — agent error, max_turns
    exhaustion (final_output is prose, not JSON), or schema mismatch.
    """
    try:
        raw = await run_agent_raw(spec, user_input)
    except Exception as e:
        logger.error("Agent %r failed to run: %s", spec.name, e)
        return _fallback(spec, "", str(e))

    try:
        return spec.result_model.model_validate_json(raw)
    except Exception as e:
        logger.error("Agent %r produced unparseable output: %s", spec.name, e)
        logger.debug("Raw output was: %s", raw[:2000])
        return _fallback(spec, raw, str(e))


async def run_agents(
    specs: Sequence[AgentSpec],
    inputs: Sequence[str],
    max_concurrent: int | None = None,
) -> list[BaseModel]:
    """Run several agents concurrently, preserving input order.

    Safe because nothing is shared: agents are per-task objects, the Firecrawl
    client is stateless HTTP, and delegation depth is a ContextVar. This is the
    path that lets four companies run inside one 45-minute window.
    """
    if len(specs) != len(inputs):
        raise ValueError(
            f"specs and inputs must be the same length "
            f"({len(specs)} vs {len(inputs)})"
        )

    limit = max_concurrent or settings.max_concurrent_agents
    sem = asyncio.Semaphore(limit)

    async def _one(spec: AgentSpec, text: str) -> BaseModel:
        async with sem:
            return await run_agent(spec, text)

    return list(
        await asyncio.gather(*(_one(s, i) for s, i in zip(specs, inputs)))
    )
