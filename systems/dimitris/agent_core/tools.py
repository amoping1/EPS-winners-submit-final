"""tools.py — the reusable tool set.

Three groups:
  web        firecrawl_search / firecrawl_scrape (plain @function_tool, REST)
  terminal   make_submit_result(ModelCls) — the structured-output factory
  delegate   make_delegate_to_subagent(spec) — fresh agent, own context

The structured-output pattern (agent-spec.md §6): the terminal tool takes a
Pydantic model as its argument, so *function calling* enforces the schema. We
never use `output_type=` — that sets response_format=json_schema on the LLM
call, which collides with function tools on several providers.
"""

from __future__ import annotations

import contextvars
import json
import logging
from typing import TYPE_CHECKING, Any, Callable

from agents import function_tool
from pydantic import BaseModel

from .config import settings
from .firecrawl_rest import scrape_markdown, search_web
from .truncate import truncate_head

if TYPE_CHECKING:  # pragma: no cover
    from .spec import AgentSpec

logger = logging.getLogger(__name__)

# Delegation depth. A ContextVar rather than agent-spec.md §4.5's module global:
# a global is wrong under asyncio concurrency, where several agents share one
# event loop. ContextVars are per-task, so run_agents() fan-out is safe.
_call_depth: contextvars.ContextVar[int] = contextvars.ContextVar(
    "agent_core_call_depth", default=0
)


# --------------------------------------------------------------------------
# Web tools
# --------------------------------------------------------------------------


@function_tool
async def firecrawl_search(query: str, limit: int = 10) -> str:
    """Search the web. Returns a JSON list of {url, title, description}.

    Use sparingly — prefer 1-3 well-formed queries over many narrow ones.

    Args:
        query: The search query.
        limit: Max results to return (1-10).
    """
    try:
        results = search_web(query, limit=limit)
    except Exception as e:
        logger.warning("firecrawl_search error for %r: %s", query, e)
        return f"Error: search failed ({e})"
    if not results:
        return f"No results for '{query}'."
    return json.dumps(results, indent=2)


@function_tool
async def firecrawl_scrape(url: str) -> str:
    """Fetch a web page as clean markdown (truncated).

    Args:
        url: The full URL to fetch.
    """
    try:
        markdown = scrape_markdown(url)
    except Exception as e:
        logger.warning("firecrawl_scrape error for %s: %s", url, e)
        return f"Error: scrape failed ({e})"
    return markdown or f"No content scraped from {url}."


WEB_TOOLS: list[Any] = [firecrawl_search, firecrawl_scrape]


# --------------------------------------------------------------------------
# Terminal tool — the structured-output factory
# --------------------------------------------------------------------------


def make_submit_result(model_cls: type[BaseModel]) -> Any:
    """Build a `submit_result` tool whose argument is `model_cls`.

    agent-spec.md §4.6 hardcodes a module-level ResultModel, which welds the
    agent to a single task. Closing over the caller's model is the change that
    makes this package reusable: `@function_tool` derives the JSON schema from
    the annotation, and the closure supplies a concrete one.
    """

    async def submit_result(result: model_cls) -> str:  # type: ignore[valid-type]
        """Submit your structured findings. Call this exactly once when done.

        Args:
            result: The structured result for this task.
        """
        return result.model_dump_json()

    # Name the function before decorating — the SDK reads __name__ for the
    # tool name, and StopAtTools matches on it.
    submit_result.__name__ = "submit_result"
    # `from __future__ import annotations` makes every annotation a *string*,
    # and the SDK resolves it against module globals — where a closure local
    # like `model_cls` does not exist. Bind the real class object instead.
    submit_result.__annotations__ = {"result": model_cls, "return": str}
    return function_tool(submit_result)


# --------------------------------------------------------------------------
# Delegation
# --------------------------------------------------------------------------


def make_delegate_to_subagent(spec: "AgentSpec") -> Any:
    """Build a `delegate_to_subagent` tool bound to `spec`.

    The subagent is built from the same spec, so it gets the same tools --
    including web access. Under agent-spec.md's MCP design this was impossible
    (§9.6: subagents spawn without search/scrape and hallucinate); with plain
    REST function tools it falls out for free.

    The subagent sees only the task string, never the parent's conversation.
    """

    async def delegate_to_subagent(task: str) -> str:
        """Delegate a focused task to a fresh subagent in its own context.

        Use for deep searches or bulk reading where you only want the result,
        not the intermediate tool output. Write a self-contained task — the
        subagent cannot see your conversation. End by telling it to call
        submit_result.

        Args:
            task: A self-contained task description for the subagent.
        """
        depth = _call_depth.get()
        if depth >= settings.max_delegation_depth:
            return "ERROR: max delegation depth reached. Do the task yourself."

        token = _call_depth.set(depth + 1)
        try:
            from .runner import run_agent_raw  # local import: circular at module level

            sub_spec = spec.for_subagent()
            logger.info("Subagent starting: %s", task[:200])
            answer = await run_agent_raw(sub_spec, task)
            logger.info("Subagent completed.")
            return truncate_head(answer)
        except Exception as e:
            logger.exception("Subagent failed")
            return f"ERROR: subagent failed - {e}"
        finally:
            _call_depth.reset(token)

    delegate_to_subagent.__name__ = "delegate_to_subagent"
    return function_tool(delegate_to_subagent)


def current_depth() -> int:
    """Delegation depth of the running task. Exposed for tests."""
    return _call_depth.get()
