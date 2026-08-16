"""agent_core — a reusable headless tool-calling agent.

Give it a Pydantic model, a system prompt and optional tools; get back a
validated instance. Nothing task-specific lives in here.

    from pydantic import BaseModel
    from agent_core import AgentSpec, run_agent

    class Answer(BaseModel):
        city: str = ""
        population: int = 0

    spec = AgentSpec(
        name="Researcher",
        instructions=SYSTEM_PROMPT,
        result_model=Answer,
    )
    answer = await run_agent(spec, "What is the population of Lisbon?")

Built on openai-agents[litellm]. Structured output comes from a
`submit_result` terminal tool whose argument is the caller's Pydantic model,
never `output_type=`.
"""

from .config import UnknownProfileError, settings
from .firecrawl_rest import FirecrawlNotConfigured, scrape_markdown, search_web
from .runner import (
    estimate_cost,
    get_last_usage,
    get_last_tool_calls,
    run_agent,
    run_agent_raw,
    run_agents,
    use_selector_event_loop,
)
from .spec import AgentSpec
from .tools import (
    WEB_TOOLS,
    firecrawl_scrape,
    firecrawl_search,
    make_delegate_to_subagent,
    make_submit_result,
)
from .truncate import truncate_90_10, truncate_head

__all__ = [
    "AgentSpec",
    "run_agent",
    "run_agent_raw",
    "run_agents",
    "use_selector_event_loop",
    "get_last_usage",
    "get_last_tool_calls",
    "estimate_cost",
    "settings",
    "UnknownProfileError",
    "FirecrawlNotConfigured",
    "make_submit_result",
    "make_delegate_to_subagent",
    "firecrawl_search",
    "firecrawl_scrape",
    "WEB_TOOLS",
    "search_web",
    "scrape_markdown",
    "truncate_head",
    "truncate_90_10",
]
