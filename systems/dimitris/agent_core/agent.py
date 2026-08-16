"""agent.py — the agent factory.

No singleton. agent-spec.md §3.1 caches one agent per process, which is what
forces its "single-process, single-agent" caveat. Agents are cheap objects;
building one per task is what makes run_agents() concurrency safe.
"""

from __future__ import annotations

import logging
from typing import Any

import litellm
from agents import Agent, ModelSettings, StopAtTools, set_tracing_disabled
from agents.extensions.models.litellm_model import LitellmModel
from openai.types.shared import Reasoning

from .config import settings
from .spec import AgentSpec
from .tools import WEB_TOOLS, make_delegate_to_subagent, make_submit_result

logger = logging.getLogger(__name__)

# Drop provider-unsupported params (e.g. reasoning_effort on DeepInfra/NIM)
# instead of raising UnsupportedParamsError. This is the escape hatch
# agent-spec.md §3.2 names, and it is what lets us send reasoning on OpenAI
# while keeping the non-OpenAI profiles usable.
litellm.drop_params = True

# LiteLLM's aiohttp transport GCs its ClientSession on the proactor loop at
# interpreter exit, which segfaults on Windows. httpx is the transport the rest
# of the stack already uses.
litellm.disable_aiohttp_transport = True

TERMINAL_TOOL_NAME = "submit_result"


def build_model(profile: str) -> LitellmModel:
    """LitellmModel for a named profile. Routing is by model-string prefix."""
    kwargs: dict[str, Any] = {
        "model": settings.model_for(profile),
        "api_key": settings.api_key_for(profile),
    }
    api_base = settings.api_base_for(profile)
    if api_base:
        kwargs["base_url"] = api_base
    return LitellmModel(**kwargs)


def build_model_settings(reasoning_effort: str) -> ModelSettings:
    """ModelSettings, with reasoning only when an effort is set.

    Empty string means send nothing — the bare-ModelSettings behaviour
    agent-spec.md §3.1 defaults to.
    """
    kwargs: dict[str, Any] = {"tool_choice": "auto"}
    if reasoning_effort:
        kwargs["reasoning"] = Reasoning(effort=reasoning_effort)
    return ModelSettings(**kwargs)


def create_agent(spec: AgentSpec) -> Agent:
    """Build an agent from a spec. Fresh object every call."""
    set_tracing_disabled(disabled=True)  # otherwise the SDK spams trace warnings

    tools: list[Any] = []
    if spec.use_web:
        tools.extend(WEB_TOOLS)
    tools.extend(spec.tools)
    if spec.allow_delegation:
        tools.append(make_delegate_to_subagent(spec))
    tools.append(make_submit_result(spec.result_model))

    profile = spec.resolved_profile
    agent = Agent(
        name=spec.name,
        instructions=spec.instructions,
        model=build_model(profile),
        model_settings=build_model_settings(spec.resolved_reasoning_effort),
        tools=tools,
        # Without this the agent keeps calling tools after submitting.
        tool_use_behavior=StopAtTools(stop_at_tool_names=[TERMINAL_TOOL_NAME]),
    )
    logger.info(
        "Agent %r built — profile=%s model=%s reasoning=%s tools=%d",
        spec.name,
        profile,
        settings.model_for(profile),
        spec.resolved_reasoning_effort or "(none)",
        len(tools),
    )
    return agent
