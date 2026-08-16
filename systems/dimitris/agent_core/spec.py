"""spec.py — AgentSpec, the module's only public input type.

A caller describes what it wants; `run_agent` does the rest. Nothing
task-specific lives inside agent_core.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from pydantic import BaseModel

from .config import settings


@dataclass(frozen=True)
class AgentSpec:
    """Everything needed to build and run one agent.

    Attributes:
        name: Display name, used in logs.
        instructions: The system prompt. NOTE: the SDK treats this as a format
            string — escape literal braces as {{ }}.
        result_model: Pydantic model the agent must submit. Becomes the
            `submit_result` argument schema.
        tools: Extra @function_tool callables (corpus readers, data clients).
        use_web: Include firecrawl_search / firecrawl_scrape.
        allow_delegation: Include delegate_to_subagent.
        profile: LLM profile override. None = settings.llm_profile.
        reasoning_effort: Override. None = settings.reasoning_effort.
            "" sends no reasoning param at all.
        max_turns: None = settings.agent_max_turns.
        fallback: Instance returned when the agent fails to produce valid
            output. None = a best-effort minimal instance is constructed.
    """

    name: str
    instructions: str
    result_model: type[BaseModel]
    tools: list[Any] = field(default_factory=list)
    use_web: bool = True
    allow_delegation: bool = False
    profile: str | None = None
    reasoning_effort: str | None = None
    max_turns: int | None = None
    fallback: BaseModel | None = None

    # --- resolution ---

    @property
    def resolved_profile(self) -> str:
        return self.profile or settings.llm_profile

    @property
    def resolved_reasoning_effort(self) -> str:
        return (
            self.reasoning_effort
            if self.reasoning_effort is not None
            else settings.reasoning_effort
        )

    @property
    def resolved_max_turns(self) -> int:
        return self.max_turns or settings.agent_max_turns

    def for_subagent(self) -> "AgentSpec":
        """Spec for a delegated child.

        Same tools and prompt, its own turn budget, and delegation switched off
        so the depth cap is enforced structurally as well as by the counter.
        """
        return replace(
            self,
            name=f"{self.name} (sub)",
            allow_delegation=False,
            max_turns=settings.agent_subagent_max_turns,
        )

    def bulk(self) -> "AgentSpec":
        """Same spec on the cheap tier — for bulk summarisation / heavy reads."""
        return replace(self, profile=settings.bulk_profile)
