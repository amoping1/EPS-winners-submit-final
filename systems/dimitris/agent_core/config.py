"""config.py — profile-driven, provider-agnostic LLM config.

Follows agent-spec.md §2. Adding a provider is one dict entry, no code change.
The whole stack flips on the LLM_PROFILE env var.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load .env into os.environ at import, before anything resolves a key.
#
# This line is load-bearing. Provider keys are read with os.getenv (they are
# not Settings fields), but pydantic-settings only reads .env into the Settings
# OBJECT -- it never exports to the process environment. Without this, a correct
# key in a correct .env is invisible to the code, every model call fails auth,
# the runner's fallback path catches it exactly as designed, and the run
# COMPLETES WITH BLANK RESULTS instead of failing. That is the worst possible
# failure shape: it uploads cleanly and scores the maximum penalty.
#
# Anchored to the repo root rather than cwd, because find_dotenv() resolves
# differently under `python -m pkg.mod` than under `python -c`, which is how
# this stayed hidden.
_REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_REPO_ROOT / ".env", override=False)

# Adding a provider = one dict entry. No code change elsewhere.
#
# OpenAI GPT-5.6 family (GA 9 Jul 2026), priced per 1M input/output tokens:
#   sol   $5.00 / $30.00  flagship
#   terra $2.00 / $12.00  balanced      <- our default driver
#   luna  $0.20 /  $1.20  bulk/cheap    <- summarisation, high-volume reads
# The bare "gpt-5.6" alias routes to sol; we always name the tier explicitly.
_LLM_PROFILES: dict[str, dict[str, str]] = {
    # --- OpenAI (primary) ---
    "openai_terra": {
        "model": "openai/gpt-5.6-terra",
        "api_key_env": "OPENAI_API_KEY",
        "api_base_env": "",
    },
    "openai_luna": {
        "model": "openai/gpt-5.6-luna",
        "api_key_env": "OPENAI_API_KEY",
        "api_base_env": "",
    },
    "openai_sol": {
        "model": "openai/gpt-5.6-sol",
        "api_key_env": "OPENAI_API_KEY",
        "api_base_env": "",
    },
    # --- Fallbacks carried over from agent-spec.md §2.1 ---
    "gemini": {
        "model": "gemini/gemini-3.1-flash-lite-preview",
        "api_key_env": "GEMINI_API_KEY",
        "api_base_env": "",
    },
    "deepinfra": {
        "model": "deepinfra/google/gemma-4-31B-it",
        "api_key_env": "DEEPINFRA_API_KEY",
        "api_base_env": "",
    },
    "nim": {
        "model": "nvidia_nim/google/gemma-4-31b-it",
        "api_key_env": "NVIDIA_NIM_API_KEY",
        "api_base_env": "",
    },
    "together": {
        "model": "together_ai/google/gemma-4-31B-it",
        "api_key_env": "TOGETHER_API_KEY",
        "api_base_env": "",
    },
    "local": {
        # llama.cpp llama-server (OpenAI-compatible). "openai/<name>" prefix
        # tells LiteLLM to route via api_base.
        "model": "openai/qwen",
        "api_key_env": "LOCAL_API_KEY",
        "api_base_env": "LOCAL_API_BASE",
    },
    "local_gemma": {
        "model": "openai/gemma",
        "api_key_env": "LOCAL_API_KEY",
        "api_base_env": "LOCAL_API_BASE_GEMMA",
    },
}

# GPT-5.6 accepts these. Anything else is dropped by litellm.drop_params.
VALID_REASONING_EFFORTS = frozenset(
    {"none", "low", "medium", "high", "xhigh", "max"}
)


class UnknownProfileError(ValueError):
    """Raised at import time for a bad LLM_PROFILE, never mid-run."""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Master switch ---
    llm_profile: str = "openai_terra"

    # --- Cheap tier for bulk summarisation / high-volume document reads ---
    bulk_profile: str = "openai_luna"

    # --- Optional overrides (blank = use profile default) ---
    agent_model: str = ""
    agent_api_base: str = ""

    # --- Reasoning ---
    # agent-spec.md §3.1 mandates bare ModelSettings() for six-provider
    # portability. We are OpenAI-first, so we take the §3.2 escape hatch:
    # litellm.drop_params=True plus an explicit effort. Set to "" to send
    # nothing at all (spec-default behaviour).
    reasoning_effort: str = "medium"

    # --- Agent loop ---
    agent_max_turns: int = 15
    agent_subagent_max_turns: int = 30

    # --- Concurrency (run_agents) ---
    max_concurrent_agents: int = 4

    # --- Tool output limits ---
    tool_output_max_chars: int = 8000
    firecrawl_max_content_length: int = 30000

    # --- Firecrawl REST (NOT MCP — see agent_core/README.md "Why REST") ---
    firecrawl_api_key: str = ""
    firecrawl_api_url: str = "https://api.firecrawl.dev"
    firecrawl_search_timeout: int = 30
    firecrawl_scrape_timeout: int = 45
    firecrawl_max_results: int = 10

    # --- Delegation ---
    max_delegation_depth: int = 1

    def _profile(self, name: str) -> dict[str, str]:
        try:
            return _LLM_PROFILES[name]
        except KeyError:
            raise UnknownProfileError(
                f"Unknown LLM profile {name!r}. "
                f"Known profiles: {', '.join(sorted(_LLM_PROFILES))}"
            ) from None

    # --- Main profile resolution ---

    @property
    def resolved_agent_model(self) -> str:
        return self.agent_model or self._profile(self.llm_profile)["model"]

    @property
    def resolved_api_key(self) -> str:
        env = self._profile(self.llm_profile)["api_key_env"]
        return os.getenv(env, "") if env else ""

    @property
    def resolved_api_base(self) -> str:
        if self.agent_api_base:
            return self.agent_api_base
        env = self._profile(self.llm_profile).get("api_base_env", "")
        return os.getenv(env, "") if env else ""

    # --- Named-profile resolution (for the bulk tier, or any explicit pick) ---

    def model_for(self, profile: str) -> str:
        return self._profile(profile)["model"]

    def api_key_for(self, profile: str) -> str:
        env = self._profile(profile)["api_key_env"]
        return os.getenv(env, "") if env else ""

    def api_base_for(self, profile: str) -> str:
        env = self._profile(profile).get("api_base_env", "")
        return os.getenv(env, "") if env else ""

    def validate_profiles(self) -> None:
        """Fail loudly at import, not at 17:50 mid-run."""
        self._profile(self.llm_profile)
        self._profile(self.bulk_profile)
        if self.reasoning_effort and self.reasoning_effort not in VALID_REASONING_EFFORTS:
            raise ValueError(
                f"reasoning_effort={self.reasoning_effort!r} is not one of "
                f"{sorted(VALID_REASONING_EFFORTS)} (or '' to send nothing)"
            )


settings = Settings()
settings.validate_profiles()


class MissingAPIKeyError(RuntimeError):
    """Raised before a run starts, never discovered halfway through."""


def require_api_key(profile: str | None = None) -> None:
    """Fail loudly if the active profile has no key.

    Call this before any run. Without it the failure is silent: litellm raises
    AuthenticationError, the runner's fallback returns an empty model, and the
    pipeline finishes with blank numbers that look like output.
    """
    p = profile or settings.llm_profile
    if not settings.api_key_for(p):
        env = _LLM_PROFILES[p]["api_key_env"]
        raise MissingAPIKeyError(
            f"No API key for profile {p!r}: {env} is not set.\n"
            f"Add it to {_REPO_ROOT / '.env'} as {env}=...\n"
            f"Refusing to start — a run without a key produces blank results, not an error."
        )
