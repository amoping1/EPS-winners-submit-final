"""smoke_test.py — prove the runtime works before any task depends on it.

Deliberately neutral: no challenge company, no forecasting, no corpus. Runs
before the tools layer exists.

    python -m agent_core.smoke_test           # offline checks + live LLM checks
    python -m agent_core.smoke_test --offline # no network, no spend
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import time
from typing import Any, Callable

from pydantic import BaseModel

from .agent import build_model_settings, create_agent
from .config import UnknownProfileError, Settings, settings
from .runner import run_agent, run_agents, use_selector_event_loop
from .spec import AgentSpec
from .tools import current_depth, make_submit_result
from .truncate import truncate_90_10, truncate_head

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, fn: Callable[[], Any]) -> None:
    try:
        detail = fn()
        RESULTS.append((name, True, str(detail or "ok")))
    except Exception as e:
        RESULTS.append((name, False, f"{type(e).__name__}: {e}"))


async def acheck(name: str, coro_fn: Callable[[], Any]) -> None:
    try:
        detail = await coro_fn()
        RESULTS.append((name, True, str(detail or "ok")))
    except Exception as e:
        RESULTS.append((name, False, f"{type(e).__name__}: {e}"))


# --------------------------------------------------------------------------
# Models used by the checks
# --------------------------------------------------------------------------


class CityInfo(BaseModel):
    """Every field defaulted, so the fallback path can default-construct it."""

    name: str = ""
    country: str = ""
    population: int = 0
    reasoning: str = ""


# --------------------------------------------------------------------------
# 1. Config
# --------------------------------------------------------------------------


def t_profiles() -> str:
    seen = []
    for p in ("openai_terra", "openai_luna", "openai_sol"):
        model = settings.model_for(p)
        assert model.startswith("openai/gpt-5.6-"), f"{p} -> {model}"
        seen.append(f"{p}={model.split('/')[-1]}")
    assert settings.llm_profile == "openai_terra", settings.llm_profile
    assert settings.bulk_profile == "openai_luna", settings.bulk_profile
    return ", ".join(seen)


def t_unknown_profile_fails_loudly() -> str:
    try:
        Settings(llm_profile="does-not-exist").validate_profiles()
    except UnknownProfileError:
        return "raises UnknownProfileError"
    raise AssertionError("unknown profile did not raise")


def t_api_key_present() -> str:
    assert os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY not set"
    return "OPENAI_API_KEY set"


# --------------------------------------------------------------------------
# 2. Schema derivation from the closure
# --------------------------------------------------------------------------


def t_submit_result_schema() -> str:
    tool = make_submit_result(CityInfo)
    assert tool.name == "submit_result", tool.name
    schema = tool.params_json_schema
    text = str(schema)
    for fname in CityInfo.model_fields:
        assert fname in text, f"{fname} missing from tool schema"
    return f"{len(CityInfo.model_fields)} fields carried into the tool schema"


# --------------------------------------------------------------------------
# 6. Truncation
# --------------------------------------------------------------------------


def t_truncate_head() -> str:
    assert truncate_head("abc", 10) == "abc"
    out = truncate_head("x" * 100, 10)
    assert out.startswith("x" * 10) and "[truncated]" in out
    return "head-only ok"


def t_truncate_90_10() -> str:
    text = "H" * 50_000 + "T" * 50_000
    cap = 30_000
    out = truncate_90_10(text, cap)
    head, tail = out.split("\n\n[... TRUNCATED:")[0], out[-10:]
    assert len(head) == int(cap * 0.9), len(head)
    assert set(head) == {"H"}, "head not from the start"
    assert set(tail) == {"T"}, "tail not from the end"
    assert truncate_90_10("short", 30_000) == "short"
    return f"kept {len(head)} head + {cap - len(head)} tail"


# --------------------------------------------------------------------------
# 9. Reasoning param
# --------------------------------------------------------------------------


def t_reasoning_settings() -> str:
    ms = build_model_settings("medium")
    assert ms.reasoning is not None and ms.reasoning.effort == "medium"
    assert build_model_settings("").reasoning is None
    return "medium set; '' sends nothing"


def t_agent_builds() -> str:
    spec = AgentSpec(
        name="Smoke",
        instructions="You are a test agent.",
        result_model=CityInfo,
        use_web=False,
        allow_delegation=True,
    )
    agent = create_agent(spec)
    names = [t.name for t in agent.tools]
    assert "submit_result" in names, names
    assert "delegate_to_subagent" in names, names
    assert agent.model_settings.reasoning.effort == "medium"
    return f"tools={names}"


def t_depth_default() -> str:
    assert current_depth() == 0
    return "depth starts at 0"


# --------------------------------------------------------------------------
# 3/4/8. Live checks
# --------------------------------------------------------------------------

TRIVIAL_PROMPT = (
    "You are a geography assistant.\n\n"
    "Call `submit_result` exactly once with: name, country, population "
    "(integer), and a one-sentence reasoning. Do not use any other tool. "
    "Answer from your own knowledge; do not search."
)


async def t_structured_roundtrip() -> str:
    spec = AgentSpec(
        name="Smoke Roundtrip",
        instructions=TRIVIAL_PROMPT,
        result_model=CityInfo,
        use_web=False,
        max_turns=6,
    )
    out = await run_agent(spec, "Tell me about Lisbon, Portugal.")
    assert isinstance(out, CityInfo), type(out)
    assert out.name, "name empty — agent likely never called submit_result"
    assert out.population > 0, f"population={out.population}"
    return f"{out.name}, {out.country}, pop={out.population:,}"


async def t_fallback_path() -> str:
    """max_turns=1 makes submit_result unreachable; must not raise."""
    spec = AgentSpec(
        name="Smoke Fallback",
        instructions=TRIVIAL_PROMPT,
        result_model=CityInfo,
        use_web=False,
        max_turns=1,
    )
    out = await run_agent(spec, "Tell me about Lisbon, Portugal.")
    assert isinstance(out, CityInfo), type(out)
    return "returned a valid instance instead of raising"


async def t_concurrency() -> str:
    cities = ["Lisbon", "Oslo", "Nairobi", "Osaka"]
    spec = AgentSpec(
        name="Smoke Concurrent",
        instructions=TRIVIAL_PROMPT,
        result_model=CityInfo,
        use_web=False,
        max_turns=6,
    )
    t0 = time.monotonic()
    outs = await run_agents([spec] * len(cities), [f"Tell me about {c}." for c in cities])
    elapsed = time.monotonic() - t0
    assert len(outs) == len(cities), len(outs)
    names = [o.name for o in outs]  # type: ignore[attr-defined]
    assert all(names), f"blank result in {names}"
    return f"{len(outs)} agents in {elapsed:.1f}s -> {names}"


async def t_web_tools() -> str:
    if not settings.firecrawl_api_key:
        raise RuntimeError("FIRECRAWL_API_KEY not set — web tools unavailable")
    from .firecrawl_rest import scrape_markdown, search_web

    hits = search_web("Python programming language history", limit=3)
    assert hits and hits[0].get("url"), hits
    md = scrape_markdown("https://example.com")
    assert md, "scrape returned nothing"
    return f"search {len(hits)} hits; scrape {len(md)} chars"


async def t_delegation() -> str:
    if not settings.firecrawl_api_key:
        raise RuntimeError("FIRECRAWL_API_KEY not set — delegation needs web tools")
    spec = AgentSpec(
        name="Smoke Delegate",
        instructions=(
            "You are a research coordinator.\n\n"
            "You MUST call `delegate_to_subagent` exactly once, asking it to "
            "find the population of Lisbon and telling it to call "
            "submit_result. Then call `submit_result` yourself with what it "
            "returned. Do not search yourself."
        ),
        result_model=CityInfo,
        allow_delegation=True,
        max_turns=10,
    )
    out = await run_agent(spec, "Get me Lisbon's population via a subagent.")
    assert isinstance(out, CityInfo), type(out)
    return f"parent got: {out.name} pop={out.population:,}"


# --------------------------------------------------------------------------


async def main(offline: bool) -> int:
    check("1a config: openai profiles resolve", t_profiles)
    check("1b config: unknown profile fails loudly", t_unknown_profile_fails_loudly)
    check("1c config: OPENAI_API_KEY present", t_api_key_present)
    check("2  schema: submit_result factory", t_submit_result_schema)
    check("6a truncate: head-only", t_truncate_head)
    check("6b truncate: 90/10", t_truncate_90_10)
    check("9  reasoning: ModelSettings", t_reasoning_settings)
    check("0  agent: builds with tools", t_agent_builds)
    check("7a delegation: depth starts 0", t_depth_default)

    if not offline:
        await acheck("3  live: structured round-trip", t_structured_roundtrip)
        await acheck("4  live: fallback on max_turns", t_fallback_path)
        await acheck("5  live: firecrawl search+scrape", t_web_tools)
        await acheck("7b live: delegation under REST", t_delegation)
        await acheck("8  live: 4 agents concurrent", t_concurrency)

    width = max(len(n) for n, _, _ in RESULTS)
    print()
    for name, ok, detail in RESULTS:
        print(f"{'PASS' if ok else 'FAIL'}  {name:<{width}}  {detail}")
    failed = [n for n, ok, _ in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} passed")
    if failed:
        print("failed: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="skip live LLM/network checks")
    args = ap.parse_args()

    use_selector_event_loop()
    raise SystemExit(asyncio.run(main(args.offline)))
