"""Model clients.

Provider-agnostic on purpose: the loop must not care who serves the tokens, and we do not
want the harness rewritten if the key we get on the day is OpenAI rather than Anthropic.

ScriptedClient replays a fixed sequence of tool calls, which lets the loop be tested for
real - retries, as_of violations, budget exhaustion - with no key and no spend.
"""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass, field
from typing import Protocol

# Rate limits are the single most likely cause of a failed final run: a fresh API account
# has low limits, and four companies running concurrently will hit them. Retry rather than
# die, and honour Retry-After when the server sends it.
RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}
MAX_RETRIES = 6


def _request_with_retry(post, describe: str = "request"):
    """Call `post()`, retrying transient failures with exponential backoff and jitter."""
    import httpx

    last = None
    for attempt in range(MAX_RETRIES):
        try:
            response = post()
            if response.status_code in RETRY_STATUS:
                retry_after = response.headers.get("retry-after")
                delay = (
                    float(retry_after)
                    if retry_after and retry_after.replace(".", "", 1).isdigit()
                    else min(2**attempt, 30) + random.random()
                )
                last = f"HTTP {response.status_code}"
                if attempt < MAX_RETRIES - 1:
                    time.sleep(delay)
                    continue
            response.raise_for_status()
            return response
        except httpx.TransportError as exc:
            last = f"{type(exc).__name__}: {exc}"
            if attempt < MAX_RETRIES - 1:
                time.sleep(min(2**attempt, 30) + random.random())
                continue
            raise
    raise RuntimeError(f"{describe} failed after {MAX_RETRIES} attempts: {last}")


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class Completion:
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)


class LLMClient(Protocol):
    name: str

    def complete(self, messages: list[dict], tools: list[dict]) -> Completion: ...


class OpenAIClient:
    """OpenAI-compatible chat completions over plain HTTP (no SDK dependency)."""

    def __init__(self, model: str = "gpt-4o", api_key: str | None = None,
                 base_url: str = "https://api.openai.com/v1"):
        self.model = model
        self.name = f"openai:{model}"
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url.rstrip("/")
        if not self.api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Put it in .env (gitignored) or export it."
            )

    def complete(self, messages: list[dict], tools: list[dict]) -> Completion:
        import httpx

        payload = {"model": self.model, "messages": messages, "temperature": 0}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        response = _request_with_retry(
            lambda: httpx.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload,
                timeout=180,
            ),
            describe=f"openai {self.model}",
        )
        message = response.json()["choices"][0]["message"]
        calls = [
            ToolCall(
                id=c["id"],
                name=c["function"]["name"],
                arguments=json.loads(c["function"]["arguments"] or "{}"),
            )
            for c in message.get("tool_calls") or []
        ]
        return Completion(content=message.get("content"), tool_calls=calls)


class AnthropicClient:
    """Anthropic messages API. Tool specs are translated from the OpenAI shape."""

    def __init__(self, model: str = "claude-sonnet-5", api_key: str | None = None):
        self.model = model
        self.name = f"anthropic:{model}"
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set.")

    def complete(self, messages: list[dict], tools: list[dict]) -> Completion:
        import httpx

        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        converted = [m for m in messages if m["role"] != "system"]
        response = _request_with_retry(lambda: httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": self.model,
                "max_tokens": 4096,
                "system": system,
                "messages": converted,
                "tools": [
                    {
                        "name": t["function"]["name"],
                        "description": t["function"]["description"],
                        "input_schema": t["function"]["parameters"],
                    }
                    for t in tools
                ],
            },
            timeout=180,
        ), describe=f"anthropic {self.model}")
        body = response.json()
        text, calls = None, []
        for block in body.get("content", []):
            if block["type"] == "text":
                text = block["text"]
            elif block["type"] == "tool_use":
                calls.append(ToolCall(id=block["id"], name=block["name"], arguments=block["input"]))
        return Completion(content=text, tool_calls=calls)


class ScriptedClient:
    """Replays a fixed plan. Lets the loop be exercised with no key and no spend.

    `plan` is a list of either:
      - (tool_name, arguments)  -> emit that tool call
      - str                     -> emit that as final content and stop
    """

    def __init__(self, plan: list):
        self.plan = list(plan)
        self.name = "scripted"
        self.calls_made = 0

    def complete(self, messages: list[dict], tools: list[dict]) -> Completion:
        if not self.plan:
            return Completion(content="No further steps planned.")
        step = self.plan.pop(0)
        if isinstance(step, str):
            return Completion(content=step)
        name, arguments = step
        self.calls_made += 1
        return Completion(
            tool_calls=[ToolCall(id=f"call_{self.calls_made}", name=name, arguments=arguments)]
        )


def load_env(path: str = ".env") -> None:
    """Load KEY=value lines from .env into the environment.

    Kept dependency-free on purpose. Existing environment variables win, so an exported
    key overrides the file rather than the other way round.
    """
    from pathlib import Path

    file = Path(path)
    if not file.exists():
        return
    for line in file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def build_client(spec: str | None = None) -> LLMClient:
    """Pick a client from config or environment. Fails loudly rather than silently."""
    load_env()
    spec = spec or os.environ.get("AGENT_MODEL", "")
    if spec.startswith("openai:"):
        return OpenAIClient(model=spec.split(":", 1)[1])
    if spec.startswith("anthropic:"):
        return AnthropicClient(model=spec.split(":", 1)[1])
    if os.environ.get("OPENAI_API_KEY"):
        return OpenAIClient()
    if os.environ.get("ANTHROPIC_API_KEY"):
        return AnthropicClient()
    raise RuntimeError(
        "No model configured. Set OPENAI_API_KEY or ANTHROPIC_API_KEY, or pass "
        "AGENT_MODEL=openai:gpt-4o. Use ScriptedClient for a dry run without a key."
    )
