"""firecrawl_rest.py — Firecrawl v2 REST client.

Adapted from the working implementation at
  fa-automation/tasks/autorss-helper-tools/firecrawl_rest.py

We use REST rather than the Firecrawl MCP server that agent-spec.md §9 calls
"production-preferred". That guidance is stale: the reference project migrated
off MCP on 2026-06-30 (commit 4ad0bdb) for three reasons, all of which apply
here —

  1. MCP teardown segfaults on Windows shutdown (this is a Windows 11 box).
  2. Token cost fell from ~17.7k to ~12.6k per run without MCP.
  3. MCP servers are context-managed per task and cannot be reused, which is
     what forced spec §9.6's "no delegation under MCP" restriction. Plain
     function tools have no such constraint, so subagents get web access for
     free.

Endpoints used:
  POST /v2/search  {query, limit, sources:[{type:"web"}]}  -> data.web[]
  POST /v2/scrape  {url, formats:["markdown"]}             -> data.markdown
"""

from __future__ import annotations

import logging

import requests

from .config import settings
from .truncate import truncate_90_10

logger = logging.getLogger(__name__)

_QUERY_CAP = 500
_TITLE_CAP = 200
_DESC_CAP = 300


class FirecrawlNotConfigured(RuntimeError):
    """Raised when FIRECRAWL_API_KEY is missing."""


def _headers() -> dict[str, str]:
    if not settings.firecrawl_api_key:
        raise FirecrawlNotConfigured(
            "FIRECRAWL_API_KEY is not set. Add it to .env before using web tools."
        )
    return {
        "Authorization": f"Bearer {settings.firecrawl_api_key}",
        "Content-Type": "application/json",
    }


def search_web(query: str, limit: int = 10) -> list[dict[str, str]]:
    """POST /v2/search -> list of {url, title, description}."""
    limit = max(1, min(limit, settings.firecrawl_max_results))
    resp = requests.post(
        f"{settings.firecrawl_api_url}/v2/search",
        headers=_headers(),
        json={
            "query": query[:_QUERY_CAP],
            "limit": limit,
            "sources": [{"type": "web"}],
        },
        timeout=settings.firecrawl_search_timeout,
    )
    resp.raise_for_status()
    data = resp.json().get("data", {})
    web = data.get("web", []) if isinstance(data, dict) else []

    results: list[dict[str, str]] = []
    for r in web[:limit]:
        if not isinstance(r, dict):
            continue
        results.append(
            {
                "url": r.get("url", ""),
                "title": (r.get("title") or "")[:_TITLE_CAP],
                "description": (r.get("description") or "")[:_DESC_CAP],
            }
        )
    return results


def scrape_markdown(url: str, max_length: int | None = None) -> str:
    """POST /v2/scrape -> page markdown, 90/10 truncated."""
    cap = (
        max_length
        if max_length is not None
        else settings.firecrawl_max_content_length
    )
    resp = requests.post(
        f"{settings.firecrawl_api_url}/v2/scrape",
        headers=_headers(),
        json={"url": url, "formats": ["markdown"]},
        timeout=settings.firecrawl_scrape_timeout,
    )
    resp.raise_for_status()
    data = resp.json().get("data", {})
    markdown = (data.get("markdown") or "") if isinstance(data, dict) else ""
    return truncate_90_10(markdown, cap)
