"""truncate.py — output trimming (agent-spec.md §7).

Two strategies:
  head-only  — cheap, predictable, deterministic. Most pages carry the point up
               top. Used for local tools and subagent returns.
  90/10      — 90% head + 10% tail. Firecrawl scrape results carry author
               bylines, publication dates and "related articles" footers that
               are useful disambiguation signal, so keep both ends.

Neither summarises on the fly: that costs tokens, adds latency, and hides
information from the agent's reasoning trace.
"""

from __future__ import annotations

from .config import settings


def truncate_head(text: str, max_chars: int | None = None) -> str:
    """Head-only truncation."""
    cap = max_chars if max_chars is not None else settings.tool_output_max_chars
    if cap <= 0 or len(text) <= cap:
        return text
    return text[:cap] + "\n... [truncated]"


def truncate_90_10(text: str, max_length: int | None = None) -> str:
    """90% head + 10% tail, with a marker naming how much was dropped.

    The result is exactly `max_length` chars of source text plus the marker.
    """
    cap = (
        max_length
        if max_length is not None
        else settings.firecrawl_max_content_length
    )
    if cap <= 0 or len(text) <= cap:
        return text
    keep_start = int(cap * 0.9)
    keep_end = cap - keep_start
    removed = len(text) - cap
    head = text[:keep_start]
    tail = text[-keep_end:] if keep_end > 0 else ""
    return f"{head}\n\n[... TRUNCATED: {removed} chars removed ...]\n\n{tail}"
