"""tools.py — market data and analyst consensus as agent tools.

Design rules, the same ones `corpus/tools.py` follows, plus what this data
demands on top:

  * Every tool output names its SOURCE and its AS-OF timestamp. Yahoo stamps no
    vintage on a consensus row, so `as_of` is the UTC time we fetched it. Saying
    so is the honest move; the evidence trail is scored and a made-up vintage is
    worse than a snapshot admitted to be one.
  * Every tool output is bounded, via `truncate_head`.
  * UNITS ARE STATED ON EVERY NUMBER. Hays quotes in pence but reports in
    pounds, so a raw feed mixes two scales in one payload. Revenue is in
    millions, percentages in points (4.5 means 4.5%). A units error is the
    cheapest way to throw a scored metric.
  * This module is BACKGROUND, not evidence. The corpus holds the filings and
    the reported actuals; these tools hold price action and the sell-side
    consensus the corpus cannot contain. No filings are fetched here.
  * Consensus is a benchmark to reason against, never an answer to copy. Yahoo's
    EPS consensus is an *adjusted* measure for HD and ADI, and its revenue
    consensus for Hays is turnover rather than the scored net-fee line.
"""

from __future__ import annotations

import json
import logging

from agents import function_tool

from agent_core.truncate import truncate_head

from . import client

logger = logging.getLogger(__name__)

_MAX = 8000


def _err(e: Exception) -> str:
    return f"Error: {type(e).__name__}: {e}"


def _header(res: client.Result) -> str:
    """Provenance block. Prepended to every tool result without exception."""
    lines = [
        f"{res.company} [{res.ticker} / Yahoo {res.yahoo_symbol}] — {res.kind}",
        f"SOURCE: {res.source}",
        f"AS OF: {res.as_of} (retrieval time; Yahoo publishes no vintage)",
    ]
    if not res.ok:
        lines.append("STATUS: NO DATA — see notes below.")
    return "\n".join(lines)


def _notes(res: client.Result) -> str:
    if not res.notes:
        return ""
    return "\nNOTES:\n" + "\n".join(f"  - {n}" for n in res.notes)


def _render(res: client.Result, body: str = "") -> str:
    parts = [_header(res)]
    if body:
        parts.append(body)
    tail = _notes(res)
    if tail:
        parts.append(tail)
    return truncate_head("\n\n".join(parts), _MAX)


def _dump(obj) -> str:
    return json.dumps(obj, indent=2, default=str)


@function_tool
async def get_quote_snapshot(ticker: str) -> str:
    """Last price, 52-week range, market cap and trailing multiples.

    Cheapest way to get the market's current read on a company. Use it to
    sanity-check scale before trusting any other figure.

    Args:
        ticker: One of HD, ADI, HAS, DE. "HAS.L" is accepted for Hays.
    """
    try:
        res = client.quote_snapshot(ticker)
        if not res.ok:
            return _render(res)
        return _render(res, _dump(res.fields))
    except Exception as e:
        logger.warning("get_quote_snapshot failed for %r: %s", ticker, e)
        return _err(e)


@function_tool
async def get_price_history(ticker: str, period: str = "1y", interval: str = "1d") -> str:
    """Price history with trailing returns and realised volatility.

    Returns a summary block first (returns over 1d/1m/3m/6m/12m, annualised
    vol, period high/low) and then the most recent bars. The summary is what
    you usually want; the bars are there to check a specific date.

    Args:
        ticker: One of HD, ADI, HAS, DE.
        period: Yahoo period string, e.g. "1mo", "6mo", "1y", "5y", "max".
        interval: Bar size, e.g. "1d", "1wk", "1mo".
    """
    try:
        res = client.price_history(ticker, period=period, interval=interval)
        if not res.ok:
            return _render(res)
        recent = res.rows[-20:]
        body = (
            _dump(res.fields)
            + f"\n\nMost recent {len(recent)} of {len(res.rows)} bars "
            f"(prices in {res.fields.get('currency')}):\n"
            + _dump(recent)
        )
        return _render(res, body)
    except Exception as e:
        logger.warning("get_price_history failed for %r: %s", ticker, e)
        return _err(e)


@function_tool
async def get_analyst_estimates(ticker: str) -> str:
    """Sell-side consensus EPS and revenue for the coming quarters and years.

    THE key benchmark tool. Periods are Yahoo's codes: "0q" is the quarter
    about to be reported, "+1q" the one after, "0y" the current fiscal year,
    "+1y" the next. Each row carries the mean, the low/high range, the analyst
    count, the year-ago actual and the implied growth.

    Read the units off the row: EPS for Hays is in PENCE, revenue is in
    millions of the reporting currency. Note that for Hays the revenue
    consensus is TURNOVER, not net fees.

    Args:
        ticker: One of HD, ADI, HAS, DE.
    """
    try:
        res = client.analyst_estimates(ticker)
        if not res.ok:
            return _render(res)
        # Uncovered periods are all-null rows. Emitting them costs ~20 lines of
        # JSON nulls each and buries the periods that do have a consensus; the
        # notes already name what is missing, which is the part worth keeping.
        covered = [row for row in res.rows if row.get("covered")]
        uncovered = [row["period"] for row in res.rows if not row.get("covered")]
        body = _dump(covered)
        if uncovered:
            body += (
                f"\n\nNo consensus published for: {', '.join(uncovered)} "
                "(omitted above rather than shown as nulls)."
            )
        return _render(res, body)
    except Exception as e:
        logger.warning("get_analyst_estimates failed for %r: %s", ticker, e)
        return _err(e)


@function_tool
async def get_estimate_revisions(ticker: str) -> str:
    """How the consensus has moved over the last 7/30/60/90 days.

    Tells you whether the consensus is stale or actively being marked up or
    down into the print, and how many analysts moved each way. A consensus
    drifting one way for 90 days is a stronger prior than a static one.

    Args:
        ticker: One of HD, ADI, HAS, DE.
    """
    try:
        res = client.estimate_revisions(ticker)
        if not res.ok:
            return _render(res)
        return _render(res, _dump(res.rows))
    except Exception as e:
        logger.warning("get_estimate_revisions failed for %r: %s", ticker, e)
        return _err(e)


@function_tool
async def get_earnings_surprise_history(ticker: str, limit: int = 12) -> str:
    """Past reported EPS vs the estimate going in — the beat/miss track record.

    Use this to calibrate. A company that has beaten consensus every quarter by
    ~5% tells you where inside the estimate range to land. Summary counts come
    first, then the per-period detail.

    Args:
        ticker: One of HD, ADI, HAS, DE.
        limit: Max periods to return, newest first.
    """
    try:
        res = client.earnings_surprise_history(ticker, limit=limit)
        if not res.ok:
            return _render(res)
        body = _dump(res.fields) + "\n\nBy period (newest first):\n" + _dump(res.rows)
        return _render(res, body)
    except Exception as e:
        logger.warning("get_earnings_surprise_history failed for %r: %s", ticker, e)
        return _err(e)


@function_tool
async def get_earnings_calendar(ticker: str) -> str:
    """Next scheduled report date, and the consensus attached to that date.

    Use it to confirm WHICH period the consensus refers to before quoting it,
    and to check whether the number you are forecasting has already been
    reported.

    Args:
        ticker: One of HD, ADI, HAS, DE.
    """
    try:
        res = client.earnings_calendar(ticker)
        if not res.ok:
            return _render(res)
        return _render(res, _dump(res.fields))
    except Exception as e:
        logger.warning("get_earnings_calendar failed for %r: %s", ticker, e)
        return _err(e)


@function_tool
async def get_reported_financials(ticker: str, freq: str = "quarterly") -> str:
    """Historical income-statement lines, as reported, with a derived GAAP
    gross margin.

    These are as-reported GAAP/IFRS figures — the actuals series to anchor a
    trend on. They are NOT adjusted or pre-exceptional: for those, and for any
    segment-level detail, use the corpus tools instead.

    If a company files no quarterly accounts, this falls back to annual and
    says so in the notes.

    Args:
        ticker: One of HD, ADI, HAS, DE.
        freq: "quarterly" or "annual".
    """
    try:
        res = client.reported_financials(ticker, freq=freq)
        if not res.ok:
            return _render(res)
        body = (
            f"frequency={res.fields.get('frequency')} | money={res.fields.get('money_unit')} "
            f"| per-share={res.fields.get('eps_unit')}\n"
            f"periods: {', '.join(str(p) for p in res.fields.get('periods', []))}\n\n"
            + _dump(res.rows)
        )
        return _render(res, body)
    except Exception as e:
        logger.warning("get_reported_financials failed for %r: %s", ticker, e)
        return _err(e)


@function_tool
async def get_analyst_coverage(ticker: str) -> str:
    """Price targets, buy/hold/sell distribution and long-run growth estimates.

    Sentiment context, not a fundamental forecast. Useful for gauging how
    contrarian a forecast is; it does not constrain a revenue or EPS number.

    Args:
        ticker: One of HD, ADI, HAS, DE.
    """
    try:
        res = client.analyst_coverage(ticker)
        if not res.ok:
            return _render(res)
        body = _dump(res.fields)
        if res.rows:
            body += "\n\nRecommendation distribution (0m = current month):\n" + _dump(res.rows)
        return _render(res, body)
    except Exception as e:
        logger.warning("get_analyst_coverage failed for %r: %s", ticker, e)
        return _err(e)


MARKET_TOOLS = [
    get_quote_snapshot,
    get_price_history,
    get_analyst_estimates,
    get_estimate_revisions,
    get_earnings_surprise_history,
    get_earnings_calendar,
    get_reported_financials,
    get_analyst_coverage,
]
