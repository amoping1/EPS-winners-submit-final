"""client.py — a thin, defensive layer over yfinance (Yahoo Finance).

Scope is deliberately narrow. The corpus holds the filings; this module holds
the two things the corpus cannot: **price history** and **sell-side consensus**.
No SEC/EDGAR access, no openstocks.com — ever.

Everything here was written against observed output, not documentation. The
facts that shaped it, all verified on 2026-08-16 with yfinance 1.2.0:

  * **HAS.L quotes in GBp (pence), reports in GBP.** `info["currency"]`
    is "GBp" while `info["financialCurrency"]` is "GBP". Confirmed
    arithmetically: price 69.1 x 1,569,982,184 shares / 100 == marketCap
    1,084,857,728 exactly. So a raw EPS estimate of 0.01102 is GBP and means
    **1.102 pence**. The challenge wants Hays EPS in pence, so every EPS for
    HAS.L is multiplied by 100 here and labelled. Getting this wrong is a
    100x error on a scored metric.

  * **Yahoo's "Gross Profit" for Hays IS net fees.** FY2024 Gross Profit came
    back as 1,113,600,000 == the GBP 1,113.6m net fees Hays reported. Yahoo's
    "Total Revenue" (GBP 6,949.1m) is *turnover*, which is NOT the scored
    metric. The consensus `revenue_estimate` is likewise turnover, so it must
    never be read as a net-fee forecast.

  * **Hays has no quarterly anything.** `quarterly_income_stmt` is empty (0x0),
    `get_earnings_history()` is empty, and the `0q`/`+1q` consensus rows are
    all NaN. Hays reports semi-annually; only the annual `0y`/`+1y` consensus
    exists (10 analysts on EPS, 7 on revenue). This is a hard limitation, not
    a bug to route around.

  * **Hays FY2025 annual is a near-empty column.** The 2025-06-30 column of
    `income_stmt` carries 4 non-null rows out of 52 — share counts and EPS
    only. Revenue, gross profit and operating income are all NaN for the most
    recent fiscal year. Take those from the corpus.

  * **Bad symbols do not raise.** yfinance returns empty DataFrames, an empty
    calendar dict and a 1-key `info`. We still wrap every call, because a
    network failure *does* raise and a tool that raises is a dead agent turn.

Provenance is scored, so every result carries `source` and `as_of`. `as_of` is
the UTC retrieval time: Yahoo stamps no vintage on a consensus row, and
inventing one would be worse than admitting it is a snapshot.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# yfinance chatters on stderr about every empty frame. That noise would end up
# interleaved with agent output, so keep it to genuine errors.
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

SOURCE = "Yahoo Finance via yfinance"

# Period codes Yahoo uses across earnings_estimate / revenue_estimate /
# eps_trend / eps_revisions / growth_estimates. Verified identical on all four.
PERIOD_LABELS: dict[str, str] = {
    "0q": "current quarter (the one about to be reported)",
    "+1q": "next quarter",
    "0y": "current fiscal year",
    "+1y": "next fiscal year",
    "LTG": "long-term growth",
}


@dataclass(frozen=True)
class Symbol:
    """One challenge company, with the unit rules that apply to it."""

    ticker: str  # challenge ticker: HD, ADI, HAS, DE
    yahoo: str  # Yahoo symbol: HD, ADI, HAS.L, DE
    name: str
    price_currency: str  # what history()/quote prices are denominated in
    money_currency: str  # what income-statement figures are denominated in
    money_unit: str  # label for scaled money figures
    eps_unit: str  # label for scaled per-share figures
    eps_scale: float  # multiply raw yfinance EPS by this
    notes: tuple[str, ...] = ()


_SYMBOLS: dict[str, Symbol] = {
    "HD": Symbol(
        ticker="HD",
        yahoo="HD",
        name="The Home Depot, Inc.",
        price_currency="USD",
        money_currency="USD",
        money_unit="USDm",
        eps_unit="USD/share",
        eps_scale=1.0,
        notes=(
            "Consensus EPS is the sell-side ADJUSTED figure; Home Depot's "
            "adjusted and GAAP diluted EPS differ. Comparable-sales % is NOT "
            "on Yahoo at all — take it from the corpus.",
        ),
    ),
    "ADI": Symbol(
        ticker="ADI",
        yahoo="ADI",
        name="Analog Devices, Inc.",
        price_currency="USD",
        money_currency="USD",
        money_unit="USDm",
        eps_unit="USD/share",
        eps_scale=1.0,
        notes=(
            "Consensus EPS is ADJUSTED (non-GAAP); ADI's GAAP diluted EPS is "
            "far lower because of acquisition amortisation. Yahoo carries no "
            "adjusted gross margin — the income statement gives GAAP gross "
            "margin only, which runs several points below ADI's adjusted "
            "figure. Use it as a floor, not as the answer.",
        ),
    ),
    "HAS": Symbol(
        ticker="HAS",
        yahoo="HAS.L",
        name="Hays plc",
        price_currency="GBp",
        money_currency="GBP",
        money_unit="GBPm",
        eps_unit="pence",
        eps_scale=100.0,
        notes=(
            "PRICES ARE IN PENCE (GBp); STATEMENTS ARE IN GBP. Raw yfinance "
            "EPS is GBP and is multiplied by 100 here to give pence.",
            "Yahoo 'Gross Profit' == Hays NET FEES (the scored metric). Yahoo "
            "'Total Revenue' == turnover, which is ~6x net fees and is NOT "
            "the scored metric.",
            "No quarterly data of any kind exists for HAS.L. Annual only.",
        ),
    ),
    "DE": Symbol(
        ticker="DE",
        yahoo="DE",
        name="Deere & Company",
        price_currency="USD",
        money_currency="USD",
        money_unit="USDm",
        eps_unit="USD/share",
        eps_scale=1.0,
        notes=(
            "Deere's consensus EPS tracks GAAP diluted EPS closely, so it is "
            "usable for the GAAP metric. Segment detail (Production & "
            "Precision Ag operating profit) is NOT on Yahoo — corpus only.",
        ),
    ),
}

# Accept the challenge ticker, the Yahoo symbol, or a sloppy variant.
_ALIASES: dict[str, str] = {
    "HD": "HD",
    "ADI": "ADI",
    "DE": "DE",
    "HAS": "HAS",
    "HAS.L": "HAS",
    "HAYS": "HAS",
    "HAS.LON": "HAS",
}

TICKERS: tuple[str, ...] = ("HD", "ADI", "HAS", "DE")


def resolve(ticker: str) -> Symbol | None:
    """Map any accepted spelling to a Symbol. None if unknown."""
    key = _ALIASES.get((ticker or "").strip().upper())
    return _SYMBOLS.get(key) if key else None


def known_tickers() -> list[str]:
    return list(TICKERS)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------
# Result types
# --------------------------------------------------------------------------


@dataclass
class Result:
    """Every client call returns one of these. It never carries an exception.

    `ok` False means "no usable data" — the caller still gets a well-formed
    object with `notes` explaining why, so tools can report instead of raise.
    """

    kind: str
    ticker: str
    yahoo_symbol: str
    company: str
    source: str = SOURCE
    as_of: str = field(default_factory=_now)
    ok: bool = False
    rows: list[dict[str, Any]] = field(default_factory=list)
    fields: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def note(self, msg: str) -> None:
        if msg and msg not in self.notes:
            self.notes.append(msg)


def _blank(kind: str, sym: Symbol | None, ticker: str, why: str) -> Result:
    r = Result(
        kind=kind,
        ticker=sym.ticker if sym else (ticker or "").upper(),
        yahoo_symbol=sym.yahoo if sym else "",
        company=sym.name if sym else "unknown",
        ok=False,
    )
    r.note(why)
    return r


# --------------------------------------------------------------------------
# yfinance plumbing
# --------------------------------------------------------------------------

_TICKER_CACHE: dict[str, Any] = {}


def _handle(sym: Symbol):
    """A cached yfinance.Ticker. Cached because each object memoises its own
    HTTP responses — 8 tools against one company then costs one fetch each,
    not eight."""
    if sym.yahoo not in _TICKER_CACHE:
        import yfinance as yf  # imported lazily so import errors surface as notes

        _TICKER_CACHE[sym.yahoo] = yf.Ticker(sym.yahoo)
    return _TICKER_CACHE[sym.yahoo]


def clear_cache() -> None:
    _TICKER_CACHE.clear()


def _num(v: Any) -> float | None:
    """Coerce to a finite float, or None. NaN is the single most common value
    in this data and must never reach the agent as the string 'nan'."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) or math.isinf(f) else f


def _money_m(v: Any) -> float | None:
    """Absolute currency units -> millions, rounded to 1dp."""
    f = _num(v)
    return None if f is None else round(f / 1e6, 1)


def _eps(v: Any, sym: Symbol) -> float | None:
    """Raw yfinance per-share value -> the challenge's unit (USD, or pence)."""
    f = _num(v)
    return None if f is None else round(f * sym.eps_scale, 4)


def _pct(v: Any) -> float | None:
    """Yahoo growth/margin fractions -> percentage POINTS (0.0433 -> 4.33)."""
    f = _num(v)
    return None if f is None else round(f * 100.0, 2)


def _date(v: Any) -> str | None:
    if v is None:
        return None
    try:
        if hasattr(v, "strftime"):
            return v.strftime("%Y-%m-%d")
        return str(v)[:10]
    except Exception:
        return None


def _empty(df: Any) -> bool:
    return df is None or not hasattr(df, "empty") or df.empty


def _cell(df: Any, row: Any, col: Any) -> Any:
    try:
        return df.loc[row, col]
    except Exception:
        return None


# --------------------------------------------------------------------------
# 1. Quote snapshot
# --------------------------------------------------------------------------


def quote_snapshot(ticker: str) -> Result:
    """Last price, 52-week range, market cap, valuation multiples."""
    sym = resolve(ticker)
    if sym is None:
        return _blank("quote", None, ticker, f"Unknown ticker {ticker!r}.")
    r = Result(kind="quote", ticker=sym.ticker, yahoo_symbol=sym.yahoo, company=sym.name)
    try:
        info = _handle(sym).info or {}
    except Exception as e:
        r.note(f"Quote fetch failed: {type(e).__name__}: {e}")
        return r
    if len(info) <= 1:
        r.note(f"Yahoo returned no quote for {sym.yahoo}.")
        return r

    price_ccy = info.get("currency") or sym.price_currency
    if price_ccy != sym.price_currency:
        r.note(
            f"Price currency from Yahoo is {price_ccy!r}, expected "
            f"{sym.price_currency!r}. Treating figures as {price_ccy!r}."
        )

    r.fields = {
        "price": _num(info.get("regularMarketPrice") or info.get("currentPrice")),
        "price_currency": price_ccy,
        "previous_close": _num(info.get("previousClose")),
        "fifty_two_week_high": _num(info.get("fiftyTwoWeekHigh")),
        "fifty_two_week_low": _num(info.get("fiftyTwoWeekLow")),
        "market_cap": _num(info.get("marketCap")),
        "market_cap_currency": info.get("financialCurrency") or sym.money_currency,
        "shares_outstanding": _num(info.get("sharesOutstanding")),
        "trailing_eps": _eps(info.get("trailingEps"), sym),
        "forward_eps": _eps(info.get("forwardEps"), sym),
        "eps_unit": sym.eps_unit,
        "trailing_pe": _num(info.get("trailingPE")),
        "forward_pe": _num(info.get("forwardPE")),
        "gross_margin_pct": _pct(info.get("grossMargins")),
        "operating_margin_pct": _pct(info.get("operatingMargins")),
        "profit_margin_pct": _pct(info.get("profitMargins")),
        "revenue_growth_pct": _pct(info.get("revenueGrowth")),
        "earnings_growth_pct": _pct(info.get("earningsGrowth")),
        "beta": _num(info.get("beta")),
        "exchange": info.get("fullExchangeName") or info.get("exchange"),
        "last_fiscal_year_end": _epoch(info.get("lastFiscalYearEnd")),
        "most_recent_quarter": _epoch(info.get("mostRecentQuarter")),
        "next_fiscal_year_end": _epoch(info.get("nextFiscalYearEnd")),
    }
    r.ok = r.fields.get("price") is not None
    if sym.ticker == "HAS":
        r.note(
            "Price is in PENCE (GBp); market cap and per-share figures are in "
            "GBP. EPS shown here is already converted to pence."
        )
        r.note(
            "grossMargin here is computed off turnover, so it reads ~3-4%. "
            "That is arithmetic, not a collapse in profitability — Hays' "
            "net-fee margin is a different ratio entirely."
        )
    for n in sym.notes:
        r.note(n)
    return r


def _epoch(v: Any) -> str | None:
    f = _num(v)
    if f is None:
        return None
    try:
        return datetime.fromtimestamp(f, tz=timezone.utc).strftime("%Y-%m-%d")
    except Exception:
        return None


# --------------------------------------------------------------------------
# 2. Price history
# --------------------------------------------------------------------------


def price_history(ticker: str, period: str = "1y", interval: str = "1d") -> Result:
    """OHLCV bars plus trailing returns and realised volatility."""
    sym = resolve(ticker)
    if sym is None:
        return _blank("price_history", None, ticker, f"Unknown ticker {ticker!r}.")
    r = Result(
        kind="price_history", ticker=sym.ticker, yahoo_symbol=sym.yahoo, company=sym.name
    )
    try:
        df = _handle(sym).history(period=period, interval=interval, auto_adjust=True)
    except Exception as e:
        r.note(f"History fetch failed: {type(e).__name__}: {e}")
        return r
    if _empty(df):
        r.note(f"No price history for {sym.yahoo} over period={period!r}.")
        return r

    closes: list[float] = []
    for idx, row in df.iterrows():
        c = _num(row.get("Close"))
        if c is None:
            continue
        closes.append(c)
        r.rows.append(
            {
                "date": _date(idx),
                "open": round(_num(row.get("Open")) or 0.0, 4),
                "high": round(_num(row.get("High")) or 0.0, 4),
                "low": round(_num(row.get("Low")) or 0.0, 4),
                "close": round(c, 4),
                "volume": int(_num(row.get("Volume")) or 0),
            }
        )
    if not closes:
        r.note("Price history contained no usable closes.")
        return r

    def ret(n: int) -> float | None:
        """Trailing return over n trading days.

        A "1y" download is ~251 bars, so a strict 252-day lookback would return
        None for the 12-month return on every US ticker — the exact number a
        forecaster wants. Where history is at least 90% of the window, anchor on
        the oldest bar instead of giving up.
        """
        if len(closes) < 2:
            return None
        idx = len(closes) - 1 - n
        if idx < 0:
            if len(closes) - 1 < n * 0.9:
                return None
            idx = 0
        if closes[idx] == 0:
            return None
        return round((closes[-1] / closes[idx] - 1.0) * 100.0, 2)

    rets = [
        (closes[i] / closes[i - 1] - 1.0)
        for i in range(1, len(closes))
        if closes[i - 1] not in (0, None)
    ]
    vol = None
    if len(rets) > 2:
        mean = sum(rets) / len(rets)
        var = sum((x - mean) ** 2 for x in rets) / (len(rets) - 1)
        vol = round(math.sqrt(var) * math.sqrt(252) * 100.0, 2)

    r.fields = {
        "period": period,
        "interval": interval,
        "currency": sym.price_currency,
        "bars": len(r.rows),
        "first_date": r.rows[0]["date"],
        "last_date": r.rows[-1]["date"],
        "last_close": round(closes[-1], 4),
        "period_high": round(max(closes), 4),
        "period_low": round(min(closes), 4),
        "return_1d_pct": ret(1),
        "return_1m_pct": ret(21),
        "return_3m_pct": ret(63),
        "return_6m_pct": ret(126),
        "return_12m_pct": ret(252),
        "annualised_vol_pct": vol,
    }
    r.ok = True
    if sym.ticker == "HAS":
        r.note("Prices are in PENCE (GBp). Divide by 100 for GBP.")
    return r


# --------------------------------------------------------------------------
# 3. Analyst estimates (the consensus anchor)
# --------------------------------------------------------------------------


def analyst_estimates(ticker: str) -> Result:
    """Consensus EPS and revenue for 0q / +1q / 0y / +1y.

    EPS is scaled to the challenge unit (pence for Hays). Revenue is converted
    to millions. Growth fractions become percentage points.
    """
    sym = resolve(ticker)
    if sym is None:
        return _blank("analyst_estimates", None, ticker, f"Unknown ticker {ticker!r}.")
    r = Result(
        kind="analyst_estimates",
        ticker=sym.ticker,
        yahoo_symbol=sym.yahoo,
        company=sym.name,
    )
    tk = _handle(sym)
    try:
        eps_df = tk.earnings_estimate
    except Exception as e:
        eps_df = None
        r.note(f"earnings_estimate failed: {type(e).__name__}: {e}")
    try:
        rev_df = tk.revenue_estimate
    except Exception as e:
        rev_df = None
        r.note(f"revenue_estimate failed: {type(e).__name__}: {e}")

    if _empty(eps_df) and _empty(rev_df):
        r.note(f"No analyst estimates published for {sym.yahoo}.")
        return r

    periods: list[str] = []
    for df in (eps_df, rev_df):
        if not _empty(df):
            for p in df.index:
                if str(p) not in periods:
                    periods.append(str(p))

    for p in periods:
        row: dict[str, Any] = {"period": p, "meaning": PERIOD_LABELS.get(p, p)}
        if not _empty(eps_df) and p in eps_df.index:
            row.update(
                {
                    "eps_avg": _eps(_cell(eps_df, p, "avg"), sym),
                    "eps_low": _eps(_cell(eps_df, p, "low"), sym),
                    "eps_high": _eps(_cell(eps_df, p, "high"), sym),
                    "eps_year_ago": _eps(_cell(eps_df, p, "yearAgoEps"), sym),
                    "eps_analysts": _num(_cell(eps_df, p, "numberOfAnalysts")),
                    "eps_growth_pct": _pct(_cell(eps_df, p, "growth")),
                    "eps_unit": sym.eps_unit,
                }
            )
        if not _empty(rev_df) and p in rev_df.index:
            rev_avg = _money_m(_cell(rev_df, p, "avg"))
            # Yahoo emits a literal 0 (not NaN) for periods it does not cover.
            if rev_avg == 0.0:
                rev_avg = None
            row.update(
                {
                    "revenue_avg_m": rev_avg,
                    "revenue_low_m": _money_m(_cell(rev_df, p, "low")) or None,
                    "revenue_high_m": _money_m(_cell(rev_df, p, "high")) or None,
                    "revenue_year_ago_m": _money_m(_cell(rev_df, p, "yearAgoRevenue")),
                    "revenue_analysts": _num(_cell(rev_df, p, "numberOfAnalysts")) or None,
                    "revenue_growth_pct": _pct(_cell(rev_df, p, "growth")),
                    "revenue_unit": sym.money_unit,
                }
            )
        has_any = any(
            row.get(k) is not None for k in ("eps_avg", "revenue_avg_m")
        )
        row["covered"] = has_any
        r.rows.append(row)

    r.ok = any(row.get("covered") for row in r.rows)
    if not r.ok:
        r.note("Estimate tables exist but every value is NaN.")

    empty_q = [
        row["period"]
        for row in r.rows
        if row["period"] in ("0q", "+1q") and not row.get("covered")
    ]
    if empty_q:
        r.note(
            f"No quarterly consensus for periods {', '.join(empty_q)} — the "
            "company is not covered on a quarterly basis. Use 0y/+1y."
        )
    r.note(
        "Consensus EPS is the sell-side ADJUSTED/normalised measure unless the "
        "company reports on a GAAP-equivalent basis. Check which basis the "
        "scored metric wants before using this number directly."
    )
    if sym.ticker == "DE":
        # Verified 2026-08-16: the 0q consensus year-ago revenue base is
        # USD 10,357m, but Yahoo's own Total Revenue for that same quarter
        # (ending 2025-07-31) is USD 11,783m — a USD 1,426m gap. The consensus
        # tracks EQUIPMENT OPERATIONS net sales; the scored metric is worldwide
        # net sales AND revenues, which includes financial services.
        r.note(
            "REVENUE BASIS MISMATCH: this consensus tracks Deere's EQUIPMENT "
            "OPERATIONS net sales, not 'worldwide net sales and revenues'. Its "
            "year-ago base is USD 10,357m for the quarter ended 2025-07-31, "
            "whereas total revenue for that quarter was USD 11,783m — a gap of "
            "~USD 1,426m (financial services). Taking this consensus as the "
            "worldwide figure understates it by well over a billion dollars. "
            "Add financial services revenue, and confirm against the corpus."
        )
    if sym.ticker == "HAS":
        # Verified 2026-08-16: the 0y year-ago EPS comes back as 1.31p while the
        # statutory FY2025 basic EPS in income_stmt is -0.49p (a loss, after
        # exceptionals). The consensus therefore runs on a PRE-EXCEPTIONAL base
        # — the same basis the challenge scores — so it is directly comparable.
        r.note(
            "The EPS consensus here is PRE-EXCEPTIONAL, not statutory: its "
            "FY2025 year-ago base is 1.31p, whereas statutory FY2025 basic EPS "
            "was -0.49p. That makes it directly comparable to a pre-exceptional "
            "basic EPS forecast — but confirm the base against the corpus."
        )
        r.note(
            "The revenue consensus is TURNOVER (~GBP 6.0bn), NOT net fees "
            "(~GBP 1.0bn). Do not read it as a net-fee forecast."
        )
    for n in sym.notes:
        r.note(n)
    return r


# --------------------------------------------------------------------------
# 4. Estimate revisions (consensus momentum)
# --------------------------------------------------------------------------


def estimate_revisions(ticker: str) -> Result:
    """How the consensus EPS has drifted over 7/30/60/90 days, plus the count
    of analysts revising up vs down. This is the single best read on whether
    the consensus is stale or moving into the print."""
    sym = resolve(ticker)
    if sym is None:
        return _blank("estimate_revisions", None, ticker, f"Unknown ticker {ticker!r}.")
    r = Result(
        kind="estimate_revisions",
        ticker=sym.ticker,
        yahoo_symbol=sym.yahoo,
        company=sym.name,
    )
    tk = _handle(sym)
    try:
        trend = tk.eps_trend
    except Exception as e:
        trend = None
        r.note(f"eps_trend failed: {type(e).__name__}: {e}")
    try:
        revs = tk.eps_revisions
    except Exception as e:
        revs = None
        r.note(f"eps_revisions failed: {type(e).__name__}: {e}")

    if _empty(trend) and _empty(revs):
        r.note(f"No revision data for {sym.yahoo}.")
        return r

    periods: list[str] = []
    for df in (trend, revs):
        if not _empty(df):
            for p in df.index:
                if str(p) not in periods:
                    periods.append(str(p))

    for p in periods:
        row: dict[str, Any] = {"period": p, "meaning": PERIOD_LABELS.get(p, p)}
        if not _empty(trend) and p in trend.index:
            cur = _eps(_cell(trend, p, "current"), sym)
            row["eps_current"] = cur
            for col, key in (
                ("7daysAgo", "eps_7d_ago"),
                ("30daysAgo", "eps_30d_ago"),
                ("60daysAgo", "eps_60d_ago"),
                ("90daysAgo", "eps_90d_ago"),
            ):
                val = _eps(_cell(trend, p, col), sym)
                row[key] = val
                # A 0.0 in the 90-day column means "no estimate existed then",
                # not "the consensus was zero" — seen on HAS.L.
                if cur is not None and val not in (None, 0.0):
                    row[key.replace("_ago", "_change_pct")] = round(
                        (cur / val - 1.0) * 100.0, 2
                    )
        if not _empty(revs) and p in revs.index:
            row.update(
                {
                    "revised_up_7d": _int(_cell(revs, p, "upLast7days")),
                    "revised_down_7d": _int(_cell(revs, p, "downLast7Days")),
                    "revised_up_30d": _int(_cell(revs, p, "upLast30days")),
                    "revised_down_30d": _int(_cell(revs, p, "downLast30days")),
                }
            )
        # Keep only periods with at least one real number. Testing this before
        # attaching the unit label matters: a non-null string like "pence" in
        # the row would make every all-NaN period look populated, which is how
        # HAS.L's empty 0q/+1q rows first slipped through.
        if any(v is not None for k, v in row.items() if k not in ("period", "meaning")):
            row["eps_unit"] = sym.eps_unit
            r.rows.append(row)

    r.ok = bool(r.rows)
    if not r.ok:
        r.note("Revision tables exist but every value is NaN.")
    r.note(
        "A 0.0 in a 60/90-day column means no estimate existed at that date, "
        "not a zero consensus."
    )
    return r


def _int(v: Any) -> int | None:
    f = _num(v)
    return None if f is None else int(f)


# --------------------------------------------------------------------------
# 5. Earnings surprise history (the beat/miss prior)
# --------------------------------------------------------------------------


def earnings_surprise_history(ticker: str, limit: int = 12) -> Result:
    """Actual vs estimated EPS for past reports.

    Two sources are tried: `get_earnings_history()` (4 quarters, clean) and
    `earnings_dates` (up to 25 rows including the upcoming, unreported one).
    They are merged because neither alone is complete.
    """
    sym = resolve(ticker)
    if sym is None:
        return _blank(
            "earnings_surprise_history", None, ticker, f"Unknown ticker {ticker!r}."
        )
    r = Result(
        kind="earnings_surprise_history",
        ticker=sym.ticker,
        yahoo_symbol=sym.yahoo,
        company=sym.name,
    )
    tk = _handle(sym)
    merged: dict[str, dict[str, Any]] = {}

    try:
        hist = tk.get_earnings_history()
    except Exception as e:
        hist = None
        r.note(f"get_earnings_history failed: {type(e).__name__}: {e}")
    if not _empty(hist):
        for idx, row in hist.iterrows():
            d = _date(idx)
            if not d:
                continue
            merged[d] = {
                "period_end": d,
                "eps_actual": _eps(row.get("epsActual"), sym),
                "eps_estimate": _eps(row.get("epsEstimate"), sym),
                "surprise_pct": _pct(row.get("surprisePercent")),
                "eps_unit": sym.eps_unit,
                "basis": "quarter end",
            }
    else:
        r.note("get_earnings_history() returned nothing.")

    try:
        dates = tk.earnings_dates
    except Exception as e:
        dates = None
        r.note(f"earnings_dates failed: {type(e).__name__}: {e}")
    if not _empty(dates):
        for idx, row in dates.iterrows():
            d = _date(idx)
            if not d:
                continue
            actual = _eps(row.get("Reported EPS"), sym)
            est = _eps(row.get("EPS Estimate"), sym)
            surprise = _num(row.get("Surprise(%)"))
            if actual is None and est is None:
                continue
            entry = merged.setdefault(d, {"period_end": d, "eps_unit": sym.eps_unit})
            entry.setdefault("basis", "report date")
            if entry.get("eps_actual") is None:
                entry["eps_actual"] = actual
            if entry.get("eps_estimate") is None:
                entry["eps_estimate"] = est
            if entry.get("surprise_pct") is None and surprise is not None:
                entry["surprise_pct"] = round(surprise, 2)
            entry["reported"] = actual is not None
    else:
        r.note("earnings_dates returned nothing.")

    rows = sorted(merged.values(), key=lambda x: x["period_end"], reverse=True)
    for row in rows:
        row.setdefault("reported", row.get("eps_actual") is not None)
        if row.get("surprise_pct") is None:
            a, e2 = row.get("eps_actual"), row.get("eps_estimate")
            if a is not None and e2 not in (None, 0):
                row["surprise_pct"] = round((a / e2 - 1.0) * 100.0, 2)
    r.rows = rows[: max(1, limit)]
    r.ok = any(row.get("eps_actual") is not None for row in r.rows)

    beats = [
        row["surprise_pct"]
        for row in r.rows
        if row.get("reported") and row.get("surprise_pct") is not None
    ]
    if beats:
        r.fields = {
            "reported_quarters": len(beats),
            "beat_count": sum(1 for b in beats if b > 0),
            "miss_count": sum(1 for b in beats if b < 0),
            "mean_surprise_pct": round(sum(beats) / len(beats), 2),
            "median_surprise_pct": round(sorted(beats)[len(beats) // 2], 2),
            "eps_unit": sym.eps_unit,
        }
    if not r.ok:
        r.note(
            f"No reported EPS surprise history available for {sym.yahoo}. "
            "For semi-annual reporters Yahoo carries none at all."
        )
    return r


# --------------------------------------------------------------------------
# 6. Earnings calendar
# --------------------------------------------------------------------------


def earnings_calendar(ticker: str) -> Result:
    """Next scheduled report date and the consensus attached to it."""
    sym = resolve(ticker)
    if sym is None:
        return _blank("earnings_calendar", None, ticker, f"Unknown ticker {ticker!r}.")
    r = Result(
        kind="earnings_calendar",
        ticker=sym.ticker,
        yahoo_symbol=sym.yahoo,
        company=sym.name,
    )
    try:
        cal = _handle(sym).calendar or {}
    except Exception as e:
        r.note(f"calendar failed: {type(e).__name__}: {e}")
        return r
    if not cal:
        r.note(f"No calendar entry for {sym.yahoo}.")
        return r

    raw_dates = cal.get("Earnings Date") or []
    if not isinstance(raw_dates, (list, tuple)):
        raw_dates = [raw_dates]
    dates = [d for d in (_date(x) for x in raw_dates) if d]

    r.fields = {
        "next_earnings_date": dates[0] if dates else None,
        "earnings_date_range": dates,
        "eps_estimate_avg": _eps(cal.get("Earnings Average"), sym),
        "eps_estimate_low": _eps(cal.get("Earnings Low"), sym),
        "eps_estimate_high": _eps(cal.get("Earnings High"), sym),
        "eps_unit": sym.eps_unit,
        "revenue_estimate_avg_m": _money_m(cal.get("Revenue Average")),
        "revenue_estimate_low_m": _money_m(cal.get("Revenue Low")),
        "revenue_estimate_high_m": _money_m(cal.get("Revenue High")),
        "revenue_unit": sym.money_unit,
        "dividend_date": _date(cal.get("Dividend Date")),
        "ex_dividend_date": _date(cal.get("Ex-Dividend Date")),
    }
    r.ok = r.fields["next_earnings_date"] is not None
    if r.fields["eps_estimate_avg"] is None:
        r.note(
            "Calendar carries a date but no consensus figures. Use "
            "analyst_estimates() for the consensus."
        )
    r.note(
        "The calendar consensus and the analyst_estimates 0q consensus are "
        "separate Yahoo snapshots and can differ slightly; prefer "
        "analyst_estimates for the number and treat any gap as noise."
    )
    return r


# --------------------------------------------------------------------------
# 7. Reported financials
# --------------------------------------------------------------------------

# Rows worth surfacing, in the order a human reads an income statement.
_ROWS_OF_INTEREST: tuple[str, ...] = (
    "Total Revenue",
    "Cost Of Revenue",
    "Gross Profit",
    "Operating Income",
    "Total Operating Income As Reported",
    "Operating Expense",
    "Net Income",
    "Net Income Common Stockholders",
    "Basic EPS",
    "Diluted EPS",
    "Basic Average Shares",
    "Diluted Average Shares",
    "EBIT",
    "Pretax Income",
    "Tax Provision",
)

_PER_SHARE_ROWS = {"Basic EPS", "Diluted EPS"}
_SHARE_COUNT_ROWS = {"Basic Average Shares", "Diluted Average Shares"}


def reported_financials(ticker: str, freq: str = "quarterly") -> Result:
    """Historical income-statement lines, as reported (GAAP/IFRS).

    This is the actuals series a forecast should be anchored to. It is NOT
    adjusted/non-GAAP — the corpus is the place to get adjusted figures.
    """
    sym = resolve(ticker)
    if sym is None:
        return _blank("reported_financials", None, ticker, f"Unknown ticker {ticker!r}.")
    r = Result(
        kind="reported_financials",
        ticker=sym.ticker,
        yahoo_symbol=sym.yahoo,
        company=sym.name,
    )
    want_q = str(freq).lower().startswith("q")
    tk = _handle(sym)
    try:
        df = tk.quarterly_income_stmt if want_q else tk.income_stmt
    except Exception as e:
        r.note(f"income statement fetch failed: {type(e).__name__}: {e}")
        return r

    if _empty(df) and want_q:
        r.note(
            f"No quarterly income statement for {sym.yahoo} — falling back to "
            "annual. (Semi-annual reporters have no quarterly data on Yahoo.)"
        )
        want_q = False
        try:
            df = tk.income_stmt
        except Exception as e:
            r.note(f"annual fallback failed: {type(e).__name__}: {e}")
            return r

    if _empty(df):
        r.note(f"No income statement of any frequency for {sym.yahoo}.")
        return r

    r.fields = {
        "frequency": "quarterly" if want_q else "annual",
        "money_unit": sym.money_unit,
        "eps_unit": sym.eps_unit,
        "periods": [_date(c) for c in df.columns],
    }

    available = [row for row in _ROWS_OF_INTEREST if row in set(df.index)]
    for label in available:
        entry: dict[str, Any] = {"line": label}
        if label in _PER_SHARE_ROWS:
            entry["unit"] = sym.eps_unit
        elif label in _SHARE_COUNT_ROWS:
            entry["unit"] = "shares (millions)"
        else:
            entry["unit"] = sym.money_unit
        values: dict[str, Any] = {}
        for col in df.columns:
            raw = _cell(df, label, col)
            if label in _PER_SHARE_ROWS:
                val = _eps(raw, sym)
            else:
                val = _money_m(raw)
            values[_date(col) or str(col)] = val
        entry["values"] = values
        if any(v is not None for v in values.values()):
            r.rows.append(entry)

    # Derived: GAAP gross margin, the only margin Yahoo supports.
    rev = next((x for x in r.rows if x["line"] == "Total Revenue"), None)
    gp = next((x for x in r.rows if x["line"] == "Gross Profit"), None)
    if rev and gp:
        margins = {}
        for period, rv in rev["values"].items():
            gv = gp["values"].get(period)
            margins[period] = (
                round(gv / rv * 100.0, 2) if rv not in (None, 0) and gv is not None else None
            )
        if any(v is not None for v in margins.values()):
            r.rows.append(
                {
                    "line": "Gross Margin (derived, GAAP)",
                    "unit": "percent",
                    "values": margins,
                }
            )

    r.ok = bool(r.rows)
    if not r.ok:
        r.note("Income statement returned, but none of the expected lines exist.")

    empty_cols = [
        p
        for i, p in enumerate(r.fields["periods"])
        if all(row["values"].get(p) is None for row in r.rows)
    ]
    if empty_cols:
        r.note(
            f"Period(s) {', '.join(empty_cols)} are entirely empty on Yahoo — "
            "the filing exists but Yahoo has not populated it. Use the corpus."
        )
    if sym.ticker == "HAS":
        r.note(
            "'Gross Profit' here IS Hays' NET FEES (verified: FY2024 = GBP "
            "1,113.6m, matching the reported net fees). 'Total Revenue' is "
            "turnover and is NOT the scored metric."
        )
        r.note("EPS rows are STATUTORY (post-exceptional) and in pence.")
    r.note(
        "These are as-reported GAAP/IFRS figures. Adjusted / pre-exceptional "
        "measures are not on Yahoo — take those from the corpus."
    )
    return r


# --------------------------------------------------------------------------
# 8. Analyst coverage
# --------------------------------------------------------------------------


def analyst_coverage(ticker: str) -> Result:
    """Price targets, recommendation distribution, and long-run growth view."""
    sym = resolve(ticker)
    if sym is None:
        return _blank("analyst_coverage", None, ticker, f"Unknown ticker {ticker!r}.")
    r = Result(
        kind="analyst_coverage",
        ticker=sym.ticker,
        yahoo_symbol=sym.yahoo,
        company=sym.name,
    )
    tk = _handle(sym)

    try:
        pt = tk.analyst_price_targets or {}
    except Exception as e:
        pt = {}
        r.note(f"analyst_price_targets failed: {type(e).__name__}: {e}")
    if pt:
        r.fields.update(
            {
                "price_current": _num(pt.get("current")),
                "target_mean": _num(pt.get("mean")),
                "target_median": _num(pt.get("median")),
                "target_high": _num(pt.get("high")),
                "target_low": _num(pt.get("low")),
                "price_currency": sym.price_currency,
            }
        )
        cur, mean = r.fields.get("price_current"), r.fields.get("target_mean")
        if cur and mean:
            r.fields["upside_to_mean_pct"] = round((mean / cur - 1.0) * 100.0, 2)
        high = r.fields.get("target_high")
        if cur and high and cur > high:
            r.note(
                "The current price is ABOVE the highest published target — the "
                "target set is stale relative to the last move. Treat targets "
                "as low-information here."
            )

    try:
        recs = tk.recommendations
    except Exception as e:
        recs = None
        r.note(f"recommendations failed: {type(e).__name__}: {e}")
    if not _empty(recs):
        for _, row in recs.iterrows():
            entry = {
                "period": row.get("period"),
                "strong_buy": _int(row.get("strongBuy")),
                "buy": _int(row.get("buy")),
                "hold": _int(row.get("hold")),
                "sell": _int(row.get("sell")),
                "strong_sell": _int(row.get("strongSell")),
            }
            entry["total"] = sum(
                v for k, v in entry.items() if k != "period" and isinstance(v, int)
            )
            r.rows.append(entry)

    try:
        growth = tk.growth_estimates
    except Exception as e:
        growth = None
        r.note(f"growth_estimates failed: {type(e).__name__}: {e}")
    if not _empty(growth):
        g: dict[str, Any] = {}
        for p in growth.index:
            v = _pct(_cell(growth, p, "stockTrend"))
            if v is not None:
                g[f"growth_{str(p).replace('+', 'plus')}_pct"] = v
        if g:
            r.fields["growth_estimates"] = g

    r.ok = bool(r.fields) or bool(r.rows)
    if not r.ok:
        r.note(f"No analyst coverage data for {sym.yahoo}.")
    if sym.ticker == "HAS":
        r.note("Price and targets are in PENCE (GBp).")
    r.note(
        "Price targets and ratings are sentiment, not a fundamental forecast. "
        "They do not constrain a revenue or EPS estimate."
    )
    return r


CLIENT_FUNCTIONS = (
    quote_snapshot,
    price_history,
    analyst_estimates,
    estimate_revisions,
    earnings_surprise_history,
    earnings_calendar,
    reported_financials,
    analyst_coverage,
)
