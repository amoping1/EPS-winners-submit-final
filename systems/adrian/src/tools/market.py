"""Market data channel - public analyst consensus and price context.

Rules explicitly permit "public information you find during the event", and consensus is
not a nice-to-have here: the accuracy score is our absolute error divided by Wall Street's
absolute error. Consensus IS the denominator. Not fetching it means forecasting blind
against a benchmark we could simply read.

Coverage is asymmetric and that is the point of having two sources:
  - US filers (HD, ADI, DE)  -> Yahoo carries analyst estimates for the current quarter
  - UK filers (Hays)         -> Yahoo has no estimates, but the company publishes its own
                                compiled consensus in the corpus (see tools/consensus.py)

Every value is tagged with its source and fetch time, and a failure returns an empty
result rather than raising - a dead network must not take the forecast run down with it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# Ticker in companies.json -> Yahoo symbol.
YAHOO_SYMBOLS = {
    "HD": "HD",
    "ADI": "ADI",
    "DE": "DE",
    "LSE:HAS": "HAS.L",
}

# Which metric labels a consensus figure maps onto, per kind.
REVENUE_KEYS = ("net sales", "revenue", "worldwide net sales")
EPS_KEYS = ("eps", "earnings per share")


@dataclass
class MarketSnapshot:
    ticker: str
    symbol: str
    available: bool = False
    earnings_date: str | None = None
    eps_avg: float | None = None
    eps_low: float | None = None
    eps_high: float | None = None
    eps_analysts: int | None = None
    eps_year_ago: float | None = None
    revenue_avg_m: float | None = None
    revenue_low_m: float | None = None
    revenue_high_m: float | None = None
    error: str | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


def _clean(value):
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


def fetch_snapshot(ticker: str, timeout: int = 20) -> MarketSnapshot:
    """Pull current-quarter analyst consensus. Never raises."""
    symbol = YAHOO_SYMBOLS.get(ticker, ticker)
    snap = MarketSnapshot(ticker=ticker, symbol=symbol)
    try:
        import yfinance as yf

        handle = yf.Ticker(symbol)
        calendar = handle.calendar or {}

        dates = calendar.get("Earnings Date") or []
        if dates:
            snap.earnings_date = str(dates[0])

        snap.eps_avg = _clean(calendar.get("Earnings Average"))
        snap.eps_low = _clean(calendar.get("Earnings Low"))
        snap.eps_high = _clean(calendar.get("Earnings High"))

        for key, attr in (("Revenue Average", "revenue_avg_m"),
                          ("Revenue Low", "revenue_low_m"),
                          ("Revenue High", "revenue_high_m")):
            raw = _clean(calendar.get(key))
            if raw is not None:
                setattr(snap, attr, raw / 1e6)     # report in millions, as the workbooks do

        try:
            estimates = handle.earnings_estimate
            if estimates is not None and "0q" in estimates.index:
                row = estimates.loc["0q"]
                snap.eps_analysts = int(_clean(row.get("numberOfAnalysts")) or 0) or None
                snap.eps_year_ago = _clean(row.get("yearAgoEps"))
                if snap.eps_avg is None:
                    snap.eps_avg = _clean(row.get("avg"))
        except Exception:                          # estimates table is optional
            pass

        snap.available = any(
            v is not None for v in (snap.eps_avg, snap.revenue_avg_m)
        )
        if not snap.available:
            snap.notes.append("no analyst coverage returned for this symbol")

    except ImportError:
        snap.error = "yfinance not installed"
    except Exception as exc:                       # network, rate limit, schema drift
        snap.error = f"{type(exc).__name__}: {exc}"[:160]
    return snap


def consensus_anchors(snap: MarketSnapshot, metric_labels: list[str],
                      target_period: str) -> list[dict]:
    """Turn a snapshot into anchors shaped like the research agent's, for reconciliation.

    Only current-quarter ("0q") estimates are used, and only when the reported earnings
    date is still ahead of us - a consensus for a quarter already reported is an actual,
    not a forecast, and would be a different kind of evidence entirely.
    """
    if not snap.available:
        return []

    anchors: list[dict] = []
    for label in metric_labels:
        low_label = label.lower()

        if any(k in low_label for k in REVENUE_KEYS) and snap.revenue_avg_m is not None:
            anchors.append({
                "metric": label, "kind": "consensus",
                "value": round(snap.revenue_avg_m, 4),
                "low": round(snap.revenue_low_m, 4) if snap.revenue_low_m else None,
                "high": round(snap.revenue_high_m, 4) if snap.revenue_high_m else None,
                "period": target_period, "units": "USDm",
                "note": (f"Yahoo Finance analyst consensus, {snap.eps_analysts or 'n/a'} "
                         f"analysts, earnings due {snap.earnings_date}"),
                "doc_id": f"yahoo:{snap.symbol}", "confidence": "high",
                "source": "market_data",
            })

        elif any(k in low_label for k in EPS_KEYS) and snap.eps_avg is not None:
            anchors.append({
                "metric": label, "kind": "consensus",
                "value": round(snap.eps_avg, 4),
                "low": snap.eps_low, "high": snap.eps_high,
                "period": target_period, "units": "USD / share",
                "note": (f"Yahoo Finance analyst consensus, {snap.eps_analysts or 'n/a'} "
                         f"analysts, prior-year {snap.eps_year_ago}, "
                         f"earnings due {snap.earnings_date}"),
                "doc_id": f"yahoo:{snap.symbol}", "confidence": "high",
                "source": "market_data",
            })

    return anchors
