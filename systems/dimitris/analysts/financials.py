"""financials.py — the FINANCIALS ANALYST.

The numeric backdrop. It produces a normalised, per-fiscal-period income
statement history for ONE company — revenue, gross profit, operating income,
net income, diluted EPS and the two derived margins — plus the recent quarterly
series, and returns a `FinancialsReport`.

It is NOT a forecast and it is NOT the evidence base. The corpus (filings
analyst) holds the reported, adjusted and segment figures with citations. This
module holds one thing the corpus does not: a single clean, unit-normalised,
same-shape table across all four companies that the central agent can TREND and
SANITY-CHECK against. If the central agent's forecast implies a 40% revenue
jump, this table is what says so.

=== WHY THIS ONE IS DETERMINISTIC, AND THE OTHER THREE ARE NOT ===

The filings and news analysts exist because their inputs require JUDGEMENT:
which document, which basis, which source is junk. There is no judgement here.
The task is "fetch an income statement and pivot it", and every hard part of it
— the GBp/GBP split, the x100 EPS scaling, absolute-units-to-millions, NaN
handling — is ALREADY SOLVED inside `marketdata/client.py`.

Handing that to an LLM buys nothing and costs three things: money, ~30 seconds
per company, and a non-zero chance it transcribes 41765 as 41.765 or applies
the Hays x100 a second time. A model that reads a number and writes it down is
a lossy channel with a bill attached.

So the core is a plain function. `analyse()` keeps the same async signature as
the other three analysts, so the pipeline cannot tell the difference, and
`build_spec()` still exists — it drives an OPTIONAL narrative pass (--llm-notes)
that may append prose to `notes` and `data_gaps` and MAY NEVER TOUCH A NUMBER.

=== WHAT THE DATA ACTUALLY IS — verified 2026-08-16, yfinance 1.2.0 ===

Read these before trusting any figure this module emits.

  * **THERE IS NO 15-YEAR HISTORY.** Yahoo returns at most FIVE annual columns
    and the oldest is empty on every one of the four (HD 2022-01-31, ADI and DE
    2021-10-31 all come back all-null). So the real depth is FOUR fiscal years,
    everywhere. `years_available` reports what is actually populated and
    `data_gaps` says so out loud. A long-horizon table is not available from
    this source at any price; the corpus goes back to 2015 and is where a
    decade-plus series has to come from.

  * **"Total Operating Income As Reported" is the row that ties. "Operating
    Income" is not.** Verified against the corpus on two companies and both
    frequencies: ADI FY2025 reported GAAP operating income 2,932.496 USDm
    matches Yahoo's As-Reported row (2,932.5) while Yahoo's "Operating Income"
    says 3,002.5; ADI Q3 FY2025 reported 818.028 matches 818.0 against 822.4;
    Hays FY2024 post-exceptional operating profit 25.1 GBPm matches 25.1
    against 69.6. This module therefore prefers the As-Reported row and records
    the discrepancy when the two disagree. Deere has no As-Reported row at all.

  * **Hays mixes two currencies in one payload.** `info["currency"]` is GBp
    (pence, the quote) and `financialCurrency` is GBP (the statements). A raw
    yfinance EPS of 0.01102 is GBP and means 1.102 PENCE. `client._eps()`
    already applies the x100. Nothing here multiplies again, and
    `_sanity_checks` fails loudly if a Hays EPS lands outside the plausible
    pence band — a double-conversion would show up as ~859 rather than 8.59.

  * **Yahoo's "Gross Profit" for Hays IS net fees.** FY2024 comes back as
    exactly 1,113.6 GBPm, which is the £1,113.6m net fees Hays reported. That
    is a directly scored metric, so it is labelled as such in every output.
    Yahoo's "Total Revenue" for Hays is TURNOVER (6,949.1 GBPm, ~6x net fees)
    and is NOT the scored line. The Hays rows also carry the conversion rate,
    Hays' own KPI, because operating-profit-over-net-fees is how that company
    is actually analysed.

  * **Deere's revenue does not tie to anything Deere reports.** Yahoo's Total
    Revenue is BELOW the reported "Total net sales and revenues" by 1,019 USDm
    in FY2025 (44,665 vs 45,684), 1,198 USDm in FY2024 (50,518 vs 51,716) and
    235 USDm in the quarter ended 2025-07-31 (11,783 vs 12,018) — it looks like
    it drops "Other revenues". It is ALSO not the equipment-operations basis
    that Yahoo's own revenue CONSENSUS tracks (10,357 for that same quarter).
    Three different numbers, none of them interchangeable. Deere's net income
    and diluted EPS do tie exactly, so those are safe. The scored revenue must
    come from the corpus.

  * **HAS.L is semi-annual and has no quarterly anything.**
    `quarterly_income_stmt` is empty, `get_earnings_history()` returns a 0x0
    frame, `earnings_dates` has one row, and the 0q/+1q consensus rows are all
    NaN. That is a reporting-frequency fact, not a failure — `quarterly` comes
    back empty and `data_gaps` names each empty endpoint. Note that
    `client.reported_financials(freq="quarterly")` FALLS BACK to annual for
    Hays, so the fallback is detected and the annual rows are NOT laundered
    into the quarterly series.

  * **Hays FY2025 is a near-empty column** — 4 non-null rows of 52, EPS and
    share counts only. Revenue, gross profit, operating income and net income
    are all null for the most recent fiscal year.

Standalone:

    python -m analysts.financials --ticker HD
    python -m analysts.financials --ticker HAS --json has.json --no-store
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

# agent_core.config load_dotenv()s the repo .env at import from an absolute
# path, so the entry point no longer matters and filings.py's local
# ensure_env_loaded() shim is not duplicated here.
from agent_core import get_last_usage, AgentSpec, run_agent, settings, use_selector_event_loop
from marketdata import client
from marketdata.tools import MARKET_TOOLS
from runstore import RunStore, get_store

from .models import FinancialsReport, FinancialYear

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent

#: The narrative pass is one short call — it reads a table that is already
#: built and writes prose. It never needs a long tool loop.
DEFAULT_MAX_TURNS = 8

#: Yahoo's hard ceiling, measured on all four symbols. Five annual columns are
#: returned and the oldest is always empty.
YAHOO_ANNUAL_COLUMNS = 5
YAHOO_POPULATED_ANNUAL_YEARS = 4

#: What the brief asked for, so the shortfall is stated rather than implied.
REQUESTED_HORIZON_YEARS = 15


# ---------------------------------------------------------------------------
# Fiscal calendars
#
# Yahoo labels a column by its period-END date. Turning that into the company's
# own fiscal-year name needs two facts per company, and they genuinely differ:
# Home Depot names a fiscal year for the calendar year it STARTS in (the year
# ended 2026-01-31 is HD's fiscal 2025), while Hays, ADI and Deere name it for
# the year it ENDS in.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FiscalCalendar:
    fy_end_month: int  # calendar month the fiscal year ends in
    fy_label_offset: int  # label = fiscal-year-end calendar year + this
    frequency: str  # "quarterly" | "semi-annual"
    calendar_note: str


CALENDARS: dict[str, FiscalCalendar] = {
    "HD": FiscalCalendar(
        fy_end_month=1,
        fy_label_offset=-1,
        frequency="quarterly",
        calendar_note=(
            "Fiscal year ends late January or early February. Home Depot names "
            "a fiscal year for the calendar year it STARTS in, so the year "
            "ended 2026-01-31 is fiscal 2025 and the quarter ending ~2 August "
            "2026 is Q2 of fiscal 2026. 52/53-week retail calendar."
        ),
    ),
    "ADI": FiscalCalendar(
        fy_end_month=10,
        fy_label_offset=0,
        frequency="quarterly",
        calendar_note=(
            "Fiscal year ends late October or early November; the year ended "
            "2025-11-01 is fiscal 2025. 52/53-week calendar, so period ends "
            "drift a few days either side of month end."
        ),
    ),
    "HAS": FiscalCalendar(
        fy_end_month=6,
        fy_label_offset=0,
        frequency="semi-annual",
        calendar_note=(
            "UK filer, fiscal year ends 30 June; the year ended 2025-06-30 is "
            "FY2025. Reports HALVES, not quarters, with quarterly trading "
            "statements that carry net fee growth but no income statement."
        ),
    ),
    "DE": FiscalCalendar(
        fy_end_month=10,
        fy_label_offset=0,
        frequency="quarterly",
        calendar_note=(
            "Fiscal year ends late October or early November; the year ended "
            "2025-11-02 is fiscal 2025. 52/53-week calendar."
        ),
    ),
}


def normalise_ticker(ticker: str) -> str:
    """Accept 'LSE:HAS', 'has', 'HAS.L' — return the challenge ticker.

    Delegates the alias table to `marketdata.client` so this module and the
    market tools can never disagree about what 'HAS.L' means.
    """
    raw = (ticker or "").strip().upper()
    if raw.startswith("LSE:"):
        raw = raw.split(":", 1)[1]
    sym = client.resolve(raw)
    if sym is None:
        raise ValueError(
            f"Unknown ticker {ticker!r}. Known: {', '.join(client.known_tickers())} "
            "(HAS.L and LSE:HAS are accepted for Hays)."
        )
    return sym.ticker


# ---------------------------------------------------------------------------
# Period labelling
# ---------------------------------------------------------------------------


def _effective_month_year(period_end: str) -> tuple[int, int] | None:
    """Month and year of a period end, robust to a 52/53-week calendar.

    A 52/53-week fiscal period can end a few days INTO the following month —
    Home Depot's fiscal 2025 ended 1 February 2026, ADI's fiscal 2025 ended
    1 November 2025 — which would push a naive `date.month` into the wrong
    quarter. Backing up 15 days snaps the date into the month the period
    substantively belongs to, and is a no-op for a normal month-end column.
    """
    try:
        d = date.fromisoformat(str(period_end)[:10])
    except (ValueError, TypeError):
        return None
    eff = d - timedelta(days=15)
    return eff.month, eff.year


def fiscal_year_label(period_end: str, cal: FiscalCalendar) -> str:
    """'FY2025' for an annual column ending `period_end`."""
    my = _effective_month_year(period_end)
    if my is None:
        return str(period_end)
    month, year = my
    fy_end_year = year if month <= cal.fy_end_month else year + 1
    return f"FY{fy_end_year + cal.fy_label_offset}"


def quarter_label(period_end: str, cal: FiscalCalendar) -> str:
    """'Q2 FY2026' for a quarterly column ending `period_end`.

    Quarter index counts backwards from the fiscal year end: the quarter whose
    effective month IS the fiscal-year-end month is Q4.
    """
    my = _effective_month_year(period_end)
    if my is None:
        return str(period_end)
    month, year = my
    offset = (month - cal.fy_end_month) % 12
    q = offset // 3
    quarter = 4 if q == 0 else q
    fy_end_year = year if month <= cal.fy_end_month else year + 1
    return f"Q{quarter} FY{fy_end_year + cal.fy_label_offset}"


# ---------------------------------------------------------------------------
# Pivot: client.Result (row-per-line) -> FinancialYear (row-per-period)
# ---------------------------------------------------------------------------

#: Income-statement line names as `marketdata.client` emits them, in the order
#: this module prefers them. The operating-income pair is deliberate — see the
#: module docstring for the tie-out evidence.
_REVENUE_LINES = ("Total Revenue",)
_GROSS_PROFIT_LINES = ("Gross Profit",)
_OPERATING_INCOME_LINES = ("Total Operating Income As Reported", "Operating Income")
_NET_INCOME_LINES = ("Net Income", "Net Income Common Stockholders")
_DILUTED_EPS_LINES = ("Diluted EPS",)

CORE_FIELDS = (
    "revenue",
    "gross_profit",
    "operating_income",
    "net_income",
    "diluted_eps",
)


def _line_values(res: client.Result, line: str) -> dict[str, Any]:
    for row in res.rows:
        if row.get("line") == line:
            return row.get("values") or {}
    return {}


def _pick(res: client.Result, lines: tuple[str, ...], period: str) -> tuple[float | None, str]:
    """First non-null value across `lines` for `period`, and which line won."""
    for line in lines:
        val = _line_values(res, line).get(period)
        if val is not None:
            return float(val), line
    return None, ""


def _margin(numerator: float | None, denominator: float | None) -> float | None:
    """Percentage POINTS. 33.32 means 33.32%, never 0.3332."""
    if numerator is None or denominator in (None, 0):
        return None
    return round(numerator / denominator * 100.0, 2)


def _units_label(sym: client.Symbol) -> str:
    """Money unit AND per-share unit in one string.

    `FinancialYear.units` is a single field but the row carries two scales, and
    for Hays they are different currencies entirely (GBP millions and pence).
    Naming only one of them is how a 100x error gets through.
    """
    return f"{sym.money_unit}; EPS in {sym.eps_unit}"


def _build_periods(
    res: client.Result,
    sym: client.Symbol,
    cal: FiscalCalendar,
    *,
    quarterly: bool,
) -> tuple[list[FinancialYear], list[str], dict[str, str]]:
    """Pivot one client Result into FinancialYear rows, newest first.

    Returns (rows, empty_period_ends, period_end_by_label). Columns where every
    core field is null are DROPPED — they carry no information and would
    inflate `years_available` — but their dates come back so `data_gaps` can
    name them.
    """
    rows: list[FinancialYear] = []
    empty: list[str] = []
    mapping: dict[str, str] = {}
    units = _units_label(sym)

    for period in res.fields.get("periods") or []:
        revenue, _ = _pick(res, _REVENUE_LINES, period)
        gross_profit, _ = _pick(res, _GROSS_PROFIT_LINES, period)
        operating_income, _ = _pick(res, _OPERATING_INCOME_LINES, period)
        net_income, _ = _pick(res, _NET_INCOME_LINES, period)
        diluted_eps, _ = _pick(res, _DILUTED_EPS_LINES, period)

        if all(
            v is None
            for v in (revenue, gross_profit, operating_income, net_income, diluted_eps)
        ):
            empty.append(str(period))
            continue

        label = (
            quarter_label(period, cal) if quarterly else fiscal_year_label(period, cal)
        )
        mapping[label] = str(period)
        rows.append(
            FinancialYear(
                fiscal_period=label,
                revenue=revenue,
                gross_profit=gross_profit,
                operating_income=operating_income,
                net_income=net_income,
                diluted_eps=diluted_eps,
                gross_margin_pct=_margin(gross_profit, revenue),
                operating_margin_pct=_margin(operating_income, revenue),
                units=units,
            )
        )

    # Yahoo already returns newest-first, but sort on the period-end date so a
    # change of upstream ordering cannot silently reverse the table.
    rows.sort(key=lambda r: mapping.get(r.fiscal_period, ""), reverse=True)
    return rows, empty, mapping


# ---------------------------------------------------------------------------
# Sanity checks — the empirically-established facts, enforced in code
#
# Each of these is a number that was verified against the corpus or against
# arithmetic. They are asserted at run time so a change in Yahoo's feed, or a
# regression in this module's unit handling, surfaces as a visible warning
# rather than as a wrong figure handed to the central agent.
# ---------------------------------------------------------------------------

#: Hays FY2024 net fees, from the 2024-08-22 full-year results announcement.
HAS_FY2024_NET_FEES_GBPM = 1113.6

#: Deere's reported "Total net sales and revenues", from the FY2025 Q4 8-K.
DE_REPORTED_TOTAL_REVENUE = {"FY2025": 45684.0, "FY2024": 51716.0}

#: Home Depot Q1 FY2026 net sales, from the statement of earnings in the
#: 2026-05-19 8-K — the figure the release renders as "$41.8 billion" in prose.
HD_Q1_FY2026_NET_SALES_USDM = 41765.0


def _find(rows: list[FinancialYear], label: str) -> FinancialYear | None:
    return next((r for r in rows if r.fiscal_period == label), None)


def _sanity_checks(
    tk: str, annual: list[FinancialYear], quarterly: list[FinancialYear]
) -> list[str]:
    """Tie-outs against known-good figures. Returns human-readable results."""
    out: list[str] = []

    if tk == "HAS":
        fy24 = _find(annual, "FY2024")
        if fy24 is None:
            out.append("CHECK FAILED: no FY2024 row to tie net fees against.")
        elif fy24.gross_profit is None:
            out.append("CHECK FAILED: FY2024 gross profit (net fees) is null.")
        elif abs(fy24.gross_profit - HAS_FY2024_NET_FEES_GBPM) < 0.05:
            out.append(
                f"CHECK OK: FY2024 gross profit {fy24.gross_profit:,.1f} GBPm ties "
                f"exactly to Hays' reported net fees of "
                f"{HAS_FY2024_NET_FEES_GBPM:,.1f}m. Yahoo 'Gross Profit' IS net "
                "fees for this company."
            )
        else:
            out.append(
                f"CHECK FAILED: FY2024 gross profit {fy24.gross_profit:,.1f} GBPm "
                f"does NOT match the reported net fees "
                f"{HAS_FY2024_NET_FEES_GBPM:,.1f}m — the gross-profit-is-net-fees "
                "mapping may no longer hold."
            )

        # A double x100 would put EPS in the hundreds; a missing x100 would put
        # it in the hundredths. Hays' statutory EPS has run between -0.5p and
        # ~9.3p across the available years.
        eps = [r.diluted_eps for r in annual if r.diluted_eps is not None]
        if not eps:
            out.append("CHECK FAILED: no diluted EPS values for Hays at all.")
        elif max(abs(v) for v in eps) > 100.0:
            out.append(
                f"CHECK FAILED: Hays diluted EPS reaches {max(eps):,.4f} — far too "
                "large for pence. The x100 GBP->pence conversion has been applied "
                "TWICE. marketdata/client.py already scales it; do not scale again."
            )
        elif max(abs(v) for v in eps) < 0.1:
            out.append(
                f"CHECK FAILED: Hays diluted EPS peaks at {max(eps):,.4f} — that is "
                "GBP, not pence. The x100 conversion did not happen."
            )
        else:
            out.append(
                f"CHECK OK: Hays diluted EPS spans {min(eps):,.2f} to {max(eps):,.2f} "
                "pence, the expected magnitude — converted once, not twice."
            )

    if tk == "DE":
        for label, reported in DE_REPORTED_TOTAL_REVENUE.items():
            row = _find(annual, label)
            if row is None or row.revenue is None:
                continue
            gap = reported - row.revenue
            out.append(
                f"CHECK: {label} Yahoo revenue {row.revenue:,.1f} USDm is "
                f"{gap:,.1f} USDm BELOW Deere's reported total net sales and "
                f"revenues of {reported:,.1f} USDm. Yahoo's line is not the "
                "scored basis."
            )

    if tk == "HD":
        q = _find(quarterly, "Q1 FY2026")
        if q is not None and q.revenue is not None:
            if abs(q.revenue - HD_Q1_FY2026_NET_SALES_USDM) < 0.5:
                out.append(
                    f"CHECK OK: Q1 FY2026 revenue {q.revenue:,.1f} USDm ties "
                    "exactly to the net sales line in Home Depot's statement of "
                    "earnings. Reported in MILLIONS, not billions."
                )
            else:
                out.append(
                    f"CHECK FAILED: Q1 FY2026 revenue {q.revenue:,.1f} USDm does "
                    f"not match the reported {HD_Q1_FY2026_NET_SALES_USDM:,.1f}."
                )

    # Universal: margins must be in percentage POINTS, never fractions.
    all_rows = annual + quarterly
    margins = [
        m
        for r in all_rows
        for m in (r.gross_margin_pct, r.operating_margin_pct)
        if m is not None
    ]
    if margins and max(abs(m) for m in margins) <= 1.0:
        out.append(
            "CHECK FAILED: every margin is below 1.0 — they look like fractions. "
            "Percentages must be in POINTS (33.32 means 33.32%)."
        )
    return out


# ---------------------------------------------------------------------------
# Per-company basis notes
# ---------------------------------------------------------------------------


def _basis_notes(tk: str) -> list[str]:
    """What each column actually MEANS for this company. Verified, not assumed."""
    common = [
        "BASIS: every figure here is as-reported GAAP or IFRS, taken from "
        "Yahoo's income statement. Adjusted, non-GAAP and pre-exceptional "
        "measures are NOT on Yahoo and must come from the corpus.",
        "OPERATING INCOME: taken from Yahoo's 'Total Operating Income As "
        "Reported' row where it exists, which is the row that ties to the "
        "company's own reported operating income. Yahoo's plain 'Operating "
        "Income' row is a derived figure that does not tie and is not used.",
    ]
    per_ticker = {
        "HD": [
            "REVENUE BASIS: total company net sales, GAAP, in USD MILLIONS. "
            "Ties exactly to the statement of earnings (Q1 FY2026 = 41,765).",
            "EPS BASIS: GAAP DILUTED EPS. The scored Home Depot metric is "
            "ADJUSTED diluted EPS, which is higher and is NOT on Yahoo — Q1 "
            "FY2026 was 3.30 GAAP against 3.43 adjusted. Do not use this "
            "column as the adjusted figure.",
            "Comparable sales percent, the third scored metric, does not exist "
            "on Yahoo in any form. Corpus only.",
        ],
        "ADI": [
            "REVENUE BASIS: GAAP revenue, USD MILLIONS. Quarterly columns are "
            "discrete quarters, not the cumulative six/nine-month columns that "
            "sit beside them in the 10-Q.",
            "EPS BASIS: GAAP DILUTED EPS. ADI's scored metric is ADJUSTED "
            "diluted EPS, which runs far higher because of acquisition-related "
            "amortisation. Not the same number.",
            "GROSS MARGIN: the derived margin here is GAAP. ADI's scored metric "
            "is ADJUSTED gross margin, several points higher. Treat the GAAP "
            "figure as a FLOOR, never as the answer.",
        ],
        "HAS": [
            "GROSS PROFIT IS NET FEES. Yahoo's 'Gross Profit' for Hays is the "
            "net fees line — FY2024 comes back as exactly 1,113.6 GBPm, "
            "matching the reported figure. Net fees is a directly scored "
            "metric, so this column is usable as-is.",
            "REVENUE IS TURNOVER, NOT NET FEES. Yahoo's 'Total Revenue' "
            "(FY2024: 6,949.1 GBPm) is statutory turnover, roughly six times "
            "net fees, and is NOT scored. The gross-margin column below is "
            "therefore net fees over turnover (~16%), which is arithmetic, not "
            "a profitability collapse.",
            "OPERATING PROFIT IS POST-EXCEPTIONAL. The operating income column "
            "ties to Hays' STATUTORY operating profit (FY2024 = 25.1 GBPm). "
            "The scored metric is PRE-EXCEPTIONAL operating profit, which was "
            "105.1m in FY2024 — a 80.0m exceptional charge apart. Not on "
            "Yahoo; corpus only.",
            "EPS IS STATUTORY AND IN PENCE. Post-exceptional basic/diluted EPS, "
            "already converted from GBP to pence. FY2025 statutory diluted EPS "
            "is negative. The scored metric is PRE-EXCEPTIONAL BASIC EPS, a "
            "different number on a different basis.",
            "CONVERSION RATE (pre-exceptional operating profit over net fees) "
            "is Hays' own headline KPI and the ratio the company is analysed "
            "on. It cannot be computed from this table, because the numerator "
            "is pre-exceptional and only the post-exceptional figure is here.",
        ],
        "DE": [
            "REVENUE BASIS IS AMBIGUOUS AND MUST NOT BE USED AS SCORED. Yahoo's "
            "'Total Revenue' for Deere sits BELOW the reported 'Total net sales "
            "and revenues' by 1,019 USDm in FY2025 (44,665 against 45,684), "
            "1,198 USDm in FY2024 (50,518 against 51,716) and 235 USDm in the "
            "quarter ended 2025-07-31 (11,783 against 12,018) — it appears to "
            "exclude 'Other revenues'. It is also NOT the equipment-operations "
            "basis that Yahoo's own revenue CONSENSUS tracks, which was 10,357 "
            "for that same quarter. Three distinct bases; the scored one is the "
            "reported worldwide total, from the corpus.",
            "EPS AND NET INCOME DO TIE. FY2025 net income 5,027 USDm and "
            "diluted EPS 18.50 match Deere's reported figures exactly, and the "
            "scored EPS metric is GAAP diluted — so this EPS column IS on the "
            "right basis.",
            "Segment detail — Production & Precision Ag operating profit, the "
            "third scored metric — is not on Yahoo at any frequency. Corpus "
            "only.",
        ],
    }
    return common + per_ticker.get(tk, [])


# ---------------------------------------------------------------------------
# The deterministic core
# ---------------------------------------------------------------------------


def build_report(ticker: str, as_of: str | None = None) -> FinancialsReport:
    """Fetch, pivot and validate one company's income-statement history.

    Blocking (yfinance is synchronous HTTP). `analyse()` runs it on a worker
    thread so a four-company fan-out is not serialised behind it.

    `as_of` is recorded but CANNOT filter this source: Yahoo serves whatever it
    holds now and stamps no vintage on it. The report's `as_of` is therefore
    the RETRIEVAL time, and a mismatch with a requested backtest date is raised
    in `data_gaps` rather than quietly ignored.
    """
    tk = normalise_ticker(ticker)
    sym = client.resolve(tk)
    assert sym is not None  # normalise_ticker would have raised
    cal = CALENDARS[tk]

    annual_res = client.reported_financials(tk, freq="annual")
    quarterly_res = client.reported_financials(tk, freq="quarterly")

    annual, empty_annual, annual_map = _build_periods(
        annual_res, sym, cal, quarterly=False
    )

    # client.reported_financials FALLS BACK to annual when a company files no
    # quarterly accounts. Detecting that is the whole reason Hays does not end
    # up with four annual columns mislabelled as quarters.
    quarterly_fell_back = quarterly_res.fields.get("frequency") != "quarterly"
    if quarterly_fell_back:
        quarterly: list[FinancialYear] = []
        empty_quarterly: list[str] = []
        quarterly_map: dict[str, str] = {}
    else:
        quarterly, empty_quarterly, quarterly_map = _build_periods(
            quarterly_res, sym, cal, quarterly=True
        )

    fully_populated = [
        r for r in annual if all(getattr(r, f) is not None for f in CORE_FIELDS)
    ]
    partial = [r for r in annual if r not in fully_populated]

    report = FinancialsReport(
        ticker=tk,
        yahoo_symbol=sym.yahoo,
        currency=sym.money_currency,
        annual=annual,
        quarterly=quarterly,
        years_available=len(annual),
        source=(
            f"{client.SOURCE} — income_stmt and quarterly_income_stmt for "
            f"{sym.yahoo}"
        ),
        as_of=annual_res.as_of,
    )

    report.data_gaps = _build_data_gaps(
        tk,
        sym,
        cal,
        annual=annual,
        quarterly=quarterly,
        partial=partial,
        empty_annual=empty_annual,
        empty_quarterly=empty_quarterly,
        quarterly_fell_back=quarterly_fell_back,
        annual_res=annual_res,
        quarterly_res=quarterly_res,
        as_of=as_of,
    )
    report.notes = _build_notes(
        tk,
        sym,
        cal,
        annual=annual,
        quarterly=quarterly,
        fully_populated=fully_populated,
        annual_map=annual_map,
        quarterly_map=quarterly_map,
        annual_res=annual_res,
        as_of=as_of,
    )
    return report


def _build_data_gaps(
    tk: str,
    sym: client.Symbol,
    cal: FiscalCalendar,
    *,
    annual: list[FinancialYear],
    quarterly: list[FinancialYear],
    partial: list[FinancialYear],
    empty_annual: list[str],
    empty_quarterly: list[str],
    quarterly_fell_back: bool,
    annual_res: client.Result,
    quarterly_res: client.Result,
    as_of: str | None,
) -> str:
    """Name what is genuinely missing, per ticker. One bullet per real gap."""
    gaps: list[str] = []

    # 1. The horizon shortfall — the biggest gap, and the same for everyone.
    gaps.append(
        f"HORIZON: {len(annual)} fiscal year(s) available, against the "
        f"{REQUESTED_HORIZON_YEARS}-year history this report was scoped for. "
        f"Yahoo returns at most {YAHOO_ANNUAL_COLUMNS} annual columns and the "
        f"oldest is empty on all four companies, so ~"
        f"{YAHOO_POPULATED_ANNUAL_YEARS} years is the ceiling of this source, "
        "not a fetch failure. A decade-plus series has to come from the corpus, "
        "which holds filings back to 2015."
    )

    if empty_annual:
        gaps.append(
            "EMPTY ANNUAL COLUMN(S): Yahoo returned "
            f"{', '.join(empty_annual)} with every income-statement line null. "
            "Dropped from the table rather than shown as a row of blanks; not "
            "counted in years_available."
        )

    # 2. Partially populated years.
    for row in partial:
        missing = [f for f in CORE_FIELDS if getattr(row, f) is None]
        if missing:
            gaps.append(
                f"PARTIAL YEAR {row.fiscal_period}: "
                f"{', '.join(missing)} are null on Yahoo. "
                + (
                    "Hays' most recent fiscal year carries 4 non-null rows out "
                    "of 52 — share counts and EPS only. Take the rest from the "
                    "corpus."
                    if tk == "HAS"
                    else "Take these from the corpus."
                )
            )

    # 3. Quarterly coverage.
    if quarterly_fell_back:
        gaps.append(
            f"NO QUARTERLY DATA AT ALL for {sym.yahoo}. This is a reporting "
            "frequency fact, not an error: Hays is a SEMI-ANNUAL filer, so "
            "quarterly_income_stmt is empty (0x0), get_earnings_history() "
            "returns an empty frame, earnings_dates has a single row, and the "
            "0q and +1q consensus rows are all NaN. Only the annual 0y/+1y "
            "consensus exists. The quarterly series in this report is "
            "deliberately EMPTY rather than filled with annual rows relabelled "
            "as quarters. Half-year figures are in the corpus (h1-8k / h2-8k); "
            "the q1/q3/q4 documents are trading statements carrying net fee "
            "growth, not income statements."
        )
    else:
        if not quarterly:
            gaps.append(
                f"QUARTERLY SERIES EMPTY for {sym.yahoo} despite a quarterly "
                "frame being returned — every column was null."
            )
        if empty_quarterly:
            gaps.append(
                "EMPTY QUARTERLY COLUMN(S): "
                f"{', '.join(empty_quarterly)} came back entirely null and were "
                "dropped."
            )
        if quarterly:
            gaps.append(
                f"QUARTERLY DEPTH: {len(quarterly)} quarter(s) only — Yahoo "
                "caps quarterly_income_stmt at 5 columns, so there is no "
                "same-quarter-prior-year comparison for the oldest quarter and "
                "no multi-year seasonality from this source."
            )

    # 4. What this source structurally does not carry.
    gaps.append(
        "NOT ON YAHOO AT ANY FREQUENCY: adjusted / non-GAAP / pre-exceptional "
        "measures, segment tables, comparable-sales percentages, guidance, and "
        "management's stated reasons for any move. This report is a GAAP/IFRS "
        "numeric backdrop only — it cannot answer 'why', and two of the three "
        "scored metrics for most of these companies are on a basis that is not "
        "in this table."
    )

    if tk == "HD":
        gaps.append(
            "HD-SPECIFIC: adjusted diluted EPS and comparable sales percent — "
            "two of the three scored metrics — are absent. Only net sales is "
            "directly usable from here."
        )
    elif tk == "ADI":
        gaps.append(
            "ADI-SPECIFIC: adjusted diluted EPS and adjusted gross margin — two "
            "of the three scored metrics — are absent. The GAAP gross margin "
            "shown is a floor for the adjusted figure, not a substitute."
        )
    elif tk == "HAS":
        gaps.append(
            "HAS-SPECIFIC: pre-exceptional operating profit and pre-exceptional "
            "BASIC EPS — two of the three scored metrics — are absent; only the "
            "statutory post-exceptional versions are here. Net fees IS present, "
            "as the gross profit column."
        )
    elif tk == "DE":
        gaps.append(
            "DE-SPECIFIC: Production & Precision Ag segment operating profit is "
            "absent, and the revenue column is on a basis that matches neither "
            "the scored worldwide total nor the consensus. Only diluted EPS is "
            "directly usable from here."
        )

    # 5. Point-in-time honesty.
    gaps.append(
        "NO POINT-IN-TIME FILTERING. Yahoo serves current data and stamps no "
        "vintage on it, so this table always reflects retrieval time "
        f"({annual_res.as_of}) and includes restatements. It cannot be rewound "
        "for a backtest."
    )
    if as_of:
        gaps.append(
            f"AS-OF REQUESTED {as_of} BUT NOT APPLIED — see above. Any figure "
            "here may post-date that cutoff. The corpus is the point-in-time "
            "source."
        )

    return "\n".join(f"- {g}" for g in gaps)


def _build_notes(
    tk: str,
    sym: client.Symbol,
    cal: FiscalCalendar,
    *,
    annual: list[FinancialYear],
    quarterly: list[FinancialYear],
    fully_populated: list[FinancialYear],
    annual_map: dict[str, str],
    quarterly_map: dict[str, str],
    annual_res: client.Result,
    as_of: str | None,
) -> str:
    """Method, units, fiscal calendar, tie-outs. Everything needed to trust it."""
    parts: list[str] = []

    parts.append(
        "METHOD: built deterministically. Figures are read straight from "
        "marketdata/client.py's normalised income statement and pivoted into "
        "per-period rows. No language model touched any number in this table, "
        "so there is no transcription or unit-conversion risk from a model."
    )

    parts.append(
        f"COVERAGE: {len(annual)} annual period(s) in the table, of which "
        f"{len(fully_populated)} carry a complete set of revenue, gross profit, "
        f"operating income, net income and diluted EPS. "
        f"{len(quarterly)} quarterly period(s). years_available counts rows "
        "that carry at least one figure."
    )

    parts.append(
        f"UNITS: money in {sym.money_unit} (millions of {sym.money_currency}); "
        f"per-share figures in {sym.eps_unit}; margins in percentage POINTS "
        "(33.32 means 33.32 percent, never 0.3332)."
    )
    if tk == "HAS":
        parts.append(
            "HAYS CURRENCY SPLIT: Yahoo reports info['currency'] as GBp (pence, "
            "the share quote) and financialCurrency as GBP (the statements). "
            "The raw per-share values are GBP — a raw 0.01102 means 1.102 "
            "PENCE. marketdata/client.py applies the x100 once, on the way out; "
            "this module applies nothing further. Money lines are GBP millions "
            "and are NOT scaled."
        )

    parts.append(f"FISCAL CALENDAR: {cal.calendar_note}")

    if annual_map:
        mapping = "; ".join(
            f"{label} = FYE {end}"
            for label, end in sorted(annual_map.items(), reverse=True)
        )
        parts.append(f"ANNUAL PERIOD ENDS: {mapping}")
    if quarterly_map:
        mapping = "; ".join(
            f"{label} = ended {end}"
            for label, end in sorted(
                quarterly_map.items(), key=lambda kv: kv[1], reverse=True
            )
        )
        parts.append(f"QUARTERLY PERIOD ENDS: {mapping}")

    parts.extend(_basis_notes(tk))

    # Hays' own KPI, as close as this data can get to it.
    if tk == "HAS":
        ratios = [
            f"{r.fiscal_period} {r.operating_income / r.gross_profit * 100.0:,.1f}%"
            for r in annual
            if r.gross_profit not in (None, 0) and r.operating_income is not None
        ]
        if ratios:
            parts.append(
                "HAYS CONVERSION RATE (POST-EXCEPTIONAL operating profit over "
                f"net fees): {'; '.join(ratios)}. Hays' own KPI uses the "
                "PRE-EXCEPTIONAL numerator and the company targets 22-25 "
                "percent through the cycle, so these figures sit below the "
                "headline ratio wherever there were exceptional charges. Use "
                "them for shape, and take the real conversion rate from the "
                "corpus."
            )

    checks = _sanity_checks(tk, annual, quarterly)
    if checks:
        parts.append("TIE-OUTS AND UNIT CHECKS:\n" + "\n".join(f"  * {c}" for c in checks))

    # Anything client.py itself flagged about this payload.
    upstream = [n for n in annual_res.notes if n]
    if upstream:
        parts.append(
            "UPSTREAM NOTES FROM marketdata/client.py:\n"
            + "\n".join(f"  * {n}" for n in upstream)
        )

    parts.append(
        "USE: this is a backdrop to TREND and SANITY-CHECK against, not a "
        "forecast and not a citation. Every scored number still needs the "
        "corpus for its basis and its evidence."
    )
    return "\n\n".join(parts)


def failed_checks(report: FinancialsReport) -> list[str]:
    """Tie-out lines in `notes` that say CHECK FAILED. Cheap post-run assertion."""
    return [
        line.strip(" *")
        for line in (report.notes or "").splitlines()
        if "CHECK FAILED" in line
    ]


# ---------------------------------------------------------------------------
# System prompt — the OPTIONAL narrative pass only
#
# BRACE RULE. `AgentSpec.instructions` documents itself as a format string, but
# the Agents SDK does NOT in fact format instructions — verified empirically:
# Agent(instructions="a b").instructions round-trips verbatim, so a doubled
# brace written as an escape would reach the model as two visible braces. The
# only safe prompt under BOTH readings is one with no braces at all.
# `_assert_brace_free` enforces that at import, exactly as analysts/filings.py
# and analysts/news.py do.
# ---------------------------------------------------------------------------


FINANCIALS_SYSTEM_PROMPT = """\
You are the FINANCIALS ANALYST on a team that forecasts a company's next \
reported results. You are the NUMERIC BACKDROP, not the evidence base and not \
the forecaster.

A deterministic function has ALREADY built your table. It read the income \
statement directly from the market-data client, pivoted it into fiscal \
periods, converted units and computed margins. That table is given to you \
below and it is FINAL.

=== YOUR ONE JOB ===

Write the NARRATIVE that goes around the table. Specifically:

  - What the series actually does over the years shown. Direction, turning \
points, the size of the swings, whether margins moved with revenue or against \
it. Trend, not commentary.
  - What a forecaster should be careful about when trending it forward: \
comparability breaks, acquisitions, a 53rd week, currency translation, a loss \
year, a base effect.
  - Which columns are on the WRONG BASIS for the metrics this team is scored \
on, and therefore must not be lifted straight out of the table.
  - What is missing that a reader would otherwise assume is there.

=== ABSOLUTE RULES ===

DO NOT CHANGE, RESTATE, ROUND OR RE-DERIVE ANY NUMBER. The table is built. \
Only the prose fields you write are kept; every numeric field is overwritten \
with the deterministic values after you submit. If you quote a figure in your \
prose, copy it EXACTLY as it appears in the table, including its unit.

DO NOT FORECAST. No estimate for the coming period, no extrapolation, no \
target. Another agent does that.

DO NOT INVENT COVERAGE. If the table shows four fiscal years, say four. Never \
imply a longer history exists here than the one you were given.

UNITS ARE PART OF EVERY SENTENCE. Money is in millions of the reporting \
currency. Percentages are in POINTS, so 33.3 means 33.3 percent. For Hays, \
per-share figures are in PENCE and money lines are in GBP millions — two \
different currencies in one table. Never convert anything.

=== TOOLS ===

You have the market-data tools. Use them ONLY to cross-check context around \
the table you were handed — the sell-side consensus, the next scheduled report \
date, the analyst coverage. Two or three calls at most. You do not need to \
re-fetch the income statement; you already have it. There is no web access and \
no corpus access here.

=== FILLING THE REPORT ===

Fill notes with the trend narrative and the basis warnings. Fill data_gaps \
with what is genuinely missing and what a reader would wrongly assume is \
present. Be concrete and be honest — an admitted gap scores better than a \
confident blank.

Set ticker, yahoo_symbol and currency to whatever the brief gives you. Leave \
the annual and quarterly lists EMPTY; they are supplied deterministically and \
anything you put there is discarded.

Call submit_result EXACTLY ONCE.
"""


def _assert_brace_free(prompt: str) -> str:
    """Guard the format-string hazard at import time.

    The SDK does not format instructions, so a doubled brace would LEAK as two
    visible braces rather than collapsing to one. A brace-free prompt is
    correct under either behaviour; this makes a later edit unable to regress
    it silently.
    """
    if "{" in prompt or "}" in prompt:
        raise ValueError(
            "FINANCIALS_SYSTEM_PROMPT contains a brace. Instructions may be "
            "treated as a format string, and the SDK does not unescape doubled "
            "braces — keep the prompt brace-free."
        )
    return prompt


_assert_brace_free(FINANCIALS_SYSTEM_PROMPT)


def build_task_prompt(report: FinancialsReport) -> str:
    """The user message for the narrative pass: the finished table, as text."""
    tk = report.ticker
    cal = CALENDARS.get(tk)
    lines = [
        f"COMPANY: {tk}   YAHOO SYMBOL: {report.yahoo_symbol}   "
        f"REPORTING CURRENCY: {report.currency}",
        f"DATA AS OF: {report.as_of}   SOURCE: {report.source}",
        "",
    ]
    if cal:
        lines += [f"FISCAL CALENDAR: {cal.calendar_note}", ""]
    lines += [
        "THE TABLE — FINAL, DO NOT RESTATE ANY NUMBER:",
        "",
        format_table(report),
        "",
        "MACHINE-DERIVED GAPS ALREADY RECORDED (extend these, do not repeat "
        "them verbatim):",
        report.data_gaps or "(none)",
        "",
        "BASIS FACTS ESTABLISHED BY TIE-OUT AGAINST THE COMPANY'S OWN FILINGS:",
    ]
    lines += [f"  - {n}" for n in _basis_notes(tk)]
    lines += [
        "",
        "Write the trend narrative and the basis warnings, then call "
        "submit_result once. Leave the annual and quarterly lists empty.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Spec
# ---------------------------------------------------------------------------


def build_spec(
    ticker: str,
    as_of: str | None = None,
    *,
    max_turns: int | None = None,
) -> AgentSpec:
    """AgentSpec for the OPTIONAL narrative pass over a finished table.

    Market data only: `use_web=False` because there is nothing on the web this
    agent should reach for, and `allow_delegation=False` because writing two
    paragraphs about a table that already exists cannot need a subagent.

    The main path — `analyse()` with `use_llm=False`, the default — never
    builds this agent at all.
    """
    tk = normalise_ticker(ticker)
    sym = client.resolve(tk)
    assert sym is not None
    return AgentSpec(
        name=f"Financials Analyst ({tk})",
        instructions=FINANCIALS_SYSTEM_PROMPT,
        result_model=FinancialsReport,
        tools=MARKET_TOOLS,
        use_web=False,  # market data only. Not negotiable.
        allow_delegation=False,
        max_turns=max_turns or DEFAULT_MAX_TURNS,
        # Typed and identifiable: a caller can see WHICH company produced
        # nothing, and that the numbers are absent rather than zero.
        fallback=FinancialsReport(
            ticker=tk,
            yahoo_symbol=sym.yahoo,
            currency=sym.money_currency,
            years_available=0,
            source=client.SOURCE,
            as_of=as_of or "",
            data_gaps=(
                "Narrative pass produced no valid output. The deterministic "
                "table is unaffected; only the prose is missing."
            ),
            notes=(
                f"Financials narrative pass produced no valid output for {tk}. "
                f"as_of={as_of or 'none'}."
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def resume_key(ticker: str) -> str:
    return f"{normalise_ticker(ticker)}:financials"


def _record_evidence(
    store: RunStore,
    run_id: str,
    task_id: str | None,
    tk: str,
    report: FinancialsReport,
) -> int:
    """One evidence row per (period, metric) that carries a value.

    The source is the Yahoo endpoint rather than a corpus filename — this
    analyst has no documents. Saying that plainly is what stops a judge
    following the trail and finding a citation that was never a citation.
    """
    n = 0
    for rows, freq in ((report.annual, "annual"), (report.quarterly, "quarterly")):
        for row in rows:
            for field in CORE_FIELDS:
                value = getattr(row, field)
                if value is None:
                    continue
                store.add_evidence(
                    run_id,
                    tk,
                    field,
                    f"{client.SOURCE} [{report.yahoo_symbol}] {freq} income statement",
                    task_id=task_id,
                    claim=(
                        f"{row.fiscal_period} {field} = {value} "
                        f"({'as-reported GAAP/IFRS' if field != 'diluted_eps' else 'GAAP/IFRS diluted'})"
                    ),
                    value=float(value),
                    units=row.units,
                    locator=f"{row.fiscal_period} ({freq})",
                )
                n += 1
    return n


def _merge_narrative(
    deterministic: FinancialsReport, narrative: FinancialsReport
) -> FinancialsReport:
    """Append the model's prose. Numbers are never taken from it.

    Every numeric and identifying field stays as the deterministic pass built
    it. Only `notes` and `data_gaps` grow, and they grow under a heading that
    says which half a judge is reading.
    """
    if narrative.notes.strip():
        deterministic.notes = (
            f"{deterministic.notes}\n\n"
            "=== NARRATIVE PASS (LLM-written prose; every number above is "
            "deterministic) ===\n"
            f"{narrative.notes.strip()}"
        )
    if narrative.data_gaps.strip():
        deterministic.data_gaps = (
            f"{deterministic.data_gaps}\n"
            "- NARRATIVE PASS ADDITIONS (LLM-written, not machine-verified):\n"
            f"{narrative.data_gaps.strip()}"
        )
    return deterministic


async def analyse(
    ticker: str,
    as_of: str | None = None,
    run_id: str | None = None,
    store: RunStore | None = None,
    *,
    use_llm: bool = False,
    max_turns: int | None = None,
) -> FinancialsReport:
    """Build one company's financial history table and return a `FinancialsReport`.

    Deterministic by default: the table is fetched and pivoted with no model in
    the loop. `use_llm=True` adds a short narrative pass that may only append
    prose to `notes` and `data_gaps`.

    Same signature and same store contract as `analysts.filings.analyse`, so
    the pipeline drives all four analysts identically. With a store and a
    run_id it is resumable on the key "<TICKER>:financials".

    Args:
        ticker: HD, ADI, HAS (or LSE:HAS / HAS.L), DE.
        as_of: Recorded for traceability. It CANNOT filter this source —
            Yahoo has no vintage — and the mismatch is reported in data_gaps.
        run_id: Existing run to attach to.
        store: Optional `runstore.RunStore` for resume, events and evidence.
        use_llm: Run the optional narrative pass. Costs a model call.
        max_turns: Turn budget for the narrative pass only.
    """
    tk = normalise_ticker(ticker)
    key = resume_key(tk)

    task_id: str | None = None
    if store is not None and run_id:
        cached = store.get_output(run_id, key)
        if cached:
            try:
                logger.info("Resuming %s from run store — task already completed", key)
                return FinancialsReport.model_validate_json(cached)
            except Exception as e:  # stored row is corrupt; re-run rather than die
                logger.warning("Stored output for %s is unusable (%s); re-running", key, e)
        task_id = store.start_task(
            run_id,
            key,
            kind="analyst",
            agent_name=f"Financials Analyst ({tk})",
            model=(settings.resolved_agent_model if use_llm else "none (deterministic)"),
            input={
                "ticker": tk,
                "as_of": as_of,
                "mode": "deterministic+llm-notes" if use_llm else "deterministic",
                "source": client.SOURCE,
            },
        )
        store.log(
            run_id,
            "analyst.start",
            task_id=task_id,
            name=f"Financials Analyst ({tk})",
            payload={"ticker": tk, "as_of": as_of, "use_llm": use_llm},
        )

    started = time.monotonic()
    try:
        # yfinance is blocking HTTP; keep it off the event loop so a
        # four-company run_agents fan-out is not serialised behind it.
        report = await asyncio.to_thread(build_report, tk, as_of)
    except Exception as e:
        logger.exception("Financials analyst crashed for %s", tk)
        if store is not None and task_id:
            store.fail_task(task_id, str(e))
        raise

    if use_llm and report.annual:
        try:
            spec = build_spec(tk, as_of, max_turns=max_turns)
            narrative = await run_agent(spec, build_task_prompt(report))
            if isinstance(narrative, FinancialsReport):
                report = _merge_narrative(report, narrative)
        except Exception as e:  # prose is a nice-to-have; the table is the job
            logger.warning("Narrative pass failed for %s (%s); keeping the table", tk, e)
            report.notes += (
                f"\n\nNARRATIVE PASS FAILED ({type(e).__name__}: {e}). The "
                "deterministic table is unaffected."
            )

    elapsed_ms = int((time.monotonic() - started) * 1000)

    failures = failed_checks(report)
    if failures:
        logger.warning(
            "%s: %d tie-out/unit check(s) FAILED: %s",
            tk,
            len(failures),
            " | ".join(failures[:3]),
        )

    if store is not None and run_id and task_id:
        rows = _record_evidence(store, run_id, task_id, tk, report)
        store.log(
            run_id,
            "analyst.done",
            task_id=task_id,
            name=f"Financials Analyst ({tk})",
            payload={
                "annual_periods": len(report.annual),
                "quarterly_periods": len(report.quarterly),
                "years_available": report.years_available,
                "evidence_rows": rows,
                "failed_checks": failures,
                "use_llm": use_llm,
            },
            duration_ms=elapsed_ms,
        )
        # An empty table is a failure, not a result. Storing it 'completed'
        # would make the resume path serve the failure back forever.
        store.finish_task(
            task_id,
            report.model_dump(),
            status="completed" if report.annual else "failed",
            usage=get_last_usage(),
        )

    logger.info(
        "Financials analyst %s done in %.1fs — %d annual, %d quarterly period(s)",
        tk,
        elapsed_ms / 1000,
        len(report.annual),
        len(report.quarterly),
    )
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _money(v: float | None) -> str:
    """Thousands separators, one decimal, NEVER scientific notation.

    `:,.4g` would render 41765 as 4.176e+04, which is a real bug this repo has
    already hit once. Do not reintroduce it.
    """
    return "n/a" if v is None else f"{v:,.1f}"


def _pct(v: float | None) -> str:
    return "n/a" if v is None else f"{v:,.2f}"


def _eps(v: float | None) -> str:
    """Up to 4dp, trailing zeros trimmed. 14.23, -0.49, 1.102 — never 1.102e+00."""
    if v is None:
        return "n/a"
    text = f"{v:,.4f}"
    return text.rstrip("0").rstrip(".") if "." in text else text


_COLUMNS = (
    ("FISCAL PERIOD", 14),
    ("REVENUE", 13),
    ("GROSS PROFIT", 13),
    ("OP INCOME", 13),
    ("NET INCOME", 13),
    ("DIL EPS", 10),
    ("GM %", 8),
    ("OP M %", 8),
)


def format_table(report: FinancialsReport) -> str:
    """The readable table: one row per fiscal period, columns per metric."""
    units = report.annual[0].units if report.annual else (
        report.quarterly[0].units if report.quarterly else ""
    )
    header = "  ".join(f"{name:<{w}}" if i == 0 else f"{name:>{w}}"
                       for i, (name, w) in enumerate(_COLUMNS))
    rule = "-" * len(header)

    out: list[str] = []
    if units:
        out.append(f"units: {units}   margins in percentage points")
        out.append("")

    for title, rows in (("ANNUAL", report.annual), ("QUARTERLY", report.quarterly)):
        out.append(f"{title} ({len(rows)} period(s))")
        out.append(header)
        out.append(rule)
        if not rows:
            out.append("  (none — see data_gaps)")
        for r in rows:
            cells = [
                f"{r.fiscal_period:<14}",
                f"{_money(r.revenue):>13}",
                f"{_money(r.gross_profit):>13}",
                f"{_money(r.operating_income):>13}",
                f"{_money(r.net_income):>13}",
                f"{_eps(r.diluted_eps):>10}",
                f"{_pct(r.gross_margin_pct):>8}",
                f"{_pct(r.operating_margin_pct):>8}",
            ]
            out.append("  ".join(cells))
        out.append("")
    return "\n".join(out).rstrip()


def print_report(report: FinancialsReport, *, elapsed: float | None = None) -> None:
    bar = "=" * 106

    print()
    print(bar)
    title = (
        f"FINANCIALS REPORT  {report.ticker}  "
        f"(Yahoo {report.yahoo_symbol})  currency {report.currency}"
    )
    print(title)
    print(f"source: {report.source}")
    print(f"as of:  {report.as_of}  (retrieval time — Yahoo publishes no vintage)", end="")
    if elapsed is not None:
        print(f"   wall clock: {elapsed:.2f}s", end="")
    print()
    print(bar)
    print()
    print(format_table(report))

    print()
    print(f"years_available: {report.years_available}")
    complete = sum(
        1 for r in report.annual if all(getattr(r, f) is not None for f in CORE_FIELDS)
    )
    print(f"  of which fully populated (all five core metrics): {complete}")
    print(f"  quarterly periods: {len(report.quarterly)}")

    print()
    print("-- DATA GAPS " + "-" * 92)
    print(report.data_gaps or "  (none reported)")

    print()
    print("-- NOTES " + "-" * 96)
    print(report.notes or "  (none)")

    failures = failed_checks(report)
    if failures:
        print()
        print("!! TIE-OUT / UNIT CHECKS FAILED:")
        for f in failures:
            print(f"   {f}")
    print()


def _default_json_path(tk: str) -> Path:
    return REPO_ROOT / "runs" / "financials" / f"{tk}.json"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m analysts.financials",
        description=(
            "Financials analyst: normalised income-statement history for one "
            "company, from Yahoo Finance. Deterministic by default."
        ),
    )
    ap.add_argument("--ticker", required=True, help="HD, ADI, HAS (or LSE:HAS), DE")
    ap.add_argument(
        "--as-of",
        dest="as_of",
        default=None,
        metavar="YYYY-MM-DD",
        help=(
            "Recorded for traceability only. Yahoo cannot be filtered "
            "point-in-time; the mismatch is reported in data_gaps."
        ),
    )
    ap.add_argument(
        "--json",
        dest="json_path",
        default=None,
        metavar="PATH",
        help="Where to write the report JSON. Default runs/financials/<TICKER>.json",
    )
    ap.add_argument(
        "--llm-notes",
        action="store_true",
        help=(
            "Add an LLM narrative pass over the finished table. It may only "
            "append prose to notes/data_gaps; numbers are never taken from it."
        ),
    )
    ap.add_argument(
        "--max-turns",
        dest="max_turns",
        type=int,
        default=None,
        help=f"Turn budget for the narrative pass (default {DEFAULT_MAX_TURNS}).",
    )
    ap.add_argument("--run-id", dest="run_id", default=None, help="Reuse a run id (enables resume).")
    ap.add_argument("--db", default=None, help="Run store SQLite path. Default runs/runs.sqlite.")
    ap.add_argument("--no-store", action="store_true", help="Skip the run store entirely.")
    ap.add_argument("-v", "--verbose", action="store_true", help="INFO logging.")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except Exception:
            pass

    try:
        tk = normalise_ticker(args.ticker)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if args.as_of:
        try:
            datetime.strptime(args.as_of, "%Y-%m-%d")
        except ValueError:
            print(f"error: --as-of must be YYYY-MM-DD, got {args.as_of!r}", file=sys.stderr)
            return 2

    store: RunStore | None = None
    run_id: str | None = args.run_id
    if not args.no_store:
        store = get_store(args.db)
        if not run_id:
            stamp = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}"
            run_id = f"financials_{tk}_{stamp}"
        store.start_run(
            f"financials-analyst {tk}",
            config={
                "ticker": tk,
                "as_of": args.as_of,
                "mode": "deterministic+llm-notes" if args.llm_notes else "deterministic",
                "source": client.SOURCE,
            },
            run_id=run_id,
        )

    sym = client.resolve(tk)
    mode = "deterministic + LLM narrative" if args.llm_notes else "deterministic"
    print(
        f"Financials analyst: {sym.name if sym else tk} [{tk} / Yahoo "
        f"{sym.yahoo if sym else '?'}]  mode={mode}  as_of={args.as_of or 'none'}"
    )
    if args.llm_notes:
        print(f"  narrative model={settings.resolved_agent_model}")
    if run_id:
        print(f"run_id={run_id}")

    use_selector_event_loop()
    started = time.monotonic()
    try:
        report = asyncio.run(
            analyse(
                tk,
                as_of=args.as_of,
                run_id=run_id,
                store=store,
                use_llm=args.llm_notes,
                max_turns=args.max_turns,
            )
        )
    except Exception as e:
        print(f"error: financials analyst failed: {type(e).__name__}: {e}", file=sys.stderr)
        if store is not None and run_id:
            store.end_run(run_id, status="failed", notes=str(e)[:2000])
        return 1
    elapsed = time.monotonic() - started

    print_report(report, elapsed=elapsed)

    out_path = Path(args.json_path) if args.json_path else _default_json_path(tk)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ticker": tk,
        "as_of_requested": args.as_of,
        "mode": "deterministic+llm-notes" if args.llm_notes else "deterministic",
        "run_id": run_id,
        "elapsed_seconds": round(elapsed, 2),
        "failed_checks": failed_checks(report),
        "report": report.model_dump(),
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Full report written to {out_path}")

    if store is not None and run_id:
        store.end_run(run_id, status="completed")
        summary = store.run_summary(run_id)
        print(
            f"Run store: tasks={summary['tasks']} evidence_rows="
            f"{summary['evidence_rows']} db={store.db_path}"
        )

    # Non-zero if the table is empty or a tie-out failed, so a shell can tell.
    return 0 if (report.annual and not failed_checks(report)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
