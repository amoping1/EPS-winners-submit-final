"""models.py — the contracts between the three analysts and the central agent.

Two rules govern every model here:

  1. EVERY FIELD IS DEFAULTED. `run_agent`'s fallback path constructs an empty
     instance when a model fails or exhausts its turns. A required field would
     turn a recoverable failure into a crash at 17:50.

  2. EVERY NUMBER CARRIES ITS SOURCE. `Citation.source` is the corpus FILENAME
     or the URL -- 1,027 of 1,139 corpus documents have `source_url: null`, so
     the filename is the citation. The rubric scores whether judges can follow
     evidence to each of the 12 final numbers.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Citation(BaseModel):
    """Where a claim came from. Never omit."""

    source: str = ""  # corpus filename, or URL for web sources
    locator: str = ""  # line number, section title, page, or publish date
    quote: str = ""  # short verbatim snippet supporting the claim


# ---------------------------------------------------------------------------
# 1. Filings analyst — offline corpus
# ---------------------------------------------------------------------------


class MetricObservation(BaseModel):
    """One historically REPORTED figure. Not a forecast."""

    metric: str = ""
    period: str = ""  # e.g. "Q2 FY2025"
    value: float | None = None
    units: str = ""
    basis: str = ""  # reported | adjusted | pre-exceptional | GAAP | non-GAAP
    citation: Citation = Field(default_factory=Citation)


class GuidanceItem(BaseModel):
    """Company-issued guidance — a first-party forecast, often the best anchor."""

    metric: str = ""
    period: str = ""
    text: str = ""
    low: float | None = None
    high: float | None = None
    units: str = ""
    issued_on: str = ""
    citation: Citation = Field(default_factory=Citation)


class Driver(BaseModel):
    """A management-stated reason a metric moved. The 'why', from MD&A."""

    name: str = ""
    affects_metric: str = ""
    direction: str = ""  # tailwind | headwind | neutral
    magnitude: str = ""  # quantified where management quantified it
    management_explanation: str = ""
    persistence: str = ""  # one-off | continuing | fading
    citations: list[Citation] = Field(default_factory=list)


class FilingsReport(BaseModel):
    ticker: str = ""
    period_target: str = ""
    metrics_history: list[MetricObservation] = Field(default_factory=list)
    guidance: list[GuidanceItem] = Field(default_factory=list)
    drivers: list[Driver] = Field(default_factory=list)
    seasonality: str = ""
    # How the business BEHAVED when its cycle last turned. `seasonality` covers
    # the within-year pattern; this covers the across-years one, and the central
    # agent's cyclicality step has nowhere else to draw from.
    cyclical_behaviour: str = ""
    business_factors: str = ""  # what structurally moves this business, over years
    accounting_notes: str = ""  # basis definitions, restatements, 53rd week, FX
    risks: str = ""
    documents_read: list[str] = Field(default_factory=list)
    confidence: str = ""
    notes: str = ""


# ---------------------------------------------------------------------------
# 2. News analyst — web, value-investing lens
# ---------------------------------------------------------------------------


class NewsItem(BaseModel):
    headline: str = ""
    url: str = ""
    published: str = ""
    publisher: str = ""
    # high  = primary filings/IR, Reuters/Bloomberg/FT/WSJ, real trade press
    # medium= reputable secondary reporting, factual sell-side summaries
    # low   = everything thin; include only if nothing better exists
    source_quality: str = ""
    quality_reason: str = ""  # WHY this rating — forces judgement, not vibes
    fundamental_relevance: str = ""  # which metric/driver it bears on
    implication: str = ""  # what it implies for the reported numbers
    direction: str = ""  # positive | negative | neutral


class ExcludedSource(BaseModel):
    """What was rejected and why. Self-knowledge is explicitly scored."""

    url_or_publisher: str = ""
    reason: str = ""


class NewsReport(BaseModel):
    ticker: str = ""
    as_of: str = ""  # the run date, passed in — never hardcoded
    corpus_freeze: str = ""  # what this agent is filling the gap after
    items: list[NewsItem] = Field(default_factory=list)
    valuation_view: str = ""  # fundamentals/valuation, NOT price action
    demand_signals: str = ""
    cost_and_margin_signals: str = ""
    guidance_changes: str = ""
    peer_and_industry_readacross: str = ""
    excluded_sources: list[ExcludedSource] = Field(default_factory=list)
    confidence: str = ""
    notes: str = ""


# ---------------------------------------------------------------------------
# 3. Financials analyst — Yahoo Finance long-horizon table
# ---------------------------------------------------------------------------


class FinancialYear(BaseModel):
    fiscal_period: str = ""  # "FY2024" or "Q2 FY2025"
    revenue: float | None = None
    gross_profit: float | None = None
    operating_income: float | None = None
    net_income: float | None = None
    diluted_eps: float | None = None
    gross_margin_pct: float | None = None
    operating_margin_pct: float | None = None
    units: str = ""  # e.g. "USDm" / "GBPm"


class FinancialsReport(BaseModel):
    ticker: str = ""
    yahoo_symbol: str = ""  # HAS.L for Hays
    currency: str = ""
    annual: list[FinancialYear] = Field(default_factory=list)
    quarterly: list[FinancialYear] = Field(default_factory=list)
    years_available: int = 0
    data_gaps: str = ""  # be honest — coverage is thin outside the US
    source: str = ""
    as_of: str = ""
    notes: str = ""


# ---------------------------------------------------------------------------
# 4. Central agent — the 12 numbers
# ---------------------------------------------------------------------------


class MetricForecast(BaseModel):
    """One of the 12. `label` and `units` must match challenge/companies.json."""

    label: str = ""
    units: str = ""
    value: float | None = None
    low: float | None = None
    high: float | None = None
    method: str = ""  # how it was derived — not "the model said so"
    assumptions: list[str] = Field(default_factory=list)
    drivers_applied: list[str] = Field(default_factory=list)
    evidence: list[Citation] = Field(default_factory=list)
    conflicts_resolved: str = ""  # where the three reports disagreed, and why this won
    confidence: str = ""


class CompanyForecast(BaseModel):
    ticker: str = ""
    period: str = ""
    metrics: list[MetricForecast] = Field(default_factory=list)
    reconciliation: str = ""  # how filings / news / financials were combined
    sanity_checks: str = ""  # units, magnitudes, YoY plausibility
    known_weaknesses: str = ""  # honesty is 6 rubric points
    confidence: str = ""
