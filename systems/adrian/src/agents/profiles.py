"""IndustryProfile registry.

Industry is data, not a code path. Adding an industry means adding a profile here; no
agent changes. The classifier picks a profile, and every downstream agent reads it.

Deliberately five profiles, not ten: four that match the companies in play plus a
fallback. Profiles we cannot exercise are untested claims, and the write-up scores
honesty.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class IndustryProfile:
    key: str
    label: str
    # Phrases that identify this industry in a company's filings.
    signals: list[str]
    # Seed queries the research agent expands on. Not exhaustive - it writes its own too.
    query_templates: list[str]
    # KPIs that matter for this industry. Used to shape extraction and sanity checks.
    kpis: list[str]
    # What actually drives the forecast for this kind of business.
    forecast_drivers: list[str]
    # Known ways a forecast goes wrong here.
    risks: list[str] = field(default_factory=list)
    # Events that move the number away from trend - what to hunt for in the calls channel.
    catalysts: list[str] = field(default_factory=list)
    # Variables the forecaster should reason over explicitly, beyond the headline series.
    predictors: list[str] = field(default_factory=list)


RETAIL = IndustryProfile(
    key="retail",
    label="Consumer / Retail",
    signals=["comparable sales", "store count", "retailer", "merchandising", "same-store"],
    query_templates=[
        "comparable sales total company",
        "fiscal year guidance comparable sales",
        "net sales quarter results",
        "average ticket transactions",
        "acquisition contribution to net sales",
    ],
    kpis=["comparable sales", "store count", "average ticket", "transactions", "e-commerce mix"],
    forecast_drivers=[
        "comparable sales growth",
        "store base and square footage",
        "acquisition contribution (inorganic)",
        "seasonality - which quarter carries the year",
        "FX impact on comps",
    ],
    catalysts=['housing turnover and mortgage rates', 'weather and seasonal timing', 'promotional cadence', 'acquisition close dates', 'tariff pass-through to pricing'],
    predictors=['ticket vs transaction split', 'US vs total comp spread', 'FX contribution to comps', 'inorganic (acquired) revenue contribution', 'quarter share of full-year sales', 'adjusted operating margin guide vs realised'],
    risks=[
        "Acquisitions inflate net sales while comps exclude them - the two diverge.",
        "53-week fiscal years distort YoY growth.",
        "Comps are reported in percentage points: 4.5 means 4.5%, not 0.045.",
    ],
)

SEMICONDUCTORS = IndustryProfile(
    key="semiconductors",
    label="Semiconductors",
    signals=["semiconductor", "wafer", "fab", "analog", "bookings", "design win"],
    query_templates=[
        "outlook for the next quarter revenue EPS",
        "adjusted gross margin quarter",
        "bookings backlog inventory",
        "revenue by end market industrial automotive communications",
        "utilization capacity",
    ],
    kpis=["revenue by end market", "adjusted gross margin", "bookings", "inventory days", "utilization"],
    forecast_drivers=[
        "explicit next-quarter company guidance (usually a point estimate +/- a band)",
        "end-market mix - industrial vs automotive vs communications",
        "utilization and its pull-through to gross margin",
        "inventory correction cycle stage",
    ],
    catalysts=['inventory correction stage', 'data-centre and AI capex cycle', 'export restrictions', 'end-market inflection in automotive or industrial'],
    predictors=['book-to-bill and bookings growth', 'end-market revenue mix', 'utilisation rate', 'guided midpoint vs prior-quarter beat/miss pattern', 'opex as percent of revenue', 'gross-to-operating margin gap'],
    risks=[
        "Companies guide operating margin but not always gross margin - derive it.",
        "Adjusted vs GAAP diverge sharply on acquisition amortisation.",
        "Export restrictions can reset demand mid-quarter.",
    ],
)

INDUSTRIAL = IndustryProfile(
    key="industrial",
    label="Industrial / Machinery",
    signals=["equipment", "machinery", "dealer", "order book", "backlog", "agriculture"],
    query_templates=[
        "segment operating profit",
        "worldwide net sales and revenues",
        "company outlook fiscal year net income",
        "industry sales outlook units",
        "price realization currency translation",
    ],
    kpis=["segment operating profit", "order backlog", "price realization", "unit volumes", "dealer inventory"],
    forecast_drivers=[
        "full-year company net income outlook, phased to the quarter",
        "segment-level sales and operating margin trend",
        "industry unit outlook",
        "price realization vs currency translation",
        "seasonality by fiscal quarter",
    ],
    catalysts=['farm income and crop prices', 'dealer inventory destocking', 'interest rates on equipment finance', 'industry unit outlook revisions'],
    predictors=['segment share of group sales', 'price realisation vs volume', 'currency translation impact', 'quarter share of full-year net income', 'segment operating margin trend', 'order backlog cover'],
    risks=[
        "Segment operating profit lives in the segment note, not the income statement.",
        "Fiscal quarters are offset from calendar quarters - verify period end dates.",
        "Financial Services is a separate segment; do not fold it into equipment results.",
    ],
)

STAFFING = IndustryProfile(
    key="staffing",
    label="Staffing / Professional services",
    signals=["net fees", "consultant", "perm", "temp and contracting", "recruitment", "placement"],
    query_templates=[
        "net fees like-for-like growth",
        "pre-exceptional operating profit consensus",
        "company compiled consensus analysts",
        "trading update quarter net fees",
        "divestment disposal country exit net fees",
    ],
    kpis=["net fees", "LFL net fee growth", "consultant headcount", "perm vs temp mix", "productivity"],
    forecast_drivers=[
        "LFL net fee growth by division and geography",
        "actual vs LFL basis - FX and disposals sit in the gap",
        "consultant headcount and productivity",
        "cost savings programme run-rate",
        "published company-compiled consensus, where available",
    ],
    catalysts=['white-collar hiring confidence', 'German average hours worked', 'country exits and disposals', 'cost-savings programme run-rate'],
    predictors=['LFL vs actual net fee growth gap', 'perm vs temp mix', 'consultant headcount and productivity', 'net fees to operating profit conversion', 'disposal contribution to prior-year base', 'exceptional items excluded from the measure'],
    risks=[
        "Net fees is gross profit, NOT revenue. Revenue includes contractor pay-through.",
        "UK companies report EPS in PENCE. 45p is 45, not 0.45.",
        "Pre-exceptional excludes restructuring and impairment - check what was excluded.",
        "Country disposals break naive YoY: reported and LFL diverge.",
    ],
)

FALLBACK = IndustryProfile(
    key="fallback",
    label="General",
    signals=[],
    query_templates=[
        "quarterly results revenue earnings per share",
        "guidance outlook",
        "management discussion and analysis",
        "segment results",
    ],
    kpis=["revenue", "operating income", "net income", "EPS"],
    forecast_drivers=["revenue trend", "margin trend", "share count", "guidance"],
    risks=["No industry-specific coverage - treat every extracted figure as unverified."],
)

PROFILES = {p.key: p for p in (RETAIL, SEMICONDUCTORS, INDUSTRIAL, STAFFING, FALLBACK)}


def classify(text: str) -> tuple[IndustryProfile, float, list[str]]:
    """Score a company's filing text against profile signals.

    Returns (profile, confidence, matched signals). Confidence is the matched-signal
    share; the agent can override with a reasoned choice, but this gives it a prior and
    something to disagree with.
    """
    lowered = text.lower()
    best, best_hits = FALLBACK, []
    for profile in PROFILES.values():
        if not profile.signals:
            continue
        hits = [s for s in profile.signals if s in lowered]
        if len(hits) > len(best_hits):
            best, best_hits = profile, hits
    confidence = len(best_hits) / len(best.signals) if best.signals else 0.0
    return best, round(confidence, 2), best_hits
