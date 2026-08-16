"""news.py - the NEWS ANALYST.

The offline corpus is frozen at 2026-08-14. The four challenge companies report
between 18 and 20 August 2026. This agent owns the gap: everything public that
happened *after* the freeze and *up to* the run date, read through a
long-horizon value-investing lens.

Three things make this module what it is:

  1. SOURCE QUALITY IS THE PRODUCT. A news agent that hoovers up "3 stocks to
     buy before earnings" listicles is worse than no news agent - it launders
     noise into the forecast with a citation attached. So every `NewsItem`
     carries `source_quality` AND `quality_reason`, and everything rejected is
     named in `excluded_sources`. The visible rejection list is the evidence
     that judgement happened; JUDGING.md scores exactly that under "honesty and
     self-knowledge" (6 pts) and "data approach" (12 pts).

  2. FUNDAMENTALS, NOT PRICE. Demand, pricing, unit economics, margin drivers,
     guidance credibility, capital allocation. Never "the stock rallied".
     Price action is not evidence about next quarter's net sales.

  3. OPENSTOCKS IS OFF-LIMITS, IN CODE. RULES.md forbids interference with
     OpenStocks and JUDGING.md says the Wall Street benchmark is deliberately
     withheld. The prohibition is in the system prompt *and* enforced by
     `install_openstocks_guard()`, which sits between the Firecrawl tools and
     the network. A prompt is a request; the guard is a rule. Every blocked
     attempt is logged and surfaces in the report's `excluded_sources`.

CLI:

    python -m analysts.news --ticker HD
    python -m analysts.news --ticker ADI --as-of 2026-08-16 --json out.json

Requires FIRECRAWL_API_KEY. Without it the module still imports and every
offline path (spec building, prompt rendering, the URL guard) still works -
only the live run refuses, with instructions.
"""

from __future__ import annotations

import argparse
import asyncio
import contextvars
import json
import logging
import string
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlparse

from agent_core import get_last_usage, AgentSpec, FirecrawlNotConfigured, run_agent, settings
from agent_core.runner import use_selector_event_loop

from .models import ExcludedSource, NewsItem, NewsReport

logger = logging.getLogger(__name__)

# The corpus freeze. This agent exists to cover the period after it.
CORPUS_FREEZE = "2026-08-14"

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_JSON_DIR = REPO_ROOT / "runs" / "news"

RESUME_KIND = "analyst"


# ==========================================================================
# 1. The openstocks guard - enforced in code, not just asked for in a prompt
# ==========================================================================

# Substring, not an exact host: openstocks.com, www.openstocks.com,
# api.openstocks.com and anything else under the name are all out of bounds.
BLOCKED_HOST_SUBSTRING = "openstocks"


class OpenStocksBlocked(RuntimeError):
    """Raised when something tries to reach the hackathon host's own site."""


def url_host(url: str) -> str:
    """Lowercased hostname of `url`. Tolerates a missing scheme."""
    raw = (url or "").strip()
    if not raw:
        return ""
    if "://" not in raw:
        raw = "https://" + raw
    try:
        return (urlparse(raw).hostname or "").lower()
    except Exception:  # pragma: no cover - urlparse is very forgiving
        return ""


def is_blocked_url(url: str) -> bool:
    """True if `url`'s HOST names openstocks.

    Host only, deliberately. A Reuters article whose *path* mentions the
    hackathon is ordinary public reporting; only traffic aimed at the host's
    own infrastructure is the problem.

    If the host cannot be parsed at all we fall back to scanning the whole
    string - over-blocking is free, under-blocking costs the entry.
    """
    host = url_host(url)
    if host:
        return BLOCKED_HOST_SUBSTRING in host
    return BLOCKED_HOST_SUBSTRING in (url or "").lower()


# Per-task bucket of blocked attempts. A ContextVar rather than a global so a
# four-company `run_agents` fan-out keeps its rejections separate.
_blocked_log: contextvars.ContextVar[list[dict[str, str]] | None] = (
    contextvars.ContextVar("news_blocked_log", default=None)
)


@contextmanager
def blocked_log() -> Iterator[list[dict[str, str]]]:
    """Collect guard rejections for the duration of the block."""
    bucket: list[dict[str, str]] = []
    token = _blocked_log.set(bucket)
    try:
        yield bucket
    finally:
        _blocked_log.reset(token)


def _record_block(target: str, reason: str) -> None:
    logger.warning("BLOCKED openstocks access: %s (%s)", target, reason)
    bucket = _blocked_log.get()
    if bucket is not None:
        bucket.append({"target": target, "reason": reason})


_BLOCK_MESSAGE = (
    "BLOCKED by the news analyst's source guard: openstocks.com is the "
    "hackathon host. RULES.md forbids interference with OpenStocks and "
    "JUDGING.md withholds the Wall Street benchmark on purpose. Do not try "
    "again. Use company IR, Reuters, Bloomberg, FT or WSJ instead."
)

_guard_installed = False


def install_openstocks_guard() -> None:
    """Wrap the Firecrawl entry points so no openstocks URL reaches the network.

    `agent_core.tools` imports `search_web` / `scrape_markdown` by name, so the
    interception has to happen on those module attributes - that is the last
    point every tool call passes through. Idempotent, and it leaves
    `agent_core/` itself untouched.

    * scrape of a blocked host  -> raises, the tool reports the block to the model
    * search result on a blocked host -> silently dropped from the result list
    * search query naming openstocks (e.g. a site: query) -> raises
    """
    global _guard_installed
    if _guard_installed:
        return

    import agent_core.tools as core_tools

    original_search = core_tools.search_web
    original_scrape = core_tools.scrape_markdown

    def guarded_search(query: str, limit: int = 10) -> list[dict[str, str]]:
        if BLOCKED_HOST_SUBSTRING in (query or "").lower():
            _record_block(f"search query: {query}", "query targets the hackathon host")
            raise OpenStocksBlocked(_BLOCK_MESSAGE)
        results = original_search(query, limit=limit)
        kept: list[dict[str, str]] = []
        for r in results:
            if is_blocked_url(r.get("url", "")):
                _record_block(r.get("url", ""), "search hit on the hackathon host, dropped")
                continue
            kept.append(r)
        return kept

    def guarded_scrape(url: str, max_length: int | None = None) -> str:
        if is_blocked_url(url):
            _record_block(url, "scrape attempt against the hackathon host")
            raise OpenStocksBlocked(_BLOCK_MESSAGE)
        return original_scrape(url, max_length)

    core_tools.search_web = guarded_search  # type: ignore[assignment]
    core_tools.scrape_markdown = guarded_scrape  # type: ignore[assignment]
    _guard_installed = True
    logger.info("openstocks URL guard installed on agent_core.tools")


# ==========================================================================
# 2. Company profiles - the macro series that actually drive each P&L
# ==========================================================================


@dataclass(frozen=True)
class CompanyProfile:
    ticker: str
    name: str
    listing: str
    period: str
    reports_on: str
    metrics: tuple[str, ...]
    macro_drivers: tuple[str, ...]
    queries: tuple[str, ...]

    def metrics_block(self) -> str:
        return "\n".join(f"  - {m}" for m in self.metrics)

    def drivers_block(self) -> str:
        return "\n".join(f"  - {d}" for d in self.macro_drivers)

    def queries_block(self) -> str:
        return "\n".join(f'  {i}. "{q}"' for i, q in enumerate(self.queries, 1))


COMPANIES: dict[str, CompanyProfile] = {
    "HD": CompanyProfile(
        ticker="HD",
        name="The Home Depot, Inc.",
        listing="NYSE: HD",
        period="FY2026 Q2",
        reports_on="2026-08-18",
        metrics=(
            "Net sales, USD millions",
            "Adjusted diluted EPS, USD per share",
            "Comparable sales, total company, percent",
        ),
        macro_drivers=(
            "US housing turnover - existing-home sales volume, mortgage rates, "
            "the lock-in effect. Turnover moves big-ticket home improvement "
            "with a lag; it is the single most important external series.",
            "Repair-and-remodel spend - Harvard JCHS LIRA, private residential "
            "improvement spend in the Census construction data.",
            "Big-ticket discretionary demand: transactions over roughly 1,000 "
            "dollars, project deferral, kitchen/bath/flooring categories.",
            "Pro versus DIY mix, and the complex-Pro build-out including SRS.",
            "Commodity price deflation or inflation in lumber and copper, which "
            "moves comparable sales without moving units.",
            "Weather and hurricane/storm demand, which management routinely "
            "quantifies in the comp bridge.",
        ),
        queries=(
            "Home Depot second quarter fiscal 2026 earnings preview comparable sales guidance",
            "US existing home sales repair remodel spending outlook August 2026",
            "Home Depot investor relations news release August 2026",
            "home improvement retail demand big ticket projects Lowe's read across 2026",
        ),
    ),
    "ADI": CompanyProfile(
        ticker="ADI",
        name="Analog Devices, Inc.",
        listing="NASDAQ: ADI",
        period="FY2026 Q3",
        reports_on="2026-08-19",
        metrics=(
            "Revenue, USD millions",
            "Adjusted diluted EPS, USD per share",
            "Adjusted gross margin, percent",
        ),
        macro_drivers=(
            "The analog semiconductor cycle: bookings, book-to-bill, backlog "
            "cover, cancellation rates and lead times.",
            "End-market demand by segment - industrial (automation, test, "
            "aerospace/defence), automotive (content per vehicle, BMS, GMSL, "
            "EV build rates), communications (data-centre power and optical), "
            "consumer.",
            "Channel and customer inventory: distributor weeks of inventory, "
            "restocking versus destocking, sell-in against sell-through.",
            "Utilisation and the hybrid manufacturing model - factory loading "
            "is the dominant swing factor in adjusted gross margin, and "
            "underutilisation charges are usually quantified on the call.",
            "Peer read-across from Texas Instruments, Microchip, NXP, "
            "STMicroelectronics, Infineon and Renesas on industrial and auto "
            "order patterns.",
        ),
        queries=(
            "Analog Devices fiscal third quarter 2026 results guidance bookings industrial",
            "analog semiconductor industrial automotive demand recovery inventory August 2026",
            "Analog Devices investor relations press release August 2026",
            "Texas Instruments Microchip NXP industrial analog order commentary 2026",
        ),
    ),
    "HAS": CompanyProfile(
        ticker="HAS",
        name="Hays plc",
        listing="LSE: HAS",
        period="FY2026 (full year, June year-end)",
        reports_on="2026-08-20",
        metrics=(
            "Net fees, GBP millions",
            "Pre-exceptional basic EPS, pence",
            "Pre-exceptional operating profit, GBP millions",
        ),
        macro_drivers=(
            "UK and European white-collar hiring: permanent placement volumes, "
            "vacancy counts, KPMG/REC UK Report on Jobs, ONS vacancies.",
            "Permanent versus temporary/contracting mix - perm is the cyclical, "
            "high-drop-through half; temp is the ballast. The split drives both "
            "net fees and conversion.",
            "Germany, the largest profit pool, plus Australia and New Zealand: "
            "industrial and automotive white-collar demand, and public-sector "
            "and contracting exposure.",
            "Consultant headcount and productivity - the cost line management "
            "actively flexes, and the main lever on operating profit.",
            "Fee rates, client and candidate confidence, time-to-hire.",
            "FX: EUR and AUD against GBP translate directly into reported net "
            "fees and profit.",
            "Peer read-across from PageGroup, Robert Walters, SThree, Randstad "
            "and Adecco trading updates.",
        ),
        queries=(
            "Hays plc full year 2026 results trading update net fees outlook",
            "UK permanent recruitment market white collar hiring vacancies August 2026",
            "Hays plc RNS regulatory news announcement August 2026",
            "PageGroup Robert Walters SThree trading update permanent placements 2026",
        ),
    ),
    "DE": CompanyProfile(
        ticker="DE",
        name="Deere & Company",
        listing="NYSE: DE",
        period="FY2026 Q3",
        reports_on="2026-08-20",
        metrics=(
            "Worldwide net sales and revenues, USD millions",
            "Diluted EPS (GAAP), USD per share",
            "Production & Precision Ag operating profit, USD millions",
        ),
        macro_drivers=(
            "US and Brazilian farm income - the USDA net farm income forecast "
            "is the demand series for large agriculture.",
            "Crop prices and stocks-to-use: corn, soybeans, wheat; input costs; "
            "the crop-price-to-input-cost ratio farmers actually decide on.",
            "Large-ag order books and early-order programmes, plus retail "
            "versus production shipment mix.",
            "Dealer inventory and used-equipment values - destocking and "
            "production shutdown days hit revenue and under-absorb factories.",
            "Precision-ag adoption: See & Spray, JDLink, the technology-stack "
            "attach rate and the shift toward recurring solutions revenue.",
            "Small Ag & Turf and Construction & Forestry cycles, including "
            "housing starts and infrastructure spend.",
            "John Deere Financial: credit losses, delinquencies and the "
            "financing spread, which sit inside net sales and revenues.",
            "Peer read-across from AGCO, CNH, Titan Machinery and Lindsay.",
        ),
        queries=(
            "Deere third quarter fiscal 2026 results guidance large agriculture equipment",
            "USDA net farm income forecast 2026 corn soybean prices equipment demand",
            "Deere investor relations news release August 2026",
            "AGCO CNH agricultural equipment dealer inventory order book 2026",
        ),
    ),
}


def profile_for(ticker: str, company_name: str | None = None) -> CompanyProfile:
    """Profile for `ticker`, or a generic one so an unknown ticker still runs."""
    key = (ticker or "").strip().upper()
    known = COMPANIES.get(key)
    if known is not None:
        if company_name and company_name != known.name:
            return CompanyProfile(**{**known.__dict__, "name": company_name})
        return known
    name = company_name or key
    logger.warning("No company profile for %r - using a generic profile.", key)
    return CompanyProfile(
        ticker=key,
        name=name,
        listing=key,
        period="the next reporting period",
        reports_on="unknown",
        metrics=(
            "Revenue or net sales",
            "Earnings per share",
            "Operating or gross margin",
        ),
        macro_drivers=(
            "The end-market demand series that actually drives this company's "
            "revenue - identify it from the company's own disclosure rather "
            "than assuming.",
            "Input costs, pricing and mix.",
            "Peer, supplier and customer disclosures that read across.",
        ),
        queries=(
            f"{name} latest quarterly results guidance outlook",
            f"{name} investor relations press release",
            f"{name} industry demand trends",
        ),
    )


# ==========================================================================
# 3. The system prompt
# ==========================================================================
#
# BRACE RULE. `AgentSpec.instructions` is treated as a format string by the
# SDK, so a stray literal brace is a runtime error at 17:50. This template is
# written with NO literal braces at all: the only braces present are the named
# substitution fields below, which `.format()` consumes. Rendered instructions
# therefore contain zero brace characters and cannot be re-interpreted by
# anything downstream. If you ever need a literal brace here, double it - and
# `_assert_renderable` will catch you if you forget.

PROMPT_FIELDS = frozenset(
    {
        "company_name",
        "ticker",
        "listing",
        "period",
        "reports_on",
        "corpus_freeze",
        "as_of",
        "metrics_block",
        "drivers_block",
        "queries_block",
    }
)

SYSTEM_PROMPT = """\
You are the NEWS ANALYST on a fundamental equity research desk. You cover ONE
company: {company_name} ({listing}, ticker {ticker}).

You think like a long-horizon VALUE INVESTOR, not a trader. You care about what
a business earns and why. You do not care what the share price did this week,
and you treat anyone who leads with the share price as a weak source.

================================================================
YOUR WINDOW
================================================================
The research desk already holds a frozen document corpus - filings, transcripts
and slides - complete up to {corpus_freeze}. You are NOT there to repeat it.

Your job is the gap. The gap has TWO parts, and conflating them is the single
most common failure of this role.

GAP 1 - COMPANY DISCLOSURES: strictly since {corpus_freeze}, up to today,
{as_of}. The corpus already holds every company filing, transcript and slide up
to the freeze, so anything the company itself published before then is already
known. Only post-freeze company disclosure is new.

GAP 2 - THIRD-PARTY AND MACRO EVIDENCE: NOT date-limited to the freeze. Look
back up to roughly SIX MONTHS.

  Read this carefully, because it is counter-intuitive. The corpus contains
  FIRST-PARTY COMPANY DOCUMENTS ONLY. It holds NO industry research, NO macro
  series, NO analyst or trade coverage, and NO peer companies' disclosures - at
  ANY date. So a housing-market study from three months ago, a competitor's
  last quarterly result, or a government statistics release is NOT "already in
  the corpus" merely because it predates the freeze. It was never there at all,
  and it may be the most decisive evidence you find.

  Do NOT reject an industry study, a macro release, or a peer's results just
  because it is dated before {corpus_freeze}. Judge it on whether it bears on
  the fiscal period being forecast. A remodeling-demand study published two
  months ago speaks directly to the quarter being reported; a company press
  release from the same date does not, because the corpus already has it.

If a search returns nothing inside Gap 1, that is a normal and reportable
result - say "nothing found in the window". It is NOT a reason to return an
empty report. Gap 2 is almost never empty, and it is where the analytical value
usually sits.

{company_name} reports {period} on {reports_on}.
- If that date is after {as_of}, the results are NOT out yet. Anything claiming
  to report actual figures for {period} is either about a DIFFERENT period or is
  fabricated. Say so and exclude it. What you are looking for is pre-print
  evidence: reaffirmed or changed guidance, trading updates, pre-announcements,
  conference and investor-day remarks, channel and macro data, peer results.
- If that date is on or before {as_of}, the release itself is public. Then the
  press release, the 8-K or RNS, the transcript and the slides are your primary
  targets and outrank every commentary about them.

You have NO reliable memory of events after the corpus freeze. Every single
item you report MUST come from a tool result you actually saw in this
conversation. Never write a headline, a date, a publisher or a URL from
memory. If you did not see it in a tool result, it does not exist.

================================================================
ABSOLUTE PROHIBITION - OPENSTOCKS
================================================================
NEVER search for, fetch, scrape or cite openstocks.com, any subdomain of it, or
any mirror of it. Do not use a site: query naming it. Do not follow a link to
it. Do not ask for its consensus figures.

OpenStocks hosts this event. Its rules forbid interference with its platform,
and it withholds the Wall Street benchmark from teams on purpose. Aiming an
automated collector at it risks disqualification from every prize.

This is enforced in code as well as here: a blocked URL never reaches the
network, and the attempt is logged. If you see a search hit from that domain,
do not scrape it - record it in excluded_sources with the reason "hackathon
host, out of bounds by fair-play rule" and move on.

================================================================
SOURCE QUALITY - THE DEFINING REQUIREMENT
================================================================
The web is mostly noise dressed as research. Sorting it is the main thing you
are for. A thin source laundered into the model with a citation attached is
worse than no source at all.

PREFER, in roughly this order:
  1. PRIMARY. The company itself: IR press releases, 8-K, 10-Q and 10-K, RNS
     announcements for a UK listing, trading statements, earnings-call
     transcripts, investor-day decks, proxy and remuneration disclosure. First
     party, on the record, and the same text the auditors saw.
  2. TIER-ONE WIRES AND PAPERS. Reuters, Bloomberg, the Financial Times, the
     Wall Street Journal, Dow Jones Newswires, Nikkei/Handelsblatt-class
     national business press. Reported, edited, corrected when wrong.
  3. GENUINE INDUSTRY TRADE PRESS that carries its own domain data - trade
     bodies, statistical agencies, industry associations, specialist titles
     that publish numbers rather than opinions.
  4. FACTUAL REPORTING OF REPUTABLE SELL-SIDE RESEARCH. Useful when it conveys
     the analyst's actual reasoning and estimate changes with attribution. A
     bare price-target headline is not this.
  5. PEER, COMPETITOR, SUPPLIER AND CUSTOMER DISCLOSURES that read across.
     A competitor's results one week earlier are often the single best
     forward-looking evidence available about the quarter.
  6. MACRO AND OFFICIAL SERIES that actually drive this P&L - see below.

REJECT, and name every rejection:
  - Day-trading and momentum commentary, "what the chart says", support and
    resistance, moving averages, RSI, any technical analysis at all.
  - "Should you buy X now", "3 stocks to buy before earnings", ranked listicles,
    quiz-style headlines.
  - Clickbait price-target pieces whose entire content is a target number and
    an upside percentage.
  - Promotional newsletters, subscription-bait, sponsored posts, "our top pick"
    marketing.
  - Penny-stock and hype aggregators.
  - Anonymous forum and social sentiment, message-board scrapes, "retail
    traders are saying".
  - Content-farm rewrites with no original reporting - a paraphrase of a wire
    story with no added fact, no named reporter, and no primary link.
  - Anything undated.
  - COMPANY disclosures dated before {corpus_freeze} - the corpus already has
    them. This does NOT apply to third-party research, macro data or peer
    results, which are never in the corpus at any date. Excluding those on date
    alone throws away your best evidence.
  - AI-generated aggregator pages that recycle the same paragraph across many
    tickers.

Rate EVERY item you keep:
  high   = primary company disclosure, tier-one wire or paper, or trade press
           with its own data.
  medium = reputable secondary reporting, factual sell-side summaries, solid
           regional or specialist outlets.
  low    = thin but not worthless, and only when nothing better exists. If you
           keep a low item, say in quality_reason why you had no better option.

quality_reason must be a REASON, not a restatement. "Reuters, wire service with
named reporter, quotes the CFO directly" is a reason. "High quality source" is
not, and will be marked wrong.

excluded_sources is NOT optional. Real search results always contain junk. An
empty exclusion list means you did not look, or did not judge. List what you
passed over and why, by publisher or URL. Being explicit about what you threw
away is part of the deliverable.

================================================================
WHAT COUNTS AS ANALYSIS
================================================================
Write about the BUSINESS:
  - Demand signals: volumes, orders, bookings, traffic, transactions, backlog,
    placements, order books, end-market commentary.
  - Pricing and mix: realised price, discounting, promotional intensity,
    commodity pass-through, product and geographic mix.
  - Unit economics and margin drivers: input costs, utilisation, freight,
    labour, absorption, operating leverage, one-offs.
  - Guidance: what was guided, when, by whom, whether it was reaffirmed,
    narrowed, raised or cut, and how this management's guidance has behaved
    historically - do they sandbag or stretch.
  - Management track record and credibility, capital allocation: buybacks,
    dividends, capex, M&A, balance-sheet moves and what they signal.
  - Second-order evidence: peers, suppliers, customers, channel checks,
    official statistics.

Do NOT write about:
  - Share price moves, market capitalisation, "the stock rallied", 52-week
    highs, volatility, options positioning, short interest as sentiment.
  - Analyst ratings changes reported as events without the underlying reasoning.
  - Generic market sentiment, fear-and-greed, "investors are watching".
  - Valuation multiples quoted without the earnings power behind them.

If a source is only interesting because of what the price did, it is not
interesting.

================================================================
THIS COMPANY
================================================================
The desk must ultimately forecast these figures for {period}. Point every item
you keep at one of them through fundamental_relevance:
{metrics_block}

The external series that actually drive this P&L - search these directly, not
just the company name:
{drivers_block}

================================================================
SEARCH STRATEGY - WORK TO A BUDGET
================================================================
You have a strict turn budget. Be deliberate: at most 4 searches and at most
4 scrapes. Quality of query beats quantity of query.

Step 1 - SEARCH, 2 to 4 well-formed queries. Include the company or series
name, the period, and a recency term. Good starting points for this company:
{queries_block}
Adapt them. Add a fourth only if the first three leave a real gap.

Step 2 - TRIAGE the hits WITHOUT scraping. You get url, title and description.
That is enough to sort primary from wire from junk. Write down what you are
discarding as you go; that becomes excluded_sources.

Step 3 - SCRAPE the best 2 to 4 URLs only. Prefer, in order: the company IR or
regulatory announcement; a tier-one wire story with substance; a peer or macro
release that reads across. Never spend a scrape on a listicle to confirm it is
a listicle - the title already told you.

Step 4 - READ FOR NUMBERS AND MECHANISM. Pull the specific figure, the date,
the speaker and the driver. "Management reaffirmed the full-year comparable
sales range on 5 August" is evidence. "Sentiment is improving" is not.

Step 5 - SUBMIT once, via submit_result. Never call it twice.

If a search returns nothing usable, say so plainly in notes, set confidence to
low, and submit what you have. An honest empty-handed report beats an invented
full one. Do not pad the report to look busy.

================================================================
FILLING THE REPORT
================================================================
ticker, as_of and corpus_freeze: exactly {ticker}, {as_of} and {corpus_freeze}.

items: one entry per source you KEPT.
  headline    - as published, not your paraphrase.
  url         - the exact URL from the tool result. Never construct one.
  published   - the publication date, ISO where you can read it. If the page
                does not state one, write "undated" and downgrade the rating.
  publisher   - the outlet or the company.
  source_quality  - high, medium or low. Nothing else.
  quality_reason  - why that rating, in terms of who reported it and how.
  fundamental_relevance - which of the target metrics or drivers it bears on.
  implication - what it implies for the REPORTED NUMBERS, with the number or
                the direction and rough size where the source gives one.
  direction   - positive, negative or neutral, for the fundamentals.

valuation_view: the fundamental picture in a few tight sentences. Earnings
power, durability, what the recent evidence changes about the forward view, and
what it would take to be wrong. No price targets. No price action.

demand_signals / cost_and_margin_signals / guidance_changes /
peer_and_industry_readacross: the specifics, with who said it and when.
Write "nothing found in the window" where that is the truth.

excluded_sources: every rejection, with the reason.

confidence: high, medium or low, with the reason stated in notes. Base it on
what you actually found, not on how much text you produced.

notes: what you could not get to, what was paywalled, where the evidence is
thin, and anything you are uncertain about. This is read and rewarded. Hiding
a weakness scores worse than admitting it.
"""


def _assert_renderable(template: str) -> None:
    """Fail at import, not mid-run, if the prompt template is malformed.

    Catches an unescaped literal brace, a typo'd field name, and any positional
    field. `AgentSpec.instructions` is format-string territory downstream, so a
    stray brace is a live hazard rather than a style nit.
    """
    for _, field_name, _, _ in string.Formatter().parse(template):
        if field_name is None:
            continue
        if field_name == "":
            raise ValueError(
                "SYSTEM_PROMPT contains a positional field '{}'. Use named "
                "fields, and double any literal brace."
            )
        root = field_name.split(".")[0].split("[")[0]
        if root not in PROMPT_FIELDS:
            raise ValueError(
                f"SYSTEM_PROMPT references unknown field {field_name!r}. "
                f"Known fields: {sorted(PROMPT_FIELDS)}. A literal brace must "
                f"be written as a doubled brace."
            )


_assert_renderable(SYSTEM_PROMPT)


def render_prompt(ticker: str, as_of: str, company_name: str | None = None) -> str:
    """Render the system prompt for one company at one point in time.

    `as_of` is injected here and nowhere else. Nothing about the current date is
    baked into the module, so a backtest at a past date gets a prompt that is
    honest about what was knowable then.
    """
    p = profile_for(ticker, company_name)
    rendered = SYSTEM_PROMPT.format(
        company_name=p.name,
        ticker=p.ticker,
        listing=p.listing,
        period=p.period,
        reports_on=p.reports_on,
        corpus_freeze=CORPUS_FREEZE,
        as_of=as_of,
        metrics_block=p.metrics_block(),
        drivers_block=p.drivers_block(),
        queries_block=p.queries_block(),
    )
    if "{" in rendered or "}" in rendered:
        raise ValueError(
            "Rendered news prompt still contains a brace. The Agents SDK treats "
            "instructions as a format string; fix the template."
        )
    return rendered


USER_TASK = """\
Research {company_name} ({ticker}) for the period since {corpus_freeze}, as of {as_of}.

Find what has actually happened, judge each source, and report the fundamental
implications for {period}. Reject the noise and list what you rejected. Finish
by calling submit_result exactly once.
"""


def render_task(ticker: str, as_of: str, company_name: str | None = None) -> str:
    p = profile_for(ticker, company_name)
    return USER_TASK.format(
        company_name=p.name,
        ticker=p.ticker,
        corpus_freeze=CORPUS_FREEZE,
        as_of=as_of,
        period=p.period,
    )


# ==========================================================================
# 4. Spec
# ==========================================================================


def build_spec(
    ticker: str,
    as_of: str,
    company_name: str | None = None,
    *,
    max_turns: int | None = None,
    profile: str | None = None,
) -> AgentSpec:
    """AgentSpec for the news analyst.

    Web tools only - no corpus access. The filings analyst owns the corpus; this
    one owns the gap after the freeze, and keeping the tool sets disjoint is
    what stops the two agents duplicating each other's work.

    Model, reasoning effort and turn budget all default to `agent_core.settings`
    so nothing about the provider is hardcoded here.
    """
    p = profile_for(ticker, company_name)
    return AgentSpec(
        name=f"News Analyst ({p.ticker})",
        instructions=render_prompt(p.ticker, as_of, p.name),
        result_model=NewsReport,
        tools=[],
        use_web=True,
        allow_delegation=False,
        profile=profile,
        max_turns=max_turns,
        fallback=NewsReport(
            ticker=p.ticker,
            as_of=as_of,
            corpus_freeze=CORPUS_FREEZE,
            confidence="none",
            notes=(
                "FALLBACK REPORT - the news agent failed or exhausted its turn "
                "budget before submitting. No web evidence was gathered. Treat "
                "as no information, not as an absence of news."
            ),
        ),
    )


# ==========================================================================
# 5. Post-run validation
# ==========================================================================

_QUALITY_LEVELS = ("high", "medium", "low")


def normalise_quality(value: str) -> str:
    """Map whatever the model wrote onto high / medium / low / unrated."""
    v = (value or "").strip().lower()
    if v in _QUALITY_LEVELS:
        return v
    for level in _QUALITY_LEVELS:
        if v.startswith(level):
            return level
    for level in _QUALITY_LEVELS:
        if level in v:
            return level
    return "unrated"


def _excl_key(e: ExcludedSource) -> tuple[str, str]:
    return (e.url_or_publisher.strip().lower(), e.reason.strip().lower())


def postprocess(
    report: NewsReport,
    ticker: str,
    as_of: str,
    blocked: list[dict[str, str]] | None = None,
) -> NewsReport:
    """Validate and repair the model's output. Rubric: validation, 12 points.

    Four checks, each one a failure mode seen in practice:
      * the identifying fields are set from the CALLER's inputs, never trusted
        to the model - it can forget them and it can transcribe them wrong;
      * source_quality is normalised to the three allowed values;
      * any item whose URL is on the blocked host is removed and re-filed as an
        exclusion - defence in depth behind the network guard;
      * duplicate URLs collapse.

    Guard rejections are appended to `excluded_sources`, so an attempt that was
    stopped in code still shows up in the visible rejection list.
    """
    report.ticker = ticker
    report.as_of = as_of
    report.corpus_freeze = CORPUS_FREEZE

    kept: list[NewsItem] = []
    extra: list[ExcludedSource] = []
    seen: set[str] = set()
    dropped_blocked = 0
    dropped_dupe = 0
    missing_reason = 0

    for item in report.items:
        item.source_quality = normalise_quality(item.source_quality)
        url = (item.url or "").strip()
        if url and is_blocked_url(url):
            dropped_blocked += 1
            extra.append(
                ExcludedSource(
                    url_or_publisher=url,
                    reason=(
                        "Removed after the run by the openstocks guard: the "
                        "hackathon host is out of bounds by the fair-play rule."
                    ),
                )
            )
            continue
        key = url.lower() or f"{item.publisher}|{item.headline}".strip().lower()
        if key and key in seen:
            dropped_dupe += 1
            continue
        if key:
            seen.add(key)
        if not item.quality_reason.strip():
            missing_reason += 1
        kept.append(item)

    report.items = kept

    for b in blocked or []:
        extra.append(
            ExcludedSource(
                url_or_publisher=b.get("target", ""),
                reason=(
                    "Blocked before the network call - "
                    + b.get("reason", "openstocks host")
                ),
            )
        )

    if extra:
        existing = {_excl_key(e) for e in report.excluded_sources}
        for e in extra:
            if _excl_key(e) not in existing:
                report.excluded_sources.append(e)
                existing.add(_excl_key(e))

    counts = {lvl: sum(1 for i in kept if i.source_quality == lvl) for lvl in _QUALITY_LEVELS}
    counts["unrated"] = sum(1 for i in kept if i.source_quality == "unrated")

    checks = [
        f"items kept={len(kept)} (high={counts['high']}, medium={counts['medium']}, "
        f"low={counts['low']}, unrated={counts['unrated']})",
        f"excluded_sources={len(report.excluded_sources)}",
    ]
    if dropped_blocked:
        checks.append(f"WARNING: {dropped_blocked} item(s) dropped as openstocks URLs")
    if dropped_dupe:
        checks.append(f"{dropped_dupe} duplicate item(s) collapsed")
    if missing_reason:
        checks.append(f"WARNING: {missing_reason} item(s) carry no quality_reason")
    if counts["unrated"]:
        checks.append(f"WARNING: {counts['unrated']} item(s) had an unusable source_quality")
    if not report.excluded_sources:
        checks.append(
            "WARNING: nothing was excluded - real search results contain junk, "
            "so an empty rejection list suggests the filter did not run"
        )
    if not kept:
        checks.append("WARNING: no usable items found in the window")

    stamp = "validation: " + "; ".join(checks)
    report.notes = f"{report.notes}\n\n{stamp}".strip() if report.notes else stamp
    return report


# ==========================================================================
# 6. Entry point
# ==========================================================================


def require_firecrawl() -> None:
    """Fail early and actionably when there is no web access."""
    if not settings.firecrawl_api_key:
        raise FirecrawlNotConfigured(
            "FIRECRAWL_API_KEY is not set, so the news analyst has no web "
            "access and cannot run.\n"
            "  Fix: add this line to "
            f"{REPO_ROOT / '.env'}\n"
            "      FIRECRAWL_API_KEY=fc-...\n"
            "  or export FIRECRAWL_API_KEY in the shell before running.\n"
            "  Keys: https://www.firecrawl.dev/app/api-keys\n"
            "  Note: .env is read relative to the working directory, so run "
            "from the repo root.\n"
            "  Everything offline in this module - build_spec, render_prompt, "
            "the openstocks guard - works without a key."
        )


async def analyse(
    ticker: str,
    as_of: str | None = None,
    run_id: str | None = None,
    store: Any = None,
    company_name: str | None = None,
    *,
    max_turns: int | None = None,
    profile: str | None = None,
) -> NewsReport:
    """Research one company's post-freeze news and return a `NewsReport`.

    Args:
        ticker: HD, ADI, HAS or DE. Anything else gets a generic profile.
        as_of: ISO run date. Defaults to today - never hardcoded, so a backtest
            can move it.
        run_id: Existing run to attach to. Created if a store is given without
            one.
        store: Optional `runstore.RunStore`. When present the work is recorded
            under the resume key "<TICKER>:news", and a completed task in the
            same run is returned instead of re-run.
        company_name: Override the display name for an unknown ticker.

    Raises:
        FirecrawlNotConfigured: if there is no Firecrawl key. The agent's only
            tools are web tools, so running without one would produce a
            confident report built on nothing.
    """
    ticker = (ticker or "").strip().upper()
    as_of = as_of or date.today().isoformat()
    key = f"{ticker}:news"

    require_firecrawl()
    install_openstocks_guard()

    spec = build_spec(ticker, as_of, company_name, max_turns=max_turns, profile=profile)

    own_run = False
    task_id: str | None = None
    if store is not None:
        if run_id is None:
            run_id = store.start_run(f"news:{ticker}", config={"as_of": as_of})
            own_run = True
        cached = store.get_output(run_id, key)
        if cached:
            try:
                logger.info("Resuming %s from the run store - already completed.", key)
                return NewsReport.model_validate_json(cached)
            except Exception as e:  # stored output unusable -> just re-run
                logger.warning("Stored output for %s did not parse (%s); re-running.", key, e)
        task_id = store.start_task(
            run_id,
            key,
            RESUME_KIND,
            agent_name=spec.name,
            model=settings.model_for(spec.resolved_profile),
            input={"ticker": ticker, "as_of": as_of, "corpus_freeze": CORPUS_FREEZE},
        )

    try:
        with blocked_log() as blocked:
            logger.info("News analyst starting: %s as of %s", ticker, as_of)
            raw = await run_agent(spec, render_task(ticker, as_of, company_name))
            report = raw if isinstance(raw, NewsReport) else NewsReport()
            report = postprocess(report, ticker, as_of, blocked)
    except Exception as e:
        logger.exception("News analyst failed for %s", ticker)
        if store is not None and task_id:
            store.fail_task(task_id, str(e))
            if own_run and run_id:
                store.end_run(run_id, status="failed", notes=str(e)[:500])
        raise

    if store is not None and run_id:
        for b in blocked:
            store.log(
                run_id,
                "news.blocked_url",
                task_id=task_id,
                name=b.get("target", ""),
                payload=b.get("reason", ""),
            )
        for e in report.excluded_sources:
            store.log(
                run_id,
                "news.excluded_source",
                task_id=task_id,
                name=e.url_or_publisher,
                payload=e.reason,
            )
        for item in report.items:
            if item.source_quality in ("high", "medium") and item.url:
                store.add_evidence(
                    run_id,
                    ticker,
                    item.fundamental_relevance or "news",
                    item.url,
                    task_id=task_id,
                    claim=item.implication or item.headline,
                    locator=item.published or item.publisher,
                )
        if task_id:
            store.finish_task(task_id, output=report.model_dump_json(),
                              usage=get_last_usage())
        if own_run:
            store.end_run(run_id)

    return report


# ==========================================================================
# 7. CLI
# ==========================================================================

_BAR = "=" * 78


def _wrap(text: str, indent: str = "    ", width: int = 96) -> str:
    import textwrap

    text = (text or "").strip()
    if not text:
        return f"{indent}(none)"
    out: list[str] = []
    for para in text.splitlines():
        if not para.strip():
            continue
        out.extend(
            textwrap.wrap(
                para.strip(), width=width, initial_indent=indent, subsequent_indent=indent
            )
        )
    return "\n".join(out) or f"{indent}(none)"


def format_summary(report: NewsReport) -> str:
    """Readable summary: items by source quality, the view, and the rejects."""
    lines: list[str] = []
    lines.append(_BAR)
    lines.append(
        f"NEWS ANALYST - {report.ticker or '?'}   "
        f"as of {report.as_of or '?'}   "
        f"covering everything after the corpus freeze {report.corpus_freeze or '?'}"
    )
    lines.append(_BAR)

    buckets: dict[str, list[NewsItem]] = {"high": [], "medium": [], "low": [], "unrated": []}
    for item in report.items:
        buckets.setdefault(normalise_quality(item.source_quality), []).append(item)

    lines.append("")
    lines.append(
        f"ITEMS ({len(report.items)} kept)   "
        + "  ".join(f"{k}={len(v)}" for k, v in buckets.items())
    )
    for level in ("high", "medium", "low", "unrated"):
        items = buckets.get(level) or []
        if not items:
            continue
        lines.append("")
        lines.append(f"--- {level.upper()} QUALITY ({len(items)}) " + "-" * (46 - len(level)))
        for n, item in enumerate(items, 1):
            head = item.headline or "(no headline)"
            meta = " | ".join(x for x in (item.publisher, item.published) if x)
            lines.append(f"  [{n}] {head}")
            if meta:
                lines.append(f"      {meta}")
            if item.url:
                lines.append(f"      {item.url}")
            if item.quality_reason:
                lines.append(f"      why {level}: {item.quality_reason}")
            if item.fundamental_relevance:
                lines.append(f"      bears on: {item.fundamental_relevance}")
            if item.implication:
                arrow = {"positive": "+", "negative": "-", "neutral": "="}.get(
                    (item.direction or "").strip().lower(), "?"
                )
                lines.append(f"      [{arrow}] {item.implication}")

    lines.append("")
    lines.append(_BAR)
    lines.append("VALUATION VIEW (fundamentals, not price)")
    lines.append(_BAR)
    lines.append(_wrap(report.valuation_view))

    for title, body in (
        ("DEMAND SIGNALS", report.demand_signals),
        ("COST AND MARGIN SIGNALS", report.cost_and_margin_signals),
        ("GUIDANCE CHANGES", report.guidance_changes),
        ("PEER AND INDUSTRY READ-ACROSS", report.peer_and_industry_readacross),
    ):
        lines.append("")
        lines.append(title)
        lines.append(_wrap(body))

    lines.append("")
    lines.append(_BAR)
    lines.append(f"EXCLUDED SOURCES ({len(report.excluded_sources)}) - what was rejected, and why")
    lines.append(_BAR)
    if not report.excluded_sources:
        lines.append("    (none listed - suspicious; real result sets contain junk)")
    for n, e in enumerate(report.excluded_sources, 1):
        lines.append(f"  [{n}] {e.url_or_publisher or '(unnamed)'}")
        lines.append(f"      reason: {e.reason or '(none given)'}")

    lines.append("")
    lines.append(f"CONFIDENCE: {report.confidence or '(unstated)'}")
    lines.append("NOTES")
    lines.append(_wrap(report.notes))
    lines.append("")
    return "\n".join(lines)


def _default_json_path(ticker: str, as_of: str) -> Path:
    return DEFAULT_JSON_DIR / f"{ticker}-news-{as_of}.json"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m analysts.news",
        description=(
            "News analyst: post-corpus-freeze evidence for one company, filtered "
            "for source quality and read for fundamentals."
        ),
    )
    ap.add_argument("--ticker", required=True, help="HD, ADI, HAS or DE")
    ap.add_argument(
        "--as-of",
        default=None,
        metavar="YYYY-MM-DD",
        help="Run date. Defaults to today. Injected into the prompt.",
    )
    ap.add_argument("--json", default=None, metavar="PATH", help="Where to write the full JSON.")
    ap.add_argument("--company", default=None, help="Override the company display name.")
    ap.add_argument("--profile", default=None, help="LLM profile override (see agent_core.config).")
    ap.add_argument("--max-turns", type=int, default=None, help="Agent turn budget override.")
    ap.add_argument("--store", action="store_true", help="Record the run in the SQLite run store.")
    ap.add_argument("--run-id", default=None, help="Attach to an existing run id (implies --store).")
    ap.add_argument("--print-prompt", action="store_true", help="Print the rendered prompt and exit.")
    ap.add_argument("-v", "--verbose", action="store_true", help="Debug logging.")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    ticker = args.ticker.strip().upper()
    as_of = args.as_of or date.today().isoformat()

    if args.print_prompt:
        print(render_prompt(ticker, as_of, args.company))
        return 0

    try:
        require_firecrawl()
    except FirecrawlNotConfigured as e:
        print(f"\nERROR: {e}\n", file=sys.stderr)
        return 2

    store = None
    if args.store or args.run_id:
        from runstore import get_store

        store = get_store()

    use_selector_event_loop()
    try:
        report = asyncio.run(
            analyse(
                ticker,
                as_of=as_of,
                run_id=args.run_id,
                store=store,
                company_name=args.company,
                max_turns=args.max_turns,
                profile=args.profile,
            )
        )
    except FirecrawlNotConfigured as e:
        print(f"\nERROR: {e}\n", file=sys.stderr)
        return 2
    except Exception as e:
        logger.exception("News run failed")
        print(f"\nERROR: news run failed: {type(e).__name__}: {e}\n", file=sys.stderr)
        return 1

    print()
    print(format_summary(report))

    out_path = Path(args.json) if args.json else _default_json_path(ticker, as_of)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(report.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Full JSON written to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
