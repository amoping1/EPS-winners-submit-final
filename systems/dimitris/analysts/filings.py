"""filings.py — the FILINGS ANALYST.

The primary evidence gatherer. It reads ONE company's first-party documents in
the frozen offline corpus (1,139 docs, freeze 2026-08-14) and returns a
`FilingsReport`. It does not forecast; it supplies what a forecast is built on.

Four layers, in the order they matter:

  (a) the historical REPORTED series, pulled from the corpus' clean pipe tables
      via `extract_tables`, so no model ever reads digits off prose;
  (b) MANAGEMENT'S STATED REASONS the metric moved — MD&A in the 10-Q/10-K and
      the prepared remarks of the earnings call. This causal layer is the whole
      point of the agent: a number without a driver is not evidence, it is a
      data point;
  (c) live company guidance — a first-party forecast and usually the single
      best anchor available;
  (d) seasonality and accounting notes — basis definitions, the 53rd week, FX,
      restatements. This is what stops a correct number being scored wrong.

Two invariants are load-bearing and appear again in the system prompt:

  * `source_url` is null on 1,027 of 1,139 documents, so the FILENAME is the
    citation. `verify_citations()` re-resolves every cited filename against the
    index, which is how we find a hallucinated source before a judge does.

  * `as_of` filters on `published_at`, NEVER on `period`. They diverge — Home
    Depot has a document published 2026-05-21 carrying period "Q2 2027" — so
    filtering on `period` leaks the future and destroys any backtest.

Standalone:

    python -m analysts.filings --ticker HD
    python -m analysts.filings --ticker ADI --as-of 2026-08-16 --json adi.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent


def ensure_env_loaded(path: Path | None = None) -> bool:
    """Export the repo `.env` into os.environ. Idempotent, never overrides.

    agent_core.config declares `env_file=".env"`, but `Settings.api_key_for()`
    reads `os.getenv(...)` — so the key only reaches LiteLLM if something has
    already exported `.env` into the PROCESS environment. That happens
    incidentally, via litellm's own `load_dotenv()`, whose path resolution
    depends on the entry point: it works under `python -c` and fails under
    `python -m analysts.filings`, where the first attempt at this agent died
    with "Incorrect API key provided: None".

    Loading it here, from an absolute path, makes the entry point irrelevant.
    Must run before the first `build_model()`, hence at import.
    """
    env_path = path or (REPO_ROOT / ".env")
    if not env_path.is_file():
        return False
    try:
        from dotenv import load_dotenv  # in requirements.txt

        return bool(load_dotenv(env_path, override=False))
    except ImportError:  # tiny fallback so a missing dep is not fatal
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))
        return True


ensure_env_loaded()

from agent_core import get_last_usage, AgentSpec, run_agent, settings, use_selector_event_loop  # noqa: E402
from corpus.index import CORPUS_ROOT, TICKERS, get  # noqa: E402
from corpus.tools import CORPUS_TOOLS  # noqa: E402
from runstore import RunStore, get_store  # noqa: E402

from .models import FilingsReport  # noqa: E402

#: Upstream freeze date of the offline corpus. Nothing published on or after
#: this date exists in it, so the news analyst owns the gap after it.
CORPUS_FREEZE = "2026-08-14"

#: Reading four layers across ~6 documents does not fit in the 15-turn global
#: default. Measured: HD completes in the low twenties.
DEFAULT_MAX_TURNS = 40


# ---------------------------------------------------------------------------
# Targets — labels and units are verbatim from challenge/companies.json.
# Getting a label or a unit wrong throws the metric regardless of the number.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetricTarget:
    label: str  # exact label from companies.json
    units: str  # exact units from companies.json
    basis: str  # which basis the series must be on
    trap: str  # the specific way this metric gets read wrong


@dataclass(frozen=True)
class CompanyTarget:
    ticker: str
    company: str
    period: str  # the period being forecast, e.g. "FY2026Q2"
    calendar: str  # fiscal calendar + seasonality in one line
    doc_hints: str  # which doc_kinds actually carry the answers
    metrics: tuple[MetricTarget, ...]


TARGETS: dict[str, CompanyTarget] = {
    "HD": CompanyTarget(
        ticker="HD",
        company="The Home Depot, Inc.",
        period="FY2026Q2",
        calendar=(
            "Fiscal year ends late January/early February; fiscal 2026 Q2 is the "
            "three months ending ~2 August 2026. Q1 and Q2 are the spring/summer "
            "peak. Watch for 53-week fiscal years and for the GMS and SRS "
            "acquisitions changing the sales base."
        ),
        doc_hints=(
            "q2-8k and q1-8k are the earnings releases: headline figures, the "
            "non-GAAP reconciliation and the full-year guidance bullets all sit "
            "there. q2-10q / q1-10q / fy-10k carry the MD&A. call-pres is the "
            "prepared remarks, call-qna the analyst Q&A."
        ),
        metrics=(
            MetricTarget(
                label="Net sales",
                units="USDm",
                basis="reported GAAP, total company, quarterly",
                trap=(
                    "Report in MILLIONS. The release says 'sales of $41.8 billion' "
                    "in prose but the statement of earnings says 41,765 — use the "
                    "table. Do not mix the quarter with the year-to-date column."
                ),
            ),
            MetricTarget(
                label="Adjusted diluted EPS",
                units="USD / share",
                basis="ADJUSTED (non-GAAP) diluted, quarterly",
                trap=(
                    "Home Depot reports BOTH: GAAP diluted EPS and adjusted "
                    "diluted EPS, in adjacent sentences (Q1 FY2026: 3.30 GAAP vs "
                    "3.43 adjusted). The target is the ADJUSTED one. Capture the "
                    "GAAP figure too, but label its basis honestly."
                ),
            ),
            MetricTarget(
                label="Comparable sales, total company",
                units="%",
                basis="total company comparable sales growth, quarterly, in percent",
                trap=(
                    "TOTAL COMPANY, not 'comparable sales in the U.S.' — both are "
                    "quoted in the same sentence. Percentage points: 0.6 means "
                    "0.6%, never 0.006. Note any FX contribution in basis points."
                ),
            ),
        ),
    ),
    "ADI": CompanyTarget(
        ticker="ADI",
        company="Analog Devices, Inc.",
        period="FY2026Q3",
        calendar=(
            "Fiscal year ends late October/early November; fiscal 2026 Q3 is the "
            "three months ending ~1 August 2026. Semiconductor cycle, inventory "
            "correction and channel replenishment dominate; segment mix "
            "(Industrial, Automotive, Communications, Consumer) drives margin."
        ),
        doc_hints=(
            "q3-8k / q2-8k are the earnings releases and carry the NEXT-QUARTER "
            "OUTLOOK, which for ADI is explicit revenue, adjusted EPS and "
            "adjusted gross margin guidance with a plus/minus band. q2-10q / "
            "q3-10q / fy-10k carry the MD&A. call-pres is the prepared remarks."
        ),
        metrics=(
            MetricTarget(
                label="Revenue",
                units="USDm",
                basis="reported GAAP revenue, quarterly",
                trap=(
                    "Report in MILLIONS. Quarterly, not the six-month or "
                    "nine-month cumulative column that sits beside it in the 10-Q."
                ),
            ),
            MetricTarget(
                label="Adjusted diluted EPS",
                units="USD / share",
                basis="ADJUSTED (non-GAAP) diluted, quarterly",
                trap=(
                    "ADI's GAAP and adjusted EPS differ by a large margin because "
                    "of acquisition-related amortisation. The target is ADJUSTED. "
                    "Find the non-GAAP reconciliation table in the 8-K."
                ),
            ),
            MetricTarget(
                label="Adjusted gross margin",
                units="%",
                basis="ADJUSTED (non-GAAP) gross margin, quarterly, in percent",
                trap=(
                    "Adjusted, not GAAP gross margin. Percentage points: 69.4 "
                    "means 69.4%. ADI guides this metric directly — get the "
                    "guided range as well as the reported history."
                ),
            ),
        ),
    ),
    "HAS": CompanyTarget(
        ticker="HAS",
        company="Hays plc (LSE:HAS)",
        period="FY2026",
        calendar=(
            "UK company, fiscal year ends 30 June, so FY2026 ENDED 30 June 2026 "
            "and is reported in late August/September 2026 — the 10 July 2026 Q4 "
            "trading statement already carries the full-year outturn language. "
            "Reports in GBP, halves not quarters, with quarterly trading updates. "
            "Staffing is cyclical: net fees track white-collar hiring; there are "
            "working-day and FX effects between periods."
        ),
        doc_hints=(
            "No 10-Q or 10-K — this is a UK filer. h2-8k / h1-8k are the "
            "full-year and half-year results announcements; q1-8k / q3-8k / "
            "q4-8k are quarterly TRADING STATEMENTS carrying net fee growth by "
            "region. 'filing' includes the annual report. call-pres is the "
            "results presentation, call-qna the analyst Q&A."
        ),
        metrics=(
            MetricTarget(
                label="Net fees",
                units="GBPm",
                basis="group net fees, FULL YEAR, GBP millions",
                trap=(
                    "Net fees is the staffing-industry gross-profit measure — NOT "
                    "turnover/revenue, which is several times larger. Full year, "
                    "not H1 or a single quarter. Watch like-for-like vs actual "
                    "growth and the currency effect, which Hays states separately."
                ),
            ),
            MetricTarget(
                label="Pre-exceptional basic EPS",
                units="GBp",
                basis="PRE-EXCEPTIONAL BASIC earnings per share, full year, in PENCE",
                trap=(
                    "Three traps in one metric. (1) Pre-exceptional, not "
                    "statutory/reported. (2) BASIC, not diluted. (3) PENCE — 6.2 "
                    "means 6.2p, never 0.062 pounds. Hays also quotes a diluted "
                    "and a statutory number nearby; do not take those."
                ),
            ),
            MetricTarget(
                label="Pre-exceptional operating profit",
                units="GBPm",
                basis="PRE-EXCEPTIONAL operating profit, full year, GBP millions",
                trap=(
                    "Pre-exceptional, not statutory operating profit. Hays "
                    "separates exceptional items (restructuring, impairment) "
                    "explicitly — record what they were and how large."
                ),
            ),
        ),
    ),
    "DE": CompanyTarget(
        ticker="DE",
        company="Deere & Company",
        period="FY2026Q3",
        calendar=(
            "Fiscal year ends late October; fiscal 2026 Q3 is the three months "
            "ending ~2 August 2026. Deep ag-equipment down-cycle context: order "
            "books, dealer inventory, used equipment and tariffs drive the story. "
            "Deere guides FULL-YEAR NET INCOME, not quarterly EPS."
        ),
        doc_hints=(
            "q3-8k / q2-8k are the earnings releases with the segment tables and "
            "the full-year outlook. q2-10q / q3-10q / fy-10k carry the MD&A. "
            "'slide' decks are the quarterly earnings-call presentations and hold "
            "the clearest segment walk. call-pres is the prepared remarks."
        ),
        metrics=(
            MetricTarget(
                label="Worldwide net sales and revenues",
                units="USDm",
                basis="TOTAL worldwide net sales AND revenues, quarterly, USD millions",
                trap=(
                    "Deere's income statement shows several stacked totals: "
                    "'Net sales' (equipment only), 'Finance and interest income', "
                    "and 'Total net sales and revenues'. The target is the "
                    "WORLDWIDE NET SALES AND REVENUES total, which includes "
                    "financial services. Quarterly, not year-to-date."
                ),
            ),
            MetricTarget(
                label="Diluted EPS (GAAP)",
                units="USD / share",
                basis="GAAP diluted EPS, quarterly",
                trap=(
                    "This one is GAAP, unlike HD's and ADI's EPS targets. Take "
                    "diluted, not basic. Quarterly, not year-to-date."
                ),
            ),
            MetricTarget(
                label="Production & Precision Ag operating profit",
                units="USDm",
                basis="Production and Precision Agriculture SEGMENT operating profit, quarterly, USD millions",
                trap=(
                    "One SEGMENT only. Deere reports four: Production & Precision "
                    "Ag, Small Ag & Turf, Construction & Forestry, and Financial "
                    "Services. Take operating profit, not net sales and not the "
                    "operating margin percentage."
                ),
            ),
        ),
    ),
}

#: challenge/companies.json calls Hays "LSE:HAS"; the corpus index calls it "HAS".
_TICKER_ALIASES = {"LSE:HAS": "HAS", "HAS.L": "HAS", "HD.US": "HD", "DE.US": "DE"}


def normalise_ticker(ticker: str) -> str:
    """Accept 'LSE:HAS', 'has', 'HAS.L' — return the corpus ticker."""
    raw = (ticker or "").strip().upper()
    tk = _TICKER_ALIASES.get(raw, raw)
    if tk not in TARGETS:
        raise ValueError(
            f"Unknown ticker {ticker!r}. Known: {', '.join(TARGETS)} "
            f"(corpus tickers: {', '.join(TICKERS)})."
        )
    return tk


# ---------------------------------------------------------------------------
# System prompt
#
# NOTE: AgentSpec documents that instructions are treated as a format string,
# so a literal brace must be escaped as a doubled brace. This prompt is written
# with NO braces at all, which is correct under either interpretation — doubling
# would leak visible doubled braces if the string is in fact never formatted.
# `_assert_brace_free` enforces that at import so a later edit cannot regress it.
# ---------------------------------------------------------------------------


FILINGS_SYSTEM_PROMPT = """\
You are the FILINGS ANALYST on a team that forecasts a company's next reported \
results. You are the team's primary evidence gatherer.

You read ONE company inside a FROZEN OFFLINE CORPUS of first-party documents — \
SEC filings, London Stock Exchange announcements, earnings-call transcripts and \
slide decks. You have no web access and you must not answer from memory. If it \
is not in a document you opened with a tool, it does not exist.

YOU DO NOT FORECAST. You do not predict the target period. You report what the \
company already reported, WHY management said it moved, and what the company \
itself has guided to. Another agent does the forecasting, using only what you \
hand it.

=== WHAT YOU MUST PRODUCE, IN PRIORITY ORDER ===

(a) THE REPORTED SERIES — metrics_history.
    For EACH target metric, the last 6 to 8 comparable reported periods, newest \
first. Same-quarter-prior-year values matter most: they are the base a forecast \
grows from. Every observation carries metric, period, value, units, basis and a \
citation.

(b) WHAT DRIVES THIS BUSINESS, AND HOW IT HAS BEHAVED — drivers, \
business_factors, cyclical_behaviour. THIS IS THE POINT OF YOUR JOB.

    A number with no explanation is a data point, not evidence. Your job is not \
only to explain the last quarter — it is to hand the forecaster an understanding \
of WHAT MOVES THIS COMPANY and HOW IT HAS BEHAVED WHEN THOSE THINGS MOVED. Look \
across SEVERAL YEARS, not just the most recent print.

    drivers — find MANAGEMENT'S OWN STATED REASON in:
      - Item 2, Management's Discussion and Analysis, of the 10-Q or 10-K. The \
        sections you want are named for the metrics: EXECUTIVE SUMMARY, RESULTS \
        OF OPERATIONS, Sales, Gross Profit, Operating Expenses, Diluted Earnings \
        per Share.
      - the PREPARED REMARKS of the earnings call, which is the call-pres \
        document, and the analyst Q&A, which is call-qna. Prepared remarks are \
        where management explains itself in its own words, and the Q&A is where \
        analysts ask the SAME two or three questions every single quarter — \
        which is how you find what actually decides beat or miss here.
    Quote management, do not paraphrase into vagueness. "Weak demand" is \
useless. "Comparable sales grew 0.6 percent, with a 55 basis point tailwind \
from foreign exchange rates, while big-ticket discretionary demand remained \
soft" is evidence. Where management quantified a driver, put the number in \
magnitude. Judge persistence honestly: one-off, continuing, or fading.
    Aim for 10 to 18 drivers spanning SEVERAL periods, not 5 from the latest \
quarter. Tie each to a metric by name where it maps to one, but a driver that \
moves the business without mapping to a target metric is still worth reporting.

    business_factors — in prose: what structurally determines this company's \
results. The demand drivers, the pricing mechanism, the cost base and what \
moves it, the mix effects, the operating leverage, the exposures (FX, rates, \
commodities, a single large customer or channel). Which two or three of these \
have REPEATEDLY decided whether the company beat or missed. Ground every claim \
in something a document says, not in general knowledge about the industry.

    cyclical_behaviour — in prose: how this business behaved the LAST TIME its \
cycle turned. Find the downturn and the recovery in the reported history you \
gathered, and describe what happened: how far revenue fell and how much further \
margin fell with it, how many quarters it took to trough, what management said \
at the time, what recovered first, and whether the damage proved structural or \
cyclical. If the corpus does not reach back far enough to see a full turn, say \
so plainly rather than inventing one. This is the single most useful thing you \
can hand a forecaster about a cyclical company, and no other agent can get it.

(c) LIVE GUIDANCE — guidance.
    The company's own outlook, as most recently issued and still standing. Give \
the low and high of every range, the units, the exact date it was issued, and \
whether it was raised, lowered, reaffirmed or withdrawn. A reaffirmation is \
itself information. Include next-quarter outlook where the company gives one, \
and full-year guidance always. Note the segment or basis it applies to.

(d) SEASONALITY AND ACCOUNTING — seasonality, accounting_notes.
    seasonality: how this metric normally behaves in this quarter, in this \
company's fiscal calendar, with the same-quarter history that shows it.
    accounting_notes: how each target metric is DEFINED by this company. What \
is excluded from the adjusted or pre-exceptional figure. A 53rd week. Currency \
translation and the FX contribution management called out. Restatements, \
segment redefinitions, acquisitions or divestitures that break comparability. \
Share count changes from buybacks, which move EPS without moving profit.

Also fill risks — what could make the target period diverge from the pattern — \
and notes, where you record what you could NOT establish. Stating a gap is \
worth more than filling it with a guess.

=== TOOL WORKFLOW — FOLLOW THIS ORDER ===

1. list_documents to see what exists. Filter with doc_kinds. Start with the \
   earnings releases, which are the 8-K kinds, because they carry headline \
   figures, non-GAAP reconciliations and guidance in one short document.
2. document_outline on a document BEFORE reading it, to choose where to look. \
   Never read a document blind — several run to hundreds of KB.
3. extract_tables for ANY NUMBER, with the contains argument set to the line \
   label, for example contains equal to Net sales. The financial statements in \
   this corpus are clean pipe tables, so this returns the real reported figures \
   with no model in the loop. THIS IS ALWAYS THE RIGHT WAY TO GET A FIGURE.
   search_document to locate discussion; read_document_section to read the MD&A \
   or the prepared remarks once the outline told you which section; \
   read_document_lines to continue past a section boundary.
   search_corpus when you do not know WHICH document says something.
   find_numbers gives SEARCH LEADS ONLY. Never cite a find_numbers value \
   without confirming it in a table or a read section.
4. read_document_section for the causal layer. This is where you spend your \
   reading budget.

If a tool returns an error string, read it, fix the argument and retry once. Do \
not loop on the same failing call.

=== RULES THAT ARE NOT NEGOTIABLE ===

CITE THE FILENAME. source_url is null on 1,027 of the 1,139 corpus documents, \
so the filename IS the citation. Every citation source must be the exact \
filename a tool showed you, ending in .md, for example \
2026-05-19__hd-us-20260519-q1-8k__1038584.md. Never invent one, never shorten \
one, never cite a document you did not open. Put the line number or the section \
title in locator, and a SHORT VERBATIM snippet in quote. Every number and every \
driver needs one. List every filename you opened in documents_read.

KEEP THE BASES STRICTLY APART. Reported versus adjusted. GAAP versus non-GAAP. \
Pre-exceptional versus statutory. Basic versus diluted. Quarterly versus \
year-to-date versus full-year. These figures sit in ADJACENT COLUMNS AND \
ADJACENT SENTENCES in these documents, and swapping them is the single most \
likely way this work goes wrong. State the basis on every observation, in the \
basis field, using the company's own wording. If you are not certain which \
basis a figure is on, say so in notes rather than guessing.

UNITS ARE PART OF THE ANSWER. Report money in the units the target asks for — \
millions means millions, so 41765 and not 41.765 and not 41765000000. \
Percentages in POINTS: 4.8 means 4.8 percent, never 0.048. Pence means pence: \
6.2 means 6.2p, never 0.062.

POINT-IN-TIME. If the task gives you an as-of date, pass it as the as_of \
argument on EVERY list_documents and search_corpus call. That filters on \
published_at, which is the date the document became public. NEVER filter on \
period instead — published_at and period diverge in this corpus, a document \
published 2026-05-21 can carry period Q2 2027, and filtering on period would \
leak the future into the answer.

BUDGET. You have a limited number of turns. Do not read whole documents when a \
table or a section will do, and do not re-read what you already have. A \
realistic plan: the two most recent earnings releases for the numbers and the \
guidance; the most recent 10-Q or results announcement for the MD&A drivers; \
the most recent prepared remarks and Q&A for management's voice and for the \
questions analysts keep asking; the prior-year same-quarter release for the \
comparable base; and — for business_factors and cyclical_behaviour — two or \
three OLDER documents from a period when the business was clearly moving the \
other way, which is where the causal material lives. That is roughly eight to \
twelve documents. Prefer extract_tables and read_document_section over reading \
whole files; that is what buys you the extra reads. If you are running short of \
turns, submit what you have with an honest confidence rating — a partial report \
with real citations beats an empty one.

You may call delegate_to_subagent for ONE self-contained deep read, such as \
extracting a long history of one metric across several years of filings. The \
subagent has the same corpus tools but cannot see your conversation, so give it \
the ticker, the exact metric, the basis, the as-of date and the instruction to \
cite filenames. Use it at most once or twice; doing the read yourself is \
usually cheaper.

FINISH. Call submit_result EXACTLY ONCE, with every field you can support. Set \
confidence to high, medium or low, and justify it in notes. If something went \
wrong, still call submit_result — an honest partial report is the deliverable.
"""


def _assert_brace_free(prompt: str) -> str:
    """Guard the format-string hazard AgentSpec warns about, at import time."""
    if "{" in prompt or "}" in prompt:
        raise ValueError(
            "FILINGS_SYSTEM_PROMPT contains a brace. The SDK may treat "
            "instructions as a format string; keep the prompt brace-free."
        )
    return prompt


_assert_brace_free(FILINGS_SYSTEM_PROMPT)


# ---------------------------------------------------------------------------
# Task prompt — the per-company brief
# ---------------------------------------------------------------------------


def build_task_prompt(ticker: str, as_of: str | None = None) -> str:
    """The user message: which company, which metrics, which cutoff."""
    tk = normalise_ticker(ticker)
    t = TARGETS[tk]

    lines: list[str] = [
        f"COMPANY: {t.company}   TICKER: {t.ticker}",
        f"PERIOD THE TEAM MUST FORECAST: {t.period} "
        "(you do NOT forecast it — you gather the evidence for it)",
        "",
        "TARGET METRICS — the labels and units below are verbatim from the "
        "challenge specification. Use them exactly as written.",
        "",
    ]
    for i, m in enumerate(t.metrics, start=1):
        lines += [
            f"{i}. {m.label}   [units: {m.units}]",
            f"   basis required: {m.basis}",
            f"   watch out: {m.trap}",
            "",
        ]

    lines += [
        f"FISCAL CALENDAR AND SEASONALITY: {t.calendar}",
        "",
        f"WHERE TO LOOK: {t.doc_hints}",
        "",
        f"CORPUS FREEZE: the corpus contains nothing published on or after "
        f"{CORPUS_FREEZE}. Do not claim knowledge of events after it.",
    ]

    if as_of:
        lines += [
            "",
            f"AS-OF DATE: {as_of}. Pass as_of={as_of} on every list_documents "
            "and search_corpus call. It filters on published_at. Do NOT filter "
            "on period — period and published_at diverge in this corpus and "
            "filtering on period leaks the future.",
        ]
    else:
        lines += [
            "",
            "AS-OF DATE: none — use the whole frozen corpus. Leave the as_of "
            "tool argument blank.",
        ]

    lines += [
        "",
        "Work the four layers in order: the reported series, then management's "
        "stated reasons it moved, then live guidance, then seasonality and "
        "accounting basis. The reasons are the part the team cannot get "
        "anywhere else. When you are done, call submit_result once.",
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
    """The AgentSpec for the filings analyst on one company.

    Corpus-only by construction: `use_web=False`. This agent must never reach
    the network — the corpus is the evidence base, and the news analyst owns
    everything after the freeze date.
    """
    tk = normalise_ticker(ticker)
    t = TARGETS[tk]
    return AgentSpec(
        name=f"Filings Analyst ({tk})",
        instructions=FILINGS_SYSTEM_PROMPT,
        result_model=FilingsReport,
        tools=CORPUS_TOOLS,
        use_web=False,  # corpus-only. Not negotiable.
        allow_delegation=True,
        max_turns=max_turns or DEFAULT_MAX_TURNS,
        # A typed, identifiable empty report beats a bare default instance when
        # the agent dies: the caller can see WHICH company produced nothing.
        fallback=FilingsReport(
            ticker=tk,
            period_target=t.period,
            confidence="none",
            notes=(
                f"Filings analyst produced no valid output for {tk}. "
                f"as_of={as_of or 'none'}."
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Citation verification — the cheap check that catches a hallucinated source
# ---------------------------------------------------------------------------


def _cited_sources(report: FilingsReport) -> list[str]:
    seen: list[str] = []
    for obs in report.metrics_history:
        if obs.citation.source:
            seen.append(obs.citation.source)
    for g in report.guidance:
        if g.citation.source:
            seen.append(g.citation.source)
    for d in report.drivers:
        for c in d.citations:
            if c.source:
                seen.append(c.source)
    seen.extend(s for s in report.documents_read if s)
    out: list[str] = []
    for s in seen:
        if s not in out:
            out.append(s)
    return out


def verify_citations(report: FilingsReport) -> tuple[list[str], list[str]]:
    """Resolve every cited filename against the corpus index and the filesystem.

    Returns (resolved, unresolved). Anything in `unresolved` is a source the
    agent named but that does not exist — i.e. a fabricated citation.
    """
    resolved: list[str] = []
    unresolved: list[str] = []
    for src in _cited_sources(report):
        try:
            doc = get(src)
        except KeyError:
            unresolved.append(src)
            continue
        if doc.path.exists():
            resolved.append(doc.name)
        else:
            unresolved.append(src)
    return resolved, unresolved


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Usage meter — turns and spend, because the final run is a 45-minute window
# and "it was slow" is not a diagnosis.
# ---------------------------------------------------------------------------

#: USD per 1M tokens (input, output), mirroring the table in agent_core/config.py.
_PRICES: dict[str, tuple[float, float]] = {
    "gpt-5.6-sol": (5.00, 30.00),
    "gpt-5.6-terra": (2.00, 12.00),
    "gpt-5.6-luna": (0.20, 1.20),
}


class UsageMeter:
    """Counts model calls and tokens across a run via a LiteLLM callback.

    One model call is one agent turn, so `calls` is the turn count actually
    spent — including any turns burnt inside a delegated subagent.
    """

    def __init__(self) -> None:
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.reasoning_tokens = 0
        self.cached_tokens = 0
        self._logger: object | None = None

    def _record(self, response_obj: object) -> None:
        usage = getattr(response_obj, "usage", None)
        if usage is None:
            return
        self.calls += 1
        self.prompt_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
        self.completion_tokens += int(getattr(usage, "completion_tokens", 0) or 0)
        details = getattr(usage, "completion_tokens_details", None)
        self.reasoning_tokens += int(getattr(details, "reasoning_tokens", 0) or 0)
        pdetails = getattr(usage, "prompt_tokens_details", None)
        self.cached_tokens += int(getattr(pdetails, "cached_tokens", 0) or 0)

    def estimated_cost_usd(self, model: str) -> float | None:
        for name, (pin, pout) in _PRICES.items():
            if name in model:
                return (
                    self.prompt_tokens * pin + self.completion_tokens * pout
                ) / 1_000_000
        return None

    def attach(self) -> "UsageMeter":
        """Register with LiteLLM. Silently no-ops if the hook is unavailable."""
        try:
            import litellm
            from litellm.integrations.custom_logger import CustomLogger

            meter = self

            class _Hook(CustomLogger):  # type: ignore[misc]
                def log_success_event(self, kwargs, response_obj, start_time, end_time):
                    meter._record(response_obj)

                async def async_log_success_event(
                    self, kwargs, response_obj, start_time, end_time
                ):
                    meter._record(response_obj)

            self._logger = _Hook()
            litellm.callbacks.append(self._logger)
        except Exception as e:  # a metering failure must never kill a run
            logger.debug("Usage metering unavailable: %s", e)
        return self

    def summary(self, model: str) -> str:
        cost = self.estimated_cost_usd(model)
        money = "n/a" if cost is None else f"${cost:.4f}"
        return (
            f"model calls (turns): {self.calls}  |  prompt tokens: "
            f"{self.prompt_tokens:,} (cached {self.cached_tokens:,})  |  "
            f"completion tokens: {self.completion_tokens:,} "
            f"(reasoning {self.reasoning_tokens:,})  |  est. cost: {money}"
        )


def resume_key(ticker: str) -> str:
    return f"{normalise_ticker(ticker)}:filings"


def _record_evidence(
    store: RunStore,
    run_id: str,
    task_id: str | None,
    tk: str,
    report: FilingsReport,
) -> int:
    """One evidence row per metric observation, plus one per guidance range."""
    n = 0
    for obs in report.metrics_history:
        store.add_evidence(
            run_id,
            tk,
            obs.metric or "(unlabelled)",
            obs.citation.source or "(no source)",
            task_id=task_id,
            claim=f"{obs.period} {obs.basis} — {obs.citation.quote}".strip(" —"),
            value=obs.value,
            units=obs.units,
            locator=obs.citation.locator,
        )
        n += 1
    for g in report.guidance:
        store.add_evidence(
            run_id,
            tk,
            f"guidance:{g.metric}" if g.metric else "guidance",
            g.citation.source or "(no source)",
            task_id=task_id,
            claim=f"{g.period} guidance issued {g.issued_on}: {g.text}".strip(),
            value=g.high if g.high is not None else g.low,
            units=g.units,
            locator=g.citation.locator,
        )
        n += 1
    return n


async def analyse(
    ticker: str,
    as_of: str | None = None,
    run_id: str | None = None,
    store: RunStore | None = None,
    *,
    max_turns: int | None = None,
) -> FilingsReport:
    """Run the filings analyst for one company.

    With a store and a run_id it is resumable: a completed
    `<TICKER>:filings` task is returned from SQLite instead of re-run, which is
    what makes a crash at minute 40 of a 45-minute window survivable.
    """
    tk = normalise_ticker(ticker)
    target = TARGETS[tk]
    spec = build_spec(tk, as_of, max_turns=max_turns)
    task_prompt = build_task_prompt(tk, as_of)
    key = resume_key(tk)

    task_id: str | None = None
    if store is not None and run_id:
        cached = store.get_output(run_id, key)
        if cached:
            try:
                logger.info("Resuming %s from run store — task already completed", key)
                return FilingsReport.model_validate_json(cached)
            except Exception as e:  # stored row is corrupt; re-run rather than die
                logger.warning("Stored output for %s is unusable (%s); re-running", key, e)
        task_id = store.start_task(
            run_id,
            key,
            kind="analyst",
            agent_name=spec.name,
            model=settings.model_for(spec.resolved_profile),
            input={
                "ticker": tk,
                "as_of": as_of,
                "period_target": target.period,
                "metrics": [m.label for m in target.metrics],
                "max_turns": spec.resolved_max_turns,
                "corpus_freeze": CORPUS_FREEZE,
            },
        )
        store.log(
            run_id,
            "analyst.start",
            task_id=task_id,
            name=spec.name,
            payload={"ticker": tk, "as_of": as_of},
        )

    started = time.monotonic()
    try:
        result = await run_agent(spec, task_prompt)
    except Exception as e:  # run_agent swallows most failures; belt and braces
        logger.exception("Filings analyst crashed for %s", tk)
        if store is not None and task_id:
            store.fail_task(task_id, str(e))
        raise
    elapsed_ms = int((time.monotonic() - started) * 1000)

    report = result if isinstance(result, FilingsReport) else FilingsReport(**result.model_dump())

    # Normalise the two identity fields — the agent sometimes leaves them blank
    # and downstream joins on them.
    if not report.ticker:
        report.ticker = tk
    if not report.period_target:
        report.period_target = target.period

    resolved, unresolved = verify_citations(report)
    if unresolved:
        logger.warning(
            "%s cited %d source(s) that do not exist in the corpus: %s",
            tk,
            len(unresolved),
            ", ".join(unresolved[:5]),
        )

    if store is not None and run_id and task_id:
        rows = _record_evidence(store, run_id, task_id, tk, report)
        store.log(
            run_id,
            "analyst.done",
            task_id=task_id,
            name=spec.name,
            payload={
                "observations": len(report.metrics_history),
                "drivers": len(report.drivers),
                "guidance": len(report.guidance),
                "documents_read": len(report.documents_read),
                "evidence_rows": rows,
                "citations_resolved": len(resolved),
                "citations_unresolved": unresolved,
                "confidence": report.confidence,
            },
            duration_ms=elapsed_ms,
        )
        # An empty report is the fallback instance, not a result. Storing it as
        # 'completed' would make the resume path serve the failure back forever.
        store.finish_task(
            task_id,
            report.model_dump(),
            status="completed" if report.metrics_history else "failed",
            usage=get_last_usage(),
        )

    logger.info(
        "Filings analyst %s done in %.1fs — %d observations, %d drivers, "
        "%d guidance, %d documents",
        tk,
        elapsed_ms / 1000,
        len(report.metrics_history),
        len(report.drivers),
        len(report.guidance),
        len(report.documents_read),
    )
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _w(text: str) -> str:
    """Console-safe: the corpus carries stray non-cp1252 bytes."""
    return text.encode("utf-8", "replace").decode("utf-8")


def _clip(text: str, n: int = 220) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[: n - 1] + "…"


def print_report(report: FilingsReport, *, elapsed: float | None = None) -> None:
    """A readable summary. The JSON file carries everything."""
    t = TARGETS.get(normalise_ticker(report.ticker or "HD"))
    bar = "=" * 78

    print()
    print(bar)
    print(f"FILINGS REPORT  {report.ticker}  target period {report.period_target}")
    print(f"confidence: {report.confidence or '(unset)'}", end="")
    if elapsed is not None:
        print(f"   wall clock: {elapsed:.1f}s", end="")
    print()
    print(bar)

    # --- (a) reported series, grouped by metric ---
    print("\n-- REPORTED SERIES ------------------------------------------------")
    if not report.metrics_history:
        print("   (none)")
    else:
        by_metric: dict[str, list] = {}
        for obs in report.metrics_history:
            by_metric.setdefault(obs.metric or "(unlabelled)", []).append(obs)
        for metric, rows in by_metric.items():
            print(f"\n   {metric}   ({len(rows)} observation(s))")
            for obs in rows:
                val = "None" if obs.value is None else f"{obs.value:,.4g}"
                print(
                    _w(
                        f"      {obs.period:<14} {val:>12} {obs.units:<12} "
                        f"[{obs.basis}]"
                    )
                )
                print(_w(f"         src: {obs.citation.source} @ {obs.citation.locator}"))

    if t:
        found = {o.metric.strip().lower() for o in report.metrics_history}
        missing = [
            m.label for m in t.metrics if m.label.strip().lower() not in found
        ]
        if missing:
            print(f"\n   !! target metrics with no exact-label history: {missing}")

    # --- (c) guidance ---
    print("\n-- COMPANY GUIDANCE -----------------------------------------------")
    if not report.guidance:
        print("   (none)")
    for g in report.guidance:
        rng = ""
        if g.low is not None or g.high is not None:
            rng = f"  [{g.low} .. {g.high} {g.units}]"
        print(_w(f"   * {g.metric} ({g.period}), issued {g.issued_on}{rng}"))
        print(_w(f"     {_clip(g.text)}"))
        print(_w(f"     src: {g.citation.source}"))

    # --- (b) drivers — the causal layer ---
    print("\n-- DRIVERS (management's stated reasons) --------------------------")
    if not report.drivers:
        print("   (none)")
    for d in report.drivers:
        mag = f", {d.magnitude}" if d.magnitude else ""
        print(
            _w(
                f"   * {d.name}  -> {d.affects_metric}  "
                f"[{d.direction}{mag}; {d.persistence}]"
            )
        )
        print(_w(f"     {_clip(d.management_explanation, 300)}"))
        for c in d.citations[:2]:
            print(_w(f"     src: {c.source} @ {c.locator}"))

    # --- (d) seasonality / accounting / risks ---
    for title, body in (
        ("SEASONALITY", report.seasonality),
        ("ACCOUNTING NOTES", report.accounting_notes),
        ("RISKS", report.risks),
        ("NOTES", report.notes),
    ):
        print(f"\n-- {title} " + "-" * max(0, 64 - len(title)))
        print(_w("   " + (_clip(body, 900) or "(none)")))

    # --- documents + citation check ---
    resolved, unresolved = verify_citations(report)
    print("\n-- DOCUMENTS READ " + "-" * 48)
    for name in report.documents_read:
        print(_w(f"   {name}"))
    if not report.documents_read:
        print("   (none reported)")
    print(
        f"\n   citation check: {len(resolved)} of {len(resolved) + len(unresolved)} "
        f"cited filenames resolve to real files under "
        f"{CORPUS_ROOT.relative_to(REPO_ROOT).as_posix()}"
    )
    for bad in unresolved:
        print(_w(f"   !! UNRESOLVED CITATION: {bad}"))
    print()


def _default_json_path(tk: str) -> Path:
    return REPO_ROOT / "runs" / "filings" / f"{tk}.json"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m analysts.filings",
        description=(
            "Filings analyst: read the frozen offline corpus for one company "
            "and emit a FilingsReport."
        ),
    )
    ap.add_argument(
        "--ticker",
        required=True,
        help="HD, ADI, HAS (or LSE:HAS), DE",
    )
    ap.add_argument(
        "--as-of",
        dest="as_of",
        default=None,
        metavar="YYYY-MM-DD",
        help=(
            "Point-in-time cutoff. Filters documents on published_at (never on "
            "period). Omit to use the whole frozen corpus."
        ),
    )
    ap.add_argument(
        "--json",
        dest="json_path",
        default=None,
        metavar="PATH",
        help="Where to write the full report JSON. Default runs/filings/<TICKER>.json",
    )
    ap.add_argument(
        "--max-turns",
        dest="max_turns",
        type=int,
        default=None,
        help=f"Agent turn budget (default {DEFAULT_MAX_TURNS}).",
    )
    ap.add_argument(
        "--run-id",
        dest="run_id",
        default=None,
        help="Reuse an existing run id in the run store (enables resume).",
    )
    ap.add_argument(
        "--db",
        default=None,
        help="Run store SQLite path. Default runs/runs.sqlite.",
    )
    ap.add_argument(
        "--no-store",
        action="store_true",
        help="Skip the run store entirely.",
    )
    ap.add_argument("-v", "--verbose", action="store_true", help="INFO logging.")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    # Corpus text contains characters the Windows console codepage cannot encode.
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
    if not CORPUS_ROOT.exists():
        print(f"error: corpus not found at {CORPUS_ROOT}", file=sys.stderr)
        return 2

    store: RunStore | None = None
    run_id: str | None = args.run_id
    if not args.no_store:
        store = get_store(args.db)
        if not run_id:
            stamp = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}"
            run_id = f"filings_{tk}_{stamp}"
        store.start_run(
            f"filings-analyst {tk}",
            config={
                "ticker": tk,
                "as_of": args.as_of,
                "profile": settings.llm_profile,
                "model": settings.resolved_agent_model,
                "max_turns": args.max_turns or DEFAULT_MAX_TURNS,
            },
            run_id=run_id,
        )

    target = TARGETS[tk]
    print(
        f"Filings analyst: {target.company} [{tk}] target {target.period}  "
        f"as_of={args.as_of or 'none'}  model={settings.resolved_agent_model}  "
        f"max_turns={args.max_turns or DEFAULT_MAX_TURNS}"
    )
    if run_id:
        print(f"run_id={run_id}")

    meter = UsageMeter().attach()
    use_selector_event_loop()
    started = time.monotonic()
    try:
        report = asyncio.run(
            analyse(
                tk,
                as_of=args.as_of,
                run_id=run_id,
                store=store,
                max_turns=args.max_turns,
            )
        )
    except Exception as e:
        print(f"error: filings analyst failed: {type(e).__name__}: {e}", file=sys.stderr)
        if store is not None and run_id:
            store.end_run(run_id, status="failed", notes=str(e)[:2000])
        return 1
    elapsed = time.monotonic() - started

    print_report(report, elapsed=elapsed)
    model = settings.resolved_agent_model
    print(f"-- USAGE " + "-" * 58)
    print(f"   {meter.summary(model)}")
    print()

    out_path = Path(args.json_path) if args.json_path else _default_json_path(tk)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    resolved, unresolved = verify_citations(report)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ticker": tk,
        "as_of": args.as_of,
        "corpus_freeze": CORPUS_FREEZE,
        "model": model,
        "run_id": run_id,
        "elapsed_seconds": round(elapsed, 1),
        "usage": {
            "model_calls": meter.calls,
            "prompt_tokens": meter.prompt_tokens,
            "cached_prompt_tokens": meter.cached_tokens,
            "completion_tokens": meter.completion_tokens,
            "reasoning_tokens": meter.reasoning_tokens,
            "estimated_cost_usd": meter.estimated_cost_usd(model),
        },
        "citation_check": {
            "resolved": resolved,
            "unresolved": unresolved,
        },
        "report": report.model_dump(),
    }
    out_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Full report written to {out_path}")

    if store is not None and run_id:
        store.end_run(run_id, status="completed")
        summary = store.run_summary(run_id)
        print(
            f"Run store: tasks={summary['tasks']} evidence_rows="
            f"{summary['evidence_rows']} db={store.db_path}"
        )

    # Non-zero only if the agent produced nothing usable, so a shell can tell.
    return 0 if report.metrics_history else 1


if __name__ == "__main__":
    raise SystemExit(main())
