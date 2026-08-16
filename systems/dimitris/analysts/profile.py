"""profile.py — the long-term company study. Built once, cached, refreshable.

This is the deep multi-year background the other analysts and the central agent
read. It answers questions no single filing does: how cyclical is this business,
which quarter is strong and why, and -- the one with real forecasting edge --
does management habitually guide low and beat, or miss?

WHY IT IS CACHED. The corpus is frozen at 2026-08-14, so the same inputs always
produce the same study. Recomputing it on every run would waste money and,
worse, waste the 45-minute final window. It is stored in the run store keyed on
(ticker, corpus_freeze, prompt_version) -- so a change to either the corpus or
this prompt invalidates it automatically. A silently stale study would be worse
than no study at all.

Because it runs once and off the clock, this is the right place to spend the
expensive tier: `--profile openai_sol` while the hot path stays on Terra.

    python -m analysts.profile --ticker HD              # cached if present
    python -m analysts.profile --ticker HD --refresh    # rebuild this one
    python -m analysts.profile --all --refresh          # rebuild all four
    python -m analysts.profile --ticker HD --show       # print cache, no LLM
    python -m analysts.profile --list                   # what is cached
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path

from pydantic import BaseModel, Field

from agent_core import get_last_usage, AgentSpec, run_agent, settings, use_selector_event_loop
from corpus.index import TICKERS, load_index
from corpus.tools import CORPUS_TOOLS
from runstore import get_store

logger = logging.getLogger(__name__)

# Bump this whenever PROFILE_PROMPT or ProfileReport changes. It is half the
# cache key, so bumping it is what forces a rebuild.
PROMPT_VERSION = "v1"

COMPANY_NAMES = {
    "HD": "Home Depot",
    "ADI": "Analog Devices",
    "HAS": "Hays plc",
    "DE": "Deere & Company",
}

# What the central agent will eventually have to forecast. The study exists to
# make these three per company predictable, so it is told what they are.
TARGET_METRICS = {
    "HD": "Net sales (USDm); Adjusted diluted EPS (USD/share); Comparable sales, total company (%)",
    "ADI": "Revenue (USDm); Adjusted diluted EPS (USD/share); Adjusted gross margin (%)",
    "HAS": "Net fees (GBPm); Pre-exceptional basic EPS (GBp, pence); Pre-exceptional operating profit (GBPm)",
    "DE": "Worldwide net sales and revenues (USDm); Diluted EPS GAAP (USD/share); Production & Precision Ag operating profit (USDm)",
}


# --------------------------------------------------------------------------
# Contract
# --------------------------------------------------------------------------


class TrendPoint(BaseModel):
    period: str = ""
    metric: str = ""
    value: float | None = None
    units: str = ""
    source: str = ""  # corpus FILENAME


class GuidanceTrackRecord(BaseModel):
    """Guided vs actual for one period. The forecasting edge lives here."""

    period: str = ""
    metric: str = ""
    guided: str = ""
    actual: str = ""
    beat_or_miss: str = ""  # beat | miss | in-line
    magnitude: str = ""
    source: str = ""


class SeasonalPattern(BaseModel):
    metric: str = ""
    pattern: str = ""  # which quarter is strong/weak and by roughly how much
    typical_qoq: str = ""
    typical_yoy: str = ""
    caveats: str = ""  # 53rd weeks, calendar shifts, FX


class ProfileReport(BaseModel):
    """Every field defaulted so the fallback path can always construct one."""

    ticker: str = ""
    company: str = ""
    corpus_freeze: str = ""
    years_covered: str = ""

    business_model: str = ""
    revenue_drivers: str = ""
    cost_structure: str = ""
    segment_mix: str = ""

    long_term_trends: list[TrendPoint] = Field(default_factory=list)
    trajectory_narrative: str = ""

    cyclicality: str = ""
    downturn_behaviour: str = ""
    seasonality: list[SeasonalPattern] = Field(default_factory=list)

    guidance_track_record: list[GuidanceTrackRecord] = Field(default_factory=list)
    guidance_bias: str = ""  # the punchline: habitually conservative, or not

    accounting_basis_notes: str = ""
    capital_allocation: str = ""
    structural_risks: str = ""

    forecasting_implications: str = ""  # how to USE all of the above
    documents_read: list[str] = Field(default_factory=list)
    confidence: str = ""


# --------------------------------------------------------------------------
# Prompt
# --------------------------------------------------------------------------

PROFILE_PROMPT = """You are a long-horizon equity analyst building a durable reference study of one company.

This study is built ONCE and reused. Spend the effort now: later agents rely on it
instead of re-deriving history. Depth and accuracy matter more than speed here.

## COMPANY
{company} ({ticker})

## WHAT THIS STUDY MUST ULTIMATELY SUPPORT
Later agents forecast these reported figures:
{metrics}
Everything you write should make those three more predictable.

## YOUR TOOLS — offline corpus only
1. `list_documents(ticker, doc_kinds, as_of, period_contains, limit)` — browse the index. START HERE.
   Useful kinds: "10k,10q,8k" for filings. Leave blank for transcripts and slides too.
2. `document_outline(document)` — sections with line ranges. Use BEFORE reading, to choose where.
3. `read_document_section(document, section_index)` / `read_document_lines(document, start_line, count)`
4. `search_document(document, query)` — find a line inside one document.
5. `search_corpus(ticker, query, doc_kinds, as_of)` — find WHICH document discusses something.
6. `extract_tables(document, contains)` — financial tables verbatim. **Use this for every number.**
7. `find_numbers(document, query)` — metric-to-number leads. Verify in the document.
8. `delegate_to_subagent(task)` — hand a deep sub-study to a fresh agent. Write a SELF-CONTAINED
   brief; it cannot see your conversation. End it with "call submit_result with the result".

## METHOD
1. `list_documents` for annual filings (10-K / FY) across as many years as exist. Note the range.
2. For each of several years, `extract_tables` the income statement. Build the multi-year series
   for revenue, margins and EPS. **Never read digits out of prose when a table exists.**
3. Read MD&A sections to learn WHY the trajectory bent — not just that it did.
4. Work the guidance track record: find guidance issued in one period, then the actual reported in
   the next, and record both. Do this for as many periods as you can. This is the single most
   valuable output of the study — a company that habitually guides low and beats is predictable in
   a way its consensus is not.
5. Establish seasonality per metric: which quarter is strong, roughly how large the swing, and what
   distorts it (53rd weeks, calendar shifts, FX, acquisitions).
6. Finish with `forecasting_implications`: concrete, usable guidance for a later agent.

## RULES
- **Cite the FILENAME for every figure.** `source_url` is null on 1,027 of 1,139 documents, so the
  filename IS the citation. A number without one is unusable.
- Keep bases strictly apart: reported vs adjusted vs pre-exceptional vs GAAP, and quarterly vs
  annual. These sit side by side in the same filings and are the easiest way to be wrong.
- Hays reports in GBP and its EPS is in PENCE. The others report in USD.
- Where the corpus does not support a claim, say so in `confidence`. Do not invent history.
- Delegate the guidance-track-record sweep if it is long — it is repetitive lookup work and the
  parent only needs the table.
- Call `submit_result` exactly once, when done.

## OUTPUT
Populate every field of the study. `guidance_bias` and `forecasting_implications` are the two the
later agents will read first — make them specific and actionable, not generic."""


def corpus_freeze() -> str:
    """Freeze date from the corpus itself, so the cache key tracks the data."""
    docs = load_index()
    for d in docs:
        text = d.path.read_text(encoding="utf-8", errors="replace")[:1024]
        for line in text.splitlines():
            if line.startswith("corpus_frozen_at:"):
                return line.split(":", 1)[1].strip().strip('"')
        break
    return "unknown"


def build_spec(ticker: str, profile_name: str | None = None) -> AgentSpec:
    ticker = ticker.upper()
    return AgentSpec(
        name=f"{ticker} Long-Term Profile",
        instructions=PROFILE_PROMPT.format(
            company=COMPANY_NAMES.get(ticker, ticker),
            ticker=ticker,
            metrics=TARGET_METRICS.get(ticker, ""),
        ),
        result_model=ProfileReport,
        tools=list(CORPUS_TOOLS),
        use_web=False,  # corpus only — this is history, not news
        allow_delegation=True,
        profile=profile_name,
        max_turns=60,  # deep study; it runs once and off the clock
        fallback=ProfileReport(ticker=ticker, confidence="failed"),
    )


async def build_profile(
    ticker: str,
    *,
    refresh: bool = False,
    profile_name: str | None = None,
    store=None,
    run_id: str | None = None,
) -> tuple[ProfileReport, bool]:
    """Return (study, from_cache). Set refresh=True to force a rebuild."""
    ticker = ticker.upper()
    store = store or get_store()
    freeze = corpus_freeze()

    if not refresh:
        cached = store.get_profile(ticker, freeze, PROMPT_VERSION)
        if cached:
            logger.info("Profile cache HIT for %s (built %s)", ticker, cached["built_at"])
            return ProfileReport.model_validate(cached["payload"]), True

    logger.info("Profile cache MISS for %s — building (this is the slow one)", ticker)
    spec = build_spec(ticker, profile_name)
    own_run = run_id is None
    if own_run:
        run_id = store.start_run(f"profile:{ticker}", config={"prompt_version": PROMPT_VERSION})

    task_id = store.start_task(
        run_id, f"{ticker}:profile", "profile",
        agent_name=spec.name, model=settings.model_for(spec.resolved_profile),
    )
    try:
        report = await run_agent(
            spec,
            f"Build the long-term reference study for {COMPANY_NAMES.get(ticker, ticker)} ({ticker}). "
            f"Use as many years of history as the corpus contains.",
        )
        report.ticker = report.ticker or ticker
        report.company = report.company or COMPANY_NAMES.get(ticker, ticker)
        report.corpus_freeze = freeze

        store.save_profile(
            ticker, freeze, PROMPT_VERSION, report.model_dump(),
            model=settings.model_for(spec.resolved_profile), build_run_id=run_id,
        )
        store.finish_task(task_id, {"documents_read": len(report.documents_read)},
                          usage=get_last_usage())
        return report, False
    except Exception as e:
        store.fail_task(task_id, str(e))
        raise
    finally:
        if own_run:
            store.end_run(run_id)


def summarise(r: ProfileReport) -> str:
    lines = [
        f"=== {r.company} ({r.ticker}) — long-term profile ===",
        f"corpus freeze : {r.corpus_freeze}",
        f"years covered : {r.years_covered}",
        f"documents read: {len(r.documents_read)}",
        f"confidence    : {r.confidence}",
        "",
        f"GUIDANCE BIAS : {r.guidance_bias or '(none)'}",
        f"CYCLICALITY   : {(r.cyclicality or '(none)')[:300]}",
        "",
        f"trend points        : {len(r.long_term_trends)}",
        f"seasonal patterns   : {len(r.seasonality)}",
        f"guidance track rows : {len(r.guidance_track_record)}",
        "",
        "FORECASTING IMPLICATIONS:",
        r.forecasting_implications or "(none)",
    ]
    return "\n".join(lines)


async def _main(args: argparse.Namespace) -> int:
    store = get_store()

    if args.list:
        rows = store.list_profiles()
        if not rows:
            print("No cached profiles.")
            return 0
        print(f"{'ticker':<8}{'freeze':<14}{'ver':<6}{'model':<22}{'built':<28}bytes")
        for r in rows:
            print(
                f"{r['ticker']:<8}{r['corpus_freeze']:<14}{r['prompt_version']:<6}"
                f"{(r['model'] or ''):<22}{r['built_at']:<28}{r['bytes']:,}"
            )
        return 0

    tickers = list(TICKERS) if args.all else [t.strip().upper() for t in args.ticker.split(",")]
    bad = [t for t in tickers if t not in TICKERS]
    if bad:
        print(f"Unknown ticker(s): {', '.join(bad)}. Use {', '.join(TICKERS)}.")
        return 2

    freeze = corpus_freeze()
    for t in tickers:
        if args.show:
            cached = store.get_profile(t, freeze, PROMPT_VERSION)
            if not cached:
                print(f"{t}: no cached profile (freeze={freeze} v={PROMPT_VERSION})")
                continue
            print(summarise(ProfileReport.model_validate(cached["payload"])))
            print()
            continue

        report, from_cache = await build_profile(
            t, refresh=args.refresh, profile_name=args.profile, store=store
        )
        print(f"[{'cache' if from_cache else 'built'}] {t}")
        print(summarise(report))
        print()
        if args.json:
            out = Path(args.json)
            path = out / f"{t}-profile.json" if out.is_dir() or not out.suffix else out
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
            print(f"  wrote {path}")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="Long-term company profile (cached).")
    ap.add_argument("--ticker", default="HD", help="Ticker, or comma-separated list")
    ap.add_argument("--all", action="store_true", help="All four challenge companies")
    ap.add_argument("--refresh", action="store_true", help="Ignore cache and rebuild")
    ap.add_argument("--show", action="store_true", help="Print cached profile, no LLM call")
    ap.add_argument("--list", action="store_true", help="List cached profiles")
    ap.add_argument("--profile", default=None, help="LLM profile, e.g. openai_sol")
    ap.add_argument("--json", default=None, help="Directory or file to write JSON to")
    a = ap.parse_args()

    use_selector_event_loop()
    raise SystemExit(asyncio.run(_main(a)))
