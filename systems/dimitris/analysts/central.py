"""central.py — the central agent: three reports in, twelve numbers out.

The three specialists (filings / news / financials) each research one evidence
base and hand back a structured report. This module turns those three reports
into a `CompanyForecast` — the three numbers for one company — and then checks
its own work.

Three deliberate design choices, each traceable to the rubric:

  1. NO TOOLS. The central agent gets `tools=[], use_web=False`. It reasons over
     the reports rather than re-researching, so the chain "evidence -> report ->
     number" stays auditable ("model quality", 12 pts). If it wants a fact that
     is not in a report, that gap is a finding, not something to paper over with
     a fresh search.

  2. VALIDATE, THEN REPAIR ONCE. `validate_forecast` is a real gate, not a
     decorative one: it catches nulls, wrong labels, wrong units, fractions in
     percentage-point cells, pounds in a pence cell and money off by 1000x.
     `synthesise` feeds the ERRORs back to the agent for exactly one repair
     round-trip and merges per metric ("validation and reliability", 12 pts).

  3. NEVER BLANK. A missing forecast scores the maximum 5.0 penalty. Every path
     out of this module — agent failure, parse failure, timeout — still returns
     a `CompanyForecast` carrying the three canonical labels and units, so the
     only thing that can be missing is the number itself, and that is loudly
     flagged rather than silently dropped.
"""

from __future__ import annotations

import ast
import json
import logging
import math
import re
from difflib import get_close_matches
from pathlib import Path
from typing import Any

from agent_core import AgentSpec, run_agent, settings
from agent_core.truncate import truncate_90_10
from analysts.models import (
    Citation,
    CompanyForecast,
    FilingsReport,
    FinancialsReport,
    MetricForecast,
    NewsReport,
)

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
COMPANIES_JSON = REPO_ROOT / "starter" / "challenge" / "companies.json"


# ---------------------------------------------------------------------------
# 1. The 12 metrics — loaded from challenge/companies.json, never retyped
# ---------------------------------------------------------------------------

# The literal below is a transcription of `starter/challenge/companies.json` and
# exists ONLY so the module still knows the 12 labels if the starter checkout is
# missing. The JSON on disk is the source of truth and overwrites this at import;
# any divergence is logged loudly rather than silently accepted.
_FALLBACK_COMPANY_METRICS: dict[str, dict[str, Any]] = {
    "HD": {
        "company": "Home Depot",
        "ticker": "HD",
        "period": "FY2026Q2",
        "output_file": "HD-FY2026Q2.xlsx",
        "metrics": [
            {"label": "Net sales", "units": "USDm"},
            {"label": "Adjusted diluted EPS", "units": "USD / share"},
            {"label": "Comparable sales, total company", "units": "%"},
        ],
    },
    "ADI": {
        "company": "Analog Devices",
        "ticker": "ADI",
        "period": "FY2026Q3",
        "output_file": "ADI-FY2026Q3.xlsx",
        "metrics": [
            {"label": "Revenue", "units": "USDm"},
            {"label": "Adjusted diluted EPS", "units": "USD / share"},
            {"label": "Adjusted gross margin", "units": "%"},
        ],
    },
    "HAS": {
        "company": "Hays plc",
        "ticker": "LSE:HAS",
        "period": "FY2026",
        "output_file": "HAS-FY2026.xlsx",
        "metrics": [
            {"label": "Net fees", "units": "GBPm"},
            {"label": "Pre-exceptional basic EPS", "units": "GBp"},
            {"label": "Pre-exceptional operating profit", "units": "GBPm"},
        ],
    },
    "DE": {
        "company": "Deere & Company",
        "ticker": "DE",
        "period": "FY2026Q3",
        "output_file": "DE-FY2026Q3.xlsx",
        "metrics": [
            {"label": "Worldwide net sales and revenues", "units": "USDm"},
            {"label": "Diluted EPS (GAAP)", "units": "USD / share"},
            {"label": "Production & Precision Ag operating profit", "units": "USDm"},
        ],
    },
}


def canonical_ticker(ticker: str) -> str:
    """`LSE:HAS`, `HAS.L`, `has` -> `HAS`. The challenge key, not the exchange's."""
    t = (ticker or "").strip().upper()
    if ":" in t:
        t = t.rsplit(":", 1)[-1]
    for suffix in (".L", ".US", ".UK"):
        if t.endswith(suffix):
            t = t[: -len(suffix)]
    return t


def _load_company_metrics() -> dict[str, dict[str, Any]]:
    """Read the challenge file. Fall back to the transcription, complaining."""
    try:
        raw = json.loads(COMPANIES_JSON.read_text(encoding="utf-8"))
        out: dict[str, dict[str, Any]] = {}
        for entry in raw["companies"]:
            key = canonical_ticker(entry["ticker"])
            out[key] = {
                "company": entry.get("company", ""),
                "ticker": entry["ticker"],  # verbatim, incl. the "LSE:" prefix
                "period": entry.get("period", ""),
                "output_file": entry.get("outputFile", ""),
                "metrics": [
                    {"label": m["label"], "units": m["units"]}
                    for m in entry.get("metrics", [])
                ],
            }
        if not out:
            raise ValueError("companies.json listed no companies")
    except Exception as e:  # missing checkout, bad JSON — keep going
        logger.error(
            "Could not read %s (%s). Falling back to the in-module transcription "
            "of the 12 metrics — VERIFY the labels before submitting.",
            COMPANIES_JSON,
            e,
        )
        return _FALLBACK_COMPANY_METRICS

    # Loud on divergence: a silently changed label is a silently thrown metric.
    for tk, spec in out.items():
        ref = _FALLBACK_COMPANY_METRICS.get(tk)
        if ref and [(m["label"], m["units"]) for m in spec["metrics"]] != [
            (m["label"], m["units"]) for m in ref["metrics"]
        ]:
            logger.warning(
                "companies.json metrics for %s differ from the in-module "
                "transcription. The FILE wins; update the transcription.",
                tk,
            )
    return out


COMPANY_METRICS: dict[str, dict[str, Any]] = _load_company_metrics()
TICKERS: tuple[str, ...] = tuple(COMPANY_METRICS)


def metrics_for(ticker: str) -> list[dict[str, str]]:
    """The 3 (label, units) pairs for one company. Empty list if unknown."""
    return list(COMPANY_METRICS.get(canonical_ticker(ticker), {}).get("metrics", []))


def period_for(ticker: str) -> str:
    return COMPANY_METRICS.get(canonical_ticker(ticker), {}).get("period", "")


def output_file_for(ticker: str) -> str:
    return COMPANY_METRICS.get(canonical_ticker(ticker), {}).get("output_file", "")


def company_name_for(ticker: str) -> str:
    return COMPANY_METRICS.get(canonical_ticker(ticker), {}).get("company", "")


# ---------------------------------------------------------------------------
# 2. Sanity guardrails
# ---------------------------------------------------------------------------

# Order-of-magnitude guardrails, NOT forecasts. Each band is several times wider
# than any plausible outcome. They exist to catch a UNITS slip — billions typed
# into a millions cell, a fraction typed into a percentage-point cell, pounds
# typed into a pence cell — not to second-guess the model's judgement.
_MAGNITUDE_BOUNDS: dict[tuple[str, str], tuple[float, float]] = {
    ("HD", "Net sales"): (20_000.0, 90_000.0),
    ("HD", "Adjusted diluted EPS"): (0.5, 15.0),
    ("HD", "Comparable sales, total company"): (-25.0, 25.0),
    ("ADI", "Revenue"): (500.0, 6_000.0),
    ("ADI", "Adjusted diluted EPS"): (0.2, 8.0),
    ("ADI", "Adjusted gross margin"): (30.0, 90.0),
    ("HAS", "Net fees"): (300.0, 1_800.0),
    ("HAS", "Pre-exceptional basic EPS"): (-10.0, 30.0),
    ("HAS", "Pre-exceptional operating profit"): (-100.0, 400.0),
    ("DE", "Worldwide net sales and revenues"): (5_000.0, 25_000.0),
    ("DE", "Diluted EPS (GAAP)"): (-5.0, 20.0),
    ("DE", "Production & Precision Ag operating profit"): (-500.0, 4_000.0),
}

_PERCENT_UNITS = {"pct", "percent", "percentage points", "percentage point", "pp", "ppt"}

ERROR = "ERROR"
WARN = "WARN"

_CONFIDENCE_SCORE = {
    "high": 0.85,
    "medium": 0.60,
    "moderate": 0.60,
    "med": 0.60,
    "low": 0.30,
    "very low": 0.15,
}


# ---------------------------------------------------------------------------
# 3. The system prompt
# ---------------------------------------------------------------------------

# NOTE on braces: the SDK does NOT treat `instructions` as a format string.
# Verified — Agent(instructions="a {b} c").instructions round-trips verbatim. So
# literal braces are written plainly below; doubling them would leak visible "{{"
# into the prompt. `_escape_braces` is retained as a no-op for callers.
#
# The prompt is deliberately split in two. `build_spec` assembles it as
#     PROMPT_HEAD + load_intuition() + PROMPT_TAIL + company_brief(ticker)
# so the human analyst's judgement lands NEAR THE TOP, before any mechanics, and
# the eight-step reasoning follows immediately beneath it. Ordering is not
# cosmetic here: the previous build put a menu of canned methods second and the
# reasoning last, and the agent duly produced canned methods.
PROMPT_HEAD = """You are the CENTRAL FORECASTING AGENT of a multi-agent equity
forecasting system competing in "Agents vs Wall Street". You produce the final
numbers that get submitted. Nothing downstream will second-guess your judgement,
so exercise it.

## WHAT YOU ARE GIVEN

Three specialist analysts have already done the research. Their reports arrive in
your user message:

1. FILINGS ANALYST — read the frozen first-party corpus (10-Q/10-K/8-K, earnings
   call transcripts, slide decks). Reported history, company guidance, management's
   stated drivers, seasonality, accounting basis. This is your PRIMARY evidence.
2. NEWS ANALYST — read post-freeze web reporting through a value-investing lens.
   Demand, cost and margin signals, guidance changes, peer read-across. This is
   your CURRENCY check: what changed after the corpus froze.
3. FINANCIALS ANALYST — a long-horizon financial-statement table. This is your
   SHAPE check: multi-year growth rates, margin trends, seasonal pattern.

Any of the three may be thin, partly empty, or missing entirely. That is a fact
about the run, not an excuse. Say so in `known_weaknesses` and forecast anyway.
"""


PROMPT_TAIL = """
## HOW TO THINK — this is the work; everything after it is bookkeeping

Do not jump from the three reports to a number. Think through all eight of these
for every metric, and let them show in what you write.

1. SEASONALITY. Which quarter is structurally strong or weak for this metric, and
   by how much? Compare to the SAME QUARTER LAST YEAR, never to last quarter — a
   sequential comparison in a seasonal business is close to meaningless. Then ask
   what distorts the comparison: a 53rd week, a calendar or Easter shift, FX
   translation, an acquisition, extra selling days.
2. CYCLICALITY. Where does this business sit in its own cycle, and how has it
   behaved at this point in past cycles? Extremes mean-revert; the most common
   error is extrapolating the last two quarters in a straight line. Watch
   operating leverage — in a downturn revenue falls a little and margin falls a
   lot, and the asymmetry runs both ways.
3. COMPANY TREND. Does the multi-year trajectory actually bend here, or is this
   a wobble around it? Most are wobble.
4. INDUSTRY TREND. Separate market movement from company-specific movement. An
   industry down 8 percent with the company down 5 is a share GAIN inside a
   cyclical decline, and it forecasts very differently from a company problem.
5. LATEST SURPRISES - NEWS. What happened after the corpus froze that no filing
   can know? Ask specifically whether it changes the REPORTED number for THIS
   fiscal period. Most news does not. Weight guidance changes and order-book
   commentary far above sentiment.
6. LATEST SURPRISES - HISTORY. Where did recent prints diverge from their own
   trend, and did management call it one-off or continuing? A "one-off" that has
   recurred three quarters running is not one-off.
7. WHAT HAS REPEATEDLY MATTERED. Every company has two or three recurring swing
   factors that decide beat-or-miss. Find them in the history, not from first
   principles — they are usually what management gets asked about on every call.
8. GUIDANCE BIAS. Does this management habitually guide low and beat, or miss?
   Company guidance is a first-party forecast made by people holding the actual
   data. Respect it, then adjust by the measured historical bias. This step
   carries more edge than any other.

STRUCTURAL vs CYCLICAL is the distinction to keep sharpest. A margin decline from
mix shift persists until mix shifts back; one from underutilisation reverses when
volume returns. MD&A usually tells you which, if you read for it.

## WHAT YOU DO

You have exactly ONE tool: `submit_result`. You cannot search, scrape or read
files, and you must not ask for more data. Your job is JUDGEMENT and ARITHMETIC
over the three reports.

For EACH of the metrics listed under COMPANY BRIEF below you must:

1. RECONCILE the three sources. State plainly which report supplied which input.
2. APPROACH THE NUMBER FROM EVERY ANGLE THE EVIDENCE SUPPORTS, then combine them
   into ONE judgement. You do NOT pick a method and apply it. A real analyst
   comes at a number from several directions at once and forms a combined view.
   Angles usually available: company guidance for the period; prior-year same
   period plus quantified drivers; the seasonal shape applied to a recent base;
   a forecast margin applied to forecast revenue; a build up from segments; a
   named analogue period or peer in the same setup; an observed run rate carried
   forward. Work every angle the evidence can genuinely feed — do not invent one
   the evidence cannot support — and say what each produced.

   Where the angles agree, that is corroboration and the range narrows. Where
   they PULL APART, that disagreement is the most valuable thing you know about
   the metric. Name it. Say which way each angle pulled, how far apart they sit,
   how you weighed them, and what the ANALYST INTUITION section above says about
   which to trust here. A number sitting at the extreme of one angle has to
   justify itself against the angles that disagree with it.
3. WRITE IT UP IN `method` — the combined reasoning, carrying the actual
   arithmetic, so a judge can recompute the number by hand. Name the angles you
   worked, what each gave, how you weighed them, and which of the eight
   considerations above moved the number off its base.

   This is a write-up: "Prior-year Q2 net sales 43,175 x (1 + 0.028 comp + 0.004
   net new store) = 44,556; guidance implies 44,100-45,000; the 5-year Q2
   seasonal uplift on Q1 gives 44,800. Weighted toward guidance because this
   management has beaten its own comp guide in 6 of the last 8 quarters."

   These are not: "Based on the evidence." A bare method name. A single line of
   arithmetic with no statement of what else was considered.
4. STATE ASSUMPTIONS in `assumptions` — at least two per metric, each specific
   enough to be wrong. "FX is a 40bp headwind on reported revenue" is an
   assumption. "Conditions remain stable" is not.
5. GIVE `value`, `low` and `high`. `value` is your point estimate. `low`/`high`
   bracket roughly an 80 percent interval — the range you would actually defend,
   not a decoration. `low` <= `value` <= `high` always.
6. CITE in `evidence`. Every number needs at least one citation carried through
   from the analyst reports: the corpus FILENAME (most corpus documents have a
   null source_url, so the filename IS the citation) or the URL, plus a locator
   and a short verbatim quote.
7. RESOLVE CONFLICTS in `conflicts_resolved`. Where the reports disagreed, name
   the winner AND the reason. If they agreed, say "no conflict" and say what you
   cross-checked.

## SOURCE PRECEDENCE (default order; override it consciously and say why)

1. Company guidance for the exact target period, most recent issuance.
2. Reported figures from filings for the comparable prior periods.
3. The financial-statement table, for multi-year growth and margin shape.
4. Post-freeze news — which OUTRANKS stale guidance when it reports a genuine
   change (a warning, an acquisition, a demand inflection). Say so when it does.
5. Your own prior. Last resort. Label it as such.

A single high-quality primary source beats three secondary sources repeating each
other. Secondary sources that merely echo the same wire story are ONE source.

## UNITS — READ TWICE

A unit error throws the metric exactly as completely as a blank cell does.

- PERCENTAGES ARE IN POINTS. 4.5 means 4.5 percent. NOT 0.045. NOT 450.
- HAYS EPS IS IN PENCE (GBp). 6.2 means 6.2 pence. NOT 0.062 pounds, NOT 620.
- MONEY IS IN MILLIONS. USDm and GBPm. Home Depot quarterly net sales is a
  ~45,000 number, not 45 and not 45,000,000,000.
- Hays reports in GBP. The other three report in USD.
- Match the BASIS exactly: "Adjusted diluted EPS" is not GAAP diluted EPS;
  "Diluted EPS (GAAP)" is not adjusted; "Pre-exceptional basic EPS" is neither
  diluted nor statutory. Deere's Production & Precision Ag operating profit is a
  SEGMENT figure, not the group figure.
- Copy `label` and `units` EXACTLY as given in the COMPANY BRIEF, character for
  character. They key the submission workbook.

## NEVER LEAVE A METRIC BLANK

Scoring is: metric score = min(5.0, your miss / Wall Street's miss). A MISSING
forecast scores the maximum 5.0. Even a crude number anchored on the prior-year
figure almost always scores less than 5.0. So:

- `value` is NEVER null. Not once, not for any metric, whatever the evidence.
- If the evidence is thin, use the weakest defensible anchor (last reported
  period, or the multi-year average), widen `low`/`high`, set `confidence` to
  "low", and say exactly what you lacked in `known_weaknesses`.
- Matching consensus scores about 1.0, which does not win. Where you have a
  specific, evidenced reason to sit away from the obvious consensus, do it and
  justify it in `method`. Where you do not, do not manufacture one.

## OUTPUT

Call `submit_result` exactly once, with:

- `ticker`, `period` — copy from the COMPANY BRIEF exactly.
- `metrics` — one entry per brief metric, in the brief's order, with `label`,
  `units`, `value`, `low`, `high`, `method`, `assumptions`, `drivers_applied`,
  `evidence`, `conflicts_resolved`, `confidence` (high | medium | low).
- `reconciliation` — how the three reports were combined overall, and which one
  drove which metric.
- `sanity_checks` — for each metric, three things: (a) the implied year-on-year
  change in percent and one sentence on why that is plausible; (b) the value
  restated in words so the SCALE is unambiguous — "net sales 47,250 is USD
  millions, i.e. USD 47.25 billion", "comparable sales 2.4 is 2.4 percentage
  points, not 0.024", "EPS 6.2 is 6.2 pence, not GBP 6.20"; (c) confirmation
  that the value sits inside the expected magnitude band from the COMPANY BRIEF.
- `known_weaknesses` — what you could not verify, which report was thin, where
  you are most likely to be wrong, and what would change your mind. Honesty is
  scored; a flawless-sounding write-up that hides a hole scores worse.
- `confidence` — overall.

Example shape:
submit_result(result={"ticker": "XX", "period": "FY2026Q2", "metrics": [ ... ]})

## WORKFLOW

1. Read all three reports before writing anything.
2. For each metric: work the eight considerations under HOW TO THINK, find every
   relevant figure across the three reports, note where they conflict, work every
   angle the evidence supports, weigh them into one number, sanity-check the
   magnitude and the units, then write it up.
3. Re-read your own numbers against the UNITS section. Percentages in points.
   Pence for Hays EPS. Millions for money. Check each value against the expected
   magnitude printed beside it in the COMPANY BRIEF.
4. Sense-check every figure against the same quarter last year. If you cannot
   explain the delta in one sentence, the number is probably wrong.
5. Confirm no `value` is null. A missing forecast scores the maximum penalty —
   a reasoned estimate always beats a blank.
6. Call `submit_result` exactly once. Do not call any other tool afterwards.
"""

# Kept for callers and tests that want the prompt without the intuition layer.
# `build_spec` assembles the real thing with `load_intuition()` in between.
SYSTEM_PROMPT = PROMPT_HEAD + PROMPT_TAIL

INTUITION_PATH = REPO_ROOT / "analysts" / "intuition.md"


def load_intuition() -> str:
    """The human judgement layer, injected verbatim into the prompt.

    A file rather than a constant on purpose: it is the part most likely to be
    revised late, and a judge can read it as evidence of deliberate method
    rather than a model guessing. Absent or empty file is fine — the prompt's
    own eight-step framework still stands.
    """
    try:
        text = INTUITION_PATH.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        logger.info("No intuition file at %s — using the built-in framework only.", INTUITION_PATH)
        return ""
    except Exception as e:
        logger.warning("Could not read %s: %s", INTUITION_PATH, e)
        return ""
    if not text:
        return ""
    return (
        "\n\n## ANALYST INTUITION — THIS IS THE METHOD\n\n"
        "What follows is the human analyst's own judgement, recorded by hand over "
        "years of doing this work. It is not background reading and it is not a "
        "tie-breaker. It is how this forecast is to be made.\n\n"
        "Everything after this section — the reasoning steps, the units rules, the "
        "output contract — is mechanics in service of it. Where anything below "
        "conflicts with this section, THIS SECTION WINS, and say in `method` that "
        "it did.\n\n" + text
    )


def _escape_braces(text: str) -> str:
    """No-op. Kept so callers need not change.

    This used to double every brace, on the belief that the Agents SDK treats
    `instructions` as a format string. It does not -- verified empirically:

        Agent(instructions="literal {brace} and {{doubled}}").instructions
        -> 'literal {brace} and {{doubled}}'

    Instructions are stored verbatim. Escaping therefore CORRUPTED the prompt,
    showing the model doubled braces wherever dynamic content contained JSON or
    a dict literal. Prompts here are written brace-safe instead.
    """
    return text


def company_brief(ticker: str) -> str:
    """The per-company block appended to the system prompt."""
    tk = canonical_ticker(ticker)
    spec = COMPANY_METRICS.get(tk)
    if not spec:
        return _escape_braces(
            f"\n## COMPANY BRIEF\n\nUnknown ticker {ticker!r}. "
            f"Known: {', '.join(TICKERS)}.\n"
        )
    # The magnitude band is the same dict the validator checks against
    # (`_MAGNITUDE_BOUNDS`). Printing it here is the whole point: previously it
    # existed only in code, so a scale slip was caught AFTER the fact and cost a
    # repair round-trip. Telling the agent the expected order of magnitude up
    # front prevents the error instead of diagnosing it.
    lines: list[str] = []
    for i, m in enumerate(spec["metrics"]):
        row = f"{i + 1}. label: {m['label']}\n   units: {m['units']}"
        bounds = _MAGNITUDE_BOUNDS.get((tk, m["label"]))
        if bounds:
            lo_b, hi_b = bounds
            row += (
                f"\n   expected magnitude: a number between {lo_b:,g} and {hi_b:,g}. "
                f"Anything outside that band is a units slip, not a bold call."
            )
        lines.append(row)
    rows = "\n".join(lines)
    text = (
        "\n## COMPANY BRIEF\n\n"
        f"Company: {spec['company']}\n"
        f"Ticker (submit this exact string): {tk}\n"
        f"Fiscal period (submit this exact string): {spec['period']}\n"
        f"Submission workbook: {spec['output_file']}\n\n"
        "The metrics you must forecast, with the EXACT label and units to copy:\n\n"
        f"{rows}\n\n"
        "Submit exactly these three metrics, in this order, with these labels and "
        "these units, character for character.\n\n"
        "Before you call `submit_result`, check each `value` against its expected "
        "magnitude above. That check costs you nothing and catches the single "
        "most expensive error available to you.\n"
    )
    return _escape_braces(text)


# ---------------------------------------------------------------------------
# 4. Rendering the three reports into the user message
# ---------------------------------------------------------------------------

# Raised from 45k when the filings analyst was asked for multi-year drivers plus
# business_factors and cyclical_behaviour. Those two prose fields sit AFTER the
# long `drivers` list in field order, and `truncate_90_10` keeps head and tail —
# so a binding cap would silently eat exactly the new material. Central spent
# only ~14.6k prompt tokens on the previous run, so the headroom is affordable.
FILINGS_CHARS = 70_000
NEWS_CHARS = 25_000
FINANCIALS_CHARS = 20_000
PRIOR_ATTEMPT_CHARS = 12_000


def _is_empty(v: Any) -> bool:
    return v is None or v == "" or v == [] or v == {}


def _fmt_block(obj: Any, indent: int = 0) -> list[str]:
    """Render a model dump as compact indented text, dropping empty fields.

    JSON would work but spends a third of its characters on punctuation, and the
    reports can be large. Empty fields are dropped so the agent's attention is
    not diluted by forty blank strings.
    """
    pad = "  " * indent
    lines: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if _is_empty(v):
                continue
            if isinstance(v, (dict, list)):
                lines.append(f"{pad}{k}:")
                lines.extend(_fmt_block(v, indent + 1))
            else:
                lines.append(f"{pad}{k}: {v}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            if _is_empty(v):
                continue
            if isinstance(v, (dict, list)):
                lines.append(f"{pad}- #{i + 1}")
                lines.extend(_fmt_block(v, indent + 1))
            else:
                lines.append(f"{pad}- {v}")
    else:
        lines.append(f"{pad}{obj}")
    return lines


def render_report(title: str, report: Any, cap: int) -> str:
    """One report as a bounded, human-readable section."""
    if report is None:
        return f"### {title}\n\n(NOT AVAILABLE — this analyst did not run.)\n"
    try:
        dump = report.model_dump(mode="json")
    except AttributeError:
        dump = report
    body = "\n".join(_fmt_block(dump))
    if not body.strip():
        return (
            f"### {title}\n\n(EMPTY — the analyst ran but returned no content. "
            f"Treat this evidence base as unavailable and say so in "
            f"known_weaknesses.)\n"
        )
    return f"### {title}\n\n{truncate_90_10(body, cap)}\n"


def build_user_message(
    ticker: str,
    filings: FilingsReport | None,
    news: NewsReport | None,
    financials: FinancialsReport | None,
    *,
    as_of: str = "",
) -> str:
    tk = canonical_ticker(ticker)
    spec = COMPANY_METRICS.get(tk, {})
    header = (
        f"# FORECAST BRIEF — {spec.get('company', tk)} ({tk}), "
        f"{spec.get('period', '')}\n"
    )
    if as_of:
        header += f"\nAs-of date for this run: {as_of}\n"
    header += (
        "\nBelow are the three analyst reports. Reason over them and submit the "
        "three forecasts. You have no tools other than submit_result.\n\n"
    )
    parts = [
        header,
        render_report("REPORT 1 of 3 — FILINGS ANALYST (offline corpus)", filings, FILINGS_CHARS),
        render_report("REPORT 2 of 3 — NEWS ANALYST (post-freeze web)", news, NEWS_CHARS),
        render_report(
            "REPORT 3 of 3 — FINANCIALS ANALYST (statement history)",
            financials,
            FINANCIALS_CHARS,
        ),
        "\n### YOUR TASK\n\n"
        + "\n".join(
            f"- {m['label']}  [{m['units']}]" for m in metrics_for(tk)
        )
        + "\n\nProduce all of the above. No nulls. Percentages in points, "
        "money in millions, Hays EPS in pence. Call submit_result once.\n",
    ]
    return "\n".join(parts)


def build_repair_message(
    ticker: str,
    previous: CompanyForecast,
    problems: list[str],
    base_message: str,
) -> str:
    """Second and final round-trip: the same brief plus the validator's verdict."""
    prev = truncate_90_10(previous.model_dump_json(indent=1), PRIOR_ATTEMPT_CHARS)
    listed = "\n".join(f"- {p}" for p in problems)
    return (
        base_message
        + "\n\n---\n\n### AUTOMATED VALIDATION FAILED ON YOUR PREVIOUS SUBMISSION\n\n"
        "You already submitted the forecast below. A post-validation pass rejected "
        "it. Fix EVERY problem listed and submit the corrected forecast. Keep "
        "everything that was not flagged; do not rewrite what was already fine.\n\n"
        "PROBLEMS FOUND:\n"
        f"{listed}\n\n"
        "YOUR PREVIOUS SUBMISSION:\n"
        f"```\n{prev}\n```\n\n"
        "Re-check units before you resubmit: percentages in POINTS (4.5 = 4.5%), "
        "Hays EPS in PENCE, money in MILLIONS. No value may be null.\n"
    )


# ---------------------------------------------------------------------------
# 5. The spec
# ---------------------------------------------------------------------------


def skeleton_forecast(ticker: str) -> CompanyForecast:
    """A valid, correctly-labelled, value-less forecast.

    Used as the AgentSpec fallback. If the model dies outright we still emit the
    right three labels and units — so the failure shows up as three flagged nulls
    in validation rather than as an empty file at 17:50.
    """
    tk = canonical_ticker(ticker)
    return CompanyForecast(
        ticker=tk,
        period=period_for(tk),
        metrics=[
            MetricForecast(
                label=m["label"],
                units=m["units"],
                method="",
                confidence="",
            )
            for m in metrics_for(tk)
        ],
        reconciliation="",
        sanity_checks="",
        known_weaknesses=(
            "The central agent produced no usable output; this is the fallback "
            "skeleton. Every value is missing."
        ),
        confidence="none",
    )


def build_spec(ticker: str) -> AgentSpec:
    """The central agent's spec. No tools but the terminal one, on purpose."""
    tk = canonical_ticker(ticker)
    return AgentSpec(
        name=f"Central Forecaster ({tk})",
        # Order matters: the analyst's own judgement lands immediately after the
        # brief on what it has been given, and BEFORE any of the mechanics.
        instructions=(
            PROMPT_HEAD + load_intuition() + PROMPT_TAIL + company_brief(tk)
        ),
        result_model=CompanyForecast,
        tools=[],           # reasoning over the reports, not fresh research
        use_web=False,      # ditto — keeps the evidence chain auditable
        allow_delegation=False,
        fallback=skeleton_forecast(tk),
    )


# ---------------------------------------------------------------------------
# 6. Canonicalisation — make the output well-formed without hiding errors
# ---------------------------------------------------------------------------


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def canonicalise(
    ticker: str, forecast: CompanyForecast
) -> tuple[CompanyForecast, list[str]]:
    """Force the labels/units/ticker/period to the challenge's exact strings.

    Returns the repaired forecast and the list of repairs made. A repair is
    always REPORTED, never silent: rewriting units from "USD bn" to "USDm"
    makes the workbook well-formed but does NOT make a 45.5 into a 45,500, and
    that risk has to reach a human.
    """
    tk = canonical_ticker(ticker)
    expected = metrics_for(tk)
    notes: list[str] = []

    if not expected:
        notes.append(f"{WARN}: unknown ticker {ticker!r} — no canonical metric list.")
        return forecast, notes

    by_norm = {_norm(m["label"]): m for m in expected}
    submitted = list(forecast.metrics or [])
    matched: dict[str, MetricForecast] = {}
    leftovers: list[MetricForecast] = []

    for mf in submitted:
        key = _norm(mf.label)
        hit = by_norm.get(key)
        if hit is None:
            close = get_close_matches(key, list(by_norm), n=1, cutoff=0.72)
            if close:
                hit = by_norm[close[0]]
                notes.append(
                    f"{WARN}: metric label {mf.label!r} did not match the challenge "
                    f"label; matched to {hit['label']!r} by similarity."
                )
        if hit is None:
            leftovers.append(mf)
            continue
        if hit["label"] in matched:
            notes.append(
                f"{WARN}: duplicate submission for {hit['label']!r}; kept the first."
            )
            continue
        if mf.label != hit["label"]:
            notes.append(
                f"{WARN}: label rewritten {mf.label!r} -> {hit['label']!r} "
                f"(the workbook keys on the exact string)."
            )
        if _norm(mf.units) != _norm(hit["units"]):
            notes.append(
                f"{ERROR}: {hit['label']!r} submitted units {mf.units!r}, expected "
                f"{hit['units']!r}. Units rewritten so the workbook is well-formed "
                f"— CHECK THE SCALE OF THE VALUE, it may be off by a factor."
            )
        matched[hit["label"]] = mf.model_copy(
            update={"label": hit["label"], "units": hit["units"]}
        )

    for mf in leftovers:
        notes.append(
            f"{WARN}: dropped unrecognised metric {mf.label!r} (value={mf.value}) "
            f"— not one of the {len(expected)} challenge metrics."
        )

    ordered: list[MetricForecast] = []
    for m in expected:
        if m["label"] in matched:
            ordered.append(matched[m["label"]])
        else:
            notes.append(
                f"{ERROR}: metric {m['label']!r} was never submitted; inserted an "
                f"empty placeholder. A missing forecast scores the maximum 5.0."
            )
            ordered.append(MetricForecast(label=m["label"], units=m["units"]))

    expected_period = period_for(tk)
    if canonical_ticker(forecast.ticker) != tk:
        notes.append(
            f"{WARN}: ticker rewritten {forecast.ticker!r} -> {tk!r}."
        )
    if expected_period and _norm(forecast.period) != _norm(expected_period):
        notes.append(
            f"{ERROR}: period {forecast.period!r} is not the target period "
            f"{expected_period!r}. Rewritten — verify the forecast is for the "
            f"right fiscal period, not merely labelled as it."
        )

    return (
        forecast.model_copy(
            update={"ticker": tk, "period": expected_period or forecast.period,
                    "metrics": ordered}
        ),
        notes,
    )


# ---------------------------------------------------------------------------
# 7. Validation — the 12-point criterion, made real
# ---------------------------------------------------------------------------


def _finite(x: float | None) -> bool:
    return x is not None and math.isfinite(x)


# ---------------------------------------------------------------------------
# Arithmetic re-check
#
# The prompt requires the derivation to carry its own numbers so a judge can
# recompute the forecast by hand. Nothing verified that the numbers actually
# compute: a model can write arithmetic that reads plausibly and does not
# produce its own answer, and the digit-presence check above cannot see it.
#
# Everything here is best effort and WARN-only. Prose is not a formula language,
# so a derivation that does not parse is skipped silently rather than penalised.
# ---------------------------------------------------------------------------

# Deliberately no eval(). The expression is walked node by node over a closed
# set of arithmetic operations, so nothing outside this list can execute.
def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("non-numeric constant")
        return float(node.value)
    if isinstance(node, ast.UnaryOp):
        v = _eval_node(node.operand)
        if isinstance(node.op, ast.USub):
            return -v
        if isinstance(node.op, ast.UAdd):
            return v
        raise ValueError("unsupported unary op")
    if isinstance(node, ast.BinOp):
        a, b = _eval_node(node.left), _eval_node(node.right)
        op = node.op
        if isinstance(op, ast.Add):
            return a + b
        if isinstance(op, ast.Sub):
            return a - b
        if isinstance(op, ast.Mult):
            return a * b
        if isinstance(op, ast.Div):
            if b == 0:
                raise ValueError("division by zero")
            return a / b
        if isinstance(op, ast.Pow):
            if abs(b) > 8:  # no accidental 10**10000 while validating a forecast
                raise ValueError("exponent too large")
            return a**b
        raise ValueError("unsupported binary op")
    raise ValueError(f"unsupported node {type(node).__name__}")


def _normalise_expression(raw: str) -> str:
    """Turn a written derivation fragment into something `ast` can parse."""
    s = raw
    s = re.sub(r"[$£€]", "", s)
    s = s.replace("−", "-").replace("–", "-")  # minus sign, en dash
    s = s.replace("×", "*").replace("÷", "/")
    s = re.sub(r"(?<=[\d\s\)])[xX](?=[\s\(\d])", "*", s)  # "43,175 x 1.02"
    s = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", s)  # thousands separators only
    s = re.sub(r"(\d+(?:\.\d+)?)\s*%", r"(\1/100)", s)  # 5.7% -> (5.7/100)
    s = re.sub(r"\bbps?\b", "", s, flags=re.I)
    return s.strip()


def _safe_eval(expr: str) -> float | None:
    expr = expr.strip().rstrip("=+-*/ ")
    if not expr or not re.search(r"\d", expr):
        return None
    if expr.count("(") != expr.count(")"):
        return None
    try:
        tree = ast.parse(expr, mode="eval")
        value = _eval_node(tree.body)
    except (SyntaxError, ValueError, RecursionError, OverflowError, TypeError):
        return None
    return value if math.isfinite(value) else None


_EQUATION = re.compile(
    r"([\d][\d\s\.,\+\-\*\/\(\)%xX×÷$£€−–]*?)\s*=\s*([\-−–]?[\d,]+(?:\.\d+)?)"
)


def _num(x: float) -> str:
    """Readable number. `{:,.4g}` renders 44556 as 4.456e+04, which is actively
    misleading in a message about magnitude errors."""
    if x == int(x) and abs(x) < 1e15:
        return f"{int(x):,}"
    return f"{x:,.4f}".rstrip("0").rstrip(".")


def check_method_arithmetic(
    tag: str, method: str, value: float, tolerance: float = 0.01
) -> list[str]:
    """Recompute any `<expression> = <result>` written into the derivation.

    Two questions, both WARN-only:
      1. Does each stated sum actually equal what it claims?
      2. Does the derivation arrive anywhere near the submitted `value`?
    """
    problems: list[str] = []
    landings: list[float] = []

    for lhs_raw, rhs_raw in _EQUATION.findall(method):
        rhs = _safe_eval(_normalise_expression(rhs_raw))
        if rhs is None:
            continue
        landings.append(rhs)
        lhs = _safe_eval(_normalise_expression(lhs_raw))
        if lhs is None:
            continue  # unparseable prose, not a wrong sum
        # A ratio written as a percentage is correct, not a mistake: the agent
        # writes "12,018/12,763 = 94.2" meaning 94.2 PER CENT. Comparing the raw
        # quotient to it would flag every margin and every growth rate.
        if abs(lhs * 100.0 - rhs) / max(abs(rhs), 1e-9) <= tolerance:
            landings[-1] = lhs * 100.0
            continue
        scale = max(abs(rhs), abs(lhs), 1e-9)
        if abs(lhs - rhs) / scale > tolerance:
            problems.append(
                f"{WARN}: {tag}: the derivation states "
                f"'{lhs_raw.strip()} = {rhs_raw.strip()}', but that computes to "
                f"{_num(lhs)}. The stated arithmetic does not produce its own "
                f"result — check the sum or the wording."
            )

    # Does the derivation arrive at the submitted number? Deliberately lenient.
    # The agent is instructed to weigh several angles into one judgement, so the
    # final value legitimately need not equal any single line — a cross-check
    # ratio, a peer figure and a guidance midpoint all appear as sums too. Only
    # complain when a sum sits on the SAME SCALE as the value (so it was plainly
    # meant to be the answer) and still misses it.
    if landings and _finite(value) and abs(value) > 1e-9:
        same_scale = [x for x in landings if 0.5 <= abs(x) / abs(value) <= 2.0]
        if same_scale and all(
            abs(x - value) / abs(value) > tolerance for x in same_scale
        ):
            near = min(same_scale, key=lambda x: abs(x - value))
            problems.append(
                f"{WARN}: {tag}: the derivation computes {_num(near)} but the "
                f"submitted value is {_num(value)}. Either the write-up is missing "
                f"the step that closes that gap, or the number came from somewhere "
                f"the derivation does not show."
            )
    return problems


def validate_metric(
    ticker: str,
    mf: MetricForecast,
    expected: dict[str, str],
    reference: float | None = None,
) -> list[str]:
    """Problems with ONE metric. Prefixed ERROR (must fix) or WARN (look at it)."""
    tk = canonical_ticker(ticker)
    label = expected["label"]
    units = expected["units"]
    p: list[str] = []
    tag = f"{tk} / {label}"

    # --- identity -----------------------------------------------------------
    if mf.label != label:
        p.append(f"{ERROR}: {tag}: label is {mf.label!r}, must be {label!r}.")
    if mf.units != units:
        p.append(f"{ERROR}: {tag}: units are {mf.units!r}, must be {units!r}.")

    # --- the number itself --------------------------------------------------
    v = mf.value
    if v is None:
        p.append(
            f"{ERROR}: {tag}: NO FORECAST. A missing value scores the maximum 5.0 "
            f"penalty — the worst possible outcome for this metric."
        )
        return p
    if not math.isfinite(v):
        p.append(f"{ERROR}: {tag}: value is {v!r}, not a finite number.")
        return p

    is_pct = units.strip() == "%" or _norm(units) in _PERCENT_UNITS
    is_pence = units.strip() == "GBp"
    is_eps = "eps" in _norm(label) or "share" in _norm(units)
    is_money = _norm(units) in {"usdm", "gbpm"}

    # --- interval -----------------------------------------------------------
    if not _finite(mf.low) or not _finite(mf.high):
        p.append(
            f"{WARN}: {tag}: no low/high interval (low={mf.low}, high={mf.high}). "
            f"An unbracketed point estimate hides its own uncertainty."
        )
    else:
        lo, hi = float(mf.low), float(mf.high)  # type: ignore[arg-type]
        if lo > hi:
            p.append(f"{ERROR}: {tag}: low {lo} is above high {hi}.")
            lo, hi = hi, lo
        if not (lo <= v <= hi):
            p.append(
                f"{ERROR}: {tag}: point estimate {v} sits outside its own interval "
                f"[{lo}, {hi}]."
            )
        if lo == hi:
            p.append(f"{WARN}: {tag}: low equals high — that is not an interval.")
        else:
            width = hi - lo
            if is_pct and width > 12.0:
                p.append(
                    f"{WARN}: {tag}: interval spans {width:.1f} percentage points "
                    f"— too wide to be informative."
                )
            elif not is_pct and v != 0 and width / abs(v) > 0.5:
                p.append(
                    f"{WARN}: {tag}: interval is {width / abs(v) * 100:.0f}% of the "
                    f"point estimate — too wide to be informative."
                )

    # --- unit discipline ----------------------------------------------------
    if is_pct:
        if abs(v) < 1.0 and v != 0.0:
            p.append(
                f"{WARN}: {tag}: value {v} is below 1 for a PERCENTAGE metric. "
                f"Percentages must be in POINTS — {v} means {v}%, so if you meant "
                f"{v * 100:.1f}% this is wrong by 100x. Confirm which it is."
            )
        if abs(v) > 100.0:
            p.append(
                f"{ERROR}: {tag}: value {v} exceeds 100 percentage points — this "
                f"looks like a fraction multiplied out, or basis points."
            )
    if tk == "HAS" and is_pence:
        if 0 < abs(v) < 0.5:
            p.append(
                f"{ERROR}: {tag}: {v} in a PENCE (GBp) cell looks like POUNDS. "
                f"{v} pounds is {v * 100:.1f}p — Hays EPS is quoted in pence."
            )
        elif 0.5 <= abs(v) < 1.0:
            p.append(
                f"{WARN}: {tag}: {v}p is very low for Hays. Confirm this is pence "
                f"and not pounds ({v * 100:.0f}p)."
            )
    # (A pounds-for-pence UNITS string is caught in `canonicalise`, which sees
    # the agent's original string before it is rewritten to the workbook's.)

    # --- magnitude ----------------------------------------------------------
    bounds = _MAGNITUDE_BOUNDS.get((tk, label))
    if bounds:
        lo_b, hi_b = bounds
        if not (lo_b <= v <= hi_b):
            hint = ""
            if is_money and v != 0:
                if v < lo_b and lo_b / max(abs(v), 1e-9) > 100:
                    hint = " Looks like BILLIONS in a millions cell (multiply by 1000)."
                elif v > hi_b and abs(v) / hi_b > 100:
                    hint = " Looks like units/thousands in a millions cell (divide by 1000)."
            p.append(
                f"{ERROR}: {tag}: value {v:,g} {units} is outside the sanity band "
                f"[{lo_b:,g}, {hi_b:,g}] — this is an order-of-magnitude guardrail, "
                f"so breaching it means a units slip or a serious mis-estimate.{hint}"
            )
    elif is_money and abs(v) >= 1_000_000:
        p.append(
            f"{ERROR}: {tag}: value {v:,g} in a MILLIONS cell — that is "
            f"{v / 1_000_000:,.1f} trillion. Money is in millions."
        )
    elif is_eps and abs(v) > 100:
        p.append(f"{ERROR}: {tag}: EPS of {v} is implausible for a per-share figure.")

    # --- consistency against the actuals we were given ----------------------
    if reference is not None and reference != 0 and not is_pct:
        ratio = v / reference
        if ratio <= 0:
            p.append(
                f"{WARN}: {tag}: forecast {v:,g} has the opposite sign to the latest "
                f"reported comparable ({reference:,g}). Deliberate?"
            )
        elif ratio > 2.0 or ratio < 0.5:
            p.append(
                f"{WARN}: {tag}: forecast {v:,g} is {ratio:.2f}x the latest reported "
                f"comparable ({reference:,g}). Justify the step change or check units."
            )

    # --- auditability (this is what "model quality" is scored on) -----------
    method = (mf.method or "").strip()
    if not method:
        p.append(f"{ERROR}: {tag}: no method stated — the number is unauditable.")
    else:
        if len(method) < 40:
            p.append(f"{WARN}: {tag}: method is only {len(method)} chars — too thin to follow.")
        if not re.search(r"\d", method):
            p.append(
                f"{WARN}: {tag}: method contains no arithmetic. A judge must be able "
                f"to recompute the number by hand."
            )
        p.extend(check_method_arithmetic(tag, method, v))
    if not mf.assumptions:
        p.append(f"{WARN}: {tag}: no assumptions listed.")
    if not mf.evidence:
        p.append(f"{WARN}: {tag}: no citations — the evidence trail stops here.")
    else:
        blank = [i for i, c in enumerate(mf.evidence) if not (c.source or "").strip()]
        if blank:
            p.append(
                f"{WARN}: {tag}: {len(blank)} citation(s) with an empty source. "
                f"Cite the corpus FILENAME or the URL."
            )
    if not (mf.confidence or "").strip():
        p.append(f"{WARN}: {tag}: no confidence stated.")

    return p


def validate_forecast(
    ticker: str,
    forecast: CompanyForecast,
    *,
    reference: dict[str, float] | None = None,
) -> list[str]:
    """Every problem with one company's forecast, worst first.

    Each line is prefixed ERROR (would throw the metric, or make it unauditable)
    or WARN (worth a human's eye). `reference` optionally maps a metric label to
    the latest reported comparable, which turns on a step-change check.

    This is deliberately paranoid. Of the twelve ways to lose a metric, eleven
    are mechanical — a null, a fraction where points were wanted, pounds where
    pence were wanted, a label the workbook does not recognise — and all eleven
    are catchable here for free.
    """
    tk = canonical_ticker(ticker)
    expected = metrics_for(tk)
    problems: list[str] = []

    if not expected:
        return [f"{ERROR}: unknown ticker {ticker!r}; expected one of {', '.join(TICKERS)}."]

    if canonical_ticker(forecast.ticker) != tk:
        problems.append(
            f"{ERROR}: forecast ticker is {forecast.ticker!r}, expected {tk!r}."
        )
    expected_period = period_for(tk)
    if expected_period and _norm(forecast.period) != _norm(expected_period):
        problems.append(
            f"{ERROR}: forecast period is {forecast.period!r}, expected "
            f"{expected_period!r}."
        )

    submitted = list(forecast.metrics or [])
    by_label = {m.label: m for m in submitted}
    seen: set[str] = set()
    for m in submitted:
        if m.label in seen:
            problems.append(f"{ERROR}: metric {m.label!r} submitted more than once.")
        seen.add(m.label)

    wanted = {m["label"] for m in expected}
    for extra in sorted(seen - wanted):
        problems.append(
            f"{ERROR}: metric {extra!r} is not one of this company's metrics "
            f"({', '.join(sorted(wanted))})."
        )

    for exp in expected:
        mf = by_label.get(exp["label"])
        if mf is None:
            problems.append(
                f"{ERROR}: metric {exp['label']!r} [{exp['units']}] is MISSING. "
                f"A missing forecast scores the maximum 5.0 penalty."
            )
            continue
        problems.extend(
            validate_metric(
                tk, mf, exp, reference=(reference or {}).get(exp["label"])
            )
        )

    # --- the write-up fields the architecture rubric reads ------------------
    if not (forecast.reconciliation or "").strip():
        problems.append(
            f"{WARN}: {tk}: `reconciliation` is empty — how the three reports were "
            f"combined is not recorded."
        )
    if not (forecast.sanity_checks or "").strip():
        problems.append(f"{WARN}: {tk}: `sanity_checks` is empty.")
    if not (forecast.known_weaknesses or "").strip():
        problems.append(
            f"{WARN}: {tk}: `known_weaknesses` is empty — honesty is explicitly scored."
        )

    problems.sort(key=lambda s: 0 if s.startswith(ERROR) else 1)
    return problems


def errors(problems: list[str]) -> list[str]:
    return [p for p in problems if p.startswith(ERROR)]


def warnings(problems: list[str]) -> list[str]:
    return [p for p in problems if p.startswith(WARN)]


# ---------------------------------------------------------------------------
# 8. Reference values, pulled from the analysts' own reports
# ---------------------------------------------------------------------------


def reference_values(
    ticker: str,
    filings: FilingsReport | None,
    financials: FinancialsReport | None,
) -> dict[str, float]:
    """Latest reported comparable per metric label, for the step-change check.

    Best effort: the filings analyst's history is matched on the metric label,
    and the financials table supplies revenue/EPS as a backstop. A miss here is
    harmless — the check simply does not fire.
    """
    tk = canonical_ticker(ticker)
    out: dict[str, float] = {}
    wanted = {_norm(m["label"]): m["label"] for m in metrics_for(tk)}
    if not wanted:
        return out

    for obs in (filings.metrics_history if filings else []) or []:
        if obs.value is None or not math.isfinite(obs.value):
            continue
        key = _norm(obs.metric)
        label = wanted.get(key)
        if label is None:
            close = get_close_matches(key, list(wanted), n=1, cutoff=0.8)
            label = wanted[close[0]] if close else None
        if label and label not in out:
            out[label] = float(obs.value)

    rows = list((financials.annual if financials else []) or []) + list(
        (financials.quarterly if financials else []) or []
    )
    for row in rows:
        for label_norm, label in wanted.items():
            if label in out:
                continue
            if "revenue" in label_norm or "net sales" in label_norm:
                if row.revenue is not None and math.isfinite(row.revenue):
                    out[label] = float(row.revenue)
            elif "eps" in label_norm and row.diluted_eps is not None:
                if math.isfinite(row.diluted_eps):
                    out[label] = float(row.diluted_eps)
    return out


# ---------------------------------------------------------------------------
# 9. Evidence recording
# ---------------------------------------------------------------------------


def _conf_score(text: str) -> float | None:
    return _CONFIDENCE_SCORE.get((text or "").strip().lower())


def record_evidence(
    store: Any,
    run_id: str,
    ticker: str,
    forecast: CompanyForecast,
    *,
    task_id: str | None = None,
) -> int:
    """Write every forecast metric into the runstore `evidence` table.

    One row per (metric, citation), so "how did this number happen" is a single
    SELECT. A metric with no citations still gets a row — the gap is recorded,
    not hidden.
    """
    if store is None or not run_id:
        return 0
    tk = canonical_ticker(ticker)
    written = 0
    for mf in forecast.metrics or []:
        method = (mf.method or "").strip() or "(no method stated)"
        cits: list[Citation] = list(mf.evidence or []) or [
            Citation(source="(uncited)", locator="", quote="")
        ]
        for cit in cits:
            claim = f"{mf.label} = {mf.value} {mf.units} [{mf.low}, {mf.high}] | {method}"
            if cit.quote:
                claim += f" | quote: {cit.quote}"
            try:
                store.add_evidence(
                    run_id,
                    tk,
                    mf.label,
                    (cit.source or "(uncited)"),
                    task_id=task_id,
                    claim=claim[:4000],
                    value=mf.value,
                    units=mf.units,
                    locator=cit.locator or None,
                    confidence=_conf_score(mf.confidence),
                )
                written += 1
            except Exception as e:  # never let bookkeeping kill a forecast
                logger.warning("add_evidence failed for %s/%s: %s", tk, mf.label, e)
    return written


# ---------------------------------------------------------------------------
# 10. The entry point
# ---------------------------------------------------------------------------


def _merge_repair(
    ticker: str,
    original: CompanyForecast,
    repaired: CompanyForecast,
    reference: dict[str, float] | None = None,
) -> CompanyForecast:
    """Per metric, keep whichever attempt the validator likes better.

    A repair pass can fix one metric and damage another. Merging at metric
    granularity means the second attempt can only help.
    """
    tk = canonical_ticker(ticker)
    ref = reference or {}
    orig_by = {m.label: m for m in original.metrics or []}
    rep_by = {m.label: m for m in repaired.metrics or []}
    merged: list[MetricForecast] = []
    for exp in metrics_for(tk):
        a = orig_by.get(exp["label"])
        b = rep_by.get(exp["label"])
        if a is None:
            merged.append(b or MetricForecast(label=exp["label"], units=exp["units"]))
            continue
        if b is None:
            merged.append(a)
            continue
        ea = len(errors(validate_metric(tk, a, exp, ref.get(exp["label"]))))
        eb = len(errors(validate_metric(tk, b, exp, ref.get(exp["label"]))))
        merged.append(b if eb <= ea else a)
    # Prose fields: prefer the repair when it wrote something.
    def pick(x: str, y: str) -> str:
        return y if (y or "").strip() else x

    return repaired.model_copy(
        update={
            "metrics": merged,
            "reconciliation": pick(original.reconciliation, repaired.reconciliation),
            "sanity_checks": pick(original.sanity_checks, repaired.sanity_checks),
            "known_weaknesses": pick(original.known_weaknesses, repaired.known_weaknesses),
        }
    )


async def synthesise(
    ticker: str,
    filings: FilingsReport | None = None,
    news: NewsReport | None = None,
    financials: FinancialsReport | None = None,
    run_id: str | None = None,
    store: Any = None,
    *,
    as_of: str = "",
    task_id: str | None = None,
    repair: bool = True,
) -> CompanyForecast:
    """Three reports in, one validated `CompanyForecast` out.

    Never raises: `run_agent` already returns the spec fallback on any agent
    failure, and everything after it is pure Python over defaulted models. The
    caller always gets a forecast carrying the correct labels and units.
    """
    tk = canonical_ticker(ticker)
    spec = build_spec(tk)
    ref = reference_values(tk, filings, financials)
    user_message = build_user_message(tk, filings, news, financials, as_of=as_of)

    def _log(kind: str, **kw: Any) -> None:
        if store is not None and run_id:
            try:
                store.log(run_id, kind, task_id=task_id, name=tk, **kw)
            except Exception:  # pragma: no cover - bookkeeping must never win
                logger.debug("runstore log failed", exc_info=True)

    _log(
        "central.input",
        payload={
            "chars": len(user_message),
            "filings": filings is not None,
            "news": news is not None,
            "financials": financials is not None,
            "model": settings.resolved_agent_model,
            "reference_values": ref,
        },
    )

    result = await run_agent(spec, user_message)
    forecast = result if isinstance(result, CompanyForecast) else skeleton_forecast(tk)
    forecast, repairs = canonicalise(tk, forecast)
    problems = repairs + validate_forecast(tk, forecast, reference=ref)
    _log(
        "central.validate",
        name=f"{tk} attempt 1",
        payload={"errors": errors(problems), "warnings": warnings(problems)},
    )

    hard = errors(problems)
    if hard and repair:
        logger.warning(
            "%s: %d validation error(s) on attempt 1 — one repair round-trip.",
            tk,
            len(hard),
        )
        _log("central.repair", payload={"errors": hard})
        retry = await run_agent(
            spec, build_repair_message(tk, forecast, problems, user_message)
        )
        if isinstance(retry, CompanyForecast):
            repaired, repairs2 = canonicalise(tk, retry)
            merged = _merge_repair(tk, forecast, repaired, ref)
            merged, repairs3 = canonicalise(tk, merged)
            new_problems = repairs2 + repairs3 + validate_forecast(tk, merged, reference=ref)
            _log(
                "central.validate",
                name=f"{tk} attempt 2",
                payload={
                    "errors": errors(new_problems),
                    "warnings": warnings(new_problems),
                },
            )
            if len(errors(new_problems)) <= len(hard):
                forecast, problems = merged, new_problems
            else:
                logger.warning("%s: repair made it worse; keeping attempt 1.", tk)
                _log("central.repair.rejected", payload={"errors": errors(new_problems)})

    rows = record_evidence(store, run_id or "", tk, forecast, task_id=task_id)
    _log(
        "central.done",
        payload={
            "evidence_rows": rows,
            "errors": len(errors(problems)),
            "warnings": len(warnings(problems)),
            "values": {m.label: m.value for m in forecast.metrics or []},
        },
    )
    return forecast


__all__ = [
    "COMPANY_METRICS",
    "TICKERS",
    "SYSTEM_PROMPT",
    "build_spec",
    "build_user_message",
    "canonical_ticker",
    "canonicalise",
    "company_brief",
    "company_name_for",
    "errors",
    "metrics_for",
    "output_file_for",
    "period_for",
    "record_evidence",
    "reference_values",
    "skeleton_forecast",
    "synthesise",
    "validate_forecast",
    "validate_metric",
    "warnings",
]
