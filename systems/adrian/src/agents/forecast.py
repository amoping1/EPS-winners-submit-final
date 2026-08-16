"""Forecaster agents and the adversarial critic.

Three forecasters, each with a different method, run on the SAME evidence pack in
SEPARATE contexts. The separation is the point: one agent asked for three numbers returns
three correlated numbers, and an ensemble of correlated estimates is theatre. Independence
is what makes the reconciliation worth anything.

The critic never sees the forecasters' reasoning - only their numbers and the evidence.
A critic shown the argument tends to agree with it.
"""

from __future__ import annotations

import json

from .llm import LLMClient

_SHARED = """\
Company: {company} ({ticker})
Target period: {period}, ending {period_end}
Metrics to forecast (use these EXACT labels):
{metric_lines}

Evidence gathered by the research agent (every figure is cited):

HISTORY (reported actuals):
{history}

ANCHORS (guidance, consensus, most recent actuals):
{anchors}

CYCLICAL / SEASONAL (derived from the reported series, not quoted from any document):
{cyclical}

VARIABLES TO REASON OVER (industry-specific):
{predictors}

DRIVERS:
{drivers}

KNOWN GAPS:
{gaps}

Unit rules - violating these is the single most costly error possible:
- Percentages are POINTS: enter 4.5 for 4.5%, never 0.045.
- UK EPS is in PENCE: enter 45 for 45p, never 0.45.
- Match the stated basis exactly: adjusted, GAAP and pre-exceptional are different numbers.

The "metric" field must be ONLY the quoted label text, e.g. "Revenue" - never with units
or basis appended.

Reply with JSON only:
{{"forecasts": [{{"metric": "<exact label>", "value": <number>, "units": "...",
  "reasoning": "<2-3 sentences showing the arithmetic>", "confidence": "high|medium|low"}}]}}

Every metric must get a number. If evidence is thin, say so in reasoning and lower
confidence - but never return null. A missing forecast scores worse than a poor one.
"""

METHODS = {
    "guidance": """\
You are a guidance-anchored forecaster. Company guidance is the strongest signal available
because management sees the quarter from the inside.

Method:
- If explicit guidance exists for the target period and metric, start at the midpoint of
  the guided range and adjust only for a stated reason.
- If only annual guidance exists, phase it to the target period using the historical share
  that period takes of the full year.
- If published analyst consensus exists, treat it as a second anchor. Where management has
  signalled a position within a consensus range, follow the signal.
- Where no anchor exists, say so and fall back to the most recent actual adjusted for the
  observed trend.
""",
    "statistical": """\
You are a statistical forecaster. Ignore narrative and management tone; work the numbers.

Method:
- Fit the trend across the same fiscal period in prior years - this controls for
  seasonality, which is the dominant effect in quarterly data.
- Compute YoY growth rates for each prior period and look at whether the rate is
  accelerating or decelerating.
- Apply the recent growth rate to the most recent comparable period.
- Sanity-check against sequential (quarter-on-quarter) movement.
- State the growth rate you applied and why.
""",
    "qualitative": """\
You are a qualitative forecaster. Read the business situation, not just the series.

Method:
- Weigh what management said about demand, pricing, costs, and mix.
- Account for structural changes: disposals, acquisitions, restructuring, country exits.
  These break naive year-on-year comparisons and are a common source of large errors.
- Consider whether the trend in the numbers is likely to continue, inflect, or reverse
  given the stated conditions.
- Be explicit about which structural factor moved your number away from the trend.
""",
}

CRITIC = """\
You are an adversarial reviewer. Your job is to find what is WRONG with these forecasts.
You are shown the numbers and the evidence, but deliberately not the reasoning behind them.

Company: {company} ({ticker}), target period {period}

Evidence:
HISTORY:
{history}
ANCHORS:
{anchors}

Proposed forecasts. Each metric shows the FINAL reconciled value that will be submitted,
plus the individual method proposals for context. Judge the FINAL value - a single method
being off does not matter if reconciliation already corrected it.
{proposals}

For each metric, check specifically:
1. UNITS - is it in the right unit? Percentages as points (4.5 not 0.045), UK EPS in pence
   (45 not 0.45), millions vs billions. A unit error is a catastrophic scoring failure.
2. BASIS - adjusted vs GAAP vs pre-exceptional. Are they forecasting the right measure?
3. MAGNITUDE - is it plausible against the history? Flag anything more than ~30% away from
   the most recent comparable period without a stated structural reason.
4. SIGN AND DIRECTION - does it move the right way given the evidence?
5. CONFUSION - is a margin being reported where a profit is required, or vice versa?

Reply with JSON only:
{{"verdicts": [{{"metric": "<exact label>", "plausible": true|false,
  "concern": "<what is wrong, or 'none'>", "suggested_low": <number>,
  "suggested_high": <number>}}]}}

Default to plausible:true when you have no specific objection. Do not manufacture doubt -
but a unit or basis error must always be flagged.
"""


def _fmt(rows: list[dict], keys: tuple[str, ...]) -> str:
    if not rows:
        return "  (none found)"
    out = []
    for r in rows:
        parts = [f"{k}={r[k]}" for k in keys if r.get(k) is not None]
        out.append("  - " + ", ".join(parts))
    return "\n".join(out)


def _fmt_cyclical(cyclical) -> str:
    """Render the derived seasonal channel. Labelled as derived so it is never mistaken
    for something a document said."""
    if not cyclical:
        return "  (not derived)"
    out = []
    for metric, shape in cyclical.items():
        if not shape.get("available"):
            out.append(f"  - {metric}: insufficient series ({shape.get('reason')})")
            continue
        out.append(
            f"  - {metric}: {shape['points']} points, latest={shape['latest']}, "
            f"recent growth={shape['recent_growth']:+.1%}, mean={shape['mean_growth']:+.1%}, "
            f"{shape['direction']}, naive projection={shape['naive_projection']}"
        )
    return "\n".join(out) or "  (none)"


def _context(company: dict, pack, profile=None) -> dict:
    return {
        "company": company["company"],
        "ticker": company["ticker"],
        "period": company["period"],
        "period_end": company.get("periodEnd") or "unknown",
        "metric_lines": "\n".join(
            f"  - label: \"{m['label']}\"  |  units: {m['units']}  |  basis: {m.get('basis', 'reported')}"
            for m in company["metrics"]
        ),
        "history": _fmt(pack.history, ("metric", "period", "value", "units", "basis", "doc_id")),
        "anchors": _fmt(pack.anchors, ("metric", "kind", "value", "low", "high", "units", "note")),
        "cyclical": _fmt_cyclical(getattr(pack, "cyclical", None)),
        "drivers": "\n".join(f"  - {d}" for d in pack.drivers) or "  (none)",
        "gaps": "\n".join(f"  - {g}" for g in pack.gaps) or "  (none)",
        "predictors": ("\n".join(f"  - {p}" for p in profile.predictors)
                       if profile and profile.predictors else "  (none)"),
    }


def _json_reply(client: LLMClient, system: str, user: str) -> dict:
    completion = client.complete(
        [{"role": "system", "content": system}, {"role": "user", "content": user}], []
    )
    text = (completion.content or "").strip()
    if text.startswith("```"):
        text = text.split("```")[1].removeprefix("json").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def run_forecaster(client: LLMClient, company: dict, pack, method: str, profile=None) -> dict:
    """One forecaster. Returns {metric label: {value, units, reasoning, confidence}}."""
    ctx = _context(company, pack, profile)
    system = METHODS[method] + "\n" + _SHARED.format(**ctx)
    data = _json_reply(client, system, f"Forecast {company['company']} {company['period']}.")
    out = {}
    for f in data.get("forecasts", []):
        label = f.get("metric")
        if label and isinstance(f.get("value"), (int, float)):
            out[label] = {
                "value": float(f["value"]),
                "units": f.get("units"),
                "reasoning": f.get("reasoning", ""),
                "confidence": f.get("confidence", "medium"),
                "method": method,
            }
    return out


def run_critic(client: LLMClient, company: dict, pack, proposals: dict, profile=None,
               reconciled: dict | None = None) -> dict:
    """Adversarial pass on the FINAL numbers.

    Runs after reconciliation, not before. Reviewing raw method proposals produced false
    alarms: Home Depot's statistical method proposed comparable sales of 2.3% against a
    guided ceiling of 2.0%, the critic objected, and the reconciled value was 1.59% - well
    inside guidance. The critic was right about the proposal and wrong about the entry.
    """
    ctx = _context(company, pack, profile)
    stripped = {}
    for label, by_method in proposals.items():
        entry = {"METHODS": {m: round(v["value"], 4) for m, v in by_method.items()}}
        if reconciled and label in reconciled:
            entry["FINAL_SUBMITTED_VALUE"] = reconciled[label].get("value")
        stripped[label] = entry
    system = CRITIC.format(
        company=ctx["company"], ticker=ctx["ticker"], period=ctx["period"],
        history=ctx["history"], anchors=ctx["anchors"],
        proposals=json.dumps(stripped, indent=2),
    )
    data = _json_reply(client, system, "Review these forecasts.")
    return {v["metric"]: v for v in data.get("verdicts", []) if v.get("metric")}
