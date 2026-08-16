"""The research agent.

An LLM in a loop with the toolbelt. It decides its own next move from what the last tool
returned, and stops when it judges the evidence sufficient - or when the budget runs out.

The loop is what makes this an agent rather than a pipeline. The path to a figure like
"Production & Precision Ag operating profit" is not knowable in advance: the agent has to
notice that the MD&A prose lacks the number, pivot to the segment note, notice the segment
note only has Q2, and pivot again to drivers.

Every tool call is recorded in a trace. The trace is the evidence that the system reasoned
its way to a forecast, and it is what we show the judges.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime, timezone

from ..tools.documents import DocumentTools
from ..tools.schema import TOOL_SPECS, dispatch
from .llm import Completion, LLMClient
from .profiles import PROFILES, IndustryProfile

SYSTEM_PROMPT = """\
You are a financial research agent. Your job is to gather evidence for a forecast, not to
produce the forecast itself. Another agent does that from your evidence pack.

Company: {company} ({ticker})
Target period: {period} ({period_type})
Metrics to gather evidence for:
{metric_lines}

Industry profile: {profile_label}
Known drivers for this industry:
{drivers}
Variables that move this metric - reason over these explicitly, not just the headline series:
{predictors}
Catalysts that push a number away from trend - hunt for these in the calls channel:
{catalysts}
Known failure modes for this industry:
{risks}

Hard rules:
- Today's as_of date is {as_of}. Documents published after it do not exist for you. The
  tools enforce this; an as_of_violation error means you tried to read the future.
- The catalog's period_label is unreliable. Prefer period_from_slug and verify against the
  document body. A period_conflict flag means check before trusting.
- Never state a figure you have not read from a document. Every number in your evidence
  pack must carry its doc_id.
- Units matter and are a common failure: percentages are points (4.5 = 4.5%), UK EPS is in
  pence, and adjusted/GAAP/pre-exceptional are different measures. Record which basis each
  figure is on.

Method:
1. Start with list_index to find the most recent earnings documents.
2. Read guidance and outlook sections - explicit company guidance is the strongest signal
   available and often gives the target figure directly.
3. Use extract_table for anything you expect in a table (segment results, income
   statement, margins). Do not read whole filings looking for a number in a table.
4. Search for published analyst consensus ("company compiled consensus", "consensus
   range"). If it exists it is extremely valuable - say so and record it.
5. Build 8-12 periods of history for each metric so a trend can be fitted.

When you have enough, reply with a JSON object and no other text. Keep reported history
separate from forward-looking anchors - they are used differently downstream.

{{"status": "complete",
  "history": [
    {{"metric": "<exact metric label>", "period": "FY2025Q3", "value": 12018,
      "units": "USDm", "basis": "reported", "doc_id": "..."}}
  ],
  "anchors": [
    {{"metric": "<exact metric label>", "kind": "guidance|consensus|last_actual|derived",
      "value": 3900, "low": 3800, "high": 4000, "units": "USDm", "period": "FY2026Q3",
      "note": "company guided revenue $3.9bn +/- $100m", "doc_id": "...",
      "confidence": "high|medium|low"}}
  ],
  "drivers": ["..."], "gaps": ["..."], "consensus": "... or null"}}

Rules for these two lists:
- history: ONE ENTRY PER PERIOD PER METRIC, value must be a number, never null. This is the
  time series a trend is fitted to, so give at least 4 prior comparable periods per metric
  (same fiscal quarter in prior years for a quarterly target, prior full years for an
  annual target). Do not put prose in 'value'.
- anchors: forward-looking or most-recent reference points. 'guidance' means the company
  stated it. 'consensus' means analysts' published estimate. 'last_actual' is the most
  recent reported figure for that metric. Use low/high when a range was given.
- Never put a margin in a metric whose label is a profit, or vice versa. Match the exact
  metric label given above.
- The "metric" field must be ONLY the quoted label text, e.g. "Revenue". Do not append
  units or basis to it.
- An anchor is only "guidance" if the company guided THAT EXACT measure. Operating margin
  is not gross margin; full-year guidance is not quarterly guidance. If the company
  guided a different measure, record kind="derived" and say so in the note.
"""


@dataclass
class TraceStep:
    n: int
    tool: str
    arguments: dict
    result_summary: str
    error: str | None = None


@dataclass
class EvidencePack:
    company: str
    ticker: str
    period: str
    profile: str
    as_of: str
    history: list[dict] = field(default_factory=list)
    anchors: list[dict] = field(default_factory=list)
    drivers: list[str] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)
    consensus: str | None = None
    trace: list[TraceStep] = field(default_factory=list)
    raw_final: str | None = None
    stopped_because: str = "complete"

    def to_dict(self) -> dict:
        return {
            "company": self.company,
            "ticker": self.ticker,
            "period": self.period,
            "profile": self.profile,
            "as_of": self.as_of,
            "history": self.history,
            "anchors": self.anchors,
            "drivers": self.drivers,
            "gaps": self.gaps,
            "consensus": self.consensus,
            "stopped_because": self.stopped_because,
            "trace": [
                {"n": s.n, "tool": s.tool, "arguments": s.arguments,
                 "result": s.result_summary, "error": s.error}
                for s in self.trace
            ],
        }


def _summarise(name: str, result: dict) -> str:
    if "error" in result:
        return f"ERROR {result.get('kind', '')}: {result['error'][:120]}"
    if name in ("list_index", "search"):
        docs = result.get("documents", [])
        conflicts = sum(1 for d in docs if d.get("period_conflict"))
        head = ", ".join(d["doc_id"][:34] for d in docs[:3])
        extra = f" ({conflicts} period conflicts)" if conflicts else ""
        return f"{result.get('count', 0)} docs{extra}: {head}"
    if name == "read_document":
        return f"matched={result.get('matched')} chars={len(result.get('text', ''))}"
    if name == "extract_table":
        return f"{result.get('count', 0)} table(s), {len(str(result.get('tables', '')))} chars"
    return str(result)[:120]


def run_research(
    client: LLMClient,
    corpus_root: str,
    company: dict,
    profile: IndustryProfile,
    as_of: date | str,
    max_steps: int = 20,
    verbose: bool = True,
    followup: str | None = None,
) -> EvidencePack:
    """Run the research loop for one company.

    `followup` names a specific hole to fill; used for the second pass when the
    aggregator reports a metric has too little history to trend or clamp.
    """
    tools = DocumentTools(corpus_root, company["corpusDir"], as_of=as_of)
    as_of_str = str(as_of)

    metric_lines = "\n".join(
        f"  - label: \"{m['label']}\"  |  units: {m['units']}  |  basis: {m.get('basis', 'reported')}"
        for m in company["metrics"]
    )
    system = SYSTEM_PROMPT.format(
        company=company["company"],
        ticker=company["ticker"],
        period=company["period"],
        period_type=company.get("periodType", "quarter"),
        metric_lines=metric_lines,
        profile_label=profile.label,
        drivers="\n".join(f"  - {d}" for d in profile.forecast_drivers),
        predictors="\n".join(f"  - {p}" for p in profile.predictors) or "  (none)",
        catalysts="\n".join(f"  - {c}" for c in profile.catalysts) or "  (none)",
        risks="\n".join(f"  - {r}" for r in profile.risks),
        as_of=as_of_str,
    )

    messages: list[dict] = [
        {"role": "system", "content": system},
        {"role": "user", "content": followup or (
            f"Gather evidence for {company['company']} {company['period']}. "
            "Begin with list_index."
        )},
    ]

    pack = EvidencePack(
        company=company["company"], ticker=company["ticker"], period=company["period"],
        profile=profile.key, as_of=as_of_str,
    )

    warned = False
    for step in range(1, max_steps + 1):
        # Budget nudge. Without this the loop gets guillotined mid-research and returns
        # nothing, which scores 5.0. Warn early enough that it can still write findings.
        remaining = max_steps - step
        if remaining <= 3 and not warned:
            warned = True
            messages.append({
                "role": "user",
                "content": (
                    f"You have {remaining} steps left. Stop gathering and reply NOW with "
                    "the final JSON object described in your instructions, using whatever "
                    "evidence you already have. Record anything still missing in 'gaps'. "
                    "Partial findings are far better than none."
                ),
            })

        completion: Completion = client.complete(messages, TOOL_SPECS)

        if not completion.tool_calls:
            pack.raw_final = completion.content
            _parse_final(pack, completion.content)
            return pack

        messages.append({
            "role": "assistant",
            "content": completion.content,
            "tool_calls": [
                {"id": c.id, "type": "function",
                 "function": {"name": c.name, "arguments": json.dumps(c.arguments)}}
                for c in completion.tool_calls
            ],
        })

        for call in completion.tool_calls:
            result = dispatch(call.name, call.arguments, tools)
            summary = _summarise(call.name, result)
            pack.trace.append(
                TraceStep(n=step, tool=call.name, arguments=call.arguments,
                          result_summary=summary, error=result.get("error"))
            )
            if verbose:
                args = json.dumps(call.arguments, ensure_ascii=False)[:88]
                print(f"  [{step:2}] {call.name}({args})")
                print(f"       -> {summary}")
            messages.append({
                "role": "tool", "tool_call_id": call.id,
                "content": json.dumps(result, ensure_ascii=False)[:12000],
            })

    pack.stopped_because = f"hit max_steps ({max_steps})"
    return pack


def _parse_final(pack: EvidencePack, content: str | None) -> None:
    if not content:
        pack.stopped_because = "model returned no content"
        return
    text = content.strip()
    if text.startswith("```"):
        text = text.split("```")[1].removeprefix("json").strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        pack.stopped_because = "final message was not valid JSON"
        return
    pack.history = [h for h in data.get("history", []) if isinstance(h.get("value"), (int, float))]
    pack.anchors = data.get("anchors", [])
    pack.drivers = data.get("drivers", [])
    pack.gaps = data.get("gaps", [])
    pack.consensus = data.get("consensus")


def profile_for(company: dict, tools: DocumentTools, sample_chars: int = 40_000):
    """Classify from filing text. Falls back cleanly when nothing matches."""
    from .profiles import classify

    recent = tools.list_index(doc_type="Filing", limit=3)
    text = ""
    for entry in recent:
        text += tools.read_document(entry.doc_id, window=sample_chars // 3)["text"]
    profile, confidence, hits = classify(text)
    return profile, confidence, hits


__all__ = ["run_research", "EvidencePack", "TraceStep", "profile_for", "PROFILES"]
