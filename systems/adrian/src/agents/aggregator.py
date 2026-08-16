"""Evidence aggregator.

Four channels feed a forecast, and they are not the same kind of thing. Treating them as
one undifferentiated pile is how Hays ended up with two rows of history and a forecast of
25.85 against a published consensus of 43.5.

  FINANCE    reported figures from filings - income statement, segment notes, tables.
             The only channel that carries audited actuals.
  CALLS      earnings-call transcripts. Where guidance, tone and forward statements live.
             Management says things here that never appear in a table.
  MARKET     external, public, fetched live during the event: analyst consensus for the
             target quarter. This is the denominator of the accuracy score. Yahoo covers
             the US filers; the UK filer publishes its own compiled consensus instead.
  CYCLICAL   seasonality and trend. DERIVED, not ingested - computed from the finance
             channel's history. Kept as its own channel because it answers a different
             question: not "what happened" but "what shape does this business repeat".

Each channel reports its own coverage, so thin evidence is visible before it becomes a bad
number rather than after.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

CHANNELS = ("finance", "calls", "market", "cyclical")

# Below this many historical points, a metric cannot be trended or clamped and the
# statistical forecaster is guessing. Two is what Hays returned.
MIN_HISTORY = 4


@dataclass
class ChannelCoverage:
    channel: str
    available: bool
    rows: int = 0
    note: str = ""

    def to_dict(self) -> dict:
        return {"channel": self.channel, "available": self.available,
                "rows": self.rows, "note": self.note}


@dataclass
class MetricEvidence:
    """Everything known about one metric, split by channel."""

    metric: str
    finance: list[dict] = field(default_factory=list)   # reported actuals, cited
    calls: list[dict] = field(default_factory=list)     # guidance / commentary anchors
    news: list[dict] = field(default_factory=list)      # external context
    cyclical: dict = field(default_factory=dict)        # derived shape

    @property
    def sufficient(self) -> bool:
        return len(self.finance) >= MIN_HISTORY

    def gaps(self) -> list[str]:
        out = []
        if len(self.finance) < MIN_HISTORY:
            out.append(
                f"{self.metric}: only {len(self.finance)} historical point(s), "
                f"need {MIN_HISTORY} to fit a trend or clamp"
            )
        if not self.calls:
            out.append(f"{self.metric}: no guidance or management commentary found")
        return out

    def to_dict(self) -> dict:
        return {"metric": self.metric, "finance": self.finance, "calls": self.calls,
                "news": self.news, "cyclical": self.cyclical,
                "sufficient": self.sufficient}


def derive_cyclical(series: list[dict]) -> dict:
    """Derive seasonality and trend from a reported series.

    Not ingested from anywhere - this is the channel the system computes for itself. Growth
    rates are period-over-period across the series as given (which the research agent
    supplies as the same fiscal quarter in prior years, so it is already seasonally
    aligned).
    """
    values = [float(r["value"]) for r in series if isinstance(r.get("value"), (int, float))]
    if len(values) < 2:
        return {"available": False, "reason": f"need 2+ points, have {len(values)}"}

    ordered = list(reversed(values)) if len(values) > 1 else values
    growth = [
        (ordered[i] - ordered[i - 1]) / abs(ordered[i - 1])
        for i in range(1, len(ordered))
        if ordered[i - 1]
    ]
    if not growth:
        return {"available": False, "reason": "series contains zeros"}

    recent = growth[-1]
    mean_growth = statistics.fmean(growth)
    if len(growth) >= 2:
        if recent >= 0 and growth[-2] >= 0:
            trend = "growth accelerating" if recent > growth[-2] else "growth slowing"
        elif recent < 0 and growth[-2] < 0:
            trend = "decline easing" if recent > growth[-2] else "decline deepening"
        else:
            trend = "inflected positive" if recent >= 0 else "inflected negative"
    else:
        trend = "single interval - direction unknown"

    return {
        "available": True,
        "points": len(values),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
        "latest": round(ordered[-1], 4),
        "growth_rates": [round(g, 4) for g in growth],
        "recent_growth": round(recent, 4),
        "mean_growth": round(mean_growth, 4),
        "direction": trend,
        "naive_projection": round(ordered[-1] * (1 + recent), 4),
    }


class EvidenceAggregator:
    """Normalises an EvidencePack into per-metric, per-channel evidence."""

    def __init__(self, metric_labels: list[str]):
        self.metric_labels = metric_labels

    def aggregate(self, pack) -> dict:
        from ..rails.reconcile import align_label

        by_metric = {label: MetricEvidence(metric=label) for label in self.metric_labels}

        for row in pack.history:
            label = align_label(row.get("metric", ""), self.metric_labels)
            if label:
                by_metric[label].finance.append(row)

        for anchor in pack.anchors:
            label = align_label(anchor.get("metric", ""), self.metric_labels)
            if not label:
                continue
            # Guidance and consensus originate in calls/statements, not the accounts.
            if anchor.get("source") == "market_data":
                by_metric[label].news.append(anchor)   # external market channel
            elif anchor.get("kind") in ("guidance", "consensus"):
                by_metric[label].calls.append(anchor)
            else:
                by_metric[label].finance.append(anchor)

        for label, evidence in by_metric.items():
            evidence.cyclical = derive_cyclical(evidence.finance)

        coverage = [
            ChannelCoverage("finance", True,
                            sum(len(e.finance) for e in by_metric.values()),
                            "reported actuals from filings"),
            ChannelCoverage("calls", True,
                            sum(len(e.calls) for e in by_metric.values()),
                            "guidance and consensus from transcripts and statements"),
            ChannelCoverage("market", True,
                            sum(len(e.news) for e in by_metric.values()),
                            "public analyst consensus (Yahoo Finance), fetched live"),
            ChannelCoverage("cyclical", True,
                            sum(1 for e in by_metric.values() if e.cyclical.get("available")),
                            "derived from the finance channel, not ingested"),
        ]

        gaps = [gap for evidence in by_metric.values() for gap in evidence.gaps()]
        thin = [label for label, e in by_metric.items() if not e.sufficient]

        return {
            "metrics": {label: e.to_dict() for label, e in by_metric.items()},
            "coverage": [c.to_dict() for c in coverage],
            "gaps": gaps,
            "thin_metrics": thin,
            "sufficient": not thin,
        }


def followup_brief(thin_metrics: list[str], aggregated: dict, period: str) -> str:
    """A targeted second research brief naming exactly what is missing.

    The first pass returns what it happened to find. This names the hole."""
    lines = [
        "Your evidence pack is incomplete. Do NOT re-gather what you already have.",
        f"Target period is {period}. Find ONLY the following, then return the JSON:",
        "",
    ]
    for label in thin_metrics:
        have = len(aggregated["metrics"][label]["finance"])
        lines.append(
            f"  - \"{label}\": you returned {have} historical value(s). Find at least "
            f"{MIN_HISTORY} prior comparable periods with a number and a doc_id."
        )
    lines += [
        "",
        "Where to look: annual and interim results statements carry the full-year and",
        "half-year figures; segment notes carry divisional lines; trading updates carry",
        "recent commentary. For a full-year target, prior FULL YEARS are the comparable",
        "periods, not quarters.",
    ]
    return "\n".join(lines)
