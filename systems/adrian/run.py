"""End-to-end run.

Phase 1: placeholder anchors, so the submission path is proven before any intelligence
exists. Phase 2+ replaces `placeholder_forecasts` with the research agent's output; the
rest of this file does not change.

    python3 run.py            # writes four workbooks to submission/
    npm run check:submission  # organizer validator, must print PASS x4
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.rails.workbook import write_all

SETTINGS = json.loads(Path("config/settings.json").read_text(encoding="utf-8"))
DEFINITIONS = json.loads(Path("challenge/companies.json").read_text(encoding="utf-8"))

# Placeholder anchors. Deliberately crude and clearly labelled - these exist only to prove
# the workbook path end-to-end. Every one is replaced by the agent in Phase 2.
#
# Where a number is directly guided in the corpus it is noted, because those become the
# guidance-anchored forecaster's starting point rather than a guess.
PLACEHOLDERS: dict[str, dict[str, float]] = {
    "HD": {
        "Net sales": 45000.0,
        "Adjusted diluted EPS": 4.70,
        "Comparable sales, total company": 1.0,      # FY guide: flat to +2.0%
    },
    "ADI": {
        "Revenue": 3900.0,                            # guided: $3.9bn +/- $100m
        "Adjusted diluted EPS": 3.30,                 # guided: $3.30 +/- $0.15
        "Adjusted gross margin": 69.0,                # NOT guided - derive from history
    },
    "LSE:HAS": {
        "Net fees": 1000.0,
        "Pre-exceptional basic EPS": 2.0,             # PENCE
        "Pre-exceptional operating profit": 45.5,     # consensus 43.5, mgmt "top of range"
    },
    "DE": {
        "Worldwide net sales and revenues": 12000.0,
        "Diluted EPS (GAAP)": 5.50,
        "Production & Precision Ag operating profit": 1200.0,
    },
}


def placeholder_forecasts() -> dict[str, dict[str, float]]:
    return PLACEHOLDERS


def agent_forecasts(log) -> dict:
    """Run the real agent pipeline. Falls back to placeholders only if it produces nothing."""
    from src.agents.llm import build_client
    from src.pipeline import run_all

    client = build_client(os.environ.get("AGENT_MODEL", "openai:gpt-4.1"))
    log(f"model: {client.name}")
    companies = json.loads(Path("config/metrics.json").read_text(encoding="utf-8"))["companies"]
    runs = run_all(client, SETTINGS["corpus"]["root"], companies,
                   as_of=SETTINGS["run"]["asOf"], max_steps=20, workers=2)

    Path(SETTINGS["output"]["logDir"]).mkdir(parents=True, exist_ok=True)
    Path(SETTINGS["output"]["logDir"], "full-run.json").write_text(
        json.dumps(runs, indent=2, default=str), encoding="utf-8")

    forecasts: dict[str, dict[str, float]] = {}
    for run in runs:
        log(f"{run['ticker']:8} profile={run['profile']:15} "
            f"{run['elapsed_s']}s {run['tool_calls']} tool calls")
        values = {}
        for label, res in run["results"].items():
            value = res.get("value")
            if value is None:
                value = PLACEHOLDERS.get(run["ticker"], {}).get(label)
                log(f"  FALLBACK {label}: agent produced nothing, using anchor {value}")
            values[label] = value
            notes = []
            if res.get("anchor_rejected"): notes.append("anchor rejected")
            if res.get("clamped"): notes.append("clamped")
            if res.get("outliers"): notes.append(f"{len(res['outliers'])} outlier(s)")
            verdict = res.get("verdict") or {}
            if verdict.get("plausible") is False: notes.append("critic flagged")
            log(f"  {label[:44]:44} {value!s:>12} {res['units']}"
                + (f"   [{', '.join(notes)}]" if notes else ""))
        forecasts[run["ticker"]] = values
    return forecasts


def main() -> int:
    started = datetime.now(timezone.utc)
    log_dir = Path(SETTINGS["output"]["logDir"])
    log_dir.mkdir(parents=True, exist_ok=True)

    lines = [f"run start (utc): {started.isoformat()}", f"as_of: {SETTINGS['run']['asOf']}"]
    print("\n".join(lines))

    use_agent = "--placeholder" not in sys.argv
    def log(msg):
        print(msg); lines.append(msg)
    forecasts = agent_forecasts(log) if use_agent else placeholder_forecasts()

    written = write_all(forecasts, out_dir=SETTINGS["output"]["submissionDir"])
    for path in written:
        print(f"  wrote {path}")
        lines.append(f"wrote {path}")

    for company in DEFINITIONS["companies"]:
        values = forecasts[company["ticker"]]
        for metric in company["metrics"]:
            line = (
                f"  {company['ticker']:8} {metric['label'][:44]:44} "
                f"{values[metric['label']]:>12,.2f}  {metric['units']}"
            )
            print(line)
            lines.append(line.strip())

    finished = datetime.now(timezone.utc)
    lines.append(f"run end (utc): {finished.isoformat()}")
    lines.append(f"elapsed: {(finished - started).total_seconds():.1f}s")
    (log_dir / "clear-run.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if use_agent:
        try:
            from src.rails.report import build as build_dashboard
            page = build_dashboard(as_of=SETTINGS["run"]["asOf"],
                                   model=os.environ.get("AGENT_MODEL", "openai:gpt-4.1"))
            print(f"dashboard: {page}")
            lines.append(f"dashboard: {page}")
        except Exception as exc:                       # never let the report break a run
            print(f"dashboard generation skipped: {exc}")

    (log_dir / "clear-run.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nlog: {log_dir / 'clear-run.log'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
