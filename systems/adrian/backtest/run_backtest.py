"""Backtest harness.

Forecast a period that has already been reported, using only evidence that existed before
it was reported, then score against the actual. This is the only way to know whether the
method works before results publish weeks from now.

The whole exercise depends on one thing: `as_of`. Retrieval hard-excludes anything
published after that date, and reading a document past it raises rather than returning
data. Without that guard a backtest retrieves the release containing the answer, scores
brilliantly, and tells us nothing.

    python3 backtest/run_backtest.py            # all configured cases
    python3 backtest/run_backtest.py DE         # one company

Scoring mirrors the competition: min(5.0, our_error / max(reference_error, floor)).
Where no analyst reference is available the run still reports absolute and percentage
error, which is the honest fallback.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents.llm import build_client                      # noqa: E402
from src.pipeline import run_company                          # noqa: E402

SETTINGS = json.loads(Path("config/settings.json").read_text(encoding="utf-8"))
CORPUS = SETTINGS["corpus"]["root"]

# Cases are prior periods whose actuals sit in the corpus. as_of is set to the day before
# the results were published, so the system sees exactly what a forecaster saw that day.
CASES = [
    {
        "ticker": "DE", "company": "Deere & Company", "corpusDir": "deere",
        "period": "FY2025Q3", "periodType": "quarter",
        # Deere's Q3 FY2025 10-Q is published 2025-08-14. The retrieval filter excludes
        # documents published AFTER as_of, so an as_of of 2025-08-14 lets the results
        # filing through and the "forecast" becomes a lookup - the first run returned the
        # segment profit exactly. as_of must sit strictly before the release date.
        "as_of": "2025-08-13",
        "actuals": {
            "Worldwide net sales and revenues": 12018.0,
            "Diluted EPS (GAAP)": 4.76,
            "Production & Precision Ag operating profit": 580.0,
        },
        "metrics": [
            {"label": "Worldwide net sales and revenues", "units": "USDm", "basis": "reported"},
            {"label": "Diluted EPS (GAAP)", "units": "USD / share", "basis": "gaap"},
            {"label": "Production & Precision Ag operating profit", "units": "USDm",
             "basis": "reported"},
        ],
    },
    {
        "ticker": "ADI", "company": "Analog Devices", "corpusDir": "analog-devices",
        "period": "FY2026Q2", "periodType": "quarter",
        "as_of": "2026-05-19",          # ADI reported Q2 FY2026 on 2026-05-20
        "actuals": {"Adjusted gross margin": 73.0, "Revenue": 3620.0},
        "metrics": [
            {"label": "Revenue", "units": "USDm", "basis": "reported"},
            {"label": "Adjusted diluted EPS", "units": "USD / share", "basis": "adjusted"},
            {"label": "Adjusted gross margin", "units": "%", "basis": "adjusted"},
        ],
    },
]

# Competition floors: percentage metrics 0.5pp, money/EPS 0.5% of the reported result.
def floor_for(units: str, actual: float) -> float:
    if "%" in units:
        return 0.5
    return max(abs(actual) * 0.005, 1e-6)


def score(our_value: float, actual: float, units: str,
          reference_error: float | None) -> dict:
    our_error = abs(our_value - actual)
    fl = floor_for(units, actual)
    result = {
        "our_error": round(our_error, 4),
        "pct_error": round(100 * our_error / abs(actual), 3) if actual else None,
        "floor": round(fl, 4),
    }
    if reference_error is not None:
        result["competition_score"] = round(min(5.0, our_error / max(reference_error, fl)), 3)
        result["reference_error"] = round(reference_error, 4)
    else:
        # No analyst reference for a historical period - report error only, and say so.
        result["competition_score"] = None
        result["note"] = "no analyst reference available for this period"
    return result


def run_case(client, case: dict, max_steps: int = 20) -> dict:
    company = {k: case[k] for k in ("ticker", "company", "corpusDir", "period",
                                    "periodType", "metrics")}
    started = time.time()
    run = run_company(client, CORPUS, company, case["as_of"], max_steps=max_steps)

    scored = {}
    for label, res in run["results"].items():
        actual = case["actuals"].get(label)
        value = res.get("value")
        if actual is None or value is None:
            scored[label] = {"forecast": value, "actual": actual,
                             "note": "no actual recorded for this metric"}
            continue
        scored[label] = {
            "forecast": round(value, 4), "actual": actual,
            "units": res.get("units"),
            **score(value, actual, res.get("units", ""), reference_error=None),
        }

    return {
        "ticker": case["ticker"], "period": case["period"], "as_of": case["as_of"],
        "elapsed_s": round(time.time() - started, 1),
        "tool_calls": run["tool_calls"],
        "leak_check": leak_check(run, case["as_of"]),
        "metrics": scored,
    }


def leak_check(run: dict, as_of: str) -> dict:
    """Confirm nothing published after as_of reached the evidence pack.

    This is the assertion that makes the backtest meaningful, so it is checked and
    reported rather than assumed.
    """
    # Same-day counts as a leak: a results filing published on the cutoff date is the
    # answer, not evidence available beforehand.
    cutoff = date.fromisoformat(as_of)
    leaked = []
    for row in (run.get("evidence") or {}).get("history", []):
        published = row.get("published_at")
        if published:
            try:
                if date.fromisoformat(str(published)[:10]) >= cutoff:
                    leaked.append(row.get("doc_id"))
            except ValueError:
                pass
    return {"cutoff": as_of, "leaked_documents": leaked, "clean": not leaked}


def main() -> int:
    only = sys.argv[1].upper() if len(sys.argv) > 1 else None
    cases = [c for c in CASES if not only or c["ticker"] == only]
    if not cases:
        print(f"No case for {only}. Available: {', '.join(c['ticker'] for c in CASES)}")
        return 1

    client = build_client("openai:gpt-4.1")
    print(f"model: {client.name}\n")

    results = []
    for case in cases:
        print(f"=== {case['ticker']} {case['period']}  (as_of {case['as_of']}) ===")
        outcome = run_case(client, case)
        results.append(outcome)

        leak = outcome["leak_check"]
        print(f"  leak check: {'CLEAN' if leak['clean'] else 'LEAKED ' + str(leak['leaked_documents'])}")
        for label, m in outcome["metrics"].items():
            if m.get("actual") is None:
                print(f"  {label[:44]:44} {str(m.get('forecast')):>10}   (no actual on file)")
            else:
                print(f"  {label[:44]:44} {m['forecast']:>10}  actual {m['actual']:>10}"
                      f"  err {m['pct_error']:>6.2f}%")
        print(f"  {outcome['tool_calls']} tool calls, {outcome['elapsed_s']}s\n")

    out = Path("logs/backtest.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
