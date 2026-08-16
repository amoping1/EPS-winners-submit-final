#!/usr/bin/env python3
"""Collect every team member's twelve forecasts into one normalised file.

Three systems were built independently on the same brief, from the same corpus,
against the same twelve targets. That makes them genuine independent estimates
of the same quantities, which is exactly the input an ensemble needs.

Nothing is merged at the code level. Each system stays in its own repository
with its own dependencies; only the outputs are read. Merging three codebases an
hour before a deadline risks all three, and buys nothing the numbers do not
already give us.

    python scripts/collect_team_forecasts.py

Writes runs/team-forecasts.json.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_companies  # noqa: E402

# The other two systems now live in the repository, under systems/, so the
# ensemble reproduces from a single clone. Previously these were sibling folders
# that existed on one laptop and in no repository: anywhere else both resolved to
# nothing, collect reported zero forecasts for two of the three systems, and the
# "ensemble" silently voted on one member while check:submission still passed.
#
# The sibling paths are kept as a fallback so an existing local layout keeps
# working unchanged.
EPS_DIR = ROOT.parent


def _first_existing(*candidates: Path) -> Path:
    """The first path that exists, else the last (so error messages name it)."""
    for path in candidates:
        if path.exists():
            return path
    return candidates[-1]


ADRIAN_RUN = _first_existing(
    ROOT / "systems" / "adrian" / "logs" / "full-run.json",
    EPS_DIR / "team-adrian" / "logs" / "full-run.json",
)
DIMITRIS_DASHBOARD = _first_existing(
    ROOT / "systems" / "dimitris" / "dashboard.html",
    EPS_DIR / "team-dimitris" / "dashboard.html",
)

# Dimitris' dashboard prints one row per metric:
#   <tr><td><strong>ADI</strong></td><td>Revenue</td>
#       <td class="num"><strong>3,950.00</strong></td>
#       <td class="num">3,850.00</td><td class="num">4,050.00</td>
#       <td>USDm</td><td><code>GUIDANCE_ADJUSTED</code></td>...
ROW_RE = re.compile(
    r"<tr><td><strong>(?P<slug>[A-Z:]+)</strong></td>"
    r"<td>(?P<metric>[^<]+)</td>"
    r'<td class="num"><strong>(?P<value>[-\d,.]+)</strong></td>'
    r'<td class="num">(?P<low>[-\d,.]+)</td>'
    r'<td class="num">(?P<high>[-\d,.]+)</td>'
    r"<td>(?P<units>[^<]*)</td>"
    r"<td><code>(?P<method>[^<]*)</code></td>",
    re.S,
)


def number(text: str) -> float | None:
    try:
        return float(text.replace(",", "").strip())
    except (ValueError, AttributeError):
        return None


def unescape(text: str) -> str:
    return (
        text.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&#39;", "'")
        .strip()
    )


def latest_local_run() -> Path | None:
    runs = sorted((ROOT / "runs").glob("run-*"), key=lambda p: p.stat().st_mtime)
    return runs[-1] if runs else None


def read_neva() -> dict[tuple[str, str], dict[str, Any]]:
    """Our own forecasts, from the most recent run."""
    run_dir = latest_local_run()
    if run_dir is None:
        return {}
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for path in sorted(run_dir.glob("*/baseline.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        slug = payload["company"]["slug"]
        accepted = {
            item["metric"] for item in payload.get("agent", []) if item.get("accepted")
        }
        for estimate in payload["estimates"]:
            out[(slug, estimate["metric"])] = {
                "value": estimate["value"],
                "units": estimate["units"],
                "confidence": estimate["confidence"],
                "method": ("reasoning agent" if estimate["metric"] in accepted else estimate["method"])[:120],
            }
    return out


def read_adrian() -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, dict[str, Any]]]:
    """Adrian's forecasts, plus the analyst consensus his system fetched."""
    if not ADRIAN_RUN.exists():
        return {}, {}
    entries = json.loads(ADRIAN_RUN.read_text(encoding="utf-8"))
    out: dict[tuple[str, str], dict[str, Any]] = {}
    market: dict[str, dict[str, Any]] = {}

    for entry in entries:
        slug = str(entry.get("ticker", "")).rsplit(":", 1)[-1].upper()
        for label, result in (entry.get("results") or {}).items():
            value = result.get("value") if isinstance(result, dict) else result
            if value is None:
                continue
            out[(slug, label)] = {
                "value": float(value),
                "units": (result or {}).get("units", "") if isinstance(result, dict) else "",
                "confidence": (result or {}).get("confidence", "") if isinstance(result, dict) else "",
                "method": str((result or {}).get("method", ""))[:120] if isinstance(result, dict) else "",
            }
        info = entry.get("market") or {}
        if info.get("available"):
            market[slug] = {
                "eps_avg": info.get("eps_avg"),
                "eps_low": info.get("eps_low"),
                "eps_high": info.get("eps_high"),
                "analysts": info.get("eps_analysts"),
                "revenue_avg_m": info.get("revenue_avg_m"),
            }
    return out, market


def read_dimitris() -> dict[tuple[str, str], dict[str, Any]]:
    """Dimitris' forecasts, parsed out of his generated dashboard.

    His repository commits no run artifacts, and reproducing them would mean
    installing his dependencies and supplying his credentials. The dashboard is
    generated from his run and carries the same twelve figures with bands.
    """
    if not DIMITRIS_DASHBOARD.exists():
        return {}
    text = DIMITRIS_DASHBOARD.read_text(encoding="utf-8", errors="replace")
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for match in ROW_RE.finditer(text):
        value = number(match.group("value"))
        if value is None:
            continue
        slug = match.group("slug").rsplit(":", 1)[-1].upper()
        out[(slug, unescape(match.group("metric")))] = {
            "value": value,
            "low": number(match.group("low")),
            "high": number(match.group("high")),
            "units": unescape(match.group("units")),
            "method": unescape(match.group("method"))[:120],
        }
    return out


def main() -> int:
    companies = load_companies()
    neva = read_neva()
    adrian, market = read_adrian()
    dimitris = read_dimitris()

    sources = {"neva": neva, "adrian": adrian, "dimitris": dimitris}
    print("Sources")
    for name, values in sources.items():
        print(f"  {name:<10} {len(values)} forecasts")
    print(f"  market     {len(market)} companies with analyst consensus")

    collected: list[dict[str, Any]] = []
    missing: list[str] = []
    for company in companies:
        for metric in company.metrics:
            key = (company.slug, metric.label)
            entry: dict[str, Any] = {
                "company": company.slug,
                "metric": metric.label,
                "units": metric.units,
                "kind": metric.kind,
                "estimates": {},
            }
            for name, values in sources.items():
                if key in values:
                    entry["estimates"][name] = values[key]
                else:
                    missing.append(f"{name}: {company.slug} / {metric.label}")

            info = market.get(company.slug, {})
            if metric.kind == "per_share" and info.get("eps_avg") is not None:
                entry["market_consensus"] = {
                    "value": info["eps_avg"],
                    "low": info.get("eps_low"),
                    "high": info.get("eps_high"),
                    "analysts": info.get("analysts"),
                    "source": "sell-side consensus via Adrian's market channel",
                }
            elif (
                metric.kind == "money"
                and any(word in metric.label.lower() for word in ("sales", "revenue"))
                and info.get("revenue_avg_m") is not None
            ):
                entry["market_consensus"] = {
                    "value": info["revenue_avg_m"],
                    "analysts": info.get("analysts"),
                    "source": "sell-side consensus via Adrian's market channel",
                }
            collected.append(entry)

    target = ROOT / "runs" / "team-forecasts.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps({"metrics": collected}, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    covered = sum(1 for e in collected if len(e["estimates"]) == 3)
    with_market = sum(1 for e in collected if "market_consensus" in e)
    print(f"\n{len(collected)} targets: {covered} have all three systems, "
          f"{with_market} have analyst consensus")
    if missing:
        print("\nMissing:")
        for item in missing:
            print(f"  {item}")
    print(f"\nWritten: {target.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
