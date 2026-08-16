"""check.py — run every market-data function against all four companies.

Run from the repo root:

    python -m marketdata.check          # pass/fail table + key values
    python -m marketdata.check -v       # also dump the full result payloads

There is no assertion of "correct" here — Yahoo's numbers move. What this
checks is that every function returns a well-formed Result for every ticker,
never raises, and that the fields a forecast actually depends on are populated.
A row marked EMPTY is a real, confirmed gap in Yahoo's coverage, not a failure
of this module; those are listed explicitly at the bottom.
"""

from __future__ import annotations

import json
import sys
import traceback
from typing import Any, Callable

from . import client

TICKERS = ("HD", "ADI", "HAS", "DE")


def _fmt(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:,.4g}"
    if isinstance(v, (list, dict)):
        return f"<{len(v)}>"
    return str(v)


def _probe(fn: Callable[..., client.Result], ticker: str, **kw) -> tuple[str, str, Any]:
    """Returns (status, headline, result). Never raises."""
    try:
        res = fn(ticker, **kw)
    except Exception as e:  # a client function raising is itself the bug
        traceback.print_exc()
        return "RAISED", f"{type(e).__name__}: {e}", None

    if not isinstance(res, client.Result):
        return "BADTYPE", type(res).__name__, res
    if not res.source or not res.as_of:
        return "NOPROV", "missing source/as_of", res
    if not res.ok:
        why = res.notes[0] if res.notes else "no data"
        return "EMPTY", why[:70], res
    return "PASS", _headline(res), res


def _headline(res: client.Result) -> str:
    """One line of real data, so the table proves the values are live."""
    k, f, rows = res.kind, res.fields, res.rows

    if k == "quote":
        return (
            f"px {_fmt(f.get('price'))} {f.get('price_currency')}  "
            f"fwdEPS {_fmt(f.get('forward_eps'))} {f.get('eps_unit')}"
        )
    if k == "price_history":
        return (
            f"{f.get('bars')} bars to {f.get('last_date')}  "
            f"close {_fmt(f.get('last_close'))} {f.get('currency')}  "
            f"12m {_fmt(f.get('return_12m_pct'))}%  vol {_fmt(f.get('annualised_vol_pct'))}%"
        )
    if k == "analyst_estimates":
        covered = [r for r in rows if r.get("covered")]
        if not covered:
            return "no covered periods"
        r0 = covered[0]
        return (
            f"{r0['period']}: EPS {_fmt(r0.get('eps_avg'))} {r0.get('eps_unit')} "
            f"(n={_fmt(r0.get('eps_analysts'))})  rev {_fmt(r0.get('revenue_avg_m'))} "
            f"{r0.get('revenue_unit')}  [{len(covered)}/{len(rows)} periods]"
        )
    if k == "estimate_revisions":
        r0 = rows[0]
        return (
            f"{r0['period']}: now {_fmt(r0.get('eps_current'))} vs 90d "
            f"{_fmt(r0.get('eps_90d_ago'))}  up30 {_fmt(r0.get('revised_up_30d'))} "
            f"down30 {_fmt(r0.get('revised_down_30d'))}"
        )
    if k == "earnings_surprise_history":
        return (
            f"{f.get('reported_quarters')} reported  "
            f"{f.get('beat_count')}beat/{f.get('miss_count')}miss  "
            f"mean {_fmt(f.get('mean_surprise_pct'))}%"
        )
    if k == "earnings_calendar":
        return (
            f"next {f.get('next_earnings_date')}  "
            f"EPS est {_fmt(f.get('eps_estimate_avg'))} {f.get('eps_unit')}"
        )
    if k == "reported_financials":
        lines = [r["line"] for r in rows]
        rev = next((r for r in rows if r["line"] == "Total Revenue"), None)
        latest = ""
        if rev:
            p = res.fields["periods"][0]
            latest = f"  rev[{p}] {_fmt(rev['values'].get(p))} {f.get('money_unit')}"
        return f"{f.get('frequency')}  {len(lines)} lines x {len(f.get('periods', []))} periods{latest}"
    if k == "analyst_coverage":
        return (
            f"target mean {_fmt(f.get('target_mean'))} vs px "
            f"{_fmt(f.get('price_current'))}  {len(rows)} rating periods"
        )
    return f"{len(rows)} rows"


CHECKS: tuple[tuple[str, Callable[..., client.Result], dict], ...] = (
    ("quote_snapshot", client.quote_snapshot, {}),
    ("price_history", client.price_history, {"period": "1y"}),
    ("analyst_estimates", client.analyst_estimates, {}),
    ("estimate_revisions", client.estimate_revisions, {}),
    ("earnings_surprise_history", client.earnings_surprise_history, {}),
    ("earnings_calendar", client.earnings_calendar, {}),
    ("reported_financials(q)", client.reported_financials, {"freq": "quarterly"}),
    ("reported_financials(a)", client.reported_financials, {"freq": "annual"}),
    ("analyst_coverage", client.analyst_coverage, {}),
)


def main(argv: list[str]) -> int:
    verbose = "-v" in argv or "--verbose" in argv

    # The house style uses em dashes; the default Windows console is cp1252 and
    # would raise or mangle on them. Force UTF-8 rather than flatten the prose.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    print("=" * 108)
    print("marketdata check — source:", client.SOURCE)
    print("=" * 108)

    results: dict[tuple[str, str], client.Result] = {}
    statuses: dict[tuple[str, str], str] = {}

    for ticker in TICKERS:
        sym = client.resolve(ticker)
        print(f"\n### {ticker} -> {sym.yahoo if sym else '??'} ({sym.name if sym else '??'})")
        print(f"{'function':<28} {'status':<8} detail")
        print("-" * 108)
        for name, fn, kw in CHECKS:
            status, headline, res = _probe(fn, ticker, **kw)
            statuses[(ticker, name)] = status
            if res is not None:
                results[(ticker, name)] = res
            print(f"{name:<28} {status:<8} {headline}")
            if verbose and res is not None:
                print(
                    json.dumps(
                        {
                            "as_of": res.as_of,
                            "fields": res.fields,
                            "rows": res.rows[:4],
                            "notes": res.notes,
                        },
                        indent=2,
                        default=str,
                    )
                )

    # ---- summary -------------------------------------------------------
    print("\n" + "=" * 108)
    print("SUMMARY MATRIX")
    print("=" * 108)
    header = f"{'function':<28}" + "".join(f"{t:<10}" for t in TICKERS)
    print(header)
    print("-" * 108)
    for name, _, _ in CHECKS:
        line = f"{name:<28}"
        for t in TICKERS:
            line += f"{statuses.get((t, name), '?'):<10}"
        print(line)

    counts: dict[str, int] = {}
    for s in statuses.values():
        counts[s] = counts.get(s, 0) + 1
    total = len(statuses)
    print("-" * 108)
    print("  ".join(f"{k}={v}" for k, v in sorted(counts.items())) + f"  of {total}")

    hard_fail = counts.get("RAISED", 0) + counts.get("BADTYPE", 0) + counts.get("NOPROV", 0)

    print("\n" + "=" * 108)
    print("CONFIRMED COVERAGE GAPS (EMPTY = Yahoo has no data; not a code fault)")
    print("=" * 108)
    gaps = [(t, n) for (t, n), s in statuses.items() if s == "EMPTY"]
    if not gaps:
        print("  none — every function returned data for every ticker")
    for t, n in gaps:
        res = results.get((t, n))
        why = res.notes[0] if res and res.notes else "unknown"
        print(f"  {t:<5} {n:<28} {why}")

    # Partial coverage is invisible in a pass/fail table but matters a lot.
    print("\n" + "=" * 108)
    print("PARTIAL COVERAGE (PASS, but some periods unavailable)")
    print("=" * 108)
    found = False
    for t in TICKERS:
        res = results.get((t, "analyst_estimates"))
        if not res:
            continue
        missing = [r["period"] for r in res.rows if not r.get("covered")]
        if missing:
            found = True
            print(f"  {t:<5} analyst_estimates: no consensus for {', '.join(missing)}")
    for t in TICKERS:
        res = results.get((t, "reported_financials(a)"))
        if not res:
            continue
        for note in res.notes:
            if "entirely empty" in note:
                found = True
                print(f"  {t:<5} reported_financials(a): {note}")
    if not found:
        print("  none")

    tool_fail = _check_tool_layer()

    print("\n" + "=" * 108)
    total_fail = hard_fail + tool_fail
    print("RESULT:", "PASS" if total_fail == 0 else f"FAIL ({total_fail} hard failures)")
    print("=" * 108)
    return 0 if total_fail == 0 else 1


def _check_tool_layer() -> int:
    """Exercise the @function_tool wrappers themselves.

    The client layer passing is not enough. `function_tool` builds a strict JSON
    schema from each signature and parses the Args: block out of the docstring
    at import time, so a wrapper can be broken while the function under it is
    fine. This invokes each tool exactly as the runner would and checks the
    output carries provenance and respects the truncation bound.
    """
    import asyncio

    print("\n" + "=" * 108)
    print("TOOL LAYER (@function_tool wrappers, invoked as the agent runner would)")
    print("=" * 108)

    try:
        # ToolContext, not RunContextWrapper: the SDK reads ctx.tool_name inside
        # on_invoke_tool, and a bare RunContextWrapper makes every call fail with
        # a swallowed AttributeError that looks exactly like a broken tool.
        from agents.tool_context import ToolContext

        from .tools import MARKET_TOOLS
    except Exception as e:
        print(f"  IMPORT FAILED: {type(e).__name__}: {e}")
        return 1

    failures = 0
    print(f"{'tool':<32} {'ticker':<7} {'status':<8} chars  provenance")
    print("-" * 108)

    async def run_all() -> None:
        nonlocal failures
        for tool in MARKET_TOOLS:
            for ticker in TICKERS:
                args = json.dumps({"ticker": ticker})
                ctx = ToolContext(
                    context=None,
                    tool_name=tool.name,
                    tool_call_id=f"check-{tool.name}-{ticker}",
                    tool_arguments=args,
                )
                try:
                    out = await tool.on_invoke_tool(ctx, args)
                except Exception as e:
                    print(f"{tool.name:<32} {ticker:<7} {'RAISED':<8} {type(e).__name__}: {e}")
                    failures += 1
                    continue
                text = out if isinstance(out, str) else str(out)
                has_src = "SOURCE:" in text
                has_asof = "AS OF:" in text
                bounded = len(text) <= 8000 + 40  # _MAX plus the truncation marker
                ok = has_src and has_asof and bounded and not text.startswith("Error:")
                if not ok:
                    failures += 1
                flags = (
                    f"src={'Y' if has_src else 'N'} asof={'Y' if has_asof else 'N'} "
                    f"bounded={'Y' if bounded else 'N'}"
                )
                print(
                    f"{tool.name:<32} {ticker:<7} {'PASS' if ok else 'FAIL':<8} "
                    f"{len(text):<6} {flags}"
                )

    try:
        asyncio.run(run_all())
    except Exception as e:
        print(f"  TOOL RUN FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        return failures + 1

    print("-" * 108)
    print(f"  tool-layer failures: {failures}")
    return failures


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
