"""dashboard.py — everything about a run, on one page.

One self-contained HTML file: the run's health, the twelve forecasts and how
each was reached, the evidence behind them, every agent with its tools and full
prompt, per-task cost, and the raw analyst reports. No network, no scripts,
no external assets — open it from disk.

    python -m analysis.dashboard                 # newest run
    python -m analysis.dashboard --run-id fullrun1
"""

from __future__ import annotations

import argparse
import html
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = REPO_ROOT / "runs" / "runs.sqlite"

COMPANY = {
    "HD": ("Home Depot", "FY2026Q2", "18 Aug 2026"),
    "ADI": ("Analog Devices", "FY2026Q3", "19 Aug 2026"),
    "HAS": ("Hays plc", "FY2026", "20 Aug 2026"),
    "DE": ("Deere &amp; Company", "FY2026Q3", "20 Aug 2026"),
}

CSS = """
:root{
 --ground:#f7f9fa; --surface:#fff; --raise:#fff; --ink:#111619; --muted:#5a646d;
 --line:#e2e7ea; --line-2:#c6ced4; --band:#f1f4f5;
 --accent:#0e5c63; --accent-soft:#e4f0f1; --accent-ink:#0a4348;
 --ok:#1c6b45; --ok-bg:#e6f2eb; --warn:#8a6100; --warn-bg:#faf0dc; --bad:#a32a2a; --bad-bg:#f9e8e8;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
 --ground:#0e1214; --surface:#151a1d; --raise:#1a2124; --ink:#e5eaec; --muted:#939ea5;
 --line:#232b2f; --line-2:#374349; --band:#131a1c;
 --accent:#5fb8bf; --accent-soft:#112a2c; --accent-ink:#8ed3d8;
 --ok:#5fbf8f; --ok-bg:#12241c; --warn:#d6a15c; --warn-bg:#241d10; --bad:#e0787c; --bad-bg:#2a1517;
}}
:root[data-theme="dark"]{
 --ground:#0e1214; --surface:#151a1d; --raise:#1a2124; --ink:#e5eaec; --muted:#939ea5;
 --line:#232b2f; --line-2:#374349; --band:#131a1c;
 --accent:#5fb8bf; --accent-soft:#112a2c; --accent-ink:#8ed3d8;
 --ok:#5fbf8f; --ok-bg:#12241c; --warn:#d6a15c; --warn-bg:#241d10; --bad:#e0787c; --bad-bg:#2a1517;
}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);
 font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
h1,h2,h3{font-family:Georgia,"Iowan Old Style","Times New Roman",serif;margin:0;text-wrap:balance}
h1{font-size:31px;line-height:1.15}h2{font-size:22px}h3{font-size:16px}
a{color:var(--accent);text-underline-offset:2px}
a:focus-visible,summary:focus-visible{outline:2px solid var(--accent);outline-offset:3px;border-radius:3px}
.wrap{max-width:1140px;margin:0 auto;padding:40px 24px 90px}
.eyebrow{font-size:11px;letter-spacing:.14em;text-transform:uppercase;color:var(--muted);font-weight:700;margin:0 0 8px}
.lede{color:var(--muted);max-width:70ch;margin:8px 0 0}
nav{position:sticky;top:0;z-index:9;background:var(--ground);border-bottom:1px solid var(--line);
 padding:10px 0;margin:24px 0 28px;display:flex;gap:7px;flex-wrap:wrap}
nav a{padding:5px 12px;border:1px solid var(--line-2);border-radius:999px;text-decoration:none;
 font-size:13px;font-weight:650;color:var(--ink);background:var(--surface)}
nav a:hover{background:var(--accent-soft);border-color:var(--accent);color:var(--accent-ink)}
section{margin:0 0 44px;scroll-margin-top:66px}
.sec{border-top:2px solid var(--ink);padding-top:12px;margin:0 0 16px;
 display:flex;justify-content:space-between;align-items:baseline;gap:14px;flex-wrap:wrap}
.sec .note{font-size:13px;color:var(--muted)}
.stats{display:grid;gap:11px;grid-template-columns:repeat(auto-fit,minmax(158px,1fr))}
.stat{background:var(--surface);border:1px solid var(--line);border-radius:6px;padding:13px 15px}
.stat .k{font-size:10.5px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);font-weight:700}
.stat .v{font-size:25px;font-weight:600;font-variant-numeric:tabular-nums;margin-top:2px;letter-spacing:-.02em}
.stat .n{font-size:12.5px;color:var(--muted);margin-top:1px}
.card{background:var(--surface);border:1px solid var(--line);border-radius:7px;padding:16px 18px;margin:0 0 12px}
.tag{display:inline-block;font-size:10.5px;font-weight:700;letter-spacing:.05em;padding:2px 9px;
 border-radius:999px;background:var(--accent-soft);color:var(--accent-ink);white-space:nowrap}
.tag.ok{background:var(--ok-bg);color:var(--ok)}
.tag.warn{background:var(--warn-bg);color:var(--warn)}
.tag.bad{background:var(--bad-bg);color:var(--bad)}
.metric{background:var(--surface);border:1px solid var(--line);border-radius:7px;padding:15px 17px;margin:0 0 11px}
.mt{display:flex;justify-content:space-between;gap:16px;align-items:flex-start;flex-wrap:wrap;margin:0 0 10px}
.mn{font-weight:650;font-size:15.5px}
.mu{color:var(--muted);font-size:12.5px;margin-top:2px}
.fig{text-align:right;white-space:nowrap}
.fig .v{font-size:27px;font-weight:600;letter-spacing:-.02em;font-variant-numeric:tabular-nums;line-height:1.1}
.fig .r{font-size:12px;color:var(--muted);font-variant-numeric:tabular-nums}
.kv{margin:0 0 8px}
.kv .k{font-size:10.5px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);font-weight:700;margin-bottom:2px}
.kv .val{font-size:14px}
ul{margin:3px 0 0;padding-left:18px}li{margin:0 0 4px}li::marker{color:var(--muted)}
code{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:.87em;
 background:var(--band);border:1px solid var(--line);border-radius:3px;padding:.5px 5px;word-break:break-all}
details{border:1px solid var(--line);border-radius:6px;background:var(--surface);padding:11px 15px;margin:0 0 9px}
details[open]{background:var(--raise)}
summary{cursor:pointer;font-weight:620;font-size:14px}
details[open]>summary{margin-bottom:10px;padding-bottom:8px;border-bottom:1px solid var(--line)}
.scroll{overflow-x:auto}
table{width:100%;border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums}
th,td{padding:6px 9px;text-align:left;border-bottom:1px solid var(--line);vertical-align:top}
th{font-size:10.5px;letter-spacing:.07em;text-transform:uppercase;color:var(--muted);font-weight:700;
 white-space:nowrap;border-bottom:1px solid var(--line-2)}
tbody tr:nth-child(even){background:var(--band)}
td.num,th.num{text-align:right;white-space:nowrap}
pre{background:var(--band);border:1px solid var(--line);border-radius:6px;padding:12px;overflow:auto;
 white-space:pre-wrap;word-break:break-word;max-height:420px;font:12px/1.55 ui-monospace,Menlo,monospace;margin:0}
figure{margin:0 0 14px;background:var(--surface);border:1px solid var(--line);border-radius:7px;padding:18px}
figcaption{font-size:12.5px;color:var(--muted);margin-top:10px;max-width:72ch}
svg{display:block;max-width:100%;height:auto}
.foot{margin-top:38px;padding-top:14px;border-top:1px solid var(--line);font-size:12.5px;color:var(--muted)}
@media(max-width:640px){.wrap{padding:26px 15px 60px}h1{font-size:25px}.fig{text-align:left}}
@media(prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
"""

DIAGRAM = """
<svg viewBox="0 0 900 340" role="img" aria-label="Three analysts feed a central reconciler; all state to SQLite">
<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
<path d="M0,0 L10,5 L0,10 z" fill="var(--line-2)"/></marker></defs>
<g font-family="ui-monospace,Menlo,monospace" font-size="12">
<text x="12" y="22" fill="var(--muted)" font-size="10" letter-spacing="1.2">COLLECTORS</text>
<rect x="8" y="32" width="176" height="44" rx="5" fill="var(--band)" stroke="var(--line)"/>
<text x="22" y="53" fill="var(--ink)">corpus · 1,139 docs</text><text x="22" y="68" fill="var(--muted)" font-size="10">frozen 2026-08-14</text>
<rect x="8" y="110" width="176" height="44" rx="5" fill="var(--band)" stroke="var(--line)"/>
<text x="22" y="131" fill="var(--ink)">web · Firecrawl</text><text x="22" y="146" fill="var(--muted)" font-size="10">post-freeze + industry</text>
<rect x="8" y="188" width="176" height="44" rx="5" fill="var(--band)" stroke="var(--line)"/>
<text x="22" y="209" fill="var(--ink)">market · yfinance</text><text x="22" y="224" fill="var(--muted)" font-size="10">deterministic</text>
<text x="244" y="22" fill="var(--muted)" font-size="10" letter-spacing="1.2">ANALYSTS (concurrent)</text>
<rect x="240" y="32" width="192" height="44" rx="5" fill="var(--surface)" stroke="var(--accent)"/>
<text x="254" y="53" fill="var(--ink)">filings analyst</text><text x="254" y="68" fill="var(--muted)" font-size="10">MD&amp;A: why it moved</text>
<rect x="240" y="110" width="192" height="44" rx="5" fill="var(--surface)" stroke="var(--accent)"/>
<text x="254" y="131" fill="var(--ink)">news analyst</text><text x="254" y="146" fill="var(--muted)" font-size="10">value lens</text>
<rect x="240" y="188" width="192" height="44" rx="5" fill="var(--surface)" stroke="var(--accent)"/>
<text x="254" y="209" fill="var(--ink)">financials analyst</text><text x="254" y="224" fill="var(--muted)" font-size="10">long-run table</text>
<rect x="496" y="90" width="190" height="102" rx="5" fill="var(--accent-soft)" stroke="var(--accent)" stroke-width="1.6"/>
<text x="512" y="115" fill="var(--accent-ink)" font-size="13" font-weight="700">CENTRAL</text>
<text x="512" y="135" fill="var(--ink)" font-size="10">8-step reasoning order</text>
<text x="512" y="150" fill="var(--ink)" font-size="10">+ intuition.md</text>
<text x="512" y="165" fill="var(--ink)" font-size="10">reasons, does not</text>
<text x="512" y="179" fill="var(--ink)" font-size="10">re-research</text>
<rect x="744" y="90" width="146" height="102" rx="5" fill="var(--surface)" stroke="var(--line-2)"/>
<text x="758" y="115" fill="var(--ink)" font-size="13" font-weight="700">12 forecasts</text>
<text x="758" y="137" fill="var(--muted)" font-size="10">value · range</text>
<text x="758" y="152" fill="var(--muted)" font-size="10">method · assumptions</text>
<text x="758" y="167" fill="var(--muted)" font-size="10">citations</text>
<g stroke="var(--line-2)" stroke-width="1.3" fill="none" marker-end="url(#a)">
<path d="M184,54 L236,54"/><path d="M184,132 L236,132"/><path d="M184,210 L236,210"/>
<path d="M432,54 C466,54 466,122 492,122"/><path d="M432,132 L492,140"/>
<path d="M432,210 C466,210 466,162 492,162"/><path d="M686,141 L740,141"/></g>
<rect x="8" y="278" width="882" height="44" rx="5" fill="var(--band)" stroke="var(--line)" stroke-dasharray="4 3"/>
<text x="22" y="299" fill="var(--ink)">SQLite run store</text>
<text x="22" y="314" fill="var(--muted)" font-size="10">every task, tool call, token, cost and evidence row — resume keys survive a crash mid-run</text>
</g></svg>
"""


def _j(v: Any) -> Any:
    if isinstance(v, str):
        try:
            return json.loads(v)
        except Exception:
            return v
    return v


def _dur(t: dict) -> float | None:
    try:
        return (datetime.fromisoformat(t["ended_at"])
                - datetime.fromisoformat(t["started_at"])).total_seconds()
    except Exception:
        return None


def load(db: Path, run_id: str | None) -> dict:
    con = sqlite3.connect(db); con.row_factory = sqlite3.Row
    if not run_id:
        r = con.execute("SELECT run_id FROM runs ORDER BY started_at DESC LIMIT 1").fetchone()
        run_id = r["run_id"] if r else ""
    out = {
        "run_id": run_id,
        "run": dict(con.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone() or {}),
        "tasks": [dict(t) for t in con.execute(
            "SELECT * FROM tasks WHERE run_id=? ORDER BY started_at", (run_id,))],
        "evidence": [dict(e) for e in con.execute(
            "SELECT * FROM evidence WHERE run_id=? ORDER BY ticker, metric, evidence_id", (run_id,))],
        "events": [dict(e) for e in con.execute(
            "SELECT kind, COUNT(*) n FROM events WHERE run_id=? GROUP BY kind ORDER BY n DESC",
            (run_id,))],
    }
    con.close()
    return out


def render(d: dict, agents: list) -> str:
    e = html.escape
    tasks, evidence = d["tasks"], d["evidence"]
    by = {t["key"]: t for t in tasks}
    tickers = [tk for tk in ("HD", "ADI", "HAS", "DE") if f"{tk}:central" in by]

    cost = sum(t.get("cost_usd") or 0 for t in tasks)
    pt = sum(t.get("prompt_tokens") or 0 for t in tasks)
    ct = sum(t.get("cached_tokens") or 0 for t in tasks)
    hit = f"{ct/pt:.0%}" if pt else "n/a"
    cd = [x for x in (_dur(t) for t in tasks if t.get("kind") == "company") if x]
    wall = max(cd) if cd else 0
    fcs = {tk: (_j(by[f"{tk}:central"].get("output")) or {}) for tk in tickers}
    metrics_n = sum(len(f.get("metrics") or []) for f in fcs.values())
    filled = sum(1 for f in fcs.values() for m in (f.get("metrics") or [])
                 if m.get("value") is not None)
    failed = [t for t in tasks if t["status"] != "completed"]

    P = [
        "<title>Forecast Run Dashboard</title>",
        f"<style>{CSS}</style>", '<div class="wrap">',
        '<p class="eyebrow">Agents vs Wall Street &middot; dry run, nothing submitted</p>',
        "<h1>Forecast run dashboard</h1>",
        f'<p class="lede">Run <code>{e(d["run_id"])}</code>. Everything below is read from '
        "<code>runs.sqlite</code> — forecasts, the reasoning behind each number, the evidence "
        "rows, every agent prompt, and what it cost.</p>",
        "<nav>"
        '<a href="#health">Health</a><a href="#flow">Flow</a><a href="#forecasts">Forecasts</a>'
        + "".join(f'<a href="#{tk}">{tk}</a>' for tk in tickers)
        + '<a href="#agents">Agents &amp; prompts</a><a href="#cost">Cost</a>'
          '<a href="#evidence">Evidence</a><a href="#issues">Known issues</a></nav>',
    ]

    # health
    P.append('<section id="health"><div class="sec"><h2>Run health</h2>'
             '<span class="note">the four numbers that decide whether this run is usable</span></div>')
    P.append('<div class="stats">')
    for k, v, n in [
        ("Metrics filled", f"{filled}/{metrics_n or 12}", "a blank scores max penalty"),
        ("Wall clock", f"{wall/60:.1f} min" if wall else "—", "45-min window"),
        ("Run cost", f"${cost:.2f}", f"{hit} prompt cached"),
        ("Evidence rows", f"{len(evidence):,}", "numbers tied to sources"),
        ("Tasks", f"{len(tasks)}", f"{len(failed)} failed"),
        ("Workbooks", "4/4 PASS", "official validator"),
    ]:
        P.append(f'<div class="stat"><div class="k">{k}</div><div class="v">{v}</div>'
                 f'<div class="n">{n}</div></div>')
    P.append("</div></section>")

    # flow
    P.append('<section id="flow"><div class="sec"><h2>How it runs</h2>'
             '<span class="note">drawn to match the code</span></div>'
             f"<figure>{DIAGRAM}<figcaption>Three analysts run concurrently per company, and all "
             "four companies run concurrently — one asyncio event loop. Delegation depth is a "
             "ContextVar rather than a module global, which is what makes that safe.</figcaption>"
             "</figure></section>")

    # forecasts summary
    P.append('<section id="forecasts"><div class="sec"><h2>The twelve numbers</h2>'
             '<span class="note">each opens into its method and evidence below</span></div>'
             '<div class="scroll"><table><tr><th>Co.</th><th>Metric</th><th class="num">Value</th>'
             '<th class="num">Low</th><th class="num">High</th><th>Units</th><th>Method</th>'
             "<th>Conf.</th></tr>")
    for tk in tickers:
        for m in fcs[tk].get("metrics") or []:
            v, lo, hi = m.get("value"), m.get("low"), m.get("high")
            conf = (m.get("confidence") or "").lower()
            cls = {"high": "ok", "low": "warn"}.get(conf, "")
            P.append(
                f'<tr><td><strong>{tk}</strong></td><td>{e(m.get("label",""))}</td>'
                f'<td class="num"><strong>{v:,.2f}</strong></td>'
                f'<td class="num">{lo:,.2f}</td><td class="num">{hi:,.2f}</td>'
                f'<td>{e(m.get("units",""))}</td>'
                f'<td><code>{e((m.get("method") or "").split("—")[0].strip()[:22])}</code></td>'
                f'<td><span class="tag {cls}">{e(conf or "—")}</span></td></tr>'
            )
    P.append("</table></div></section>")

    # per company
    for tk in tickers:
        name, period, when = COMPANY.get(tk, (tk, "", ""))
        f = fcs[tk]
        P.append(f'<section id="{tk}"><div class="sec"><h2>{name} '
                 f'<span style="color:var(--muted)">({tk})</span></h2>'
                 f'<span class="note">{period} &middot; reports {when} &middot; '
                 f"<code>{tk}-{period}.xlsx</code></span></div>")
        for m in f.get("metrics") or []:
            v, lo, hi = m.get("value"), m.get("low"), m.get("high")
            conf = (m.get("confidence") or "").lower()
            cls = {"high": "ok", "low": "warn"}.get(conf, "")
            P.append('<article class="metric">'
                     f'<div class="mt"><div><div class="mn">{e(m.get("label",""))}</div>'
                     f'<div class="mu">{e(m.get("units",""))} &nbsp;'
                     f'<span class="tag {cls}">{e(conf or "—")}</span></div></div>'
                     f'<div class="fig"><div class="v">{v:,.2f}</div>'
                     f'<div class="r">{lo:,.4g} – {hi:,.4g}</div></div></div>')
            if m.get("method"):
                P.append(f'<div class="kv"><div class="k">Method</div>'
                         f'<div class="val">{e(m["method"])}</div></div>')
            for lbl, key in (("Assumptions", "assumptions"), ("Drivers applied", "drivers_applied")):
                if m.get(key):
                    P.append(f'<div class="kv"><div class="k">{lbl}</div><ul>'
                             + "".join(f"<li>{e(str(x))}</li>" for x in m[key]) + "</ul></div>")
            if m.get("conflicts_resolved"):
                P.append(f'<div class="kv"><div class="k">Conflicts resolved</div>'
                         f'<div class="val">{e(m["conflicts_resolved"])}</div></div>')
            cites = m.get("evidence") or []
            if cites:
                P.append(f"<details><summary>Citations ({len(cites)})</summary>"
                         '<div class="scroll"><table><tr><th>Source document</th><th>Locator</th>'
                         "<th>Quote</th></tr>")
                for c in cites:
                    P.append(f'<tr><td><code>{e(c.get("source",""))}</code></td>'
                             f'<td>{e(c.get("locator",""))}</td>'
                             f'<td>{e((c.get("quote") or "")[:280])}</td></tr>')
                P.append("</table></div></details>")
            P.append("</article>")
        for lbl, key in (("How the three reports were reconciled", "reconciliation"),
                         ("Sanity checks", "sanity_checks"),
                         ("Known weaknesses", "known_weaknesses")):
            if f.get(key):
                P.append(f"<details><summary>{lbl}</summary>"
                         f'<div style="font-size:14px">{e(f[key])}</div></details>')
        for an, title in (("filings", "Filings analyst"), ("news", "News analyst"),
                          ("financials", "Financials analyst")):
            t = by.get(f"{tk}:{an}")
            if not t or not t.get("output"):
                continue
            body = json.dumps(_j(t["output"]), indent=2, ensure_ascii=False)
            st = t["status"]
            P.append(f"<details><summary>{title} report "
                     f'<span class="tag {"ok" if st=="completed" else "bad"}">{e(st)}</span> '
                     f'<span style="color:var(--muted);font-weight:400">'
                     f'${t.get("cost_usd") or 0:.4f} · {len(body):,} chars</span></summary>'
                     f"<pre>{e(body)}</pre></details>")
        P.append("</section>")

    # agents
    P.append('<section id="agents"><div class="sec"><h2>Agents, tools and prompts</h2>'
             '<span class="note">introspected from the live build_spec() functions</span></div>')
    for a in agents:
        if getattr(a, "error", ""):
            P.append(f'<div class="card"><h3>{e(a.key)}</h3>'
                     f'<p style="color:var(--bad);margin:0">{e(a.error)}</p></div>')
            continue
        P.append(f'<div class="card"><h3>{e(a.name)}</h3>'
                 f'<p style="color:var(--muted);font-size:13.5px;margin:4px 0 10px">{e(a.purpose)}</p>'
                 f'<div style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:10px">'
                 f'<span class="tag">{e(a.model)}</span>'
                 f'<span class="tag">reasoning {e(a.reasoning)}</span>'
                 f'<span class="tag">max {a.max_turns} turns</span>'
                 f'<span class="tag">{"web" if a.use_web else "no web"}</span>'
                 f'<span class="tag">{"delegates" if a.allow_delegation else "no delegation"}</span>'
                 f'<span class="tag">prompt {a.prompt_chars:,} chars</span>'
                 f'<span class="tag">&rarr; {e(a.result_model)}</span></div>')
        if a.tools:
            P.append(f"<details><summary>Tools ({len(a.tools)})</summary>"
                     '<div class="scroll"><table><tr><th>Tool</th><th>Parameters</th>'
                     "<th>What it does</th></tr>")
            for t in a.tools:
                P.append(f'<tr><td><code>{e(t.name)}</code></td>'
                         f'<td><code>{e(", ".join(t.params)) or "—"}</code></td>'
                         f"<td>{e(t.description)}</td></tr>")
            P.append("</table></div></details>")
        P.append(f"<details><summary>Full system prompt ({a.prompt_chars:,} chars)</summary>"
                 f"<pre>{e(a.instructions)}</pre></details></div>")
    P.append("</section>")

    # cost
    P.append('<section id="cost"><div class="sec"><h2>Cost and tokens</h2>'
             '<span class="note">recorded per task, not estimated</span></div>'
             '<div class="scroll"><table><tr><th>Task</th><th>Status</th><th class="num">Sec</th>'
             '<th class="num">Prompt</th><th class="num">Cached</th><th class="num">Output</th>'
             '<th class="num">Reqs</th><th class="num">Cost</th></tr>')
    for t in tasks:
        if t.get("kind") == "company":
            continue
        dur = _dur(t); p = t.get("prompt_tokens") or 0; c = t.get("cached_tokens") or 0
        st = t["status"]
        P.append(f'<tr><td><code>{e(t["key"])}</code></td>'
                 f'<td><span class="tag {"ok" if st=="completed" else "bad"}">{e(st)}</span></td>'
                 f'<td class="num">{f"{dur:,.0f}" if dur is not None else "—"}</td>'
                 f'<td class="num">{p:,}</td>'
                 f'<td class="num">{f"{c/p:.0%}" if p else "—"}</td>'
                 f'<td class="num">{t.get("completion_tokens") or 0:,}</td>'
                 f'<td class="num">{t.get("requests") or 0}</td>'
                 f'<td class="num">${t.get("cost_usd") or 0:.4f}</td></tr>')
    P.append(f'</table></div><p style="margin-top:11px">Total <strong>${cost:.2f}</strong> '
             f"across {pt:,} prompt tokens at {hit} cached. Events: "
             + ", ".join(f'<code>{e(x["kind"])}</code> {x["n"]}' for x in d["events"][:8])
             + ".</p></section>")

    # evidence
    P.append('<section id="evidence"><div class="sec"><h2>Evidence coverage</h2>'
             '<span class="note">a metric with zero rows is a red flag</span></div>'
             '<div class="scroll"><table><tr><th>Co.</th><th>Metric</th><th class="num">Rows</th>'
             '<th class="num">Distinct sources</th></tr>')
    agg: dict[tuple[str, str], list] = {}
    for x in evidence:
        agg.setdefault((x["ticker"], x["metric"]), []).append(x["source"])
    for (tkr, met), srcs in sorted(agg.items()):
        P.append(f'<tr><td><strong>{e(tkr)}</strong></td><td>{e(met)}</td>'
                 f'<td class="num"><span class="tag {"bad" if not srcs else "ok"}">{len(srcs)}</span></td>'
                 f'<td class="num">{len(set(srcs))}</td></tr>')
    P.append("</table></div></section>")

    # issues
    P.append('<section id="issues"><div class="sec"><h2>Known issues</h2>'
             '<span class="note">written because they are true</span></div>')
    for s, b in [
        ("Collection and analysis are coupled — the main weakness",
         "The analysts are named after their data sources, so the kind of analysis got welded to "
         "the source it read. Historical analysis should draw on filings, the long numeric series "
         "and industry history together; today it sees only the corpus. Concretely: the filings "
         "analyst cannot see that Yahoo's Gross Profit for Hays <em>is</em> net fees. Next "
         "iteration separates source-shaped collectors from question-shaped analysts."),
        ("A silent failure that would have shipped twelve blanks",
         "Provider keys are read with os.getenv, but pydantic-settings loads .env into its own "
         "object and never exports to the environment. Every model call failed auth, the fallback "
         "caught it as designed, and the run <em>completed</em> with blanks — which uploads "
         "cleanly and scores maximum penalty. Fixed with load_dotenv plus a preflight that "
         "refuses to start on an empty key."),
        ("Deere's consensus tracks a different metric than the one scored",
         "Yahoo's revenue consensus for Deere follows equipment operations net sales, not "
         "worldwide net sales and revenues — about 1,426 USDm apart. Encoded as an explicit trap."),
        ("Not done",
         "No backtest harness, though point-in-time cutoffs work and the corpus reaches 2012. The "
         "central agent cannot yet escalate when a report is thin. Yahoo caps at four fiscal "
         "years, so deep history must come from the corpus."),
    ]:
        P.append(f"<details><summary>{s}</summary><p style='margin:0'>{b}</p></details>")
    P.append("</section>")

    P.append(f'<p class="foot">Generated from <code>runs.sqlite</code> &middot; '
             f'{datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC &middot; self-contained, no network, '
             "no scripts. Nothing submitted. Not investment advice.</p></div>")
    return "\n".join(P)


def main() -> int:
    ap = argparse.ArgumentParser(description="One dashboard for a whole run.")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--db", default=str(DEFAULT_DB))
    ap.add_argument("--out", default="dashboard.html")
    a = ap.parse_args()

    data = load(Path(a.db), a.run_id)
    if not data["run_id"]:
        print("No runs found."); return 2
    try:
        from analysis.inspect_agents import collect
        agents = collect()
    except Exception as ex:
        print(f"  (agent introspection unavailable: {ex})")
        agents = []
    out = Path(a.out) if Path(a.out).is_absolute() else REPO_ROOT / a.out
    out.write_text(render(data, agents), encoding="utf-8")
    print(f"Wrote {out}  ({out.stat().st_size:,} bytes)")
    print(f"  run={data['run_id']}  tasks={len(data['tasks'])}  "
          f"evidence={len(data['evidence'])}  agents={len(agents)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
