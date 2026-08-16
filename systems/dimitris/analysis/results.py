"""results.py — a navigable results browser, built from the run store.

The forecasts, the reasoning behind each number, the evidence rows that back
it, and the three analyst reports that fed it, in one page you can actually
walk through. Zero JavaScript: navigation is anchors, drill-downs are
<details>, so it works from a file:// URL with nothing installed.

    python -m analysis.results                      # newest run
    python -m analysis.results --run-id fullrun1
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
    "HD": ("Home Depot", "FY2026 Q2", "reports 18 Aug"),
    "ADI": ("Analog Devices", "FY2026 Q3", "reports 19 Aug"),
    "HAS": ("Hays plc", "FY2026", "reports 20 Aug"),
    "DE": ("Deere &amp; Company", "FY2026 Q3", "reports 20 Aug"),
}

CSS = """
:root{
  --ground:#f9fafb; --surface:#ffffff; --raise:#ffffff; --ink:#121619; --muted:#5b6570;
  --line:#e3e7eb; --line-strong:#c8d0d6; --band:#f2f5f4;
  --accent:#0e5c63; --accent-soft:#e6f1f1; --accent-ink:#0a4348;
  --ok:#1c6b45; --warn:#8a6100; --bad:#a32a2a;
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    --ground:#0f1315; --surface:#161b1e; --raise:#1b2225; --ink:#e6ebed; --muted:#95a0a7;
    --line:#252d31; --line-strong:#39444a; --band:#141a1c;
    --accent:#5fb8bf; --accent-soft:#122b2d; --accent-ink:#8ed3d8;
    --ok:#5fbf8f; --warn:#d6a15c; --bad:#e0787c;
  }
}
:root[data-theme="dark"]{
  --ground:#0f1315; --surface:#161b1e; --raise:#1b2225; --ink:#e6ebed; --muted:#95a0a7;
  --line:#252d31; --line-strong:#39444a; --band:#141a1c;
  --accent:#5fb8bf; --accent-soft:#122b2d; --accent-ink:#8ed3d8;
  --ok:#5fbf8f; --warn:#d6a15c; --bad:#e0787c;
}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);
  font:15.5px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
h1,h2,h3{font-family:Georgia,"Iowan Old Style","Times New Roman",serif;margin:0;text-wrap:balance}
h1{font-size:32px;line-height:1.15}
h2{font-size:23px}
h3{font-size:16.5px}
a{color:var(--accent);text-underline-offset:2px}
a:focus-visible,summary:focus-visible{outline:2px solid var(--accent);outline-offset:3px;border-radius:3px}
.wrap{max-width:1080px;margin:0 auto;padding:44px 26px 96px}
.eyebrow{font-size:11.5px;letter-spacing:.13em;text-transform:uppercase;color:var(--muted);
  font-weight:700;margin:0 0 9px}
.lede{color:var(--muted);max-width:66ch;margin:8px 0 0}
nav{position:sticky;top:0;z-index:5;background:var(--ground);
  border-bottom:1px solid var(--line);padding:11px 0;margin:26px 0 30px;
  display:flex;gap:8px;flex-wrap:wrap}
nav a{display:inline-block;padding:6px 13px;border:1px solid var(--line-strong);
  border-radius:999px;text-decoration:none;font-size:13.5px;font-weight:650;
  color:var(--ink);background:var(--surface)}
nav a:hover{background:var(--accent-soft);border-color:var(--accent);color:var(--accent-ink)}
.co{margin:0 0 46px;scroll-margin-top:70px}
.co-head{display:flex;justify-content:space-between;align-items:flex-end;gap:16px;
  border-top:2px solid var(--ink);padding-top:13px;margin:0 0 18px;flex-wrap:wrap}
.co-head .meta{font-size:13px;color:var(--muted);text-align:right}
.metric{background:var(--surface);border:1px solid var(--line);border-radius:7px;
  padding:17px 19px;margin:0 0 13px}
.metric-top{display:flex;justify-content:space-between;gap:18px;align-items:flex-start;
  flex-wrap:wrap;margin:0 0 11px}
.metric-name{font-weight:650;font-size:16px}
.metric-units{color:var(--muted);font-size:13px;margin-top:2px}
.figure{text-align:right;white-space:nowrap}
.figure .v{font-size:29px;font-weight:600;letter-spacing:-.02em;
  font-variant-numeric:tabular-nums;line-height:1.1}
.figure .r{font-size:12.5px;color:var(--muted);font-variant-numeric:tabular-nums;margin-top:2px}
.tag{display:inline-block;font-size:11px;font-weight:700;letter-spacing:.05em;padding:2px 9px;
  border-radius:999px;background:var(--accent-soft);color:var(--accent-ink);white-space:nowrap}
.tag.ok{background:color-mix(in srgb,var(--ok) 15%,transparent);color:var(--ok)}
.tag.warn{background:color-mix(in srgb,var(--warn) 17%,transparent);color:var(--warn)}
.tag.bad{background:color-mix(in srgb,var(--bad) 15%,transparent);color:var(--bad)}
.kv{margin:0 0 9px}
.kv .k{font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);
  font-weight:700;margin-bottom:2px}
.kv .val{font-size:14.5px}
ul{margin:3px 0 0;padding-left:18px}
li{margin:0 0 4px}
li::marker{color:var(--muted)}
code{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:.88em;
  background:var(--band);border:1px solid var(--line);border-radius:3px;padding:.5px 5px;
  word-break:break-all}
details{border:1px solid var(--line);border-radius:6px;background:var(--surface);
  padding:11px 15px;margin:0 0 9px}
details[open]{background:var(--raise)}
summary{cursor:pointer;font-weight:620;font-size:14px}
details[open]>summary{margin-bottom:10px;padding-bottom:8px;border-bottom:1px solid var(--line)}
.scroll{overflow-x:auto}
table{width:100%;border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums}
th,td{padding:6px 9px;text-align:left;border-bottom:1px solid var(--line);vertical-align:top}
th{font-size:10.5px;letter-spacing:.07em;text-transform:uppercase;color:var(--muted);
  font-weight:700;white-space:nowrap;border-bottom:1px solid var(--line-strong)}
tbody tr:nth-child(even){background:var(--band)}
td.num{text-align:right;white-space:nowrap}
pre{background:var(--band);border:1px solid var(--line);border-radius:6px;padding:13px;
  overflow-x:auto;white-space:pre-wrap;word-break:break-word;max-height:440px;overflow-y:auto;
  font:12px/1.55 ui-monospace,Menlo,monospace;margin:0}
.stats{display:grid;gap:11px;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));margin:0 0 8px}
.stat{background:var(--surface);border:1px solid var(--line);border-radius:6px;padding:12px 14px}
.stat .k{font-size:10.5px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);font-weight:700}
.stat .v{font-size:23px;font-weight:600;font-variant-numeric:tabular-nums;margin-top:2px}
.foot{margin-top:40px;padding-top:15px;border-top:1px solid var(--line);font-size:13px;color:var(--muted)}
@media (max-width:640px){.wrap{padding:28px 16px 60px}h1{font-size:26px}.figure{text-align:left}}
@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
"""


def load(db: Path, run_id: str | None) -> dict:
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    if not run_id:
        r = con.execute("SELECT run_id FROM runs ORDER BY started_at DESC LIMIT 1").fetchone()
        run_id = r["run_id"] if r else ""
    tasks = [dict(t) for t in con.execute(
        "SELECT * FROM tasks WHERE run_id=? ORDER BY started_at", (run_id,))]
    evidence = [dict(e) for e in con.execute(
        "SELECT * FROM evidence WHERE run_id=? ORDER BY ticker, metric, evidence_id", (run_id,))]
    con.close()
    return {"run_id": run_id, "tasks": tasks, "evidence": evidence}


def _json(v: Any) -> Any:
    if isinstance(v, str):
        try:
            return json.loads(v)
        except Exception:
            return v
    return v


def render(data: dict) -> str:
    e = html.escape
    tasks, evidence = data["tasks"], data["evidence"]
    by_key = {t["key"]: t for t in tasks}
    tickers = [tk for tk in ("HD", "ADI", "HAS", "DE") if f"{tk}:central" in by_key]

    total_cost = sum(t.get("cost_usd") or 0 for t in tasks)
    P = [
        "<title>Forecast Results</title>",
        f"<style>{CSS}</style>",
        '<div class="wrap">',
        '<p class="eyebrow">Agents vs Wall Street</p>',
        "<h1>Forecast results</h1>",
        f'<p class="lede">Run <code>{e(data["run_id"])}</code>. Every number below opens into the '
        "method that produced it, the evidence rows that back it, and the three analyst reports "
        "that fed the reconciliation.</p>",
    ]

    n_metrics = sum(
        len((_json(by_key[f"{tk}:central"].get("output")) or {}).get("metrics") or [])
        for tk in tickers
    )
    P.append('<div class="stats" style="margin-top:20px">')
    for k, v in [
        ("Companies", str(len(tickers))),
        ("Metrics", str(n_metrics)),
        ("Evidence rows", f"{len(evidence):,}"),
        ("Run cost", f"${total_cost:.2f}"),
    ]:
        P.append(f'<div class="stat"><div class="k">{k}</div><div class="v">{v}</div></div>')
    P.append("</div>")

    P.append("<nav>")
    for tk in tickers:
        P.append(f'<a href="#{tk}">{tk} &middot; {COMPANY.get(tk, (tk,))[0]}</a>')
    P.append("</nav>")

    for tk in tickers:
        name, period, when = COMPANY.get(tk, (tk, "", ""))
        fc = _json(by_key[f"{tk}:central"].get("output")) or {}
        metrics = fc.get("metrics") or []
        P.append(f'<section class="co" id="{tk}">')
        P.append(
            f'<div class="co-head"><div><h2>{name} <span style="color:var(--muted)">'
            f"({tk})</span></h2><div style='color:var(--muted);font-size:13.5px'>{period}</div></div>"
            f'<div class="meta">{when}<br>'
            f"<code>{tk}-{period.replace(' ','').replace('FY','FY')}.xlsx</code></div></div>"
        )

        for m in metrics:
            v, lo, hi = m.get("value"), m.get("low"), m.get("high")
            conf = (m.get("confidence") or "").lower()
            cls = {"high": "ok", "low": "warn"}.get(conf, "")
            rng = (f"{lo:,.4g} &ndash; {hi:,.4g}" if lo is not None and hi is not None else "&mdash;")
            P.append('<article class="metric">')
            P.append(
                f'<div class="metric-top"><div><div class="metric-name">{e(m.get("label",""))}</div>'
                f'<div class="metric-units">{e(m.get("units",""))}'
                f' &nbsp;<span class="tag {cls}">{e(conf or "—")} confidence</span></div></div>'
                f'<div class="figure"><div class="v">{v:,.2f}</div>'
                f'<div class="r">range {rng}</div></div></div>'
            )
            if m.get("method"):
                P.append(f'<div class="kv"><div class="k">Method</div>'
                         f'<div class="val">{e(m["method"])}</div></div>')
            if m.get("assumptions"):
                P.append('<div class="kv"><div class="k">Assumptions</div><ul>')
                for a in m["assumptions"]:
                    P.append(f"<li>{e(str(a))}</li>")
                P.append("</ul></div>")
            if m.get("drivers_applied"):
                P.append('<div class="kv"><div class="k">Drivers applied</div><ul>')
                for d in m["drivers_applied"]:
                    P.append(f"<li>{e(str(d))}</li>")
                P.append("</ul></div>")
            if m.get("conflicts_resolved"):
                P.append(f'<div class="kv"><div class="k">Conflicts resolved</div>'
                         f'<div class="val">{e(m["conflicts_resolved"])}</div></div>')

            cites = m.get("evidence") or []
            rows = [x for x in evidence
                    if x["ticker"] == tk and x["metric"] == m.get("label")]
            if cites or rows:
                P.append(f"<details><summary>Evidence &mdash; {len(cites)} citation(s), "
                         f"{len(rows)} stored row(s)</summary>")
                if cites:
                    P.append('<div class="scroll"><table><tr><th>Source document</th>'
                             "<th>Locator</th><th>Quote</th></tr>")
                    for c in cites:
                        P.append(f'<tr><td><code>{e(c.get("source",""))}</code></td>'
                                 f'<td>{e(c.get("locator",""))}</td>'
                                 f'<td>{e((c.get("quote","") or "")[:300])}</td></tr>')
                    P.append("</table></div>")
                if rows:
                    P.append('<div class="scroll" style="margin-top:10px"><table>'
                             '<tr><th>Claim</th><th class="num">Value</th><th>Units</th>'
                             "<th>Source</th></tr>")
                    for r in rows[:40]:
                        val = f'{r["value"]:,.4g}' if r["value"] is not None else "&mdash;"
                        P.append(f'<tr><td>{e(r["claim"] or "")}</td><td class="num">{val}</td>'
                                 f'<td>{e(r["units"] or "")}</td>'
                                 f'<td><code>{e((r["source"] or "")[:56])}</code></td></tr>')
                    P.append("</table></div>")
                P.append("</details>")
            P.append("</article>")

        for label, key in [("How the three reports were reconciled", "reconciliation"),
                           ("Sanity checks", "sanity_checks"),
                           ("Known weaknesses", "known_weaknesses")]:
            if fc.get(key):
                P.append(f"<details><summary>{label}</summary>"
                         f'<div style="font-size:14.5px">{e(fc[key])}</div></details>')

        for analyst, title in [("filings", "Filings analyst report"),
                               ("news", "News analyst report"),
                               ("financials", "Financials analyst report")]:
            t = by_key.get(f"{tk}:{analyst}")
            if not t or not t.get("output"):
                continue
            cost = t.get("cost_usd") or 0
            body = json.dumps(_json(t["output"]), indent=2, ensure_ascii=False)
            P.append(
                f"<details><summary>{title} "
                f'<span class="tag {"ok" if t["status"]=="completed" else "bad"}">'
                f'{e(t["status"])}</span> '
                f'<span style="color:var(--muted);font-weight:400">'
                f"${cost:.4f} &middot; {len(body):,} chars</span></summary>"
                f"<pre>{e(body)}</pre></details>"
            )
        P.append("</section>")

    P.append(
        f'<p class="foot">Generated from <code>runs.sqlite</code> &middot; '
        f'{datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC &middot; no network, no scripts. '
        "Nothing here has been submitted. Forecasts are not investment advice.</p></div>"
    )
    return "\n".join(P)


def main() -> int:
    ap = argparse.ArgumentParser(description="Navigable results browser.")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--db", default=str(DEFAULT_DB))
    ap.add_argument("--out", default="results.html")
    a = ap.parse_args()

    data = load(Path(a.db), a.run_id)
    if not data["run_id"]:
        print("No runs found.")
        return 2
    out = Path(a.out) if Path(a.out).is_absolute() else REPO_ROOT / a.out
    out.write_text(render(data), encoding="utf-8")
    print(f"Wrote {out}  ({out.stat().st_size:,} bytes)")
    print(f"  run={data['run_id']}  tasks={len(data['tasks'])}  evidence={len(data['evidence'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
