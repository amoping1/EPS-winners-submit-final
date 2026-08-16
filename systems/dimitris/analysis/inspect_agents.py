"""inspect_agents.py — dump every agent, its tools and its full prompt.

Built for review rather than telemetry: before trusting a forecast you want to
read what each agent was actually told, with what tools, on which model. Every
value here is introspected from the live `build_spec()` functions, so it cannot
drift from what runs.

    python -m analysis.inspect_agents              # terminal summary
    python -m analysis.inspect_agents --html out.html
    python -m analysis.inspect_agents --agent news --prompt   # full prompt text
"""

from __future__ import annotations

import argparse
import html
import json
import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Callable

from agent_core import settings
from agent_core.spec import AgentSpec

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class ToolInfo:
    name: str
    description: str
    params: list[str] = field(default_factory=list)
    origin: str = ""  # where the tool came from, and any wrapping applied


@dataclass
class AgentInfo:
    key: str
    name: str
    purpose: str
    profile: str
    model: str
    reasoning: str
    max_turns: int
    use_web: bool
    allow_delegation: bool
    result_model: str
    result_fields: list[tuple[str, str]]
    tools: list[ToolInfo]
    instructions: str
    error: str = ""
    tools_are_assembled: bool = True  # False => spec-only fallback, under-reports
    prompt_parts: list[tuple[str, str, str]] = field(default_factory=list)
    ticker: str = ""

    @property
    def prompt_chars(self) -> int:
        return len(self.instructions)


# Which module each tool name comes from, so the page says where to go and edit.
_TOOL_ORIGIN: dict[str, str] = {}
for _mod, _label in (
    ("corpus.tools", "corpus/tools.py"),
    ("marketdata.tools", "marketdata/tools.py"),
):
    try:
        _m = __import__(_mod, fromlist=["*"])
        for _t in getattr(_m, "CORPUS_TOOLS", []) or getattr(_m, "MARKET_TOOLS", []):
            _TOOL_ORIGIN[getattr(_t, "name", "")] = _label
    except Exception:  # pragma: no cover - a broken module is survivable here
        pass
_TOOL_ORIGIN.setdefault("firecrawl_search", "agent_core/tools.py (WEB_TOOLS)")
_TOOL_ORIGIN.setdefault("firecrawl_scrape", "agent_core/tools.py (WEB_TOOLS)")
_TOOL_ORIGIN.setdefault("delegate_to_subagent", "agent_core/tools.py (allow_delegation)")
_TOOL_ORIGIN.setdefault("submit_result", "agent_core/tools.py (terminal tool)")


def _tool_info(tool: Any) -> ToolInfo:
    name = getattr(tool, "name", getattr(tool, "__name__", "?"))
    # The FULL docstring, not the first line: this is verbatim what the model
    # is shown, and truncating it hides the contract being reviewed.
    desc = (getattr(tool, "description", "") or "").strip()
    params: list[str] = []
    schema = getattr(tool, "params_json_schema", None)
    if isinstance(schema, dict):
        props = schema.get("properties") or {}
        required = set(schema.get("required") or [])
        for p in props:
            params.append(p if p in required else f"{p}?")
    return ToolInfo(
        name=name,
        description=desc,
        params=params,
        origin=_TOOL_ORIGIN.get(name, ""),
    )


def _assembled_tools(spec: AgentSpec) -> tuple[list[ToolInfo], bool]:
    """The tools the agent ACTUALLY runs with.

    `spec.tools` is only the extra list — `create_agent` also adds WEB_TOOLS when
    `use_web`, `delegate_to_subagent` when `allow_delegation`, and always the
    terminal `submit_result`. Reporting `spec.tools` told the reader the news
    analyst had zero tools when it runs with two. Building the real agent is the
    same code path the run uses, so this cannot drift.
    """
    try:
        from agent_core.agent import create_agent

        return [_tool_info(t) for t in create_agent(spec).tools], True
    except Exception as e:  # no API key, or a provider import problem
        logger.warning(
            "Could not build %r to read its real tool list (%s) — falling back to "
            "spec.tools, which UNDER-REPORTS.", spec.name, e,
        )
        return [_tool_info(t) for t in spec.tools], False


def _result_fields(model_cls: Any) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for fname, finfo in getattr(model_cls, "model_fields", {}).items():
        ann = getattr(finfo, "annotation", None)
        label = getattr(ann, "__name__", None) or str(ann).replace("typing.", "")
        out.append((fname, str(label)[:60]))
    return out


def _prompt_parts(key: str, ticker: str) -> list[tuple[str, str, str]]:
    """(part name, file to edit, text) — so a reader knows WHERE to change what.

    Only the central agent assembles its prompt from several sources; the others
    are a single constant and say so.
    """
    if key != "central":
        src = {
            "profile": ("PROFILE_PROMPT", "analysts/profile.py"),
            "filings": ("FILINGS_SYSTEM_PROMPT", "analysts/filings.py"),
            "news": ("SYSTEM_PROMPT (rendered per company)", "analysts/news.py"),
            "financials": ("FINANCIALS_SYSTEM_PROMPT", "analysts/financials.py"),
        }.get(key)
        return [(src[0], src[1], "")] if src else []
    from analysts import central as c

    return [
        ("PROMPT_HEAD — role and inputs", "analysts/central.py", c.PROMPT_HEAD),
        (
            "ANALYST INTUITION — human judgement, injected verbatim. EDIT THIS ONE.",
            "analysts/intuition.md",
            c.load_intuition(),
        ),
        ("PROMPT_TAIL — reasoning order, then mechanics", "analysts/central.py", c.PROMPT_TAIL),
        ("COMPANY BRIEF — labels, units, magnitude bands", "starter/challenge/companies.json",
         c.company_brief(ticker)),
    ]


def _from_spec(key: str, purpose: str, spec: AgentSpec, ticker: str) -> AgentInfo:
    profile = spec.resolved_profile
    tools, assembled = _assembled_tools(spec)
    return AgentInfo(
        key=key,
        name=spec.name,
        purpose=purpose,
        profile=profile,
        model=settings.model_for(profile),
        reasoning=spec.resolved_reasoning_effort or "(none)",
        max_turns=spec.resolved_max_turns,
        use_web=spec.use_web,
        allow_delegation=spec.allow_delegation,
        result_model=spec.result_model.__name__,
        result_fields=_result_fields(spec.result_model),
        tools=tools,
        tools_are_assembled=assembled,
        instructions=spec.instructions,
        prompt_parts=_prompt_parts(key, ticker),
        ticker=ticker,
    )


def _builders(ticker: str) -> list[tuple[str, str, Callable[[], AgentSpec]]]:
    """(key, purpose, factory). Imported lazily so one broken module is survivable."""
    today = date.today().isoformat()

    def profile_spec():
        from analysts.profile import build_spec
        return build_spec(ticker)

    def filings_spec():
        from analysts.filings import build_spec
        return build_spec(ticker)

    def news_spec():
        from analysts.news import build_spec, install_openstocks_guard
        install_openstocks_guard()  # the run does this too; the tools are wrapped
        return build_spec(ticker, today)

    def financials_spec():
        from analysts.financials import build_spec
        return build_spec(ticker)

    def central_spec():
        from analysts.central import build_spec
        return build_spec(ticker)

    return [
        ("profile", "Long-term company study. Built once, cached, feeds everything else.", profile_spec),
        ("filings", "Reads the frozen offline corpus: reported history, guidance, what drives the business, and how it behaved when its cycle turned.", filings_spec),
        ("news", "Post-freeze and industry evidence through a value-investing lens. Rejects trader content explicitly. Web tools are wrapped by install_openstocks_guard() — openstocks.com is refused in code, not just in the prompt.", news_spec),
        ("financials", "Long-run financial history from Yahoo. Deterministic — no model touches the numbers.", financials_spec),
        ("central", "Weighs the three reports into the 12 forecasts. Reasons rather than researches; the human intuition file sits near the top of its prompt.", central_spec),
    ]


def collect(ticker: str = "HD") -> list[AgentInfo]:
    out: list[AgentInfo] = []
    for key, purpose, factory in _builders(ticker):
        try:
            out.append(_from_spec(key, purpose, factory(), ticker))
        except Exception as e:
            out.append(
                AgentInfo(
                    key=key, name=key, purpose=purpose, profile="?", model="?",
                    reasoning="?", max_turns=0, use_web=False, allow_delegation=False,
                    result_model="?", result_fields=[], tools=[], instructions="",
                    error=f"{type(e).__name__}: {e}", ticker=ticker,
                )
            )
    return out


# ---------------------------------------------------------------------------
# Terminal
# ---------------------------------------------------------------------------


def print_summary(agents: list[AgentInfo], show_prompt: str | None = None) -> None:
    print()
    print("=" * 100)
    print("AGENT INVENTORY".center(100))
    print("=" * 100)
    for a in agents:
        print()
        if a.error:
            print(f"  {a.key.upper():<12} FAILED TO BUILD — {a.error}")
            continue
        web = "web" if a.use_web else "no-web"
        dele = "delegates" if a.allow_delegation else "no-delegation"
        print(f"  {a.key.upper():<12} {a.name}")
        print(f"  {'':<12} {a.purpose}")
        print(f"  {'':<12} model={a.model}  reasoning={a.reasoning}  max_turns={a.max_turns}  {web}  {dele}")
        print(f"  {'':<12} prompt={a.prompt_chars:,} chars   ->  {a.result_model} ({len(a.result_fields)} fields)")
        if not a.tools_are_assembled:
            print(f"  {'':<12} !! could not build the agent — tool list below is spec-only and UNDER-REPORTS")
        if a.tools:
            print(f"  {'':<12} tools ({len(a.tools)}, as actually assembled):")
            for t in a.tools:
                sig = ", ".join(t.params)
                origin = f"   [{t.origin}]" if t.origin else ""
                print(f"  {'':<14} - {t.name}({sig}){origin}")
                if t.description:
                    print(f"  {'':<16}   {t.description.splitlines()[0][:88]}")
        else:
            print(f"  {'':<12} tools: none")
        if a.prompt_parts:
            print(f"  {'':<12} prompt assembled from:")
            for name, path, text in a.prompt_parts:
                size = f"{len(text):,} chars" if text else "see file"
                print(f"  {'':<14} - {name}  ({size})  <- {path}")

    if show_prompt:
        match = next((a for a in agents if a.key == show_prompt), None)
        if not match:
            print(f"\nNo agent named {show_prompt!r}. Known: {', '.join(a.key for a in agents)}")
            return
        print()
        print("=" * 100)
        print(f"FULL SYSTEM PROMPT — {match.key}  ({match.prompt_chars:,} chars)".center(100))
        print("=" * 100)
        print(match.instructions)


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

_CSS = """
:root{--bg:#fbfbfa;--fg:#1a1a18;--muted:#6b6b66;--line:#e3e3df;--card:#fff;
--accent:#7a4522;--accent-soft:#f2ebe4;--ok:#2d6a4f;--warn:#9a6700;--code:#f6f6f4}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#16161a;--fg:#e9e9e4;
--muted:#9a9a93;--line:#2c2c33;--card:#1d1d22;--accent:#d9a066;--accent-soft:#2a2118;
--ok:#74c69d;--warn:#e0b252;--code:#232329}}
:root[data-theme=dark]{--bg:#16161a;--fg:#e9e9e4;--muted:#9a9a93;--line:#2c2c33;--card:#1d1d22;
--accent:#d9a066;--accent-soft:#2a2118;--ok:#74c69d;--warn:#e0b252;--code:#232329}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.6 ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:1100px;margin:0 auto;padding:40px 24px 80px}
h1{font-size:26px;margin:0 0 4px;letter-spacing:-.01em}
.sub{color:var(--muted);margin:0 0 32px;font-size:14px}
.flow{background:var(--card);border:1px solid var(--line);border-radius:12px;
padding:20px;margin:0 0 32px;overflow-x:auto}
.flow pre{margin:0;font:12.5px/1.55 ui-monospace,"Cascadia Code",Menlo,monospace;color:var(--fg)}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;
padding:20px 22px;margin:0 0 18px}
.card h2{font-size:18px;margin:0 0 2px;color:var(--accent)}
.purpose{color:var(--muted);font-size:14px;margin:0 0 14px}
.meta{display:flex;flex-wrap:wrap;gap:6px;margin:0 0 14px}
.pill{background:var(--accent-soft);color:var(--accent);border-radius:999px;
padding:3px 10px;font-size:12px;font-weight:600;white-space:nowrap}
.pill.plain{background:transparent;border:1px solid var(--line);color:var(--muted);font-weight:400}
table{width:100%;border-collapse:collapse;font-size:13.5px}
th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--muted);font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:.04em}
code{background:var(--code);border-radius:4px;padding:1px 5px;
font:12.5px ui-monospace,Menlo,monospace}
details{margin-top:14px;border-top:1px solid var(--line);padding-top:12px}
summary{cursor:pointer;font-size:13.5px;font-weight:600;color:var(--accent)}
pre.prompt{background:var(--code);border-radius:8px;padding:16px;overflow-x:auto;
white-space:pre-wrap;word-break:break-word;font:12.5px/1.6 ui-monospace,Menlo,monospace;
max-height:520px;overflow-y:auto;margin:12px 0 0}
.err{color:#b4232c;font-weight:600}
.scroll{overflow-x:auto}
pre.doc{margin:0;white-space:pre-wrap;word-break:break-word;max-width:52ch;
font:12px/1.5 ui-monospace,Menlo,monospace;color:var(--muted);max-height:12em;
overflow-y:auto}
td{vertical-align:top}
"""

_FLOW = """                      corpus tools (8)
                            |
   [ long-term profile ] ---+---> [ filings analyst ]  --> FilingsReport ---.
    built once, cached                                                       |
                                                                             |
   Firecrawl search/scrape ------> [ news analyst ]     --> NewsReport -----+--> [ CENTRAL ]
    value-investor lens                                                      |     reconciles
                                                                             |     8-step order
   yfinance (deterministic) -----> [ financials ]       --> FinancialsReport-'     + intuition.md
                                    no model involved                              |
                                                                                   v
                                                                        12 metric forecasts
                                                                        + evidence trail

   every task -> SQLite: resume keys, token cost, tool calls, parent/child flow"""


def render_html(agents: list[AgentInfo]) -> str:
    e = html.escape
    parts = [
        "<title>Forecasting Agents</title>",
        f"<style>{_CSS}</style>",
        '<div class="wrap">',
        "<h1>Forecasting agents — tools and prompts</h1>",
        '<p class="sub">Introspected from the live <code>build_spec()</code> functions, '
        "so nothing here can drift from what actually runs.</p>",
        f'<div class="flow"><pre>{e(_FLOW)}</pre></div>',
    ]
    for a in agents:
        parts.append('<div class="card">')
        parts.append(f"<h2>{e(a.name)}</h2>")
        if a.error:
            parts.append(f'<p class="err">Failed to build: {e(a.error)}</p></div>')
            continue
        parts.append(f'<p class="purpose">{e(a.purpose)}</p>')
        parts.append('<div class="meta">')
        for label in (
            a.model,
            f"reasoning: {a.reasoning}",
            f"max turns: {a.max_turns}",
            "web" if a.use_web else "no web",
            "delegates" if a.allow_delegation else "no delegation",
        ):
            parts.append(f'<span class="pill">{e(str(label))}</span>')
        parts.append(f'<span class="pill plain">prompt {a.prompt_chars:,} chars</span>')
        parts.append(
            f'<span class="pill plain">&rarr; {e(a.result_model)} '
            f"({len(a.result_fields)} fields)</span>"
        )
        parts.append("</div>")

        if not a.tools_are_assembled:
            parts.append(
                '<p class="err">Could not build this agent (usually a missing API '
                "key), so the tool list below is <code>spec.tools</code> only and "
                "under-reports the real runtime set.</p>"
            )
        if a.tools:
            parts.append(
                f'<p class="purpose">{len(a.tools)} tools, as actually assembled by '
                "<code>create_agent()</code> — this includes the web tools added by "
                "<code>use_web</code>, <code>delegate_to_subagent</code> added by "
                "<code>allow_delegation</code>, and the terminal "
                "<code>submit_result</code>.</p>"
            )
            parts.append('<div class="scroll"><table><tr><th>Tool</th><th>Parameters</th>'
                         "<th>Defined in</th><th>What the model is shown</th></tr>")
            for t in a.tools:
                parts.append(
                    f"<tr><td><code>{e(t.name)}</code></td>"
                    f'<td><code>{e(", ".join(t.params)) or "&mdash;"}</code></td>'
                    f'<td><code>{e(t.origin) or "&mdash;"}</code></td>'
                    f'<td><pre class="doc">{e(t.description)}</pre></td></tr>'
                )
            parts.append("</table></div>")
        else:
            parts.append('<p class="purpose">No tools at all.</p>')

        if a.prompt_parts:
            rows = "".join(
                f"<tr><td>{e(name)}</td>"
                f'<td>{f"{len(text):,} chars" if text else "&mdash;"}</td>'
                f"<td><code>{e(path)}</code></td></tr>"
                for name, path, text in a.prompt_parts
            )
            parts.append(
                "<details open><summary>Prompt is assembled from these parts, in "
                "this order</summary>"
                '<div class="scroll"><table><tr><th>Part</th><th>Size</th>'
                f"<th>File to edit</th></tr>{rows}</table></div></details>"
            )

        parts.append(
            f"<details><summary>Output contract — {e(a.result_model)}</summary>"
            '<div class="scroll"><table><tr><th>Field</th><th>Type</th></tr>'
            + "".join(
                f"<tr><td><code>{e(n)}</code></td><td><code>{e(t)}</code></td></tr>"
                for n, t in a.result_fields
            )
            + "</table></div></details>"
        )
        parts.append(
            f"<details><summary>Full system prompt ({a.prompt_chars:,} characters)</summary>"
            f'<pre class="prompt">{e(a.instructions)}</pre></details>'
        )
        parts.append("</div>")

    parts.append("</div>")
    return "\n".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser(description="Inspect every agent, tool and prompt.")
    ap.add_argument("--html", nargs="?", const="analysis/agents.html", default=None,
                    help="Write a self-contained HTML page")
    ap.add_argument("--agent", help="Print one agent's full prompt (with --prompt)")
    ap.add_argument("--prompt", action="store_true", help="Include the full prompt text")
    ap.add_argument("--json", action="store_true", help="Machine-readable dump")
    ap.add_argument("--ticker", default="HD",
                    help="Render the per-company prompt variables for this company "
                         "(HD, ADI, HAS, DE). HAS is the one worth checking: pence, "
                         "pre-exceptional and semi-annual all differ.")
    a = ap.parse_args()

    agents = collect(a.ticker.upper())

    if a.json:
        print(json.dumps([{
            "key": x.key, "name": x.name, "model": x.model, "max_turns": x.max_turns,
            "use_web": x.use_web, "allow_delegation": x.allow_delegation,
            "result_model": x.result_model, "prompt_chars": x.prompt_chars,
            "tools": [t.name for t in x.tools], "error": x.error,
        } for x in agents], indent=2))
        return 0

    print_summary(agents, show_prompt=a.agent if (a.prompt or a.agent) else None)

    if a.html:
        out = REPO_ROOT / a.html if not Path(a.html).is_absolute() else Path(a.html)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(render_html(agents), encoding="utf-8")
        print(f"\nWrote {out}  ({out.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
