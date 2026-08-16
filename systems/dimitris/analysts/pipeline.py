"""pipeline.py — the orchestrator. One command produces the twelve numbers.

    python -m analysts.pipeline                        # all four companies
    python -m analysts.pipeline --tickers HD           # one company
    python -m analysts.pipeline --resume               # continue the last run
    python -m analysts.pipeline --stub-analysts        # orchestration only, no research

Shape of a run:

    run_all(4 companies, concurrently)
      └─ run_company(TICKER)
           ├─ filings | news | financials      <- asyncio.gather, concurrent
           └─ central                          <- consumes all three

Four properties the day demands, in the order they matter:

  RESUMABLE. Every step is a runstore task keyed "TICKER:step". Re-running with
  the same run_id reuses whatever already completed instead of paying for it
  twice. A crash at minute 40 of a 45-minute window costs the current step, not
  the run.

  DEGRADING. An analyst that is missing, broken, or hung does not take the
  company down. It yields an empty report of the right type, the central agent
  is told the evidence base was unavailable, and the forecast still ships. Three
  numbers from two analysts beat no numbers from three.

  BOUNDED. Every step has a timeout, so one wedged agent cannot eat the window.

  OBSERVABLE. Wall-clock is printed per step, per company and overall, against
  the 45-minute budget, and the clear-run log is exported to logs/ at the end.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import inspect
import json
import logging
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from agent_core import (
    get_last_tool_calls,
    get_last_usage,
    settings,
    use_selector_event_loop,
)
from analysts.central import (
    COMPANY_METRICS,
    canonical_ticker,
    company_name_for,
    errors,
    metrics_for,
    output_file_for,
    period_for,
    reference_values,
    skeleton_forecast,
    synthesise,
    validate_forecast,
    warnings,
)
from analysts.models import (
    CompanyForecast,
    FilingsReport,
    FinancialsReport,
    NewsReport,
)
from runstore import RunStore, get_store

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
LAST_RUN_FILE = REPO_ROOT / "runs" / "last_run_id.txt"

# The final run is a 45-minute window for four companies.
BUDGET_SECONDS = 45 * 60

# Per-step ceilings. A hung step must cost its own budget, never the window.
ANALYST_TIMEOUT = 900.0   # 15 min
CENTRAL_TIMEOUT = 600.0   # 10 min

# step name -> (module, report type). The module is imported LAZILY inside the
# step, so one that does not exist yet — or fails to import — degrades to an
# empty report instead of killing the process at startup.
ANALYSTS: dict[str, tuple[str, type]] = {
    "filings": ("analysts.filings", FilingsReport),
    "news": ("analysts.news", NewsReport),
    "financials": ("analysts.financials", FinancialsReport),
}


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass
class StepResult:
    name: str
    report: Any
    seconds: float = 0.0
    reused: bool = False
    degraded: str = ""          # non-empty = why this step produced nothing


@dataclass
class CompanyResult:
    ticker: str
    period: str = ""
    forecast: CompanyForecast | None = None
    problems: list[str] = field(default_factory=list)
    steps: list[StepResult] = field(default_factory=list)
    seconds: float = 0.0
    failed: str = ""

    @property
    def degraded(self) -> list[str]:
        return [f"{s.name}: {s.degraded}" for s in self.steps if s.degraded]

    @property
    def reused(self) -> list[str]:
        return [s.name for s in self.steps if s.reused]

    @property
    def filled(self) -> int:
        if not self.forecast:
            return 0
        return sum(1 for m in self.forecast.metrics or [] if m.value is not None)


# ---------------------------------------------------------------------------
# Calling an analyst we did not write
# ---------------------------------------------------------------------------


def _start_task(store: RunStore, run_id: str, key: str, **kw: Any) -> str | None:
    """`store.start_task`, but a key that already COMPLETED cannot crash the run.

    `tasks` has UNIQUE(run_id, key) and `start_task` only clears rows that did
    not complete, so re-attempting an already-completed key raises. That happens
    on resume whenever a checkpoint exists but its stored output will not parse.
    A new attempt at an old key is legitimate, so it gets a suffixed key rather
    than an exception.
    """
    try:
        return store.start_task(run_id, key, **kw)
    except Exception:
        alt = f"{key}#{uuid.uuid4().hex[:4]}"
        logger.warning("task key %r already completed; recording this attempt as %r",
                       key, alt)
        try:
            return store.start_task(run_id, alt, **kw)
        except Exception:  # pragma: no cover - the store is unusable; keep going
            logger.exception("could not record task %r", key)
            return None


def _record_tool_calls(
    store: RunStore, run_id: str, task_id: str | None, calls: list[dict] | None
) -> None:
    """One `tool.call` event per invocation, plus a `tool.summary` per task.

    The per-call rows answer "what did it actually look at"; the summary answers
    "did it thrash" in a single query without counting rows. Telemetry, so a
    failure here must never take down a finished forecast.
    """
    if not task_id or not calls:
        return
    try:
        counts: dict[str, int] = {}
        for i, c in enumerate(calls):
            counts[c.get("tool", "?")] = counts.get(c.get("tool", "?"), 0) + 1
            store.log(
                run_id,
                "tool.call",
                task_id=task_id,
                name=c.get("tool", "?"),
                payload={
                    "seq": i + 1,
                    "arguments": c.get("arguments", ""),
                    "output_chars": c.get("output_chars"),
                },
            )
        store.log(
            run_id,
            "tool.summary",
            task_id=task_id,
            name=f"{len(calls)} calls",
            payload=counts,
        )
    except Exception as e:  # pragma: no cover - telemetry only
        logger.debug("Could not record tool calls: %s", e)


def _supported_kwargs(fn: Any, candidates: dict[str, Any]) -> dict[str, Any]:
    """Pass only the keyword arguments this analyst actually accepts.

    The three analysts are written in parallel by other agents against an agreed
    signature. Introspecting instead of assuming means a missing `store=` or an
    extra parameter is a shrug, not a TypeError at 17:50.
    """
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return dict(candidates)
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return dict(candidates)
    return {k: v for k, v in candidates.items() if k in sig.parameters}


async def _call_analyst(
    name: str, ticker: str, as_of: str, run_id: str, store: RunStore, task_id: str
) -> Any:
    """Import and invoke one analyst. Raises; the caller decides how to degrade."""
    module_name, model_cls = ANALYSTS[name]
    mod = importlib.import_module(module_name)
    fn = getattr(mod, "analyse", None)
    if fn is None:
        raise AttributeError(f"{module_name} has no `analyse`")

    kwargs = _supported_kwargs(
        fn, {"as_of": as_of, "run_id": run_id, "store": store, "task_id": task_id}
    )
    out = fn(ticker, **kwargs)
    if inspect.isawaitable(out):
        out = await out

    if isinstance(out, model_cls):
        return out
    # Tolerate a dict or a structurally-compatible model.
    try:
        if hasattr(out, "model_dump"):
            return model_cls.model_validate(out.model_dump())
        return model_cls.model_validate(out)
    except Exception as e:
        raise TypeError(
            f"{module_name}.analyse returned {type(out).__name__}, "
            f"not {model_cls.__name__} ({e})"
        ) from e


async def _run_step(
    *,
    name: str,
    ticker: str,
    as_of: str,
    run_id: str,
    store: RunStore,
    parent_id: str | None,
    completed: set[str],
) -> StepResult:
    """One analyst, with resume, timeout, checkpointing and graceful degradation."""
    module_name, model_cls = ANALYSTS[name]
    key = f"{ticker}:{name}"
    empty = model_cls(ticker=ticker)

    # --- resume ------------------------------------------------------------
    if key in completed:
        raw = store.get_output(run_id, key)
        if raw:
            try:
                report = model_cls.model_validate_json(raw)
                store.log(run_id, "step.reused", task_id=parent_id, name=key)
                logger.info("%s: reusing completed step", key)
                return StepResult(name, report, 0.0, reused=True)
            except Exception as e:
                logger.warning("%s: stored output would not parse (%s) — re-running", key, e)
        else:
            logger.warning("%s: marked complete but has no output — re-running", key)

    # --- run ---------------------------------------------------------------
    task_id = _start_task(
        store,
        run_id,
        key,
        kind=name,
        parent_id=parent_id,
        agent_name=f"{name} analyst ({ticker})",
        model=settings.resolved_agent_model,
        input={"ticker": ticker, "as_of": as_of},
    )
    t0 = time.monotonic()

    async def _call_with_usage():
        # Same context-copy problem as the central step: asyncio.wait_for
        # spawns a task, so usage must be read here, not after the await.
        out = await _call_analyst(name, ticker, as_of, run_id, store, task_id)
        return out, get_last_usage(), get_last_tool_calls()

    try:
        report, step_usage, step_tools = await asyncio.wait_for(
            _call_with_usage(), timeout=ANALYST_TIMEOUT
        )
        elapsed = time.monotonic() - t0
        store.finish_task(task_id, output=report.model_dump(mode="json"),
                          usage=step_usage)
        _record_tool_calls(store, run_id, task_id, step_tools)
        store.log(
            run_id, "step.done", task_id=task_id, name=key, duration_ms=int(elapsed * 1000)
        )
        return StepResult(name, report, elapsed)
    except asyncio.TimeoutError:
        elapsed = time.monotonic() - t0
        why = f"timed out after {ANALYST_TIMEOUT:.0f}s"
    except ModuleNotFoundError as e:
        elapsed = time.monotonic() - t0
        why = f"module not available ({e.name})"
    except Exception as e:
        elapsed = time.monotonic() - t0
        why = f"{type(e).__name__}: {e}"

    logger.warning("%s DEGRADED — %s", key, why)
    store.fail_task(task_id, why)
    store.log(run_id, "step.degraded", task_id=task_id, name=key, payload=why,
              duration_ms=int(elapsed * 1000))
    return StepResult(name, empty, elapsed, degraded=why)


# ---------------------------------------------------------------------------
# One company
# ---------------------------------------------------------------------------


async def run_company(
    ticker: str,
    as_of: str,
    run_id: str,
    store: RunStore,
    *,
    stub_analysts: bool = False,
    redo: frozenset[str] = frozenset(),
) -> CompanyResult:
    """Three analysts concurrently, then the central agent over their output."""
    tk = canonical_ticker(ticker)
    result = CompanyResult(ticker=tk, period=period_for(tk))
    t0 = time.monotonic()

    # Reuse is a property of the RUN, not of the flag: a completed key inside
    # this run_id is always reused. `--resume` merely selects an existing run_id
    # to continue; a fresh run_id has nothing completed and redoes everything.
    completed = store.completed_keys(run_id)
    # `--redo central` exists for one specific job: change the central prompt and
    # re-run it against the SAME analyst reports. The analysts are LLMs and vary
    # between runs, so re-running them alongside a prompt change moves two
    # variables at once and the comparison means nothing. This holds the evidence
    # fixed so the prompt is the only thing that changed.
    if redo:
        completed = {k for k in completed if k.split(":", 1)[-1] not in redo}
        logger.info("%s: forcing a re-run of %s", tk, ", ".join(sorted(redo)))
    parent_id = _start_task(
        store,
        run_id,
        f"{tk}:company",
        kind="company",
        agent_name=f"{company_name_for(tk) or tk} pipeline",
        model=settings.resolved_agent_model,
        input={"ticker": tk, "as_of": as_of, "period": result.period},
    )
    store.log(run_id, "company.start", task_id=parent_id, name=tk,
              payload={"as_of": as_of, "resume_from": sorted(k for k in completed
                                                             if k.startswith(f"{tk}:"))})

    # --- the three analysts, concurrently ----------------------------------
    if stub_analysts:
        # Orchestration-only mode: exercises every path below without spending
        # on research the siblings own.
        result.steps = [
            StepResult(name, cls(ticker=tk), 0.0, degraded="stubbed (--stub-analysts)")
            for name, (_, cls) in ANALYSTS.items()
        ]
    else:
        result.steps = list(
            await asyncio.gather(
                *(
                    _run_step(
                        name=name,
                        ticker=tk,
                        as_of=as_of,
                        run_id=run_id,
                        store=store,
                        parent_id=parent_id,
                        completed=completed,
                    )
                    for name in ANALYSTS
                )
            )
        )

    by_name = {s.name: s for s in result.steps}
    filings = by_name["filings"].report
    news = by_name["news"].report
    financials = by_name["financials"].report

    # --- the central agent --------------------------------------------------
    central_key = f"{tk}:central"
    forecast: CompanyForecast | None = None
    central_seconds = 0.0
    reused_central = False

    if central_key in completed:
        raw = store.get_output(run_id, central_key)
        if raw:
            try:
                forecast = CompanyForecast.model_validate_json(raw)
                reused_central = True
                store.log(run_id, "step.reused", task_id=parent_id, name=central_key)
                logger.info("%s: reusing completed step", central_key)
            except Exception as e:
                logger.warning("%s: stored output would not parse (%s)", central_key, e)

    if forecast is None:
        central_task = _start_task(
            store,
            run_id,
            central_key,
            kind="central",
            parent_id=parent_id,
            agent_name=f"Central Forecaster ({tk})",
            model=settings.resolved_agent_model,
            input={"ticker": tk, "period": result.period,
                   "degraded_inputs": result.degraded},
        )
        t1 = time.monotonic()

        async def _synthesise_with_usage():
            """Read usage INSIDE the child task.

            asyncio.wait_for spawns a task, and a task gets a COPY of the
            context — so a ContextVar set by run_agent down here never
            propagates back to the caller. Capturing before we return is what
            keeps the central agent's cost from silently recording as zero.
            """
            out = await synthesise(
                tk,
                filings,
                news,
                financials,
                run_id=run_id,
                store=store,
                as_of=as_of,
                task_id=central_task,
            )
            return out, get_last_usage(), get_last_tool_calls()

        try:
            forecast, central_usage, central_tools = await asyncio.wait_for(
                _synthesise_with_usage(), timeout=CENTRAL_TIMEOUT
            )
            central_seconds = time.monotonic() - t1
            store.finish_task(central_task, output=forecast.model_dump(mode="json"),
                              usage=central_usage)
            _record_tool_calls(store, run_id, central_task, central_tools)
            store.log(run_id, "step.done", task_id=central_task, name=central_key,
                      duration_ms=int(central_seconds * 1000))
        except Exception as e:
            central_seconds = time.monotonic() - t1
            why = (
                f"timed out after {CENTRAL_TIMEOUT:.0f}s"
                if isinstance(e, asyncio.TimeoutError)
                else f"{type(e).__name__}: {e}"
            )
            logger.error("%s FAILED — %s", central_key, why)
            store.fail_task(central_task, why)
            store.log(run_id, "step.degraded", task_id=central_task,
                      name=central_key, payload=why)
            result.failed = why
            forecast = skeleton_forecast(tk)

    result.steps.append(
        StepResult("central", forecast, central_seconds, reused=reused_central)
    )
    result.forecast = forecast

    # --- validate -----------------------------------------------------------
    ref = reference_values(tk, filings, financials)
    result.problems = validate_forecast(tk, forecast, reference=ref)
    result.seconds = time.monotonic() - t0

    store.finish_task(
        parent_id,
        output={
            "ticker": tk,
            "period": result.period,
            "values": {m.label: m.value for m in forecast.metrics or []},
            "errors": errors(result.problems),
            "warnings": warnings(result.problems),
            "degraded": result.degraded,
            "seconds": round(result.seconds, 1),
        },
    )
    store.log(
        run_id,
        "company.done",
        task_id=parent_id,
        name=tk,
        payload={
            "filled": f"{result.filled}/{len(metrics_for(tk))}",
            "errors": len(errors(result.problems)),
            "warnings": len(warnings(result.problems)),
        },
        duration_ms=int(result.seconds * 1000),
    )
    return result


# ---------------------------------------------------------------------------
# All companies
# ---------------------------------------------------------------------------


async def run_all(
    tickers: list[str],
    as_of: str,
    run_id: str,
    store: RunStore,
    *,
    stub_analysts: bool = False,
    redo: frozenset[str] = frozenset(),
) -> list[CompanyResult]:
    """Every company concurrently. One company's failure never blocks another."""
    tks = [canonical_ticker(t) for t in tickers]
    store.log(run_id, "run.start", name=",".join(tks), payload={"as_of": as_of})

    async def one(tk: str) -> CompanyResult:
        try:
            return await run_company(
                tk, as_of, run_id, store, stub_analysts=stub_analysts, redo=redo
            )
        except Exception as e:  # belt and braces: never lose the other three
            logger.exception("%s: pipeline crashed", tk)
            store.log(run_id, "company.crashed", name=tk, payload=f"{type(e).__name__}: {e}")
            fc = skeleton_forecast(tk)
            return CompanyResult(
                ticker=tk,
                period=period_for(tk),
                forecast=fc,
                problems=validate_forecast(tk, fc),
                failed=f"{type(e).__name__}: {e}",
            )

    return list(await asyncio.gather(*(one(tk) for tk in tks)))


# ---------------------------------------------------------------------------
# Presentation
# ---------------------------------------------------------------------------


def _fmt_value(value: float | None, units: str) -> str:
    if value is None:
        return "MISSING"
    if units == "%":
        return f"{value:,.2f}"
    if "share" in units or units == "GBp":
        return f"{value:,.2f}"
    return f"{value:,.1f}"


def print_company(result: CompanyResult) -> None:
    tk = result.ticker
    name = company_name_for(tk) or tk
    fc = result.forecast or skeleton_forecast(tk)
    errs, warns = errors(result.problems), warnings(result.problems)

    print()
    print("=" * 100)
    print(f"{tk}  —  {name}  —  {result.period}  —  {output_file_for(tk)}")
    print("=" * 100)

    header = f"{'Metric':<44} {'Value':>12} {'Low':>12} {'High':>12}  {'Units':<12} Conf"
    print(header)
    print("-" * 100)
    for m in fc.metrics or []:
        print(
            f"{m.label:<44} {_fmt_value(m.value, m.units):>12} "
            f"{_fmt_value(m.low, m.units):>12} {_fmt_value(m.high, m.units):>12}  "
            f"{m.units:<12} {(m.confidence or '-')}"
        )
        method = " ".join((m.method or "(no method stated)").split())
        print(f"    method: {method[:150]}{'...' if len(method) > 150 else ''}")
        cites = [c.source for c in (m.evidence or []) if (c.source or '').strip()]
        print(f"    cites : {', '.join(cites[:3]) if cites else '(none)'}"
              f"{f' (+{len(cites) - 3} more)' if len(cites) > 3 else ''}")
    print("-" * 100)

    steps = "  ".join(
        f"{s.name}={'reused' if s.reused else f'{s.seconds:.0f}s'}"
        f"{'*' if s.degraded else ''}"
        for s in result.steps
    )
    print(f"timing : {steps}   total={result.seconds:.0f}s")
    if result.degraded:
        print("degraded inputs (*):")
        for d in result.degraded:
            print(f"    - {d}")
    if result.failed:
        print(f"CENTRAL AGENT FAILED: {result.failed}")

    if not result.problems:
        print("validation: clean — no problems found.")
    else:
        print(f"validation: {len(errs)} error(s), {len(warns)} warning(s)")
        for p in result.problems:
            print(f"    {p}")


def print_summary(
    results: list[CompanyResult], run_id: str, elapsed: float, store: RunStore
) -> None:
    total_metrics = sum(len(metrics_for(r.ticker)) for r in results)
    filled = sum(r.filled for r in results)
    n_err = sum(len(errors(r.problems)) for r in results)
    n_warn = sum(len(warnings(r.problems)) for r in results)

    print()
    print("=" * 100)
    print("RUN SUMMARY")
    print("=" * 100)
    print(f"run_id          : {run_id}")
    print(f"model           : {settings.resolved_agent_model}")
    print(f"companies       : {', '.join(r.ticker for r in results)}")
    print(f"metrics filled  : {filled}/{total_metrics}"
          f"{'  <-- BLANKS SCORE 5.0 EACH' if filled < total_metrics else ''}")
    print(f"validation      : {n_err} error(s), {n_warn} warning(s)")
    for r in results:
        reused = f"  reused: {', '.join(r.reused)}" if r.reused else ""
        print(f"    {r.ticker:<5} {r.seconds:>6.0f}s  "
              f"{r.filled}/{len(metrics_for(r.ticker))} filled  "
              f"{len(errors(r.problems))}E {len(warnings(r.problems))}W{reused}")

    summary = store.run_summary(run_id)
    print(f"tasks           : {summary['tasks']}")
    print(f"evidence rows   : {summary['evidence_rows']}")

    mins, secs = divmod(int(elapsed), 60)
    headroom = BUDGET_SECONDS - elapsed
    print(f"WALL CLOCK      : {mins}m {secs:02d}s")
    print(
        f"budget          : {BUDGET_SECONDS // 60}m window — "
        + (
            f"{int(headroom // 60)}m {int(headroom % 60):02d}s headroom"
            if headroom >= 0
            else f"OVER BY {int(-headroom // 60)}m {int(-headroom % 60):02d}s"
        )
    )
    if len(results) == 1 and results[0].seconds > 0:
        print(
            f"                  one company took {results[0].seconds:.0f}s; four "
            f"concurrently should land near that, serially near "
            f"{results[0].seconds * 4 / 60:.0f}m."
        )


# ---------------------------------------------------------------------------
# Artefacts
# ---------------------------------------------------------------------------


def write_outputs(results: list[CompanyResult], out_dir: Path, run_id: str) -> list[Path]:
    """Forecast JSON per company plus a run-wide validation report.

    JSON, not xlsx: the submission workbooks must be filled from the supplied
    templates (only the yellow cells change), which is the workbook writer's job.
    This is its input, and the judges' machine-readable copy of the same numbers.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for r in results:
        fc = r.forecast or skeleton_forecast(r.ticker)
        stem = (output_file_for(r.ticker) or f"{r.ticker}.xlsx").rsplit(".", 1)[0]
        path = out_dir / f"{stem}.json"
        path.write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "workbook": output_file_for(r.ticker),
                    "forecast": fc.model_dump(mode="json"),
                    "validation": r.problems,
                    "degraded_inputs": r.degraded,
                    "seconds": round(r.seconds, 1),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        written.append(path)

    combined = out_dir / "_forecasts.json"
    combined.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "companies": [
                    {
                        "ticker": r.ticker,
                        "period": r.period,
                        "workbook": output_file_for(r.ticker),
                        "metrics": [
                            {
                                "label": m.label,
                                "units": m.units,
                                "value": m.value,
                                "low": m.low,
                                "high": m.high,
                            }
                            for m in (r.forecast.metrics if r.forecast else [])
                        ],
                    }
                    for r in results
                ],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    written.append(combined)

    report = out_dir / "_validation.md"
    lines = [f"# Validation report — {run_id}", ""]
    for r in results:
        lines.append(f"## {r.ticker} — {r.period}")
        lines.append("")
        lines.append(f"- metrics filled: {r.filled}/{len(metrics_for(r.ticker))}")
        lines.append(f"- errors: {len(errors(r.problems))}, warnings: {len(warnings(r.problems))}")
        if r.degraded:
            lines.append(f"- degraded inputs: {'; '.join(r.degraded)}")
        if r.failed:
            lines.append(f"- central agent failed: {r.failed}")
        lines.append("")
        lines.extend(f"- {p}" for p in r.problems or ["(no problems found)"])
        lines.append("")
    report.write_text("\n".join(lines), encoding="utf-8")
    written.append(report)
    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _preflight() -> None:
    """Load .env into the environment, then check we can actually reach a model.

    `agent_core` resolves the API key with `os.getenv` at model-build time, but
    pydantic-settings only reads .env into `Settings`' own fields — it never
    touches `os.environ`. A shell that did not export OPENAI_API_KEY therefore
    sails through startup and fails at every single model call. The failure is
    survivable (each agent falls back), which is precisely the problem: you get
    a complete run of blank forecasts instead of an error. Say it up front.
    """
    try:
        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / ".env", override=False)
    except ImportError:  # pragma: no cover - python-dotenv is in requirements
        pass

    if not settings.resolved_api_key:
        print("!" * 100)
        print(f"!! NO API KEY for profile {settings.llm_profile!r} "
              f"({settings.resolved_agent_model}).")
        print("!! Every agent will fall back and every metric will come out BLANK.")
        print("!! Export the key, or put it in .env, before the real run.")
        print("!" * 100)


def _resolve_run_id(args: argparse.Namespace) -> str:
    if args.run_id:
        return args.run_id
    if args.resume:
        if LAST_RUN_FILE.exists():
            rid = LAST_RUN_FILE.read_text(encoding="utf-8").strip()
            if rid:
                print(f"--resume: continuing the last run, {rid}")
                return rid
        raise SystemExit(
            "--resume needs a run to resume: pass --run-id, or run once without "
            f"--resume first ({LAST_RUN_FILE} does not name one)."
        )
    return f"run_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}"


def _remember_run_id(run_id: str) -> None:
    try:
        LAST_RUN_FILE.parent.mkdir(parents=True, exist_ok=True)
        LAST_RUN_FILE.write_text(run_id, encoding="utf-8")
    except Exception:  # pragma: no cover
        logger.debug("could not record last run id", exc_info=True)


async def main(args: argparse.Namespace) -> int:
    t0 = time.monotonic()

    tickers = [canonical_ticker(t) for t in args.tickers.split(",") if t.strip()]
    unknown = [t for t in tickers if t not in COMPANY_METRICS]
    if unknown:
        raise SystemExit(
            f"Unknown ticker(s): {', '.join(unknown)}. "
            f"Known: {', '.join(COMPANY_METRICS)}"
        )

    run_id = _resolve_run_id(args)
    store = get_store(args.db)
    store.start_run(
        f"forecast {','.join(tickers)}",
        run_id=run_id,
        config={
            "tickers": tickers,
            "as_of": args.as_of,
            "model": settings.resolved_agent_model,
            "reasoning_effort": settings.reasoning_effort,
            "resume": bool(args.resume),
            "stub_analysts": bool(args.stub_analysts),
        },
    )
    _remember_run_id(run_id)

    print(f"run_id   : {run_id}")
    print(f"tickers  : {', '.join(tickers)}")
    print(f"as-of    : {args.as_of}")
    print(f"model    : {settings.resolved_agent_model} "
          f"(reasoning={settings.reasoning_effort or 'none'})")
    print(f"resume   : {'on' if args.resume else 'off'}"
          + ("   [ANALYSTS STUBBED]" if args.stub_analysts else ""))
    print(f"db       : {store.db_path}")
    print("running…", flush=True)

    status = "completed"
    try:
        results = await run_all(
            tickers, args.as_of, run_id, store, stub_analysts=args.stub_analysts,
            redo=frozenset(
                x.strip().lower() for x in (args.redo or "").split(",") if x.strip()
            ),
        )
    except KeyboardInterrupt:
        store.log(run_id, "run.interrupted")
        store.end_run(run_id, status="interrupted")
        elapsed = time.monotonic() - t0
        print(f"\nInterrupted after {elapsed:.0f}s. "
              f"Resume with:  python -m analysts.pipeline --resume --tickers "
              f"{args.tickers}")
        return 130

    elapsed = time.monotonic() - t0

    for r in results:
        print_company(r)

    written = write_outputs(results, Path(args.out), run_id)

    log_path = Path(args.log_dir) / (
        f"clear-run_{run_id}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.log"
    )
    try:
        store.export_log(run_id, log_path)
    except Exception as e:  # pragma: no cover
        logger.error("log export failed: %s", e)
        log_path = None  # type: ignore[assignment]

    if any(errors(r.problems) for r in results):
        status = "completed_with_errors"
    store.end_run(run_id, status=status)

    print_summary(results, run_id, elapsed, store)
    print()
    print("artefacts:")
    for p in written:
        print(f"    {p}")
    if log_path:
        print(f"    {log_path}")
    print(f"\nresume this run:  python -m analysts.pipeline --run-id {run_id} "
          f"--resume --tickers {args.tickers}")

    return 1 if any(errors(r.problems) for r in results) else 0


def cli() -> int:
    ap = argparse.ArgumentParser(
        prog="python -m analysts.pipeline",
        description="Run the three analysts and the central agent, and print the "
                    "forecasts with their validation problems.",
    )
    ap.add_argument("--tickers", default=",".join(COMPANY_METRICS),
                    help="Comma-separated. Default: all four.")
    ap.add_argument("--as-of", default=date.today().isoformat(),
                    help="Point-in-time date for the run (ISO). Default: today.")
    ap.add_argument("--run-id", default="",
                    help="Reuse a run id. With --resume, completed steps are skipped.")
    ap.add_argument("--resume", action="store_true",
                    help="Reuse completed steps from --run-id (or the last run).")
    ap.add_argument("--out", default="submission",
                    help="Directory for forecast JSON and the validation report.")
    ap.add_argument("--log-dir", default="logs",
                    help="Directory for the exported clear-run log.")
    ap.add_argument("--db", default=None, help="Override the runstore SQLite path.")
    ap.add_argument("--redo", default="",
                    help="With --resume: re-run these steps even though they "
                         "completed, reusing everything else. Comma-separated "
                         "(central, filings, news, financials). Use "
                         "'--redo central' to test a central-prompt change "
                         "against the SAME analyst reports.")
    ap.add_argument("--stub-analysts", action="store_true",
                    help="Skip the three analysts and feed the central agent empty "
                         "reports. Exercises orchestration and validation only.")
    ap.add_argument("-v", "--verbose", action="store_true", help="Debug logging.")
    args = ap.parse_args()

    # Validator messages, analyst prose and corpus quotes are all UTF-8. A
    # Windows console defaults to cp1252, where printing one em dash aborts the
    # run at the last step. Never let presentation kill a finished forecast.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except Exception:  # pragma: no cover - not a real file stream
            pass

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    use_selector_event_loop()
    try:
        return asyncio.run(main(args))
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130


if __name__ == "__main__":
    raise SystemExit(cli())
