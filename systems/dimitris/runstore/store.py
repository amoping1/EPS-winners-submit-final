"""store.py — SQLite run store: durability, traceability, resume.

Three jobs, and each one maps to something the day actually demands:

  1. DURABILITY. The final run is a 45-minute window for four companies. A
     crash at minute 40 must not cost the work already done. Every completed
     unit is committed the moment it finishes, so a retry resumes instead of
     restarting.

  2. TRACEABILITY. The submission requires "a timestamped record of what the
     final system did". Beyond that, the rubric scores whether judges can
     follow how evidence became each number ("model quality", 12 pts) and
     whether failures were handled ("validation and reliability", 12 pts).
     Prose logs do not answer that; a queryable event table does.

  3. FLOW. Parent/child links across delegated subagents make the actual
     execution shape recoverable after the fact — which is what the
     architecture diagram has to match.

Deliberately dependency-free (stdlib sqlite3) and safe under the concurrent
`run_agents` fan-out: WAL mode, a short busy timeout, one connection per
thread.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)

DEFAULT_DB = Path(__file__).resolve().parent.parent / "runs" / "runs.sqlite"

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id      TEXT PRIMARY KEY,
    label       TEXT NOT NULL,
    started_at  TEXT NOT NULL,
    ended_at    TEXT,
    status      TEXT NOT NULL DEFAULT 'running',
    git_commit  TEXT,
    config      TEXT,
    notes       TEXT
);

-- One row per unit of work (e.g. one company, one filing's MD&A read).
-- `key` is the resume key: unique within a run, so a retry can skip what is
-- already done.
CREATE TABLE IF NOT EXISTS tasks (
    task_id     TEXT PRIMARY KEY,
    run_id      TEXT NOT NULL REFERENCES runs(run_id),
    parent_id   TEXT REFERENCES tasks(task_id),
    key         TEXT NOT NULL,
    kind        TEXT NOT NULL,
    agent_name  TEXT,
    model       TEXT,
    status      TEXT NOT NULL DEFAULT 'running',
    started_at  TEXT NOT NULL,
    ended_at    TEXT,
    input       TEXT,
    output      TEXT,
    error       TEXT,
    -- Usage. Without these, "did this run bloat?" is unanswerable, and the
    -- 45-minute final window has to be budgeted on guesswork.
    -- cached_tokens is the one that matters: a collapsing cache hit rate means
    -- the prompt is churning, which shows up as cost long before it shows up
    -- as a wrong number.
    prompt_tokens     INTEGER,
    cached_tokens     INTEGER,
    completion_tokens INTEGER,
    requests          INTEGER,
    cost_usd          REAL,
    UNIQUE (run_id, key)
);

-- Append-only. Tool calls, model calls, decisions, rejected values.
CREATE TABLE IF NOT EXISTS events (
    event_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      TEXT NOT NULL REFERENCES runs(run_id),
    task_id     TEXT REFERENCES tasks(task_id),
    ts          TEXT NOT NULL,
    kind        TEXT NOT NULL,
    name        TEXT,
    payload     TEXT,
    duration_ms INTEGER
);

-- The evidence trail: one row per (metric, source) so "how did this number
-- happen" is a single SELECT.
CREATE TABLE IF NOT EXISTS evidence (
    evidence_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      TEXT NOT NULL REFERENCES runs(run_id),
    task_id     TEXT REFERENCES tasks(task_id),
    ticker      TEXT NOT NULL,
    metric      TEXT NOT NULL,
    claim       TEXT,
    value       REAL,
    units       TEXT,
    source      TEXT NOT NULL,
    locator     TEXT,
    confidence  REAL,
    ts          TEXT NOT NULL
);

-- Long-term company studies. Deliberately NOT scoped to a run: the corpus is
-- frozen, so the same inputs always give the same study. Recomputing it every
-- run would waste both money and the 45-minute final window.
-- The cache key includes corpus_freeze and prompt_version so a stale study can
-- never be served silently -- staleness here is worse than a cache miss.
CREATE TABLE IF NOT EXISTS profiles (
    ticker         TEXT NOT NULL,
    corpus_freeze  TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    model          TEXT,
    built_at       TEXT NOT NULL,
    build_run_id   TEXT,
    payload        TEXT NOT NULL,
    PRIMARY KEY (ticker, corpus_freeze, prompt_version)
);

CREATE INDEX IF NOT EXISTS idx_tasks_run    ON tasks(run_id, status);
CREATE INDEX IF NOT EXISTS idx_tasks_parent ON tasks(parent_id);
CREATE INDEX IF NOT EXISTS idx_events_run   ON events(run_id, ts);
CREATE INDEX IF NOT EXISTS idx_events_task  ON events(task_id);
CREATE INDEX IF NOT EXISTS idx_evi_metric   ON evidence(run_id, ticker, metric);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _dump(obj: Any) -> str | None:
    if obj is None:
        return None
    if isinstance(obj, str):
        return obj
    try:
        return json.dumps(obj, default=str, ensure_ascii=False)
    except Exception:
        return str(obj)


class RunStore:
    """SQLite-backed run store. Thread-safe; one connection per thread."""

    def __init__(self, db_path: Path | str = DEFAULT_DB):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        with self._conn() as c:
            c.executescript(SCHEMA)
            self._migrate(c)

    @staticmethod
    def _migrate(c: sqlite3.Connection) -> None:
        """Add columns missing from an older database.

        CREATE TABLE IF NOT EXISTS silently leaves an existing table alone, so a
        store created before the usage columns existed would keep working and
        keep recording nothing. Additive only; never drops.
        """
        have = {r["name"] for r in c.execute("PRAGMA table_info(tasks)")}
        for col, decl in (
            ("prompt_tokens", "INTEGER"),
            ("cached_tokens", "INTEGER"),
            ("completion_tokens", "INTEGER"),
            ("requests", "INTEGER"),
            ("cost_usd", "REAL"),
        ):
            if col not in have:
                c.execute(f"ALTER TABLE tasks ADD COLUMN {col} {decl}")
                logger.info("Migrated runs.sqlite: added tasks.%s", col)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        # WAL lets readers proceed during writes, which matters when four
        # agents commit concurrently.
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = self._connect()
            self._local.conn = conn
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    # ---------------- runs ----------------

    def start_run(
        self,
        label: str,
        *,
        config: dict | None = None,
        git_commit: str | None = None,
        run_id: str | None = None,
    ) -> str:
        rid = run_id or f"run_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_{uuid.uuid4().hex[:6]}"
        with self._conn() as c:
            c.execute(
                "INSERT OR IGNORE INTO runs (run_id,label,started_at,status,git_commit,config)"
                " VALUES (?,?,?,?,?,?)",
                (rid, label, _now(), "running", git_commit, _dump(config)),
            )
        logger.info("Run %s started (%s)", rid, label)
        return rid

    def end_run(self, run_id: str, status: str = "completed", notes: str | None = None) -> None:
        with self._conn() as c:
            c.execute(
                "UPDATE runs SET ended_at=?, status=?, notes=? WHERE run_id=?",
                (_now(), status, notes, run_id),
            )

    # ---------------- tasks ----------------

    def completed_keys(self, run_id: str) -> set[str]:
        """Resume set: keys already completed in this run."""
        with self._conn() as c:
            rows = c.execute(
                "SELECT key FROM tasks WHERE run_id=? AND status='completed'", (run_id,)
            ).fetchall()
        return {r["key"] for r in rows}

    def get_output(self, run_id: str, key: str) -> str | None:
        with self._conn() as c:
            row = c.execute(
                "SELECT output FROM tasks WHERE run_id=? AND key=? AND status='completed'",
                (run_id, key),
            ).fetchone()
        return row["output"] if row else None

    def start_task(
        self,
        run_id: str,
        key: str,
        kind: str,
        *,
        parent_id: str | None = None,
        agent_name: str | None = None,
        model: str | None = None,
        input: Any = None,
    ) -> str:
        tid = f"task_{uuid.uuid4().hex[:12]}"
        with self._conn() as c:
            # A retry of a previously failed key replaces it.
            c.execute("DELETE FROM tasks WHERE run_id=? AND key=? AND status!='completed'", (run_id, key))
            c.execute(
                "INSERT INTO tasks (task_id,run_id,parent_id,key,kind,agent_name,model,"
                "status,started_at,input) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (tid, run_id, parent_id, key, kind, agent_name, model, "running", _now(), _dump(input)),
            )
        return tid

    def finish_task(
        self,
        task_id: str,
        output: Any = None,
        status: str = "completed",
        usage: dict | None = None,
    ) -> None:
        """Close a task, optionally recording token usage and cost.

        `usage` is the dict from agent_core.runner.get_last_usage().
        """
        with self._conn() as c:
            c.execute(
                "UPDATE tasks SET status=?, ended_at=?, output=? WHERE task_id=?",
                (status, _now(), _dump(output), task_id),
            )
            if usage:
                c.execute(
                    "UPDATE tasks SET prompt_tokens=?, cached_tokens=?, "
                    "completion_tokens=?, requests=?, cost_usd=? WHERE task_id=?",
                    (
                        usage.get("prompt_tokens"),
                        usage.get("cached_tokens"),
                        usage.get("completion_tokens"),
                        usage.get("requests"),
                        usage.get("cost_usd"),
                        task_id,
                    ),
                )

    def record_usage(self, task_id: str, usage: dict) -> None:
        """Attach usage to a task independently of finishing it."""
        with self._conn() as c:
            c.execute(
                "UPDATE tasks SET prompt_tokens=?, cached_tokens=?, "
                "completion_tokens=?, requests=?, cost_usd=? WHERE task_id=?",
                (
                    usage.get("prompt_tokens"),
                    usage.get("cached_tokens"),
                    usage.get("completion_tokens"),
                    usage.get("requests"),
                    usage.get("cost_usd"),
                    task_id,
                ),
            )

    def fail_task(self, task_id: str, error: str) -> None:
        with self._conn() as c:
            c.execute(
                "UPDATE tasks SET status='failed', ended_at=?, error=? WHERE task_id=?",
                (_now(), str(error)[:8000], task_id),
            )

    # ---------------- events ----------------

    def log(
        self,
        run_id: str,
        kind: str,
        *,
        task_id: str | None = None,
        name: str | None = None,
        payload: Any = None,
        duration_ms: int | None = None,
    ) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO events (run_id,task_id,ts,kind,name,payload,duration_ms)"
                " VALUES (?,?,?,?,?,?,?)",
                (run_id, task_id, _now(), kind, name, _dump(payload), duration_ms),
            )

    @contextmanager
    def timed(self, run_id: str, kind: str, name: str, task_id: str | None = None):
        """Log an event with its duration, and record failures rather than hiding them."""
        t0 = time.monotonic()
        try:
            yield
        except Exception as e:
            self.log(
                run_id,
                f"{kind}.error",
                task_id=task_id,
                name=name,
                payload=str(e)[:4000],
                duration_ms=int((time.monotonic() - t0) * 1000),
            )
            raise
        else:
            self.log(
                run_id,
                kind,
                task_id=task_id,
                name=name,
                duration_ms=int((time.monotonic() - t0) * 1000),
            )

    # ---------------- evidence ----------------

    def add_evidence(
        self,
        run_id: str,
        ticker: str,
        metric: str,
        source: str,
        *,
        task_id: str | None = None,
        claim: str | None = None,
        value: float | None = None,
        units: str | None = None,
        locator: str | None = None,
        confidence: float | None = None,
    ) -> None:
        """Record one (metric, source) link. `source` is the corpus FILENAME."""
        with self._conn() as c:
            c.execute(
                "INSERT INTO evidence (run_id,task_id,ticker,metric,claim,value,units,"
                "source,locator,confidence,ts) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (run_id, task_id, ticker, metric, claim, value, units, source,
                 locator, confidence, _now()),
            )

    def evidence_for(self, run_id: str, ticker: str, metric: str | None = None) -> list[dict]:
        sql = "SELECT * FROM evidence WHERE run_id=? AND ticker=?"
        args: list[Any] = [run_id, ticker]
        if metric:
            sql += " AND metric=?"
            args.append(metric)
        with self._conn() as c:
            return [dict(r) for r in c.execute(sql + " ORDER BY evidence_id", args)]

    # ---------------- profiles (long-term studies, cached) ----------------

    def get_profile(
        self, ticker: str, corpus_freeze: str, prompt_version: str
    ) -> dict | None:
        """Return a cached study, or None. Never serves a mismatched key."""
        with self._conn() as c:
            row = c.execute(
                "SELECT * FROM profiles WHERE ticker=? AND corpus_freeze=? AND prompt_version=?",
                (ticker.upper(), corpus_freeze, prompt_version),
            ).fetchone()
        if not row:
            return None
        out = dict(row)
        try:
            out["payload"] = json.loads(out["payload"])
        except Exception:
            pass
        return out

    def save_profile(
        self,
        ticker: str,
        corpus_freeze: str,
        prompt_version: str,
        payload: Any,
        *,
        model: str | None = None,
        build_run_id: str | None = None,
    ) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO profiles "
                "(ticker,corpus_freeze,prompt_version,model,built_at,build_run_id,payload)"
                " VALUES (?,?,?,?,?,?,?)",
                (ticker.upper(), corpus_freeze, prompt_version, model, _now(),
                 build_run_id, _dump(payload)),
            )
        logger.info("Saved profile for %s (freeze=%s v=%s)", ticker, corpus_freeze, prompt_version)

    def list_profiles(self) -> list[dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT ticker,corpus_freeze,prompt_version,model,built_at,"
                "LENGTH(payload) AS bytes FROM profiles ORDER BY ticker"
            ).fetchall()
        return [dict(r) for r in rows]

    def delete_profile(
        self, ticker: str, corpus_freeze: str | None = None, prompt_version: str | None = None
    ) -> int:
        sql = "DELETE FROM profiles WHERE ticker=?"
        args: list[Any] = [ticker.upper()]
        if corpus_freeze:
            sql += " AND corpus_freeze=?"
            args.append(corpus_freeze)
        if prompt_version:
            sql += " AND prompt_version=?"
            args.append(prompt_version)
        with self._conn() as c:
            cur = c.execute(sql, args)
            return cur.rowcount

    # ---------------- reporting ----------------

    def run_summary(self, run_id: str) -> dict:
        with self._conn() as c:
            run = c.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
            tasks = c.execute(
                "SELECT status, COUNT(*) n FROM tasks WHERE run_id=? GROUP BY status", (run_id,)
            ).fetchall()
            events = c.execute(
                "SELECT kind, COUNT(*) n FROM events WHERE run_id=? GROUP BY kind ORDER BY n DESC",
                (run_id,),
            ).fetchall()
            evi = c.execute(
                "SELECT COUNT(*) n FROM evidence WHERE run_id=?", (run_id,)
            ).fetchone()
        return {
            "run": dict(run) if run else None,
            "tasks": {r["status"]: r["n"] for r in tasks},
            "events": {r["kind"]: r["n"] for r in events},
            "evidence_rows": evi["n"] if evi else 0,
        }

    def export_log(self, run_id: str, path: Path | str) -> Path:
        """Write the timestamped clear-run log the submission requires."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            rows = c.execute(
                "SELECT e.ts, e.kind, e.name, e.duration_ms, t.key, t.kind AS task_kind "
                "FROM events e LEFT JOIN tasks t ON t.task_id = e.task_id "
                "WHERE e.run_id=? ORDER BY e.event_id",
                (run_id,),
            ).fetchall()
        with path.open("w", encoding="utf-8") as fh:
            fh.write(f"# Clear-run log — {run_id}\n")
            fh.write(f"# exported {_now()}\n\n")
            for r in rows:
                dur = f" ({r['duration_ms']}ms)" if r["duration_ms"] is not None else ""
                task = f" [{r['task_kind']}:{r['key']}]" if r["key"] else ""
                fh.write(f"{r['ts']}  {r['kind']}{task}  {r['name'] or ''}{dur}\n")
        logger.info("Exported %d log lines to %s", len(rows), path)
        return path


_default: RunStore | None = None


def get_store(db_path: Path | str | None = None) -> RunStore:
    """Process-wide default store. Override the path with RUNSTORE_DB."""
    global _default
    if db_path is not None:
        return RunStore(db_path)
    if _default is None:
        _default = RunStore(os.getenv("RUNSTORE_DB") or DEFAULT_DB)
    return _default
