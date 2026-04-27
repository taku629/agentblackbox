"""SQLite-backed storage for agentblackbox recordings."""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Optional

from .models import ErrorRecord, LLMCall, Session, ToolCall

DEFAULT_DB_PATH = Path.home() / ".agentblackbox" / "recordings.db"

_DDL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS sessions (
    session_id     TEXT PRIMARY KEY,
    agent_name     TEXT NOT NULL,
    start_time     REAL NOT NULL,
    end_time       REAL,
    status         TEXT NOT NULL DEFAULT 'running',
    total_cost_usd REAL NOT NULL DEFAULT 0.0,
    metadata       TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_sessions_agent  ON sessions(agent_name);
CREATE INDEX IF NOT EXISTS idx_sessions_start  ON sessions(start_time);
CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(status);

CREATE TABLE IF NOT EXISTS llm_calls (
    call_id       TEXT PRIMARY KEY,
    session_id    TEXT NOT NULL,
    timestamp     REAL NOT NULL,
    model         TEXT NOT NULL,
    input_text    TEXT NOT NULL DEFAULT '',
    output_text   TEXT NOT NULL DEFAULT '',
    input_tokens  INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd      REAL NOT NULL DEFAULT 0.0,
    latency_ms    REAL NOT NULL DEFAULT 0.0,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);
CREATE INDEX IF NOT EXISTS idx_llm_session ON llm_calls(session_id);
CREATE INDEX IF NOT EXISTS idx_llm_ts      ON llm_calls(timestamp);
CREATE INDEX IF NOT EXISTS idx_llm_model   ON llm_calls(model);

CREATE TABLE IF NOT EXISTS tool_calls (
    call_id    TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    timestamp  REAL NOT NULL,
    tool_name  TEXT NOT NULL,
    args       TEXT NOT NULL DEFAULT '{}',
    result     TEXT,
    latency_ms REAL NOT NULL DEFAULT 0.0,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);
CREATE INDEX IF NOT EXISTS idx_tool_session ON tool_calls(session_id);
CREATE INDEX IF NOT EXISTS idx_tool_ts      ON tool_calls(timestamp);
CREATE INDEX IF NOT EXISTS idx_tool_name    ON tool_calls(tool_name);

CREATE TABLE IF NOT EXISTS errors (
    error_id   TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    timestamp  REAL NOT NULL,
    error_type TEXT NOT NULL DEFAULT '',
    message    TEXT NOT NULL DEFAULT '',
    traceback  TEXT NOT NULL DEFAULT '',
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);
CREATE INDEX IF NOT EXISTS idx_err_session ON errors(session_id);
CREATE INDEX IF NOT EXISTS idx_err_ts      ON errors(timestamp);
"""


class SQLiteStorage:
    def __init__(self, db_path: Path = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return self._local.conn

    def _init_db(self) -> None:
        self._conn().executescript(_DDL)
        self._conn().commit()

    # ── public API ────────────────────────────────────────────────────────

    def save_session(self, session: Session) -> None:
        """Upsert session and all embedded calls/errors atomically."""
        conn = self._conn()
        conn.execute(
            "INSERT OR REPLACE INTO sessions VALUES (?,?,?,?,?,?,?)",
            (
                session.session_id,
                session.agent_name,
                session.start_time,
                session.end_time,
                session.status,
                session.total_cost_usd,
                json.dumps(session.metadata),
            ),
        )
        # Replace child records
        conn.execute("DELETE FROM llm_calls WHERE session_id=?", (session.session_id,))
        conn.execute("DELETE FROM tool_calls WHERE session_id=?", (session.session_id,))
        conn.execute("DELETE FROM errors WHERE session_id=?", (session.session_id,))
        for c in session.llm_calls:
            conn.execute(
                "INSERT INTO llm_calls VALUES (?,?,?,?,?,?,?,?,?,?)",
                (c.call_id, c.session_id, c.timestamp, c.model,
                 c.input_text, c.output_text, c.input_tokens, c.output_tokens,
                 c.cost_usd, c.latency_ms),
            )
        for c in session.tool_calls:
            conn.execute(
                "INSERT INTO tool_calls VALUES (?,?,?,?,?,?,?)",
                (c.call_id, c.session_id, c.timestamp, c.tool_name,
                 json.dumps(c.args), _json_result(c.result), c.latency_ms),
            )
        for e in session.errors:
            conn.execute(
                "INSERT INTO errors VALUES (?,?,?,?,?,?)",
                (e.error_id, e.session_id, e.timestamp, e.error_type,
                 e.message, e.traceback),
            )
        conn.commit()

    def get_session(self, session_id: str) -> Optional[Session]:
        """Return session with all embedded calls and errors loaded."""
        row = self._conn().execute(
            "SELECT * FROM sessions WHERE session_id=?", (session_id,)
        ).fetchone()
        if not row:
            return None
        session = _row_to_session(row)
        session.llm_calls = self._get_llm_calls(session_id)
        session.tool_calls = self._get_tool_calls(session_id)
        session.errors = self._get_errors(session_id)
        return session

    def list_sessions(
        self,
        agent_name: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
    ) -> list[Session]:
        """List sessions with summary data only (embedded lists are empty)."""
        q = "SELECT * FROM sessions WHERE 1=1"
        params: list = []
        if agent_name:
            q += " AND agent_name=?"
            params.append(agent_name)
        if status:
            q += " AND status=?"
            params.append(status)
        q += " ORDER BY start_time DESC LIMIT ?"
        params.append(limit)
        rows = self._conn().execute(q, params).fetchall()
        return [_row_to_session(r) for r in rows]

    def delete_session(self, session_id: str) -> None:
        """Delete session and all associated records."""
        conn = self._conn()
        conn.execute("DELETE FROM llm_calls WHERE session_id=?", (session_id,))
        conn.execute("DELETE FROM tool_calls WHERE session_id=?", (session_id,))
        conn.execute("DELETE FROM errors WHERE session_id=?", (session_id,))
        conn.execute("DELETE FROM sessions WHERE session_id=?", (session_id,))
        conn.commit()

    # ── analytics helpers (used by dashboard) ────────────────────────────

    def total_cost_by_model(self) -> list[dict]:
        rows = self._conn().execute(
            "SELECT model, SUM(cost_usd) as total, COUNT(*) as calls "
            "FROM llm_calls GROUP BY model ORDER BY total DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def cost_by_session(self, limit: int = 50) -> list[dict]:
        rows = self._conn().execute(
            "SELECT session_id, agent_name, total_cost_usd, status, start_time "
            "FROM sessions ORDER BY start_time DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ── internal helpers ──────────────────────────────────────────────────

    def _get_llm_calls(self, session_id: str) -> list[LLMCall]:
        rows = self._conn().execute(
            "SELECT * FROM llm_calls WHERE session_id=? ORDER BY timestamp",
            (session_id,),
        ).fetchall()
        return [_row_to_llm(r) for r in rows]

    def _get_tool_calls(self, session_id: str) -> list[ToolCall]:
        rows = self._conn().execute(
            "SELECT * FROM tool_calls WHERE session_id=? ORDER BY timestamp",
            (session_id,),
        ).fetchall()
        return [_row_to_tool(r) for r in rows]

    def _get_errors(self, session_id: str) -> list[ErrorRecord]:
        rows = self._conn().execute(
            "SELECT * FROM errors WHERE session_id=? ORDER BY timestamp",
            (session_id,),
        ).fetchall()
        return [_row_to_error(r) for r in rows]


# ── Row converters ─────────────────────────────────────────────────────────────


def _row_to_session(r: sqlite3.Row) -> Session:
    return Session(
        session_id=r["session_id"],
        agent_name=r["agent_name"],
        start_time=r["start_time"],
        end_time=r["end_time"],
        status=r["status"],
        total_cost_usd=r["total_cost_usd"],
        metadata=json.loads(r["metadata"]),
    )


def _row_to_llm(r: sqlite3.Row) -> LLMCall:
    return LLMCall(
        call_id=r["call_id"],
        session_id=r["session_id"],
        timestamp=r["timestamp"],
        model=r["model"],
        input_text=r["input_text"],
        output_text=r["output_text"],
        input_tokens=r["input_tokens"],
        output_tokens=r["output_tokens"],
        cost_usd=r["cost_usd"],
        latency_ms=r["latency_ms"],
    )


def _row_to_tool(r: sqlite3.Row) -> ToolCall:
    return ToolCall(
        call_id=r["call_id"],
        session_id=r["session_id"],
        timestamp=r["timestamp"],
        tool_name=r["tool_name"],
        args=json.loads(r["args"]),
        result=json.loads(r["result"]) if r["result"] is not None else None,
        latency_ms=r["latency_ms"],
    )


def _row_to_error(r: sqlite3.Row) -> ErrorRecord:
    return ErrorRecord(
        error_id=r["error_id"],
        session_id=r["session_id"],
        timestamp=r["timestamp"],
        error_type=r["error_type"],
        message=r["message"],
        traceback=r["traceback"],
    )


def _json_result(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        return json.dumps(value)
    except (TypeError, ValueError):
        return json.dumps(str(value))
