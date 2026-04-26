"""SQLite-backed storage for agentblackbox recordings."""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Optional

from .models import ErrorRecord, LLMCall, Session, ToolCall

DEFAULT_DB_PATH = Path.home() / ".agentblackbox" / "recordings.db"

_DDL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS sessions (
    session_id   TEXT PRIMARY KEY,
    agent_name   TEXT NOT NULL,
    start_time   INTEGER NOT NULL,
    end_time     INTEGER,
    status       TEXT NOT NULL DEFAULT 'running',
    total_cost_usd REAL NOT NULL DEFAULT 0.0,
    metadata     TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_sessions_agent ON sessions(agent_name);
CREATE INDEX IF NOT EXISTS idx_sessions_start ON sessions(start_time);
CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(status);

CREATE TABLE IF NOT EXISTS llm_calls (
    id           TEXT PRIMARY KEY,
    session_id   TEXT NOT NULL,
    timestamp    INTEGER NOT NULL,
    model        TEXT NOT NULL,
    input_tokens  INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    input_text   TEXT NOT NULL DEFAULT '',
    output_text  TEXT NOT NULL DEFAULT '',
    duration_ms  REAL NOT NULL DEFAULT 0.0,
    cost_usd     REAL NOT NULL DEFAULT 0.0,
    metadata     TEXT NOT NULL DEFAULT '{}',
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);
CREATE INDEX IF NOT EXISTS idx_llm_session ON llm_calls(session_id);
CREATE INDEX IF NOT EXISTS idx_llm_model   ON llm_calls(model);
CREATE INDEX IF NOT EXISTS idx_llm_ts      ON llm_calls(timestamp);

CREATE TABLE IF NOT EXISTS tool_calls (
    id           TEXT PRIMARY KEY,
    session_id   TEXT NOT NULL,
    timestamp    INTEGER NOT NULL,
    tool_name    TEXT NOT NULL,
    arguments    TEXT NOT NULL DEFAULT '{}',
    result       TEXT,
    duration_ms  REAL NOT NULL DEFAULT 0.0,
    error        TEXT,
    metadata     TEXT NOT NULL DEFAULT '{}',
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);
CREATE INDEX IF NOT EXISTS idx_tool_session   ON tool_calls(session_id);
CREATE INDEX IF NOT EXISTS idx_tool_name      ON tool_calls(tool_name);
CREATE INDEX IF NOT EXISTS idx_tool_ts        ON tool_calls(timestamp);

CREATE TABLE IF NOT EXISTS errors (
    id           TEXT PRIMARY KEY,
    session_id   TEXT NOT NULL,
    timestamp    INTEGER NOT NULL,
    error_type   TEXT NOT NULL DEFAULT '',
    message      TEXT NOT NULL DEFAULT '',
    traceback    TEXT NOT NULL DEFAULT '',
    metadata     TEXT NOT NULL DEFAULT '{}',
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

    # ── sessions ──────────────────────────────────────────────────────────

    def create_session(self, session: Session) -> None:
        self._conn().execute(
            "INSERT INTO sessions VALUES (?,?,?,?,?,?,?)",
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
        self._conn().commit()

    def update_session(self, session: Session) -> None:
        self._conn().execute(
            "UPDATE sessions SET end_time=?, status=?, total_cost_usd=?, metadata=? WHERE session_id=?",
            (
                session.end_time,
                session.status,
                session.total_cost_usd,
                json.dumps(session.metadata),
                session.session_id,
            ),
        )
        self._conn().commit()

    def get_session(self, session_id: str) -> Optional[Session]:
        row = self._conn().execute(
            "SELECT * FROM sessions WHERE session_id=?", (session_id,)
        ).fetchone()
        return _row_to_session(row) if row else None

    def list_sessions(
        self,
        agent_name: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> list[Session]:
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

    # ── llm_calls ─────────────────────────────────────────────────────────

    def insert_llm_call(self, call: LLMCall) -> None:
        self._conn().execute(
            "INSERT INTO llm_calls VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                call.id,
                call.session_id,
                call.timestamp,
                call.model,
                call.input_tokens,
                call.output_tokens,
                call.input_text,
                call.output_text,
                call.duration_ms,
                call.cost_usd,
                json.dumps(call.metadata),
            ),
        )
        self._conn().commit()

    def get_llm_calls(self, session_id: str) -> list[LLMCall]:
        rows = self._conn().execute(
            "SELECT * FROM llm_calls WHERE session_id=? ORDER BY timestamp", (session_id,)
        ).fetchall()
        return [_row_to_llm(r) for r in rows]

    # ── tool_calls ────────────────────────────────────────────────────────

    def insert_tool_call(self, call: ToolCall) -> None:
        self._conn().execute(
            "INSERT INTO tool_calls VALUES (?,?,?,?,?,?,?,?,?)",
            (
                call.id,
                call.session_id,
                call.timestamp,
                call.tool_name,
                json.dumps(call.arguments),
                json.dumps(call.result),
                call.duration_ms,
                call.error,
                json.dumps(call.metadata),
            ),
        )
        self._conn().commit()

    def get_tool_calls(self, session_id: str) -> list[ToolCall]:
        rows = self._conn().execute(
            "SELECT * FROM tool_calls WHERE session_id=? ORDER BY timestamp", (session_id,)
        ).fetchall()
        return [_row_to_tool(r) for r in rows]

    # ── errors ────────────────────────────────────────────────────────────

    def insert_error(self, err: ErrorRecord) -> None:
        self._conn().execute(
            "INSERT INTO errors VALUES (?,?,?,?,?,?,?)",
            (
                err.id,
                err.session_id,
                err.timestamp,
                err.error_type,
                err.message,
                err.traceback,
                json.dumps(err.metadata),
            ),
        )
        self._conn().commit()

    def get_errors(self, session_id: str) -> list[ErrorRecord]:
        rows = self._conn().execute(
            "SELECT * FROM errors WHERE session_id=? ORDER BY timestamp", (session_id,)
        ).fetchall()
        return [_row_to_error(r) for r in rows]

    # ── analytics ─────────────────────────────────────────────────────────

    def total_cost_by_model(self) -> list[dict]:
        rows = self._conn().execute(
            "SELECT model, SUM(cost_usd) as total, COUNT(*) as calls FROM llm_calls GROUP BY model ORDER BY total DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def cost_by_session(self, limit: int = 50) -> list[dict]:
        rows = self._conn().execute(
            "SELECT s.session_id, s.agent_name, s.total_cost_usd, s.status, s.start_time "
            "FROM sessions s ORDER BY s.start_time DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]


# ── Row converters ─────────────────────────────────────────────────────────


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
        id=r["id"],
        session_id=r["session_id"],
        timestamp=r["timestamp"],
        model=r["model"],
        input_tokens=r["input_tokens"],
        output_tokens=r["output_tokens"],
        input_text=r["input_text"],
        output_text=r["output_text"],
        duration_ms=r["duration_ms"],
        cost_usd=r["cost_usd"],
        metadata=json.loads(r["metadata"]),
    )


def _row_to_tool(r: sqlite3.Row) -> ToolCall:
    return ToolCall(
        id=r["id"],
        session_id=r["session_id"],
        timestamp=r["timestamp"],
        tool_name=r["tool_name"],
        arguments=json.loads(r["arguments"]),
        result=json.loads(r["result"]) if r["result"] else None,
        duration_ms=r["duration_ms"],
        error=r["error"],
        metadata=json.loads(r["metadata"]),
    )


def _row_to_error(r: sqlite3.Row) -> ErrorRecord:
    return ErrorRecord(
        id=r["id"],
        session_id=r["session_id"],
        timestamp=r["timestamp"],
        error_type=r["error_type"],
        message=r["message"],
        traceback=r["traceback"],
        metadata=json.loads(r["metadata"]),
    )
