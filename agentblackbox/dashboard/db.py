"""SQLite data access for the agentblackbox dashboard."""
from __future__ import annotations

import json
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional


class DBHelper:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _table_exists(self) -> bool:
        try:
            with self._conn() as c:
                c.execute("SELECT 1 FROM sessions LIMIT 1")
            return True
        except sqlite3.OperationalError:
            return False

    def get_sessions(
        self,
        agent_name: str = "",
        status: str = "",
        date_from: str = "",
        date_to: str = "",
        limit: int = 200,
    ) -> list[dict]:
        if not self._table_exists():
            return []
        q = """
            SELECT s.*,
                COUNT(DISTINCT l.id) AS llm_count,
                COUNT(DISTINCT t.id) AS tool_count
            FROM sessions s
            LEFT JOIN llm_calls l ON s.session_id = l.session_id
            LEFT JOIN tool_calls t ON s.session_id = t.session_id
            WHERE 1=1
        """
        params: list = []
        if agent_name:
            q += " AND s.agent_name = ?"
            params.append(agent_name)
        if status:
            q += " AND s.status = ?"
            params.append(status)
        if date_from:
            try:
                dt = datetime.strptime(date_from, "%Y-%m-%d")
                q += " AND s.start_time >= ?"
                params.append(int(dt.timestamp() * 1e9))
            except ValueError:
                pass
        if date_to:
            try:
                dt = datetime.strptime(date_to, "%Y-%m-%d") + timedelta(days=1)
                q += " AND s.start_time < ?"
                params.append(int(dt.timestamp() * 1e9))
            except ValueError:
                pass
        q += " GROUP BY s.session_id ORDER BY s.start_time DESC LIMIT ?"
        params.append(limit)
        with self._conn() as conn:
            rows = conn.execute(q, params).fetchall()
        return [_enrich_session(dict(r)) for r in rows]

    def get_summary_stats(self) -> dict:
        if not self._table_exists():
            return {"total_sessions": 0, "total_cost": 0.0, "success_rate": 0.0, "avg_duration_ms": 0.0}
        with self._conn() as conn:
            row = conn.execute("""
                SELECT
                    COUNT(*) AS total_sessions,
                    COALESCE(SUM(total_cost_usd), 0) AS total_cost,
                    COALESCE(AVG(CASE WHEN status='success' THEN 1.0 ELSE 0.0 END), 0) AS success_rate,
                    COALESCE(AVG(CASE WHEN end_time IS NOT NULL THEN (end_time - start_time) / 1000000.0 END), 0) AS avg_duration_ms
                FROM sessions
            """).fetchone()
        return dict(row) if row else {}

    def get_distinct_agents(self) -> list[str]:
        if not self._table_exists():
            return []
        with self._conn() as conn:
            rows = conn.execute("SELECT DISTINCT agent_name FROM sessions ORDER BY agent_name").fetchall()
        return [r[0] for r in rows]

    def get_session_detail(self, session_id: str) -> Optional[dict]:
        if not self._table_exists():
            return None
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM sessions WHERE session_id=?", (session_id,)).fetchone()
            if not row:
                return None
            session = _enrich_session(dict(row))

            llm_rows = conn.execute(
                "SELECT * FROM llm_calls WHERE session_id=? ORDER BY timestamp", (session_id,)
            ).fetchall()
            tool_rows = conn.execute(
                "SELECT * FROM tool_calls WHERE session_id=? ORDER BY timestamp", (session_id,)
            ).fetchall()
            error_rows = conn.execute(
                "SELECT * FROM errors WHERE session_id=? ORDER BY timestamp", (session_id,)
            ).fetchall()

        events: list[dict] = []
        for r in llm_rows:
            d = dict(r)
            d["_type"] = "llm"
            d["_dt"] = _fmt_ns(d["timestamp"])
            events.append(d)
        for r in tool_rows:
            d = dict(r)
            d["_type"] = "tool"
            d["_dt"] = _fmt_ns(d["timestamp"])
            d["arguments"] = _try_json(d.get("arguments", "{}"))
            d["result"] = _try_json(d.get("result", "null"))
            events.append(d)
        for r in error_rows:
            d = dict(r)
            d["_type"] = "error"
            d["_dt"] = _fmt_ns(d["timestamp"])
            events.append(d)

        events.sort(key=lambda x: x["timestamp"])

        # Sidebar stats
        model_costs: dict[str, float] = {}
        for r in llm_rows:
            model_costs[r["model"]] = model_costs.get(r["model"], 0.0) + r["cost_usd"]

        tool_counts: dict[str, int] = {}
        for r in tool_rows:
            tool_counts[r["tool_name"]] = tool_counts.get(r["tool_name"], 0) + 1

        return {
            "session": session,
            "events": events,
            "model_costs": model_costs,
            "tool_counts": tool_counts,
        }

    def export_session_json(self, session_id: str) -> Optional[dict]:
        if not self._table_exists():
            return None
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM sessions WHERE session_id=?", (session_id,)).fetchone()
            if not row:
                return None
            session = _enrich_session(dict(row))
            llm_rows = conn.execute(
                "SELECT * FROM llm_calls WHERE session_id=? ORDER BY timestamp", (session_id,)
            ).fetchall()
            tool_rows = conn.execute(
                "SELECT * FROM tool_calls WHERE session_id=? ORDER BY timestamp", (session_id,)
            ).fetchall()
            error_rows = conn.execute(
                "SELECT * FROM errors WHERE session_id=? ORDER BY timestamp", (session_id,)
            ).fetchall()
        return {
            "session": {k: v for k, v in session.items()},
            "llm_calls": [dict(r) for r in llm_rows],
            "tool_calls": [
                {**dict(r), "arguments": _try_json(r["arguments"]), "result": _try_json(r["result"])}
                for r in tool_rows
            ],
            "errors": [dict(r) for r in error_rows],
        }

    def get_analytics_data(self, period: str = "7d") -> dict:
        if not self._table_exists():
            return {"daily": [], "agent_costs": [], "model_costs": [], "period": period}

        now_ns = time.time_ns()
        periods = {"24h": 24 * 3600, "7d": 7 * 24 * 3600, "30d": 30 * 24 * 3600}
        cutoff_ns = now_ns - periods.get(period, 7 * 24 * 3600) * 1_000_000_000

        with self._conn() as conn:
            daily = conn.execute("""
                SELECT date(start_time/1000000000, 'unixepoch') AS day,
                    ROUND(SUM(total_cost_usd), 8) AS cost,
                    COUNT(*) AS sessions
                FROM sessions WHERE start_time > ?
                GROUP BY day ORDER BY day
            """, (cutoff_ns,)).fetchall()

            agent_costs = conn.execute("""
                SELECT agent_name,
                    ROUND(SUM(total_cost_usd), 8) AS cost,
                    COUNT(*) AS sessions,
                    ROUND(AVG(CASE WHEN status='success' THEN 1.0 ELSE 0.0 END) * 100, 1) AS success_rate
                FROM sessions WHERE start_time > ?
                GROUP BY agent_name ORDER BY cost DESC
            """, (cutoff_ns,)).fetchall()

            model_costs = conn.execute("""
                SELECT l.model,
                    ROUND(SUM(l.cost_usd), 8) AS cost,
                    SUM(l.input_tokens + l.output_tokens) AS total_tokens,
                    COUNT(*) AS calls
                FROM llm_calls l
                JOIN sessions s ON l.session_id = s.session_id
                WHERE s.start_time > ?
                GROUP BY l.model ORDER BY cost DESC
            """, (cutoff_ns,)).fetchall()

        return {
            "daily": [dict(r) for r in daily],
            "agent_costs": [dict(r) for r in agent_costs],
            "model_costs": [dict(r) for r in model_costs],
            "period": period,
        }


# ── Helpers ────────────────────────────────────────────────────────────────────

def _enrich_session(s: dict) -> dict:
    s["start_dt"] = datetime.fromtimestamp(s["start_time"] / 1e9).strftime("%Y-%m-%d %H:%M:%S")
    if s.get("end_time"):
        ms = (s["end_time"] - s["start_time"]) / 1e6
        s["duration_ms"] = ms
        s["duration_str"] = _fmt_duration(ms)
    else:
        s["duration_ms"] = None
        s["duration_str"] = "—"
    s["session_id_short"] = s["session_id"][:8]
    return s


def _fmt_duration(ms: float) -> str:
    if ms < 1000:
        return f"{ms:.0f}ms"
    elif ms < 60_000:
        return f"{ms / 1000:.1f}s"
    return f"{ms / 60_000:.1f}m"


def _fmt_ns(ns: int) -> str:
    return datetime.fromtimestamp(ns / 1e9).strftime("%H:%M:%S.%f")[:-3]


def _try_json(s) -> object:
    if s is None:
        return None
    if isinstance(s, (dict, list)):
        return s
    try:
        return json.loads(s)
    except Exception:
        return s
