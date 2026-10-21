"""FastAPI dashboard for agentblackbox."""
from __future__ import annotations

import json
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import hashlib
import hmac
import os
import secrets

from fastapi import FastAPI, Request, Query, HTTPException, Depends
from fastapi.responses import HTMLResponse, Response, JSONResponse
from fastapi.templating import Jinja2Templates

from agentblackbox.storage import DEFAULT_DB_PATH, SQLiteStorage
from .db import DBHelper

_TEMPLATES_DIR = Path(__file__).parent / "templates"


def _require_api_key(api_key: Optional[str] = None):
    """Returns a dependency that validates X-Api-Key header when cloud mode is active."""
    def _check(request: Request):
        if api_key is None:
            return  # local mode — no auth
        key = request.headers.get("X-Api-Key", "")
        if not hmac.compare_digest(key, api_key):
            raise HTTPException(status_code=401, detail="Invalid API key")
    return _check


def create_app(db_path: Optional[Path] = None, api_key: Optional[str] = None) -> FastAPI:
    db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
    db = DBHelper(db_path)
    store = SQLiteStorage(db_path)
    auth = _require_api_key(api_key)

    templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))
    templates.env.filters["pretty_json"] = _pretty_json
    templates.env.filters["fmt_cost"] = lambda v: f"${v:.6f}" if v else "$0.000000"
    templates.env.filters["fmt_ns"] = lambda ns: datetime.fromtimestamp(ns / 1e9).strftime("%Y-%m-%d %H:%M:%S") if ns else "—"

    app = FastAPI(title="AgentBlackBox Dashboard", docs_url=None, redoc_url=None)

    # ── Screen 1: Sessions list ──────────────────────────────────────────

    @app.get("/", response_class=HTMLResponse)
    async def sessions_page(
        request: Request,
        agent_name: str = "",
        status: str = "",
        date_from: str = "",
        date_to: str = "",
    ):
        filters = dict(agent_name=agent_name, status=status, date_from=date_from, date_to=date_to)
        return templates.TemplateResponse(request, "sessions.html", {
            "sessions": db.get_sessions(**filters),
            "stats": db.get_summary_stats(),
            "agents": db.get_distinct_agents(),
            "filters": filters,
        })

    @app.get("/api/sessions", response_class=HTMLResponse)
    async def api_sessions(
        request: Request,
        agent_name: str = "",
        status: str = "",
        date_from: str = "",
        date_to: str = "",
    ):
        filters = dict(agent_name=agent_name, status=status, date_from=date_from, date_to=date_to)
        return templates.TemplateResponse(request, "partials/sessions_rows.html", {
            "sessions": db.get_sessions(**filters),
        })

    # ── Screen 2: Session detail ─────────────────────────────────────────

    @app.get("/sessions/{session_id}", response_class=HTMLResponse)
    async def session_detail(request: Request, session_id: str):
        detail = db.get_session_detail(session_id)
        if not detail:
            return HTMLResponse(_not_found_html("Session not found"), status_code=404)
        return templates.TemplateResponse(request, "session_detail.html", detail)

    @app.get("/sessions/{session_id}/export")
    async def export_session(session_id: str):
        data = db.export_session_json(session_id)
        if data is None:
            return Response("Session not found", status_code=404)
        return Response(
            content=json.dumps(data, ensure_ascii=False, indent=2, default=str),
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="session_{session_id[:8]}.json"'},
        )

    # ── Screen 3: Analytics ──────────────────────────────────────────────

    @app.get("/analytics", response_class=HTMLResponse)
    async def analytics_page(request: Request, period: str = "7d"):
        if period not in ("24h", "7d", "30d"):
            period = "7d"
        analytics = db.get_analytics_data(period)
        return templates.TemplateResponse(request, "analytics.html", {
            "period": period,
            "analytics_json": json.dumps(analytics),
            "analytics": analytics,
        })

    # ── Cloud ingest API ─────────────────────────────────────────────────

    @app.post("/api/v1/ingest/session", dependencies=[Depends(auth)])
    async def ingest_session(request: Request):
        body = await request.json()
        from agentblackbox.models import Session
        s = Session(
            session_id=body["session_id"],
            agent_name=body["agent_name"],
            start_time=body["start_time"],
            status=body.get("status", "running"),
            metadata=body.get("metadata", {}),
        )
        try:
            store.create_session(s)
        except Exception:
            pass  # duplicate — ignore
        return JSONResponse({"ok": True})

    @app.post("/api/v1/ingest/session/update", dependencies=[Depends(auth)])
    async def ingest_session_update(request: Request):
        body = await request.json()
        from agentblackbox.models import Session
        existing = store.get_session(body["session_id"])
        if existing:
            existing.end_time = body.get("end_time")
            existing.status = body.get("status", existing.status)
            existing.total_cost_usd = body.get("total_cost_usd", existing.total_cost_usd)
            store.update_session(existing)
        return JSONResponse({"ok": True})

    @app.post("/api/v1/ingest/llm_call", dependencies=[Depends(auth)])
    async def ingest_llm_call(request: Request):
        body = await request.json()
        from agentblackbox.models import LLMCall
        call = LLMCall(
            id=body["id"],
            session_id=body["session_id"],
            timestamp=body["timestamp"],
            model=body["model"],
            input_tokens=body["input_tokens"],
            output_tokens=body["output_tokens"],
            input_text=body.get("input_text", ""),
            output_text=body.get("output_text", ""),
            duration_ms=body.get("duration_ms", 0.0),
            cost_usd=body.get("cost_usd", 0.0),
        )
        store.insert_llm_call(call)
        return JSONResponse({"ok": True})

    @app.post("/api/v1/ingest/tool_call", dependencies=[Depends(auth)])
    async def ingest_tool_call(request: Request):
        body = await request.json()
        from agentblackbox.models import ToolCall
        call = ToolCall(
            id=body["id"],
            session_id=body["session_id"],
            timestamp=body["timestamp"],
            tool_name=body["tool_name"],
            arguments=body.get("arguments", {}),
            result=body.get("result"),
            duration_ms=body.get("duration_ms", 0.0),
            error=body.get("error"),
        )
        store.insert_tool_call(call)
        return JSONResponse({"ok": True})

    @app.post("/api/v1/ingest/error", dependencies=[Depends(auth)])
    async def ingest_error(request: Request):
        body = await request.json()
        from agentblackbox.models import ErrorRecord
        err = ErrorRecord(
            id=body["id"],
            session_id=body["session_id"],
            timestamp=body["timestamp"],
            error_type=body.get("error_type", ""),
            message=body.get("message", ""),
            traceback=body.get("traceback", ""),
        )
        store.insert_error(err)
        return JSONResponse({"ok": True})

    @app.get("/api/v1/health")
    async def health():
        return {"status": "ok", "cloud_mode": api_key is not None}

    return app



# ── Helpers ────────────────────────────────────────────────────────────────────

def _pretty_json(value) -> str:
    try:
        return json.dumps(value, indent=2, ensure_ascii=False)
    except Exception:
        return str(value)


def _not_found_html(msg: str) -> str:
    return f'<html><body style="background:#0f1117;color:#e2e8f0;font-family:system-ui;padding:40px"><h1>{msg}</h1><a href="/" style="color:#6366f1">← Back</a></body></html>'
