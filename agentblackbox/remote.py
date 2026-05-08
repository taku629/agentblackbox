"""Remote storage: push recordings to a cloud AgentBlackBox server."""
from __future__ import annotations

import json
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any, Optional

from .models import Session, LLMCall, ToolCall, ErrorRecord
from .storage import SQLiteStorage, DEFAULT_DB_PATH


class RemoteStorage(SQLiteStorage):
    """SQLiteStorage that also forwards events to a remote AgentBlackBox cloud server.

    Usage::

        from agentblackbox.remote import RemoteStorage
        from agentblackbox import BlackBox

        store = RemoteStorage(
            api_key="abx_...",
            endpoint="https://your-cloud.example.com",
        )
        with BlackBox.session("my-agent", storage=store) as bb:
            ...
    """

    def __init__(
        self,
        api_key: str,
        endpoint: str,
        db_path: Optional[Path] = None,
        timeout_s: float = 5.0,
    ) -> None:
        super().__init__(db_path or DEFAULT_DB_PATH)
        self._api_key = api_key
        self._endpoint = endpoint.rstrip("/")
        self._timeout = timeout_s

    def _post(self, path: str, payload: dict) -> None:
        url = f"{self._endpoint}{path}"
        data = json.dumps(payload, default=str).encode()
        req = urllib.request.Request(
            url,
            data=data,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "X-Api-Key": self._api_key,
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout):
                pass
        except urllib.error.URLError:
            pass  # never break agent code

    def create_session(self, session: Session) -> None:
        super().create_session(session)
        self._post("/api/v1/ingest/session", {
            "session_id": session.session_id,
            "agent_name": session.agent_name,
            "start_time": session.start_time,
            "status": session.status,
            "metadata": session.metadata,
        })

    def update_session(self, session: Session) -> None:
        super().update_session(session)
        self._post("/api/v1/ingest/session/update", {
            "session_id": session.session_id,
            "end_time": session.end_time,
            "status": session.status,
            "total_cost_usd": session.total_cost_usd,
        })

    def insert_llm_call(self, call: LLMCall) -> None:
        super().insert_llm_call(call)
        self._post("/api/v1/ingest/llm_call", {
            "id": call.id,
            "session_id": call.session_id,
            "timestamp": call.timestamp,
            "model": call.model,
            "input_tokens": call.input_tokens,
            "output_tokens": call.output_tokens,
            "input_text": call.input_text,
            "output_text": call.output_text,
            "duration_ms": call.duration_ms,
            "cost_usd": call.cost_usd,
        })

    def insert_tool_call(self, call: ToolCall) -> None:
        super().insert_tool_call(call)
        self._post("/api/v1/ingest/tool_call", {
            "id": call.id,
            "session_id": call.session_id,
            "timestamp": call.timestamp,
            "tool_name": call.tool_name,
            "arguments": call.arguments,
            "result": call.result,
            "duration_ms": call.duration_ms,
            "error": call.error,
        })

    def insert_error(self, err: ErrorRecord) -> None:
        super().insert_error(err)
        self._post("/api/v1/ingest/error", {
            "id": err.id,
            "session_id": err.session_id,
            "timestamp": err.timestamp,
            "error_type": err.error_type,
            "message": err.message,
            "traceback": err.traceback,
        })
