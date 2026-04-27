"""Core BlackBox recorder: context manager and decorator."""
from __future__ import annotations

import contextvars
import functools
import json
import time
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

from .cost import calculate_cost
from .masking import PIIMasker
from .models import ErrorRecord, LLMCall, Session, ToolCall
from .storage import DEFAULT_DB_PATH, SQLiteStorage

F = TypeVar("F", bound=Callable[..., Any])

_storage: Optional[SQLiteStorage] = None
_masker: Optional[PIIMasker] = None
_current_bb: contextvars.ContextVar[Optional["BlackBox"]] = contextvars.ContextVar(
    "_current_bb", default=None
)


def _get_storage() -> SQLiteStorage:
    global _storage
    if _storage is None:
        _storage = SQLiteStorage(DEFAULT_DB_PATH)
    return _storage


class BlackBox:
    """Black-box recorder for AI agent sessions."""

    def __init__(self, agent_name: str, metadata: Optional[dict] = None) -> None:
        self.agent_name = agent_name
        self._session = Session(
            session_id=str(uuid.uuid4()),
            agent_name=agent_name,
            start_time=time.time(),
            metadata=metadata or {},
        )
        self._token: Optional[contextvars.Token] = None

    # ── class-level configuration ─────────────────────────────────────────

    @classmethod
    def configure(
        cls,
        db_path: Optional[str] = None,
        masking: bool = False,
        mask_patterns: Optional[list[str]] = None,
        custom_mask_patterns: Optional[dict[str, str]] = None,
    ) -> None:
        global _storage, _masker
        _storage = SQLiteStorage(Path(db_path) if db_path else DEFAULT_DB_PATH)
        if masking:
            _masker = PIIMasker(
                patterns=mask_patterns,
                custom_patterns=custom_mask_patterns,
                enabled=True,
            )
        else:
            _masker = None

    # ── factory classmethods ──────────────────────────────────────────────

    @classmethod
    def record(cls, agent_name: str) -> Callable[[F], F]:
        """Decorator: @BlackBox.record(agent_name='x')"""
        def decorator(func: F) -> F:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                with cls.session(agent_name) as bb:  # noqa: F841
                    return func(*args, **kwargs)
            return wrapper  # type: ignore[return-value]
        return decorator

    @classmethod
    def session(cls, agent_name: str, metadata: Optional[dict] = None) -> "_BlackBoxSession":
        """Context manager: with BlackBox.session('x') as bb"""
        return _BlackBoxSession(agent_name, metadata or {})

    @classmethod
    def current(cls) -> Optional["BlackBox"]:
        """Return the currently active BlackBox, or None."""
        return _current_bb.get()

    # ── recording ─────────────────────────────────────────────────────────

    def record_llm_call(
        self,
        model: str,
        input_text: str,
        output_text: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        latency_ms: float = 0.0,
    ) -> LLMCall:
        global _masker
        if _masker is not None:
            input_text = _masker.mask(input_text)
            output_text = _masker.mask(output_text)
        cost = calculate_cost(model, input_tokens, output_tokens)
        call = LLMCall(
            call_id=str(uuid.uuid4()),
            session_id=self._session.session_id,
            timestamp=time.time(),
            model=model,
            input_text=input_text,
            output_text=output_text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            latency_ms=latency_ms,
        )
        self._session.llm_calls.append(call)
        self._session.total_cost_usd += cost
        return call

    def record_tool_call(
        self,
        tool_name: str,
        args: dict,
        result: Any,
        latency_ms: float = 0.0,
    ) -> ToolCall:
        global _masker
        if _masker is not None:
            args = _masker.mask_dict(args)
            if isinstance(result, dict):
                result = _masker.mask_dict(result)
        call = ToolCall(
            call_id=str(uuid.uuid4()),
            session_id=self._session.session_id,
            timestamp=time.time(),
            tool_name=tool_name,
            args=args,
            result=result,
            latency_ms=latency_ms,
        )
        self._session.tool_calls.append(call)
        return call

    def record_error(self, error: Exception) -> ErrorRecord:
        tb_str = "".join(
            traceback.format_exception(type(error), error, error.__traceback__)
        )
        err = ErrorRecord(
            error_id=str(uuid.uuid4()),
            session_id=self._session.session_id,
            timestamp=time.time(),
            error_type=type(error).__name__,
            message=str(error),
            traceback=tb_str,
        )
        self._session.errors.append(err)
        self._session.status = "error"
        return err

    # ── replay / export ───────────────────────────────────────────────────

    def replay(self) -> None:
        """Print an annotated timeline of this session to the console."""
        session = self._session
        events: list[tuple[float, str, Any]] = []
        for c in session.llm_calls:
            events.append((c.timestamp, "llm", c))
        for c in session.tool_calls:
            events.append((c.timestamp, "tool", c))
        for e in session.errors:
            events.append((e.timestamp, "error", e))
        events.sort(key=lambda x: x[0])

        start_dt = datetime.fromtimestamp(session.start_time)
        print(f"\n{'='*70}")
        print(f"  Session: {session.session_id}")
        print(f"  Agent:   {session.agent_name}")
        print(f"  Status:  {session.status}")
        print(f"  Start:   {start_dt.strftime('%Y-%m-%d %H:%M:%S')}")
        if session.end_time:
            elapsed_s = session.end_time - session.start_time
            print(f"  Elapsed: {elapsed_s * 1000:.1f}ms")
        print(f"  Cost:    ${session.total_cost_usd:.6f}")
        print(f"{'='*70}")

        for ts, kind, ev in events:
            dt = datetime.fromtimestamp(ts)
            prefix = dt.strftime("%H:%M:%S.%f")[:-3]
            if kind == "llm":
                print(f"\n[{prefix}] 🤖 LLM  model={ev.model}  "
                      f"tokens={ev.input_tokens}→{ev.output_tokens}  "
                      f"${ev.cost_usd:.6f}  {ev.latency_ms:.1f}ms")
                _print_truncated("IN ", ev.input_text, 200)
                _print_truncated("OUT", ev.output_text, 200)
            elif kind == "tool":
                print(f"\n[{prefix}] 🔧 TOOL  {ev.tool_name}  {ev.latency_ms:.1f}ms")
                _print_truncated("ARGS  ", json.dumps(ev.args), 200)
                _print_truncated("RESULT", str(ev.result), 200)
            elif kind == "error":
                print(f"\n[{prefix}] ❌ ERROR  {ev.error_type}: {ev.message}")
                print("  " + ev.traceback.replace("\n", "\n  ").rstrip())

        if session.end_time:
            elapsed_s = session.end_time - session.start_time
            print(f"\n⬛  END — total: ${session.total_cost_usd:.4f}, {elapsed_s:.2f}s")
        print(f"{'='*70}\n")

    def export_json(self, path: str) -> None:
        """Write session data as JSON to the given file path."""
        session = self._session
        data = {
            "session": {
                "session_id": session.session_id,
                "agent_name": session.agent_name,
                "start_time": session.start_time,
                "end_time": session.end_time,
                "status": session.status,
                "total_cost_usd": session.total_cost_usd,
                "metadata": session.metadata,
            },
            "llm_calls": [
                {
                    "call_id": c.call_id,
                    "timestamp": c.timestamp,
                    "model": c.model,
                    "input_text": c.input_text,
                    "output_text": c.output_text,
                    "input_tokens": c.input_tokens,
                    "output_tokens": c.output_tokens,
                    "cost_usd": c.cost_usd,
                    "latency_ms": c.latency_ms,
                }
                for c in session.llm_calls
            ],
            "tool_calls": [
                {
                    "call_id": c.call_id,
                    "timestamp": c.timestamp,
                    "tool_name": c.tool_name,
                    "args": c.args,
                    "result": c.result,
                    "latency_ms": c.latency_ms,
                }
                for c in session.tool_calls
            ],
            "errors": [
                {
                    "error_id": e.error_id,
                    "timestamp": e.timestamp,
                    "error_type": e.error_type,
                    "message": e.message,
                    "traceback": e.traceback,
                }
                for e in session.errors
            ],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # ── query ─────────────────────────────────────────────────────────────

    @classmethod
    def list_sessions(
        cls,
        agent_name: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
    ) -> list[Session]:
        return _get_storage().list_sessions(
            agent_name=agent_name, status=status, limit=limit
        )


class _BlackBoxSession(BlackBox):
    """BlackBox subclass that acts as a context manager."""

    def __init__(self, agent_name: str, metadata: dict) -> None:
        super().__init__(agent_name, metadata)

    def __enter__(self) -> "BlackBox":
        self._token = _current_bb.set(self)
        _get_storage().save_session(self._session)  # persist initial running state
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_type is not None:
            self.record_error(exc_val)
        else:
            self._session.status = "success"
        self._session.end_time = time.time()
        _get_storage().save_session(self._session)
        if self._token is not None:
            _current_bb.reset(self._token)
            self._token = None
        return False  # don't suppress exceptions


def _print_truncated(label: str, text: str, max_len: int) -> None:
    if len(text) > max_len:
        text = text[:max_len] + f"... [{len(text) - max_len} more chars]"
    print(f"  {label}: {text}")
