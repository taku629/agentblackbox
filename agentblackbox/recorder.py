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
from .models import ErrorRecord, LLMCall, Session, ToolCall
from .storage import DEFAULT_DB_PATH, SQLiteStorage

F = TypeVar("F", bound=Callable[..., Any])

_current_bb: contextvars.ContextVar[Optional["BlackBox"]] = contextvars.ContextVar(
    "_current_bb", default=None
)

_default_storage: Optional[SQLiteStorage] = None


def _get_storage(db_path: Optional[Path] = None) -> SQLiteStorage:
    global _default_storage
    if db_path is not None:
        return SQLiteStorage(db_path)
    if _default_storage is None:
        _default_storage = SQLiteStorage(DEFAULT_DB_PATH)
    return _default_storage


class BlackBox:
    """Black-box recorder for AI agent sessions.

    Supports three usage patterns:
      1. Decorator:          @BlackBox.record(agent_name="x")
      2. Context manager:   with BlackBox.session("x") as bb:
      3. Manual:            bb = BlackBox("x"); bb.start(); ...; bb.stop()
    """

    def __init__(
        self,
        agent_name: str,
        session_id: Optional[str] = None,
        db_path: Optional[Path] = None,
        storage: Optional[SQLiteStorage] = None,
    ) -> None:
        self.agent_name = agent_name
        self.session_id = session_id or str(uuid.uuid4())
        self._db_path = db_path
        self._storage: Optional[SQLiteStorage] = storage
        self._session: Optional[Session] = None
        self._token: Optional[contextvars.Token] = None
        self._total_cost = 0.0

    # ── storage ───────────────────────────────────────────────────────────

    def _store(self) -> SQLiteStorage:
        if self._storage is None:
            self._storage = _get_storage(self._db_path)
        return self._storage

    # ── lifecycle ─────────────────────────────────────────────────────────

    def start(self) -> "BlackBox":
        self._session = Session(
            session_id=self.session_id,
            agent_name=self.agent_name,
            start_time=time.time_ns(),
            status="running",
        )
        self._store().create_session(self._session)
        self._token = _current_bb.set(self)
        return self

    def stop(self, success: bool = True) -> None:
        if self._session is None:
            return
        self._session.end_time = time.time_ns()
        self._session.status = "success" if success else "error"
        self._session.total_cost_usd = self._total_cost
        self._store().update_session(self._session)
        if self._token is not None:
            _current_bb.reset(self._token)
            self._token = None

    # ── context manager ───────────────────────────────────────────────────

    def __enter__(self) -> "BlackBox":
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_type is not None:
            self.record_error(exc_val, exc_tb)
        self.stop(success=exc_type is None)
        return False  # don't suppress exceptions

    # ── factory classmethods ──────────────────────────────────────────────

    @classmethod
    def session(
        cls,
        agent_name: str,
        session_id: Optional[str] = None,
        db_path: Optional[Path] = None,
        storage: Optional[SQLiteStorage] = None,
    ) -> "BlackBox":
        return cls(
            agent_name,
            session_id=session_id,
            db_path=db_path,
            storage=storage,
        )

    @classmethod
    def record(
        cls,
        agent_name: str,
        db_path: Optional[Path] = None,
        storage: Optional[SQLiteStorage] = None,
    ) -> Callable[[F], F]:
        def decorator(func: F) -> F:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                with cls.session(agent_name, db_path=db_path, storage=storage):
                    return func(*args, **kwargs)

            return wrapper  # type: ignore[return-value]

        return decorator

    @classmethod
    def current(cls) -> Optional["BlackBox"]:
        return _current_bb.get()

    # ── recording ─────────────────────────────────────────────────────────

    def record_llm_call(
        self,
        model: str,
        input_text: str,
        output_text: str,
        input_tokens: int,
        output_tokens: int,
        duration_ms: float,
        metadata: Optional[dict] = None,
    ) -> LLMCall:
        cost = calculate_cost(model, input_tokens, output_tokens)
        self._total_cost += cost
        call = LLMCall(
            id=str(uuid.uuid4()),
            session_id=self.session_id,
            timestamp=time.time_ns(),
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_text=input_text,
            output_text=output_text,
            duration_ms=duration_ms,
            cost_usd=cost,
            metadata=metadata or {},
        )
        self._store().insert_llm_call(call)
        return call

    def record_tool_call(
        self,
        tool_name: str,
        arguments: dict,
        result: Any,
        duration_ms: float,
        error: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> ToolCall:
        call = ToolCall(
            id=str(uuid.uuid4()),
            session_id=self.session_id,
            timestamp=time.time_ns(),
            tool_name=tool_name,
            arguments=arguments,
            result=result,
            duration_ms=duration_ms,
            error=error,
            metadata=metadata or {},
        )
        self._store().insert_tool_call(call)
        return call

    def record_error(
        self,
        exc: BaseException,
        tb=None,
        metadata: Optional[dict] = None,
    ) -> ErrorRecord:
        tb_str = "".join(traceback.format_exception(type(exc), exc, tb or exc.__traceback__))
        err = ErrorRecord(
            id=str(uuid.uuid4()),
            session_id=self.session_id,
            timestamp=time.time_ns(),
            error_type=type(exc).__name__,
            message=str(exc),
            traceback=tb_str,
            metadata=metadata or {},
        )
        self._store().insert_error(err)
        return err

    # ── events ────────────────────────────────────────────────────────────

    def iter_events(self, session_id: Optional[str] = None):
        """Yield (timestamp, kind, event) tuples in chronological order.

        kind is one of: "llm", "tool", "error"
        """
        sid = session_id or self.session_id
        store = self._store()
        events: list[tuple[int, str, Any]] = []
        for c in store.get_llm_calls(sid):
            events.append((c.timestamp, "llm", c))
        for c in store.get_tool_calls(sid):
            events.append((c.timestamp, "tool", c))
        for e in store.get_errors(sid):
            events.append((e.timestamp, "error", e))
        events.sort(key=lambda x: x[0])
        yield from events

    # ── query ─────────────────────────────────────────────────────────────

    @classmethod
    def list_sessions(
        cls,
        agent_name: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
        db_path: Optional[Path] = None,
    ) -> list[Session]:
        return _get_storage(db_path).list_sessions(agent_name=agent_name, status=status, limit=limit)

    # ── replay / export ───────────────────────────────────────────────────

    @classmethod
    def replay(cls, session_id: str, db_path: Optional[Path] = None) -> None:
        sid = session_id
        store = _get_storage(db_path)
        session = store.get_session(sid)
        if session is None:
            print(f"Session not found: {sid}")
            return

        llm_calls = store.get_llm_calls(sid)
        tool_calls = store.get_tool_calls(sid)
        errors = store.get_errors(sid)

        events: list[tuple[int, str, Any]] = []
        for c in llm_calls:
            events.append((c.timestamp, "llm", c))
        for c in tool_calls:
            events.append((c.timestamp, "tool", c))
        for e in errors:
            events.append((e.timestamp, "error", e))
        events.sort(key=lambda x: x[0])

        start_dt = datetime.fromtimestamp(session.start_time / 1e9)
        end_dt = datetime.fromtimestamp(session.end_time / 1e9) if session.end_time else None

        print(f"\n{'='*70}")
        print(f"  Session: {sid}")
        print(f"  Agent:   {session.agent_name}")
        print(f"  Status:  {session.status}")
        print(f"  Start:   {start_dt.strftime('%Y-%m-%d %H:%M:%S.%f')}")
        if end_dt:
            elapsed_ms = (session.end_time - session.start_time) / 1e6  # type: ignore[operator]
            print(f"  End:     {end_dt.strftime('%Y-%m-%d %H:%M:%S.%f')}  ({elapsed_ms:.1f}ms)")
        print(f"  Cost:    ${session.total_cost_usd:.6f}")
        print(f"{'='*70}")

        for ts, kind, ev in events:
            dt = datetime.fromtimestamp(ts / 1e9)
            prefix = dt.strftime("%H:%M:%S.%f")[:-3]
            if kind == "llm":
                print(f"\n[{prefix}] LLM  model={ev.model}  tokens={ev.input_tokens}→{ev.output_tokens}  ${ev.cost_usd:.6f}  {ev.duration_ms:.1f}ms")
                _print_truncated("  IN ", ev.input_text, 200)
                _print_truncated("  OUT", ev.output_text, 200)
            elif kind == "tool":
                status_mark = "✗" if ev.error else "✓"
                print(f"\n[{prefix}] TOOL [{status_mark}] {ev.tool_name}  {ev.duration_ms:.1f}ms")
                _print_truncated("  ARGS  ", json.dumps(ev.arguments), 200)
                if ev.error:
                    _print_truncated("  ERROR ", ev.error, 200)
                else:
                    _print_truncated("  RESULT", str(ev.result), 200)
            elif kind == "error":
                print(f"\n[{prefix}] ERROR {ev.error_type}: {ev.message}")
                print("  " + ev.traceback.replace("\n", "\n  ").rstrip())
        print(f"\n{'='*70}\n")

    @classmethod
    def export_json(cls, session_id: str, db_path: Optional[Path] = None) -> str:
        sid = session_id
        store = _get_storage(db_path)
        session = store.get_session(sid)
        if session is None:
            raise ValueError(f"Session not found: {sid}")

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
                    "id": c.id,
                    "timestamp": c.timestamp,
                    "model": c.model,
                    "input_tokens": c.input_tokens,
                    "output_tokens": c.output_tokens,
                    "input_text": c.input_text,
                    "output_text": c.output_text,
                    "duration_ms": c.duration_ms,
                    "cost_usd": c.cost_usd,
                }
                for c in store.get_llm_calls(sid)
            ],
            "tool_calls": [
                {
                    "id": c.id,
                    "timestamp": c.timestamp,
                    "tool_name": c.tool_name,
                    "arguments": c.arguments,
                    "result": c.result,
                    "duration_ms": c.duration_ms,
                    "error": c.error,
                }
                for c in store.get_tool_calls(sid)
            ],
            "errors": [
                {
                    "id": e.id,
                    "timestamp": e.timestamp,
                    "error_type": e.error_type,
                    "message": e.message,
                    "traceback": e.traceback,
                }
                for e in store.get_errors(sid)
            ],
        }
        return json.dumps(data, ensure_ascii=False, indent=2)


    @classmethod
    def load_recording(
        cls,
        session_id: str,
        db_path: Optional[Path] = None,
    ) -> "MockBlackBox":
        """Load a previously recorded session for deterministic replay in tests."""
        return MockBlackBox(session_id, db_path=db_path)


class MockBlackBox(BlackBox):
    """A BlackBox backed by pre-recorded responses for deterministic testing.

    Instead of making live API calls, agent code can call pop_llm_response()
    and pop_tool_result() to consume responses recorded in a prior session.

    Usage::

        mock = BlackBox.load_recording("session-id-abc")
        # In your agent test:
        recorded = mock.pop_llm_response()
        assert recorded.output_text == "expected answer"
    """

    def __init__(self, original_session_id: str, db_path: Optional[Path] = None) -> None:
        store = _get_storage(db_path)
        original = store.get_session(original_session_id)
        if original is None:
            raise ValueError(f"Session not found: {original_session_id}")
        super().__init__(f"{original.agent_name}__replay", db_path=db_path)
        self._original_session_id = original_session_id
        self._llm_queue = list(store.get_llm_calls(original_session_id))
        self._tool_queue = list(store.get_tool_calls(original_session_id))
        self._llm_index = 0
        self._tool_index = 0

    @property
    def original_session_id(self) -> str:
        return self._original_session_id

    def pop_llm_response(self) -> Optional[LLMCall]:
        """Return the next recorded LLM response in order, or None if exhausted."""
        if self._llm_index >= len(self._llm_queue):
            return None
        result = self._llm_queue[self._llm_index]
        self._llm_index += 1
        return result

    def pop_tool_result(self) -> Optional[ToolCall]:
        """Return the next recorded tool result in order, or None if exhausted."""
        if self._tool_index >= len(self._tool_queue):
            return None
        result = self._tool_queue[self._tool_index]
        self._tool_index += 1
        return result

    def remaining_llm_responses(self) -> int:
        return len(self._llm_queue) - self._llm_index

    def remaining_tool_results(self) -> int:
        return len(self._tool_queue) - self._tool_index


def _print_truncated(label: str, text: str, max_len: int) -> None:
    if len(text) > max_len:
        text = text[:max_len] + f"... [{len(text) - max_len} more chars]"
    print(f"  {label}: {text}")
