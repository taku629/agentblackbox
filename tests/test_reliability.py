"""Concurrency, crash safety, privacy, and deterministic replay regressions."""
from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from agentblackbox import BlackBox
from agentblackbox.models import LLMCall, Session, ToolCall
from agentblackbox.storage import SQLiteStorage


def test_concurrent_writers_have_no_lost_events(tmp_path):
    db = tmp_path / "writers.db"
    workers, per_worker = 8, 150

    def write(worker: int) -> None:
        with BlackBox.session(f"worker-{worker}", db_path=db) as bb:
            for number in range(per_worker):
                bb.record_tool_call("work", {"worker": worker, "n": number}, number, 0)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(write, range(workers)))

    store = SQLiteStorage(db)
    sessions = store.list_sessions(limit=workers + 1)
    assert len(sessions) == workers
    assert all(session.status == "success" for session in sessions)
    assert sum(len(store.get_tool_calls(session.session_id)) for session in sessions) == workers * per_worker


def test_async_decorator_preserves_isolated_contexts(tmp_path):
    db = tmp_path / "async.db"
    observed: list[tuple[str, str]] = []

    @BlackBox.record("async-agent", db_path=db)
    async def task(name: str) -> None:
        await asyncio.sleep(0)
        current = BlackBox.current()
        assert current is not None
        observed.append((name, current.session_id))
        current.record_tool_call("task", {"name": name}, "ok", 0)

    asyncio.run(_gather_tasks(task))
    assert len({session_id for _, session_id in observed}) == 12
    assert all(session.status == "success" for session in SQLiteStorage(db).list_sessions(limit=20))


async def _gather_tasks(task) -> None:
    await asyncio.gather(*(task(str(i)) for i in range(12)))


def test_equal_timestamps_replay_in_insertion_order(tmp_path, monkeypatch):
    db = tmp_path / "order.db"
    monkeypatch.setattr(time, "time_ns", lambda: 123456789)
    with BlackBox.session("ordered", db_path=db) as bb:
        first = bb.record_tool_call("first", {}, None, 0)
        second = bb.record_llm_call("model", "second", "", 0, 0, 0)
        third = bb.record_tool_call("third", {}, None, 0)
    assert [event.id for _, _, event in bb.iter_events()] == [first.id, second.id, third.id]


def test_duplicate_event_id_is_idempotent_across_event_types(tmp_path):
    store = SQLiteStorage(tmp_path / "duplicates.db")
    store.create_session(Session("s", "agent", 1))
    llm = LLMCall("same", "s", 2, "m", 0, 0, "", "", 0, 0)
    store.insert_llm_call(llm)
    store.insert_llm_call(llm)
    store.insert_tool_call(ToolCall("same", "s", 3, "must-not-appear", {}, None, 0))
    assert len(store.get_events("s")) == 1
    assert store.get_tool_calls("s") == []


def test_transaction_rolls_back_all_events_on_interruption(tmp_path):
    store = SQLiteStorage(tmp_path / "atomic.db")
    store.create_session(Session("s", "agent", 1))
    try:
        with store.transaction():
            store.insert_tool_call(ToolCall("one", "s", 2, "x", {}, None, 0))
            raise KeyboardInterrupt
    except KeyboardInterrupt:
        pass
    assert store.get_events("s") == []


def test_process_termination_keeps_committed_data_and_discards_transaction(tmp_path):
    db = tmp_path / "kill.db"
    code = """
import os, sys
from pathlib import Path
from agentblackbox.models import Session, ToolCall
from agentblackbox.storage import SQLiteStorage
s=SQLiteStorage(Path(sys.argv[1])); s.create_session(Session('s','agent',1))
s.insert_tool_call(ToolCall('committed','s',2,'ok',{},None,0))
with s.transaction():
 s.insert_tool_call(ToolCall('partial','s',3,'bad',{},None,0)); os._exit(9)
"""
    result = subprocess.run([sys.executable, "-c", code, str(db)], check=False)
    assert result.returncode == 9
    store = SQLiteStorage(db)
    assert [event.id for _, _, event in store.get_events("s")] == ["committed"]
    assert store._conn().execute("PRAGMA integrity_check").fetchone()[0] == "ok"


def test_nested_secrets_and_exception_credentials_are_redacted(tmp_path):
    db = tmp_path / "privacy.db"
    synthetic = "sk_FAKE_CREDENTIAL_123456789"
    with BlackBox.session("private", db_path=db) as bb:
        circular: dict = {"authorization": f"Bearer {synthetic}"}
        circular["self"] = circular
        bb.record_tool_call("request", {"headers": circular}, {"api_key": synthetic}, 0,
                            error=f"Bearer {synthetic}", metadata={"password": synthetic})
        bb.record_error(RuntimeError(f"request failed with {synthetic}"))
    exported = BlackBox.export_json(bb.session_id, db_path=db)
    assert synthetic not in exported
    assert "[REDACTED]" in exported
    assert "[CYCLE]" in exported


def test_partial_session_and_repeated_replay_are_stable(tmp_path, capsys):
    db = tmp_path / "partial.db"
    store = SQLiteStorage(db)
    store.create_session(Session("partial", "agent", 1))
    store.insert_tool_call(ToolCall("event", "partial", 2, "tool", {}, "ok", 0))
    BlackBox.replay("partial", db_path=db)
    first = capsys.readouterr().out
    BlackBox.replay("partial", db_path=db)
    assert capsys.readouterr().out == first
    assert "running" in first and "tool" in first


def test_corrupted_recording_fails_explicitly(tmp_path):
    db = tmp_path / "corrupt.db"
    store = SQLiteStorage(db)
    store.create_session(Session("s", "agent", 1))
    store._conn().execute("UPDATE sessions SET metadata='not-json' WHERE session_id='s'")
    store._conn().commit()
    try:
        store.get_session("s")
    except json.JSONDecodeError:
        pass
    else:
        raise AssertionError("corrupt JSON must not be silently replayed")
