"""Core test suite for agentblackbox — targets 60%+ coverage."""
from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

import pytest

from agentblackbox import BlackBox, calculate_cost, list_supported_models
from agentblackbox.remote import RemoteStorage
from agentblackbox.storage import SQLiteStorage
from agentblackbox.models import Session, LLMCall, ToolCall, ErrorRecord


# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_db(tmp_path):
    return tmp_path / "test.db"


@pytest.fixture
def storage(tmp_db):
    return SQLiteStorage(tmp_db)


@pytest.fixture
def bb(tmp_db):
    return BlackBox("test_agent", db_path=tmp_db)


# ── cost ──────────────────────────────────────────────────────────────────────


class TestCost:
    def test_known_model(self):
        cost = calculate_cost("gpt-4o", 1_000_000, 1_000_000)
        assert cost == pytest.approx(2.50 + 10.00)

    def test_known_model_case_insensitive(self):
        cost = calculate_cost("GPT-4O", 1_000_000, 0)
        assert cost == pytest.approx(2.50)

    def test_family_fallback_claude_sonnet(self):
        cost = calculate_cost("claude-sonnet-4-99-new", 1_000_000, 0)
        assert cost == pytest.approx(3.00)

    def test_unknown_model_returns_zero(self):
        assert calculate_cost("unknown-model-xyz", 100, 100) == 0.0

    def test_zero_tokens(self):
        assert calculate_cost("gpt-4o", 0, 0) == 0.0

    def test_list_supported_models(self):
        models = list_supported_models()
        assert isinstance(models, list)
        assert "gpt-4o" in models
        assert "claude-3-5-sonnet-20241022" in models

    def test_openai_mini(self):
        cost = calculate_cost("gpt-4o-mini", 0, 1_000_000)
        assert cost == pytest.approx(0.60)


# ── storage ───────────────────────────────────────────────────────────────────


class TestStorage:
    def test_create_and_get_session(self, storage):
        s = Session(
            session_id="s1",
            agent_name="agent_a",
            start_time=time.time_ns(),
            status="running",
        )
        storage.create_session(s)
        fetched = storage.get_session("s1")
        assert fetched is not None
        assert fetched.agent_name == "agent_a"
        assert fetched.status == "running"

    def test_get_nonexistent_session(self, storage):
        assert storage.get_session("nope") is None

    def test_update_session(self, storage):
        s = Session("s2", "agent_b", time.time_ns())
        storage.create_session(s)
        s.end_time = time.time_ns()
        s.status = "success"
        s.total_cost_usd = 0.001
        storage.update_session(s)
        fetched = storage.get_session("s2")
        assert fetched.status == "success"
        assert fetched.total_cost_usd == pytest.approx(0.001)

    def test_list_sessions_empty(self, storage):
        assert storage.list_sessions() == []

    def test_list_sessions_filter_by_agent(self, storage):
        storage.create_session(Session("s3", "agent_x", time.time_ns()))
        storage.create_session(Session("s4", "agent_y", time.time_ns()))
        result = storage.list_sessions(agent_name="agent_x")
        assert len(result) == 1
        assert result[0].agent_name == "agent_x"

    def test_list_sessions_filter_by_status(self, storage):
        s = Session("s5", "ag", time.time_ns())
        storage.create_session(s)
        s.status = "success"
        storage.update_session(s)
        assert len(storage.list_sessions(status="success")) == 1
        assert len(storage.list_sessions(status="running")) == 0

    def test_insert_and_get_llm_call(self, storage):
        storage.create_session(Session("s6", "ag", time.time_ns()))
        call = LLMCall(
            id="l1", session_id="s6", timestamp=time.time_ns(),
            model="gpt-4o", input_tokens=100, output_tokens=50,
            input_text="hello", output_text="world",
            duration_ms=500.0, cost_usd=0.0015,
        )
        storage.insert_llm_call(call)
        calls = storage.get_llm_calls("s6")
        assert len(calls) == 1
        assert calls[0].model == "gpt-4o"
        assert calls[0].input_tokens == 100

    def test_insert_and_get_tool_call(self, storage):
        storage.create_session(Session("s7", "ag", time.time_ns()))
        call = ToolCall(
            id="t1", session_id="s7", timestamp=time.time_ns(),
            tool_name="search", arguments={"q": "test"},
            result={"results": []}, duration_ms=100.0,
        )
        storage.insert_tool_call(call)
        calls = storage.get_tool_calls("s7")
        assert len(calls) == 1
        assert calls[0].tool_name == "search"
        assert calls[0].arguments == {"q": "test"}

    def test_insert_and_get_error(self, storage):
        storage.create_session(Session("s8", "ag", time.time_ns()))
        err = ErrorRecord(
            id="e1", session_id="s8", timestamp=time.time_ns(),
            error_type="ValueError", message="oops",
            traceback="Traceback...\nValueError: oops",
        )
        storage.insert_error(err)
        errors = storage.get_errors("s8")
        assert len(errors) == 1
        assert errors[0].error_type == "ValueError"

    def test_tool_call_with_error_field(self, storage):
        storage.create_session(Session("s9", "ag", time.time_ns()))
        call = ToolCall(
            id="t2", session_id="s9", timestamp=time.time_ns(),
            tool_name="broken_tool", arguments={}, result=None,
            duration_ms=10.0, error="connection refused",
        )
        storage.insert_tool_call(call)
        calls = storage.get_tool_calls("s9")
        assert calls[0].error == "connection refused"

    def test_total_cost_by_model(self, storage):
        storage.create_session(Session("s10", "ag", time.time_ns()))
        storage.insert_llm_call(LLMCall(
            id="l2", session_id="s10", timestamp=time.time_ns(),
            model="gpt-4o", input_tokens=100, output_tokens=50,
            input_text="", output_text="", duration_ms=0.0, cost_usd=0.002,
        ))
        rows = storage.total_cost_by_model()
        assert any(r["model"] == "gpt-4o" for r in rows)

    def test_metadata_roundtrip(self, storage):
        meta = {"user": "alice", "tags": ["test", "demo"]}
        s = Session("s11", "ag", time.time_ns(), metadata=meta)
        storage.create_session(s)
        fetched = storage.get_session("s11")
        assert fetched.metadata == meta


# ── blackbox recorder ─────────────────────────────────────────────────────────


class TestBlackBoxContextManager:
    def test_basic_context_manager(self, tmp_db):
        with BlackBox.session("test_cm", db_path=tmp_db) as bb:
            assert bb is not None
            assert BlackBox.current() is bb

        assert BlackBox.current() is None

        sessions = BlackBox.list_sessions(db_path=tmp_db)
        assert len(sessions) == 1
        assert sessions[0].status == "success"
        assert sessions[0].agent_name == "test_cm"

    def test_context_manager_records_error(self, tmp_db):
        with pytest.raises(RuntimeError):
            with BlackBox.session("error_agent", db_path=tmp_db):
                raise RuntimeError("boom")

        sessions = BlackBox.list_sessions(db_path=tmp_db)
        assert sessions[0].status == "error"

    def test_record_llm_call(self, tmp_db):
        with BlackBox.session("llm_agent", db_path=tmp_db) as bb:
            bb.record_llm_call(
                model="gpt-4o",
                input_text="hi",
                output_text="hello",
                input_tokens=10,
                output_tokens=5,
                duration_ms=200.0,
            )

        sessions = BlackBox.list_sessions(db_path=tmp_db)
        assert sessions[0].total_cost_usd > 0

        store = SQLiteStorage(tmp_db)
        calls = store.get_llm_calls(sessions[0].session_id)
        assert len(calls) == 1
        assert calls[0].model == "gpt-4o"

    def test_record_tool_call(self, tmp_db):
        with BlackBox.session("tool_agent", db_path=tmp_db) as bb:
            bb.record_tool_call(
                tool_name="calculator",
                arguments={"expr": "2+2"},
                result=4,
                duration_ms=5.0,
            )

        store = SQLiteStorage(tmp_db)
        sessions = store.list_sessions()
        calls = store.get_tool_calls(sessions[0].session_id)
        assert len(calls) == 1
        assert calls[0].tool_name == "calculator"

    def test_record_error_manually(self, tmp_db):
        with BlackBox.session("manual_err", db_path=tmp_db) as bb:
            try:
                raise ValueError("manual error")
            except ValueError as e:
                bb.record_error(e)

        store = SQLiteStorage(tmp_db)
        sessions = store.list_sessions()
        errors = store.get_errors(sessions[0].session_id)
        assert len(errors) == 1
        assert errors[0].error_type == "ValueError"
        assert "manual error" in errors[0].message

    def test_cost_accumulates(self, tmp_db):
        with BlackBox.session("cost_agent", db_path=tmp_db) as bb:
            bb.record_llm_call("gpt-4o", "a", "b", 1_000_000, 0, 100.0)
            bb.record_llm_call("gpt-4o", "c", "d", 1_000_000, 0, 100.0)

        sessions = BlackBox.list_sessions(db_path=tmp_db)
        assert sessions[0].total_cost_usd == pytest.approx(5.00)


class TestBlackBoxDecorator:
    def test_decorator_records_session(self, tmp_db):
        @BlackBox.record(agent_name="deco_agent", db_path=tmp_db)
        def my_func():
            return "ok"

        result = my_func()
        assert result == "ok"

        sessions = BlackBox.list_sessions(db_path=tmp_db)
        assert sessions[0].agent_name == "deco_agent"
        assert sessions[0].status == "success"

    def test_decorator_with_exception(self, tmp_db):
        @BlackBox.record(agent_name="deco_fail", db_path=tmp_db)
        def bad_func():
            raise TypeError("bad")

        with pytest.raises(TypeError):
            bad_func()

        sessions = BlackBox.list_sessions(db_path=tmp_db)
        assert sessions[0].status == "error"

    def test_decorator_current_session_accessible(self, tmp_db):
        captured = []

        @BlackBox.record(agent_name="deco_current", db_path=tmp_db)
        def my_func():
            captured.append(BlackBox.current())

        my_func()
        assert len(captured) == 1
        assert captured[0] is not None
        assert captured[0].agent_name == "deco_current"

    def test_decorator_preserves_function_name(self, tmp_db):
        @BlackBox.record(agent_name="x", db_path=tmp_db)
        def my_named_function():
            pass

        assert my_named_function.__name__ == "my_named_function"


class TestBlackBoxExport:
    def test_export_json(self, tmp_db):
        with BlackBox.session("export_agent", db_path=tmp_db) as bb:
            bb.record_llm_call("gpt-4o", "in", "out", 100, 50, 100.0)
            bb.record_tool_call("t", {}, "r", 10.0)

        sessions = BlackBox.list_sessions(db_path=tmp_db)
        exported = bb.export_json(sessions[0].session_id)
        data = json.loads(exported)

        assert data["session"]["agent_name"] == "export_agent"
        assert len(data["llm_calls"]) == 1
        assert len(data["tool_calls"]) == 1
        assert data["llm_calls"][0]["model"] == "gpt-4o"

    def test_export_nonexistent_raises(self, tmp_db):
        bb = BlackBox("x", db_path=tmp_db)
        bb._storage = SQLiteStorage(tmp_db)
        with pytest.raises(ValueError, match="Session not found"):
            bb.export_json("nonexistent-id")

    def test_replay_nonexistent(self, tmp_db, capsys):
        bb = BlackBox("x", db_path=tmp_db)
        bb._storage = SQLiteStorage(tmp_db)
        bb.replay("nonexistent-id")
        captured = capsys.readouterr()
        assert "not found" in captured.out

    def test_replay_prints_output(self, tmp_db, capsys):
        with BlackBox.session("replay_agent", db_path=tmp_db) as bb:
            bb.record_llm_call("gpt-4o-mini", "prompt", "response", 50, 25, 300.0)

        sessions = BlackBox.list_sessions(db_path=tmp_db)
        bb.replay(sessions[0].session_id)
        captured = capsys.readouterr()
        assert "replay_agent" in captured.out
        assert "gpt-4o-mini" in captured.out


class TestBlackBoxListSessions:
    def test_list_sessions_empty(self, tmp_db):
        sessions = BlackBox.list_sessions(db_path=tmp_db)
        assert sessions == []

    def test_list_sessions_limit(self, tmp_db):
        for i in range(5):
            with BlackBox.session(f"agent_{i}", db_path=tmp_db):
                pass
        sessions = BlackBox.list_sessions(limit=3, db_path=tmp_db)
        assert len(sessions) == 3

    def test_list_sessions_by_agent(self, tmp_db):
        with BlackBox.session("alpha", db_path=tmp_db):
            pass
        with BlackBox.session("beta", db_path=tmp_db):
            pass
        sessions = BlackBox.list_sessions(agent_name="alpha", db_path=tmp_db)
        assert len(sessions) == 1
        assert sessions[0].agent_name == "alpha"


class TestCurrentContext:
    def test_current_none_outside_session(self):
        assert BlackBox.current() is None

    def test_current_not_leaked_after_session(self, tmp_db):
        with BlackBox.session("leak_test", db_path=tmp_db):
            inside = BlackBox.current()
        assert inside is not None
        assert BlackBox.current() is None

    def test_nested_sessions_not_supported_gracefully(self, tmp_db):
        with BlackBox.session("outer", db_path=tmp_db) as outer:
            with BlackBox.session("inner", db_path=tmp_db) as inner:
                assert BlackBox.current() is inner
            # after inner exits, current should be outer again
            assert BlackBox.current() is outer


class TestCustomStorage:
    def test_session_accepts_storage_instance(self, tmp_db):
        storage = SQLiteStorage(tmp_db)
        with BlackBox.session("storage_agent", storage=storage) as bb:
            bb.record_tool_call("ping", {}, "pong", 1.0)

        sessions = storage.list_sessions()
        assert len(sessions) == 1
        assert sessions[0].agent_name == "storage_agent"

    def test_decorator_accepts_storage_instance(self, tmp_db):
        storage = SQLiteStorage(tmp_db)

        @BlackBox.record(agent_name="decorator_storage", storage=storage)
        def run():
            return "ok"

        assert run() == "ok"
        sessions = storage.list_sessions()
        assert len(sessions) == 1
        assert sessions[0].agent_name == "decorator_storage"

    def test_remote_storage_inherits_sqlite_storage_contract(self, tmp_db):
        storage = RemoteStorage(
            api_key="abx_test",
            endpoint="http://127.0.0.1:9",
            db_path=tmp_db,
            timeout_s=0.01,
        )

        with BlackBox.session("remote_agent", storage=storage):
            pass

        sessions = storage.list_sessions()
        assert len(sessions) == 1
        assert sessions[0].agent_name == "remote_agent"
