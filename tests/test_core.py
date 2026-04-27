"""Core test suite for agentblackbox — 39+ tests."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from agentblackbox import BlackBox, calculate_cost, list_supported_models
from agentblackbox.storage import SQLiteStorage
from agentblackbox.models import Session, LLMCall, ToolCall, ErrorRecord


# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def reset_blackbox(tmp_path):
    """Each test gets a fresh isolated DB via BlackBox.configure()."""
    BlackBox.configure(db_path=str(tmp_path / "test.db"), masking=False)
    yield
    BlackBox.configure(masking=False)  # reset masker


@pytest.fixture
def storage(tmp_path):
    return SQLiteStorage(tmp_path / "storage.db")


# ── models ────────────────────────────────────────────────────────────────────


class TestModels:
    def test_llm_call_fields(self):
        c = LLMCall(
            call_id="c1", session_id="s1", timestamp=1.0,
            model="gpt-4o", input_text="hi", output_text="hello",
            input_tokens=10, output_tokens=5, cost_usd=0.001, latency_ms=200.0,
        )
        assert c.call_id == "c1"
        assert c.latency_ms == 200.0
        assert c.cost_usd == 0.001

    def test_tool_call_fields(self):
        c = ToolCall(
            call_id="t1", session_id="s1", timestamp=1.0,
            tool_name="search", args={"q": "test"}, result=["r1"],
            latency_ms=50.0,
        )
        assert c.call_id == "t1"
        assert c.args == {"q": "test"}
        assert c.latency_ms == 50.0

    def test_error_record_fields(self):
        e = ErrorRecord(
            error_id="e1", session_id="s1", timestamp=1.0,
            error_type="ValueError", message="oops", traceback="tb...",
        )
        assert e.error_id == "e1"
        assert e.error_type == "ValueError"

    def test_session_defaults(self):
        s = Session(session_id="s1", agent_name="ag", start_time=time.time())
        assert s.status == "running"
        assert s.end_time is None
        assert s.llm_calls == []
        assert s.tool_calls == []
        assert s.errors == []
        assert s.total_cost_usd == 0.0
        assert s.metadata == {}

    def test_session_with_embedded_calls(self):
        call = LLMCall("c1", "s1", 1.0, "gpt-4o", "in", "out", 10, 5, 0.001, 100.0)
        s = Session(
            session_id="s1", agent_name="ag", start_time=1.0,
            llm_calls=[call], total_cost_usd=0.001,
        )
        assert len(s.llm_calls) == 1
        assert s.llm_calls[0].model == "gpt-4o"


# ── cost ──────────────────────────────────────────────────────────────────────


class TestCost:
    def test_gpt4o_price(self):
        # spec: input=5.0, output=15.0 per 1M
        cost = calculate_cost("gpt-4o", 1_000_000, 0)
        assert cost == pytest.approx(5.0)
        cost = calculate_cost("gpt-4o", 0, 1_000_000)
        assert cost == pytest.approx(15.0)

    def test_gpt4o_mini(self):
        cost = calculate_cost("gpt-4o-mini", 1_000_000, 1_000_000)
        assert cost == pytest.approx(0.15 + 0.60)

    def test_gpt4_turbo(self):
        cost = calculate_cost("gpt-4-turbo", 1_000_000, 0)
        assert cost == pytest.approx(10.0)

    def test_gpt35_turbo(self):
        cost = calculate_cost("gpt-3.5-turbo", 1_000_000, 1_000_000)
        assert cost == pytest.approx(0.50 + 1.50)

    def test_o1(self):
        cost = calculate_cost("o1", 1_000_000, 0)
        assert cost == pytest.approx(15.0)

    def test_o3_mini(self):
        cost = calculate_cost("o3-mini", 0, 1_000_000)
        assert cost == pytest.approx(4.40)

    def test_claude_35_sonnet(self):
        cost = calculate_cost("claude-3-5-sonnet-20241022", 1_000_000, 0)
        assert cost == pytest.approx(3.0)

    def test_claude_35_haiku(self):
        cost = calculate_cost("claude-3-5-haiku-20241022", 0, 1_000_000)
        assert cost == pytest.approx(4.0)

    def test_claude_3_opus(self):
        cost = calculate_cost("claude-3-opus-20240229", 1_000_000, 0)
        assert cost == pytest.approx(15.0)

    def test_claude_3_haiku(self):
        cost = calculate_cost("claude-3-haiku-20240307", 1_000_000, 0)
        assert cost == pytest.approx(0.25)

    def test_claude_opus_45(self):
        cost = calculate_cost("claude-opus-4-5", 1_000_000, 0)
        assert cost == pytest.approx(15.0)

    def test_gemini_15_pro(self):
        cost = calculate_cost("gemini-1.5-pro", 1_000_000, 0)
        assert cost == pytest.approx(3.5)

    def test_gemini_15_flash(self):
        cost = calculate_cost("gemini-1.5-flash", 0, 1_000_000)
        assert cost == pytest.approx(0.30)

    def test_gemini_20_flash(self):
        cost = calculate_cost("gemini-2.0-flash", 1_000_000, 0)
        assert cost == pytest.approx(0.10)

    def test_llama_31_70b(self):
        cost = calculate_cost("llama-3.1-70b", 1_000_000, 1_000_000)
        assert cost == pytest.approx(0.88 + 0.88)

    def test_mistral_large(self):
        cost = calculate_cost("mistral-large", 1_000_000, 0)
        assert cost == pytest.approx(3.0)

    def test_deepseek_chat(self):
        cost = calculate_cost("deepseek-chat", 0, 1_000_000)
        assert cost == pytest.approx(1.10)

    def test_grok_2(self):
        cost = calculate_cost("grok-2", 1_000_000, 0)
        assert cost == pytest.approx(2.0)

    def test_unknown_model_returns_zero(self):
        assert calculate_cost("unknown-model-xyz", 100, 100) == 0.0

    def test_zero_tokens(self):
        assert calculate_cost("gpt-4o", 0, 0) == 0.0

    def test_case_insensitive(self):
        assert calculate_cost("GPT-4O", 1_000_000, 0) == pytest.approx(5.0)

    def test_list_supported_models_includes_required(self):
        models = list_supported_models()
        assert "gpt-4o" in models
        assert "claude-3-5-sonnet-20241022" in models
        assert "gemini-1.5-pro" in models
        assert len(models) >= 18


# ── storage ───────────────────────────────────────────────────────────────────


class TestStorage:
    def test_save_and_get_session(self, storage):
        s = Session(session_id="s1", agent_name="agent_a", start_time=time.time())
        storage.save_session(s)
        fetched = storage.get_session("s1")
        assert fetched is not None
        assert fetched.agent_name == "agent_a"
        assert fetched.status == "running"

    def test_get_nonexistent_session(self, storage):
        assert storage.get_session("nope") is None

    def test_save_updates_session(self, storage):
        s = Session(session_id="s2", agent_name="agent_b", start_time=time.time())
        storage.save_session(s)
        s.end_time = time.time()
        s.status = "success"
        s.total_cost_usd = 0.001
        storage.save_session(s)
        fetched = storage.get_session("s2")
        assert fetched.status == "success"
        assert fetched.total_cost_usd == pytest.approx(0.001)

    def test_list_sessions_empty(self, storage):
        assert storage.list_sessions() == []

    def test_list_sessions_filter_by_agent(self, storage):
        storage.save_session(Session("s3", "agent_x", time.time()))
        storage.save_session(Session("s4", "agent_y", time.time()))
        result = storage.list_sessions(agent_name="agent_x")
        assert len(result) == 1
        assert result[0].agent_name == "agent_x"

    def test_list_sessions_filter_by_status(self, storage):
        s = Session("s5", "ag", time.time())
        storage.save_session(s)
        s.status = "success"
        s.end_time = time.time()
        storage.save_session(s)
        assert len(storage.list_sessions(status="success")) == 1
        assert len(storage.list_sessions(status="running")) == 0

    def test_list_sessions_limit(self, storage):
        for i in range(10):
            storage.save_session(Session(f"s{i}", "ag", time.time()))
        assert len(storage.list_sessions(limit=5)) == 5

    def test_save_with_embedded_llm_call(self, storage):
        s = Session("s6", "ag", time.time())
        call = LLMCall("c1", "s6", time.time(), "gpt-4o", "hi", "hello", 10, 5, 0.001, 100.0)
        s.llm_calls.append(call)
        storage.save_session(s)
        fetched = storage.get_session("s6")
        assert len(fetched.llm_calls) == 1
        assert fetched.llm_calls[0].model == "gpt-4o"
        assert fetched.llm_calls[0].call_id == "c1"
        assert fetched.llm_calls[0].latency_ms == pytest.approx(100.0)

    def test_save_with_embedded_tool_call(self, storage):
        s = Session("s7", "ag", time.time())
        call = ToolCall("t1", "s7", time.time(), "search", {"q": "test"}, ["r1"], 50.0)
        s.tool_calls.append(call)
        storage.save_session(s)
        fetched = storage.get_session("s7")
        assert len(fetched.tool_calls) == 1
        assert fetched.tool_calls[0].tool_name == "search"
        assert fetched.tool_calls[0].args == {"q": "test"}

    def test_save_with_embedded_error(self, storage):
        s = Session("s8", "ag", time.time())
        err = ErrorRecord("e1", "s8", time.time(), "ValueError", "oops", "tb...")
        s.errors.append(err)
        storage.save_session(s)
        fetched = storage.get_session("s8")
        assert len(fetched.errors) == 1
        assert fetched.errors[0].error_id == "e1"
        assert fetched.errors[0].error_type == "ValueError"

    def test_delete_session(self, storage):
        s = Session("s9", "ag", time.time())
        storage.save_session(s)
        assert storage.get_session("s9") is not None
        storage.delete_session("s9")
        assert storage.get_session("s9") is None

    def test_delete_cascades_child_records(self, storage):
        s = Session("s10", "ag", time.time())
        s.llm_calls.append(LLMCall("c1", "s10", time.time(), "gpt-4o", "", "", 0, 0, 0.0, 0.0))
        storage.save_session(s)
        storage.delete_session("s10")
        assert storage.get_session("s10") is None
        # verify child records gone
        fetched = storage.get_session("s10")
        assert fetched is None

    def test_metadata_roundtrip(self, storage):
        meta = {"user": "alice", "tags": ["test", "demo"]}
        s = Session("s11", "ag", time.time(), metadata=meta)
        storage.save_session(s)
        fetched = storage.get_session("s11")
        assert fetched.metadata == meta

    def test_tool_call_none_result(self, storage):
        s = Session("s12", "ag", time.time())
        call = ToolCall("t2", "s12", time.time(), "tool", {}, None, 10.0)
        s.tool_calls.append(call)
        storage.save_session(s)
        fetched = storage.get_session("s12")
        assert fetched.tool_calls[0].result is None


# ── blackbox recorder ─────────────────────────────────────────────────────────


class TestBlackBoxContextManager:
    def test_basic_context_manager(self):
        with BlackBox.session("test_cm") as bb:
            assert bb is not None
            assert BlackBox.current() is bb
        assert BlackBox.current() is None
        sessions = BlackBox.list_sessions()
        assert len(sessions) == 1
        assert sessions[0].status == "success"
        assert sessions[0].agent_name == "test_cm"

    def test_context_manager_records_error(self):
        with pytest.raises(RuntimeError):
            with BlackBox.session("error_agent"):
                raise RuntimeError("boom")
        sessions = BlackBox.list_sessions()
        assert sessions[0].status == "error"

    def test_record_llm_call(self):
        with BlackBox.session("llm_agent") as bb:
            call = bb.record_llm_call(
                model="gpt-4o",
                input_text="hi",
                output_text="hello",
                input_tokens=10,
                output_tokens=5,
                latency_ms=200.0,
            )
        assert isinstance(call, LLMCall)
        assert call.model == "gpt-4o"
        assert call.latency_ms == 200.0
        sessions = BlackBox.list_sessions()
        assert sessions[0].total_cost_usd > 0

    def test_record_tool_call(self):
        with BlackBox.session("tool_agent") as bb:
            call = bb.record_tool_call(
                tool_name="calculator",
                args={"expr": "2+2"},
                result=4,
                latency_ms=5.0,
            )
        assert isinstance(call, ToolCall)
        assert call.tool_name == "calculator"
        assert call.args == {"expr": "2+2"}
        assert call.latency_ms == 5.0

    def test_record_error_manually(self):
        with BlackBox.session("manual_err") as bb:
            try:
                raise ValueError("manual error")
            except ValueError as e:
                err = bb.record_error(e)
        assert isinstance(err, ErrorRecord)
        assert err.error_type == "ValueError"
        assert "manual error" in err.message

    def test_cost_accumulates_in_session(self):
        with BlackBox.session("cost_agent") as bb:
            bb.record_llm_call("gpt-4o", "a", "b", 1_000_000, 0, 100.0)
            bb.record_llm_call("gpt-4o", "c", "d", 1_000_000, 0, 100.0)
        sessions = BlackBox.list_sessions()
        assert sessions[0].total_cost_usd == pytest.approx(10.0)

    def test_session_persists_all_data(self):
        with BlackBox.session("full_agent") as bb:
            bb.record_llm_call("gpt-4o", "q", "a", 100, 50, 300.0)
            bb.record_tool_call("search", {"q": "test"}, ["r1"], 100.0)
        from agentblackbox.recorder import _get_storage
        sessions = BlackBox.list_sessions()
        full = _get_storage().get_session(sessions[0].session_id)
        assert len(full.llm_calls) == 1
        assert len(full.tool_calls) == 1


class TestBlackBoxDecorator:
    def test_decorator_records_session(self):
        @BlackBox.record(agent_name="deco_agent")
        def my_func():
            return "ok"

        result = my_func()
        assert result == "ok"
        sessions = BlackBox.list_sessions()
        assert sessions[0].agent_name == "deco_agent"
        assert sessions[0].status == "success"

    def test_decorator_with_exception(self):
        @BlackBox.record(agent_name="deco_fail")
        def bad_func():
            raise TypeError("bad")

        with pytest.raises(TypeError):
            bad_func()
        sessions = BlackBox.list_sessions()
        assert sessions[0].status == "error"

    def test_decorator_current_session_accessible(self):
        captured = []

        @BlackBox.record(agent_name="deco_current")
        def my_func():
            captured.append(BlackBox.current())

        my_func()
        assert len(captured) == 1
        assert captured[0] is not None
        assert captured[0].agent_name == "deco_current"

    def test_decorator_preserves_function_name(self):
        @BlackBox.record(agent_name="x")
        def my_named_function():
            pass

        assert my_named_function.__name__ == "my_named_function"

    def test_decorator_records_error_in_session(self):
        @BlackBox.record(agent_name="err_deco")
        def fail():
            raise RuntimeError("oops")

        with pytest.raises(RuntimeError):
            fail()
        from agentblackbox.recorder import _get_storage
        sessions = BlackBox.list_sessions()
        full = _get_storage().get_session(sessions[0].session_id)
        assert len(full.errors) == 1
        assert full.errors[0].error_type == "RuntimeError"


class TestBlackBoxExport:
    def test_export_json_writes_file(self, tmp_path):
        out = str(tmp_path / "out.json")
        with BlackBox.session("export_agent") as bb:
            bb.record_llm_call("gpt-4o", "in", "out", 100, 50, 100.0)
            bb.record_tool_call("t", {"k": "v"}, "r", 10.0)
        bb.export_json(out)
        with open(out) as f:
            data = json.load(f)
        assert data["session"]["agent_name"] == "export_agent"
        assert len(data["llm_calls"]) == 1
        assert len(data["tool_calls"]) == 1
        assert data["llm_calls"][0]["model"] == "gpt-4o"
        assert data["llm_calls"][0]["latency_ms"] == pytest.approx(100.0)

    def test_export_json_tool_call_args(self, tmp_path):
        out = str(tmp_path / "out2.json")
        with BlackBox.session("exp2") as bb:
            bb.record_tool_call("search", {"query": "ai"}, ["r1", "r2"], 50.0)
        bb.export_json(out)
        with open(out) as f:
            data = json.load(f)
        assert data["tool_calls"][0]["args"] == {"query": "ai"}

    def test_replay_prints_output(self, capsys):
        with BlackBox.session("replay_agent") as bb:
            bb.record_llm_call("gpt-4o-mini", "prompt", "response", 50, 25, 300.0)
        bb.replay()
        captured = capsys.readouterr()
        assert "replay_agent" in captured.out
        assert "gpt-4o-mini" in captured.out

    def test_replay_shows_tool_calls(self, capsys):
        with BlackBox.session("replay_tool") as bb:
            bb.record_tool_call("my_tool", {"k": "v"}, "result", 40.0)
        bb.replay()
        captured = capsys.readouterr()
        assert "my_tool" in captured.out

    def test_replay_shows_errors(self, capsys):
        with pytest.raises(ValueError):
            with BlackBox.session("replay_err") as bb:
                raise ValueError("test error")
        bb.replay()
        captured = capsys.readouterr()
        assert "ValueError" in captured.out


class TestBlackBoxListSessions:
    def test_list_sessions_empty(self):
        assert BlackBox.list_sessions() == []

    def test_list_sessions_limit(self):
        for i in range(5):
            with BlackBox.session(f"agent_{i}"):
                pass
        sessions = BlackBox.list_sessions(limit=3)
        assert len(sessions) == 3

    def test_list_sessions_by_agent(self):
        with BlackBox.session("alpha"):
            pass
        with BlackBox.session("beta"):
            pass
        sessions = BlackBox.list_sessions(agent_name="alpha")
        assert len(sessions) == 1
        assert sessions[0].agent_name == "alpha"

    def test_list_sessions_by_status(self):
        with BlackBox.session("ok"):
            pass
        with pytest.raises(RuntimeError):
            with BlackBox.session("fail"):
                raise RuntimeError("x")
        ok_sessions = BlackBox.list_sessions(status="success")
        err_sessions = BlackBox.list_sessions(status="error")
        assert len(ok_sessions) == 1
        assert len(err_sessions) == 1


class TestCurrentContext:
    def test_current_none_outside_session(self):
        assert BlackBox.current() is None

    def test_current_not_leaked_after_session(self):
        with BlackBox.session("leak_test"):
            inside = BlackBox.current()
        assert inside is not None
        assert BlackBox.current() is None

    def test_nested_sessions_contextvar(self):
        with BlackBox.session("outer") as outer:
            with BlackBox.session("inner") as inner:
                assert BlackBox.current() is inner
            assert BlackBox.current() is outer


class TestBlackBoxConfigure:
    def test_configure_sets_db(self, tmp_path):
        db = str(tmp_path / "custom.db")
        BlackBox.configure(db_path=db)
        with BlackBox.session("cfg_agent"):
            pass
        from pathlib import Path
        assert Path(db).exists()

    def test_configure_masking_enabled(self, tmp_path):
        BlackBox.configure(db_path=str(tmp_path / "test.db"), masking=True)
        with BlackBox.session("masked") as bb:
            call = bb.record_llm_call(
                model="gpt-4o",
                input_text="email: user@example.com",
                output_text="ok",
            )
        assert "[MASKED_EMAIL]" in call.input_text
        BlackBox.configure(masking=False)

    def test_configure_masking_false_resets(self, tmp_path):
        db = str(tmp_path / "test.db")
        BlackBox.configure(db_path=db, masking=True)
        BlackBox.configure(db_path=db, masking=False)  # keep same db, reset masking
        with BlackBox.session("unmasked") as bb:
            call = bb.record_llm_call(
                model="gpt-4o",
                input_text="email: user@example.com",
                output_text="ok",
            )
        assert "user@example.com" in call.input_text

    def test_default_kwargs(self):
        # record_llm_call with no optional args
        with BlackBox.session("defaults") as bb:
            call = bb.record_llm_call(model="gpt-4o", input_text="x", output_text="y")
        assert call.input_tokens == 0
        assert call.latency_ms == 0.0

    def test_record_tool_call_default_latency(self):
        with BlackBox.session("tool_default") as bb:
            call = bb.record_tool_call("my_tool", {}, None)
        assert call.latency_ms == 0.0
