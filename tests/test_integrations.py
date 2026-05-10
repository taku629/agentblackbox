"""Tests for integrations: Anthropic SDK, LangChain, OpenAI Agents."""
from __future__ import annotations

import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentblackbox import BlackBox
from agentblackbox.integrations.langchain_cb import BlackBoxCallbackHandler


@pytest.fixture
def tmp_db(tmp_path):
    return tmp_path / "integ_test.db"


# ── LangChain callback handler ────────────────────────────────────────────────


class TestLangChainCallbackHandler:
    def test_on_llm_end_records_call(self, tmp_db):
        handler = BlackBoxCallbackHandler()
        run_id = uuid.uuid4()

        with BlackBox.session("lc_agent", db_path=tmp_db) as bb:
            handler.on_llm_start({}, ["Hello world"], run_id=run_id)
            time.sleep(0.001)

            # Simulate a LangChain LLMResult
            generation = SimpleNamespace(text="Hi there")
            response = SimpleNamespace(
                generations=[[generation]],
                llm_output={
                    "model_name": "gpt-4o",
                    "token_usage": {"prompt_tokens": 10, "completion_tokens": 5},
                },
            )
            handler.on_llm_end(response, run_id=run_id)

        sessions = BlackBox.list_sessions(db_path=tmp_db)
        from agentblackbox.storage import SQLiteStorage
        calls = SQLiteStorage(tmp_db).get_llm_calls(sessions[0].session_id)
        assert len(calls) == 1
        assert calls[0].model == "gpt-4o"
        assert calls[0].input_tokens == 10
        assert calls[0].output_text == "Hi there"

    def test_on_llm_end_unknown_run_id_noop(self, tmp_db):
        handler = BlackBoxCallbackHandler()
        response = SimpleNamespace(generations=[], llm_output=None)
        with BlackBox.session("noop", db_path=tmp_db):
            handler.on_llm_end(response, run_id="never-started")
        sessions = BlackBox.list_sessions(db_path=tmp_db)
        from agentblackbox.storage import SQLiteStorage
        calls = SQLiteStorage(tmp_db).get_llm_calls(sessions[0].session_id)
        assert calls == []

    def test_on_llm_end_no_active_session_noop(self):
        handler = BlackBoxCallbackHandler()
        run_id = uuid.uuid4()
        handler.on_llm_start({}, ["prompt"], run_id=run_id)
        response = SimpleNamespace(generations=[], llm_output=None)
        handler.on_llm_end(response, run_id=run_id)  # no BlackBox.current() — should not raise

    def test_on_llm_error_records_error(self, tmp_db):
        handler = BlackBoxCallbackHandler()
        run_id = uuid.uuid4()

        with BlackBox.session("lc_err", db_path=tmp_db) as bb:
            handler.on_llm_start({}, ["prompt"], run_id=run_id)
            handler.on_llm_error(ValueError("API timeout"), run_id=run_id)

        from agentblackbox.storage import SQLiteStorage
        sessions = BlackBox.list_sessions(db_path=tmp_db)
        errors = SQLiteStorage(tmp_db).get_errors(sessions[0].session_id)
        assert len(errors) == 1
        assert errors[0].error_type == "ValueError"

    def test_on_llm_error_clears_start_state(self):
        handler = BlackBoxCallbackHandler()
        run_id = uuid.uuid4()
        handler.on_llm_start({}, ["x"], run_id=run_id)
        handler.on_llm_error(RuntimeError("oops"), run_id=run_id)
        assert str(run_id) not in handler._llm_starts

    def test_on_tool_end_records_tool(self, tmp_db):
        handler = BlackBoxCallbackHandler()
        run_id = uuid.uuid4()

        with BlackBox.session("lc_tool", db_path=tmp_db):
            handler.on_tool_start({"name": "web_search"}, "python tutorial", run_id=run_id)
            handler.on_tool_end("Found 10 results", run_id=run_id)

        sessions = BlackBox.list_sessions(db_path=tmp_db)
        from agentblackbox.storage import SQLiteStorage
        calls = SQLiteStorage(tmp_db).get_tool_calls(sessions[0].session_id)
        assert len(calls) == 1
        assert calls[0].tool_name == "web_search"
        assert calls[0].result == "Found 10 results"

    def test_on_tool_error_records_failed_tool(self, tmp_db):
        handler = BlackBoxCallbackHandler()
        run_id = uuid.uuid4()

        with BlackBox.session("lc_tool_err", db_path=tmp_db):
            handler.on_tool_start({"name": "broken_tool"}, "input", run_id=run_id)
            handler.on_tool_error(ConnectionError("network error"), run_id=run_id)

        sessions = BlackBox.list_sessions(db_path=tmp_db)
        from agentblackbox.storage import SQLiteStorage
        calls = SQLiteStorage(tmp_db).get_tool_calls(sessions[0].session_id)
        assert len(calls) == 1
        assert calls[0].error is not None
        assert "network error" in calls[0].error

    def test_on_tool_end_unknown_run_id_noop(self, tmp_db):
        handler = BlackBoxCallbackHandler()
        with BlackBox.session("noop_tool", db_path=tmp_db):
            handler.on_tool_end("output", run_id="never-started")  # should not raise

    def test_on_tool_error_unknown_run_id_noop(self, tmp_db):
        handler = BlackBoxCallbackHandler()
        with BlackBox.session("noop_tool_err", db_path=tmp_db):
            handler.on_tool_error(RuntimeError("x"), run_id="never-started")

    def test_noop_chain_callbacks(self):
        handler = BlackBoxCallbackHandler()
        handler.on_chain_start({}, {})
        handler.on_chain_end({})
        handler.on_chain_error(RuntimeError())
        handler.on_text("text")
        handler.on_agent_action(MagicMock())
        handler.on_agent_finish(MagicMock())


# ── Anthropic SDK integration ─────────────────────────────────────────────────


class TestAnthropicIntegration:
    def _make_anthropic_mock(self):
        """Build a minimal mock of the anthropic module structure."""
        usage = SimpleNamespace(input_tokens=20, output_tokens=10)
        content_block = SimpleNamespace(text="Hello from Claude")
        response = SimpleNamespace(usage=usage, content=[content_block], model="claude-sonnet-4-6")

        Messages = MagicMock()
        Messages.create = MagicMock(return_value=response)

        AsyncMessages = MagicMock()
        AsyncMessages.create = AsyncMock(return_value=response)

        messages_mod = SimpleNamespace(Messages=Messages, AsyncMessages=AsyncMessages)
        resources_mod = SimpleNamespace(messages=messages_mod)

        anthropic_mod = MagicMock()
        anthropic_mod.resources = resources_mod
        return anthropic_mod, response

    def test_extract_input_text_string_content(self):
        from agentblackbox.integrations.anthropic_sdk import _extract_input_text
        messages = [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]
        result = _extract_input_text(messages)
        assert "Hello" in result
        assert "Hi" in result

    def test_extract_input_text_list_content(self):
        from agentblackbox.integrations.anthropic_sdk import _extract_input_text
        messages = [{"role": "user", "content": [{"type": "text", "text": "Search for X"}]}]
        result = _extract_input_text(messages)
        assert "Search for X" in result

    def test_extract_output_text(self):
        from agentblackbox.integrations.anthropic_sdk import _extract_output_text
        block = SimpleNamespace(text="The answer is 42")
        response = SimpleNamespace(content=[block])
        assert _extract_output_text(response) == "The answer is 42"

    def test_extract_output_text_no_content(self):
        from agentblackbox.integrations.anthropic_sdk import _extract_output_text
        response = SimpleNamespace()
        assert _extract_output_text(response) == ""

    def test_extract_output_text_dict_block(self):
        from agentblackbox.integrations.anthropic_sdk import _extract_output_text
        response = SimpleNamespace(content=[{"type": "text", "text": "dict text"}])
        assert _extract_output_text(response) == "dict text"

    def test_patch_anthropic_records_call(self, tmp_db):
        from agentblackbox.integrations import anthropic_sdk as sdk_module

        anthropic_mod, response = self._make_anthropic_mock()
        sdk_module._patched_sync = False
        sdk_module._patched_async = False

        with patch.dict("sys.modules", {"anthropic": anthropic_mod}):
            sdk_module.patch_anthropic()

        Messages = anthropic_mod.resources.messages.Messages
        patched_create = Messages.create

        with BlackBox.session("anthropic_agent", db_path=tmp_db) as bb:
            fake_self = MagicMock()
            patched_create(
                fake_self,
                model="claude-sonnet-4-6",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=100,
            )

        sessions = BlackBox.list_sessions(db_path=tmp_db)
        from agentblackbox.storage import SQLiteStorage
        calls = SQLiteStorage(tmp_db).get_llm_calls(sessions[0].session_id)
        assert len(calls) == 1
        assert calls[0].model == "claude-sonnet-4-6"
        assert calls[0].input_tokens == 20
        sdk_module._patched_sync = False
        sdk_module._patched_async = False

    def test_patch_anthropic_idempotent(self, capsys):
        from agentblackbox.integrations import anthropic_sdk as sdk_module
        sdk_module._patched_sync = True
        sdk_module._patched_async = True
        sdk_module.patch_anthropic()  # should return early
        sdk_module._patched_sync = False
        sdk_module._patched_async = False

    def test_patch_anthropic_import_error(self):
        from agentblackbox.integrations import anthropic_sdk as sdk_module
        sdk_module._patched_sync = False
        sdk_module._patched_async = False
        with patch.dict("sys.modules", {"anthropic": None}):
            with pytest.raises(ImportError, match="anthropic"):
                sdk_module.patch_anthropic()

    def test_patch_sync_missing_attribute(self):
        from agentblackbox.integrations.anthropic_sdk import _patch_sync
        empty_mod = SimpleNamespace()  # no .resources
        _patch_sync(empty_mod)  # should not raise

    def test_patch_async_missing_attribute(self):
        from agentblackbox.integrations.anthropic_sdk import _patch_async
        empty_mod = SimpleNamespace()
        _patch_async(empty_mod)  # should not raise

    def test_patched_create_no_active_session(self, tmp_db):
        """Patched create should still work if no BlackBox session is active."""
        from agentblackbox.integrations import anthropic_sdk as sdk_module

        anthropic_mod, response = self._make_anthropic_mock()
        sdk_module._patched_sync = False
        sdk_module._patched_async = False

        with patch.dict("sys.modules", {"anthropic": anthropic_mod}):
            sdk_module.patch_anthropic()

        Messages = anthropic_mod.resources.messages.Messages
        patched_create = Messages.create
        fake_self = MagicMock()
        # Call outside any session — should not raise
        result = patched_create(
            fake_self,
            model="claude-sonnet-4-6",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=50,
        )
        assert result is response
        sdk_module._patched_sync = False
        sdk_module._patched_async = False


# ── OpenAI Agents integration ─────────────────────────────────────────────────


class TestOpenAIAgentsIntegration:
    def test_patch_openai_agents_import_error(self):
        from agentblackbox.integrations import openai_agents as oa_module
        oa_module._patched = False
        with patch.dict("sys.modules", {"agents": None}):
            with pytest.raises(ImportError, match="openai-agents"):
                from agentblackbox.integrations.openai_agents import patch_openai_agents
                patch_openai_agents()
        oa_module._patched = False

    def test_patch_openai_agents_idempotent(self):
        from agentblackbox.integrations import openai_agents as oa_module
        oa_module._patched = True
        from agentblackbox.integrations.openai_agents import patch_openai_agents
        patch_openai_agents()  # should return early without error
        oa_module._patched = False

    def test_async_blackbox_wrapper(self, tmp_db):
        import asyncio
        from agentblackbox.integrations.openai_agents import _AsyncBlackBoxWrapper

        async def run():
            with patch("agentblackbox.recorder._default_storage", None):
                from agentblackbox.storage import SQLiteStorage
                storage = SQLiteStorage(tmp_db)
                with patch("agentblackbox.recorder._get_storage", return_value=storage):
                    async with _AsyncBlackBoxWrapper("test_async_agent"):
                        bb = BlackBox.current()
                        assert bb is not None
                        assert bb.agent_name == "test_async_agent"

        asyncio.run(run())

    def test_async_blackbox_wrapper_records_error(self, tmp_db):
        import asyncio
        from agentblackbox.integrations.openai_agents import _AsyncBlackBoxWrapper

        async def run():
            with patch("agentblackbox.recorder._default_storage", None):
                from agentblackbox.storage import SQLiteStorage
                storage = SQLiteStorage(tmp_db)
                with patch("agentblackbox.recorder._get_storage", return_value=storage):
                    with pytest.raises(ValueError):
                        async with _AsyncBlackBoxWrapper("error_async_agent"):
                            raise ValueError("async failure")

        asyncio.run(run())
        sessions = BlackBox.list_sessions(db_path=tmp_db)
        assert sessions[0].status == "error"
