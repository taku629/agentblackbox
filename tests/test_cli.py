"""Tests for the CLI entry point."""
from __future__ import annotations

import sys
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from agentblackbox import BlackBox
from agentblackbox.cli import main


@pytest.fixture
def tmp_db(tmp_path):
    return tmp_path / "cli_test.db"


@pytest.fixture
def populated_db(tmp_db):
    with BlackBox.session("cli_agent", db_path=tmp_db) as bb:
        bb.record_llm_call("gpt-4o", "hello", "world", 100, 50, 200.0)
        bb.record_tool_call("search", {"q": "test"}, ["result"], 30.0)
    return tmp_db


class TestCLIHelp:
    def test_no_args_prints_help(self, capsys):
        with patch.object(sys, "argv", ["agentblackbox"]):
            main()
        out = capsys.readouterr().out
        assert "Usage" in out

    def test_help_flag(self, capsys):
        with patch.object(sys, "argv", ["agentblackbox", "--help"]):
            main()
        out = capsys.readouterr().out
        assert "sessions" in out

    def test_h_flag(self, capsys):
        with patch.object(sys, "argv", ["agentblackbox", "-h"]):
            main()
        out = capsys.readouterr().out
        assert "Usage" in out

    def test_help_command(self, capsys):
        with patch.object(sys, "argv", ["agentblackbox", "help"]):
            main()
        out = capsys.readouterr().out
        assert "Usage" in out

    def test_unknown_command_exits_1(self, capsys):
        with patch.object(sys, "argv", ["agentblackbox", "notacommand"]):
            with pytest.raises(SystemExit) as exc:
                main()
        assert exc.value.code == 1
        out = capsys.readouterr().out
        assert "Unknown command" in out


class TestCLISessions:
    def test_sessions_no_recordings(self, capsys, tmp_db, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_db.parent))
        with patch("agentblackbox.recorder._default_storage", None):
            from agentblackbox.storage import SQLiteStorage
            with patch("agentblackbox.recorder._get_storage", return_value=SQLiteStorage(tmp_db)):
                with patch.object(sys, "argv", ["agentblackbox", "sessions"]):
                    main()
        out = capsys.readouterr().out
        assert "No sessions" in out

    def test_sessions_with_data(self, capsys, populated_db):
        with patch("agentblackbox.recorder._default_storage", None):
            from agentblackbox.storage import SQLiteStorage
            storage = SQLiteStorage(populated_db)
            with patch("agentblackbox.recorder._get_storage", return_value=storage):
                with patch.object(sys, "argv", ["agentblackbox", "sessions"]):
                    main()
        out = capsys.readouterr().out
        assert "cli_agent" in out


class TestCLIReplay:
    def test_replay_no_args_exits(self, capsys):
        with patch.object(sys, "argv", ["agentblackbox", "replay"]):
            with pytest.raises(SystemExit) as exc:
                main()
        assert exc.value.code == 1

    def test_replay_with_session_id(self, capsys, populated_db):
        sessions = BlackBox.list_sessions(db_path=populated_db)
        sid = sessions[0].session_id

        with patch("agentblackbox.recorder._default_storage", None):
            from agentblackbox.storage import SQLiteStorage
            storage = SQLiteStorage(populated_db)
            with patch("agentblackbox.recorder._get_storage", return_value=storage):
                with patch.object(sys, "argv", ["agentblackbox", "replay", sid]):
                    main()
        out = capsys.readouterr().out
        assert "cli_agent" in out


class TestCLIExport:
    def test_export_no_args_exits(self, capsys):
        with patch.object(sys, "argv", ["agentblackbox", "export"]):
            with pytest.raises(SystemExit) as exc:
                main()
        assert exc.value.code == 1

    def test_export_with_session_id(self, capsys, populated_db):
        sessions = BlackBox.list_sessions(db_path=populated_db)
        sid = sessions[0].session_id

        with patch("agentblackbox.recorder._default_storage", None):
            from agentblackbox.storage import SQLiteStorage
            storage = SQLiteStorage(populated_db)
            with patch("agentblackbox.recorder._get_storage", return_value=storage):
                with patch.object(sys, "argv", ["agentblackbox", "export", sid]):
                    main()
        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["session"]["agent_name"] == "cli_agent"
        assert len(data["llm_calls"]) == 1


class TestCLIDashboard:
    def test_dashboard_missing_deps_exits(self, capsys):
        with patch.object(sys, "argv", ["agentblackbox", "dashboard"]):
            with patch.dict("sys.modules", {"fastapi": None, "uvicorn": None}):
                with pytest.raises((SystemExit, ImportError)):
                    main()
