from __future__ import annotations

import json
import re

import pytest

from agentblackbox import BlackBox, decode_share_payload, share_session
from agentblackbox.sharing import (
    DEFAULT_BASE_URL,
    TEXT_TRUNCATION_HINT,
    _encode,
    _truncate_payload,
)


@pytest.fixture
def tmp_db(tmp_path):
    return tmp_path / "share.db"


@pytest.fixture
def small_session_id(tmp_db):
    with BlackBox.session("share-agent", db_path=tmp_db) as bb:
        bb.record_llm_call("gpt-4o-mini", "hi", "hello", 5, 10, 50.0)
        bb.record_tool_call("search", {"q": "weather"}, {"hits": 3}, 12.0)
    sessions = BlackBox.list_sessions(db_path=tmp_db)
    return sessions[0].session_id


def test_share_session_returns_self_contained_url(tmp_db, small_session_id):
    link = share_session(small_session_id, db_path=tmp_db)
    assert link.url.startswith(DEFAULT_BASE_URL)
    assert "#z=" in link.url
    assert link.encoded_bytes > 0
    assert link.truncated is False


def test_share_round_trip_recovers_session(tmp_db, small_session_id):
    link = share_session(small_session_id, db_path=tmp_db)
    encoded = re.search(r"z=([\w-]+)", link.url).group(1)
    payload = decode_share_payload(encoded)

    assert payload["session"]["session_id"] == small_session_id
    assert payload["llm_calls"][0]["model"] == "gpt-4o-mini"
    assert payload["tool_calls"][0]["tool_name"] == "search"


def test_share_custom_base_url(tmp_db, small_session_id):
    link = share_session(small_session_id, db_path=tmp_db, base_url="https://example.dev/v")
    assert link.url.startswith("https://example.dev/v#z=")


def test_share_appends_to_existing_hash(tmp_db, small_session_id):
    link = share_session(small_session_id, db_path=tmp_db, base_url="https://example.dev/v#mode=view")
    # Existing fragment must be preserved with an & separator instead of overwritten.
    assert "#mode=view&z=" in link.url


def test_share_truncates_oversized_session(tmp_db):
    import secrets
    # Use non-repetitive content so gzip can't crush the size away.
    big_blob = secrets.token_hex(20_000)
    with BlackBox.session("big-agent", db_path=tmp_db) as bb:
        bb.record_llm_call("gpt-4o", big_blob, big_blob, 10, 10, 100.0)
    sid = BlackBox.list_sessions(db_path=tmp_db)[0].session_id

    link = share_session(sid, db_path=tmp_db, max_encoded_bytes=2_000)
    assert link.truncated is True
    assert link.encoded_bytes <= 2_000

    encoded = re.search(r"z=([\w-]+)", link.url).group(1)
    payload = decode_share_payload(encoded)
    assert payload.get("share_truncated") is True
    assert TEXT_TRUNCATION_HINT in payload["llm_calls"][0]["input_text"]


def test_share_unknown_session_raises(tmp_db):
    with pytest.raises(ValueError, match="Session not found"):
        share_session("does-not-exist", db_path=tmp_db)


def test_truncate_handles_nonstring_tool_args(tmp_db):
    payload = {
        "session": {"session_id": "x", "agent_name": "a", "status": "success"},
        "llm_calls": [],
        "tool_calls": [
            {
                "tool_name": "search",
                "arguments": {"q": "Y" * 20_000},
                "result": {"hits": ["Z" * 5_000 for _ in range(20)]},
            }
        ],
        "errors": [],
    }
    truncated = _truncate_payload(payload, target_bytes=2_000)
    assert len(_encode(truncated)) <= 2_000
    assert truncated["share_truncated"] is True


def test_decode_handles_missing_padding():
    payload = {"hello": "world", "items": list(range(50))}
    encoded = _encode(payload)
    # _encode strips padding; decoder must add it back.
    assert "=" not in encoded
    assert decode_share_payload(encoded) == payload
