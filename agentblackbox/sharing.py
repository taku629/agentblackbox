"""Share a recorded session via a self-contained URL.

The entire session is gzip-compressed and base64url-encoded into the URL fragment, so the viewer
needs no server: it loads the URL hash, decodes, and renders. Privacy-preserving — the recording
never leaves the user's control unless they hand out the URL.

Sessions that exceed a soft size cap are truncated: long input/output_text bodies are clipped, and
the result advertises the cloud waitlist for full-fidelity sharing.
"""
from __future__ import annotations

import base64
import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .recorder import BlackBox
from .storage import DEFAULT_DB_PATH, SQLiteStorage

DEFAULT_BASE_URL = "https://agentblackbox.dev/view"

# Soft cap on the encoded URL length. Most browsers handle URLs up to ~32KB; we leave headroom.
DEFAULT_MAX_ENCODED_BYTES = 24_000
TEXT_TRUNCATION_HINT = "...[truncated for share-link; see full session on Cloud waitlist]"


@dataclass
class ShareLink:
    url: str
    encoded_bytes: int
    truncated: bool


def share_session(
    session_id: str,
    *,
    db_path: Optional[Path] = None,
    base_url: str = DEFAULT_BASE_URL,
    max_encoded_bytes: int = DEFAULT_MAX_ENCODED_BYTES,
) -> ShareLink:
    """Build a share-link URL for the given session.

    Raises ValueError if the session doesn't exist.
    """
    raw = BlackBox.export_json(session_id, db_path=db_path)
    payload = json.loads(raw)
    encoded, truncated = _encode_payload(payload, max_encoded_bytes=max_encoded_bytes)
    sep = "&" if "#" in base_url else "#"
    url = f"{base_url}{sep}z={encoded}"
    return ShareLink(url=url, encoded_bytes=len(encoded), truncated=truncated)


def decode_share_payload(encoded: str) -> dict:
    """Inverse of the encoder. Used by tests and by Python-side consumers."""
    padding = "=" * (-len(encoded) % 4)
    compressed = base64.urlsafe_b64decode(encoded + padding)
    return json.loads(gzip.decompress(compressed).decode("utf-8"))


def _encode_payload(payload: dict, *, max_encoded_bytes: int) -> tuple[str, bool]:
    encoded = _encode(payload)
    if len(encoded) <= max_encoded_bytes:
        return encoded, False

    truncated = _truncate_payload(payload, max_encoded_bytes)
    encoded = _encode(truncated)
    return encoded, True


def _encode(payload: dict) -> str:
    raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    compressed = gzip.compress(raw, mtime=0)
    return base64.urlsafe_b64encode(compressed).rstrip(b"=").decode("ascii")


def _truncate_payload(payload: dict, target_bytes: int) -> dict:
    """Iteratively shrink the most expensive fields until the payload fits.

    Non-string ``arguments`` / ``result`` fields are first serialized to a JSON string so they can
    be clipped safely without producing invalid JSON. The viewer treats them as opaque strings when
    ``share_truncated`` is set.
    """
    work = json.loads(json.dumps(payload))
    work["share_truncated"] = True

    for tool in work.get("tool_calls", []):
        if tool.get("arguments") is not None and not isinstance(tool["arguments"], str):
            tool["arguments"] = json.dumps(tool["arguments"])
        if tool.get("result") is not None and not isinstance(tool["result"], str):
            tool["result"] = json.dumps(tool["result"])

    limits = [4000, 2000, 1000, 500, 250, 120, 60, 30, 10]
    for limit in limits:
        for call in work.get("llm_calls", []):
            call["input_text"] = _clip(call.get("input_text", ""), limit)
            call["output_text"] = _clip(call.get("output_text", ""), limit)
        for tool in work.get("tool_calls", []):
            if isinstance(tool.get("arguments"), str):
                tool["arguments"] = _clip(tool["arguments"], limit)
            if isinstance(tool.get("result"), str):
                tool["result"] = _clip(tool["result"], limit)
        for err in work.get("errors", []):
            err["traceback"] = _clip(err.get("traceback", ""), limit)
            err["message"] = _clip(err.get("message", ""), limit)

        if len(_encode(work)) <= target_bytes:
            return work

    return work


def _clip(text: str, limit: int) -> str:
    if not isinstance(text, str):
        return text
    if len(text) <= limit:
        return text
    head = max(1, limit - len(TEXT_TRUNCATION_HINT))
    return text[:head] + TEXT_TRUNCATION_HINT
