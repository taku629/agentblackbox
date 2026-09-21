"""Conservative, zero-dependency redaction for recorded data."""
from __future__ import annotations

import re
from typing import Any

REDACTED = "[REDACTED]"
_SENSITIVE_KEY = re.compile(
    r"(^|[_-])(api[_-]?key|authorization|auth|access[_-]?token|refresh[_-]?token|"
    r"password|passwd|secret|client[_-]?secret|cookie|set[_-]?cookie)([_-]|$)", re.I
)
_VALUE_PATTERNS = (
    re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(r"\b(?:sk|abx)_[A-Za-z0-9_-]{8,}\b"),
)


def redact(value: Any, *, _seen: set[int] | None = None, _depth: int = 0) -> Any:
    """Return a JSON-safe copy with common credentials removed.

    Cycles and pathological nesting are represented by stable markers instead of
    crashing the application being observed.
    """
    if _depth > 50:
        return "[MAX_DEPTH]"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        result = value
        for pattern in _VALUE_PATTERNS:
            result = pattern.sub(lambda m: (m.group(1) if m.lastindex else "") + REDACTED, result)
        return result
    if isinstance(value, bytes):
        return redact(value.decode("utf-8", errors="replace"), _depth=_depth + 1)

    seen = _seen if _seen is not None else set()
    identity = id(value)
    if identity in seen:
        return "[CYCLE]"
    seen.add(identity)
    try:
        if isinstance(value, dict):
            return {
                str(key): REDACTED if _SENSITIVE_KEY.search(str(key)) else redact(
                    item, _seen=seen, _depth=_depth + 1
                )
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple, set, frozenset)):
            return [redact(item, _seen=seen, _depth=_depth + 1) for item in value]
        return redact(str(value), _seen=seen, _depth=_depth + 1)
    finally:
        seen.remove(identity)
