"""Auto-instrumentation for the Anthropic Python SDK (sync + async)."""
from __future__ import annotations

import time
from typing import Any

_patched_sync = False
_patched_async = False


def patch_anthropic() -> None:
    """Monkey-patch the Anthropic SDK to auto-record all messages.create calls.

    Call once at startup — both sync and async clients are patched.
    Works with anthropic>=0.20.
    """
    global _patched_sync, _patched_async
    if _patched_sync and _patched_async:
        return

    try:
        import anthropic
    except ImportError:
        raise ImportError(
            "anthropic package is not installed. "
            "Install it with: pip install anthropic"
        )

    if not _patched_sync:
        _patch_sync(anthropic)
        _patched_sync = True

    if not _patched_async:
        _patch_async(anthropic)
        _patched_async = True

    print("[agentblackbox] Anthropic SDK patched ✓")


# ── sync ──────────────────────────────────────────────────────────────────────

def _patch_sync(anthropic: Any) -> None:
    from ..recorder import BlackBox

    try:
        Messages = anthropic.resources.messages.Messages
    except AttributeError:
        return

    original_create = Messages.create

    def _patched_create(self, *args, **kwargs):
        # Intercept streaming — wrap but don't record tokens (no usage in stream)
        if kwargs.get("stream", False):
            return original_create(self, *args, **kwargs)

        t0 = time.perf_counter()
        response = original_create(self, *args, **kwargs)
        duration_ms = (time.perf_counter() - t0) * 1000

        bb = BlackBox.current()
        if bb is not None:
            _record_response(bb, kwargs, response, duration_ms)
        return response

    Messages.create = _patched_create


# ── async ─────────────────────────────────────────────────────────────────────

def _patch_async(anthropic: Any) -> None:
    from ..recorder import BlackBox

    try:
        AsyncMessages = anthropic.resources.messages.AsyncMessages
    except AttributeError:
        return

    original_create = AsyncMessages.create

    async def _patched_create(self, *args, **kwargs):
        if kwargs.get("stream", False):
            return await original_create(self, *args, **kwargs)

        t0 = time.perf_counter()
        response = await original_create(self, *args, **kwargs)
        duration_ms = (time.perf_counter() - t0) * 1000

        bb = BlackBox.current()
        if bb is not None:
            _record_response(bb, kwargs, response, duration_ms)
        return response

    AsyncMessages.create = _patched_create


# ── helpers ───────────────────────────────────────────────────────────────────

def _record_response(bb: Any, kwargs: dict, response: Any, duration_ms: float) -> None:
    try:
        model = kwargs.get("model", getattr(response, "model", "unknown"))

        messages = kwargs.get("messages", [])
        input_text = " | ".join(
            (m.get("content", "") if isinstance(m.get("content"), str) else "")
            for m in messages
        )

        output_text = ""
        content = getattr(response, "content", [])
        if content:
            first = content[0]
            output_text = getattr(first, "text", "") or ""

        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "input_tokens", 0) if usage else 0
        output_tokens = getattr(usage, "output_tokens", 0) if usage else 0

        bb.record_llm_call(
            model=model,
            input_text=input_text,
            output_text=output_text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            duration_ms=duration_ms,
        )
    except Exception:
        pass  # never break user code
