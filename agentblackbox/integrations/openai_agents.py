"""Auto-instrumentation for the OpenAI Agents SDK."""
from __future__ import annotations

import time
from typing import Any

_patched = False


def patch_openai_agents() -> None:
    """Monkey-patch the OpenAI Agents SDK to auto-record all agent runs.

    Call this once at startup — every subsequent agent.run() will be recorded
    in the current BlackBox session (if one is active) or a new auto-session.
    """
    global _patched
    if _patched:
        return

    try:
        import agents  # openai-agents package
    except ImportError:
        raise ImportError(
            "openai-agents package is not installed. "
            "Install it with: pip install openai-agents"
        )

    _patch_runner(agents)
    _patch_model_calls(agents)
    _patched = True
    print("[agentblackbox] OpenAI Agents SDK patched ✓")


def _patch_runner(agents_module: Any) -> None:
    from ..recorder import BlackBox

    try:
        Runner = agents_module.Runner
    except AttributeError:
        return

    original_run = Runner.run

    async def _patched_run(cls_or_self, agent, input, **kwargs):  # type: ignore[override]
        agent_name = getattr(agent, "name", "openai_agent")
        current = BlackBox.current()

        if current is not None:
            return await original_run(cls_or_self, agent, input, **kwargs)

        async with _AsyncBlackBoxWrapper(agent_name):
            return await original_run(cls_or_self, agent, input, **kwargs)

    Runner.run = _patched_run


def _patch_model_calls(agents_module: Any) -> None:
    """Patch model invocations to record LLM calls."""
    from ..recorder import BlackBox

    try:
        from openai import AsyncOpenAI
    except ImportError:
        return

    original_create = AsyncOpenAI().chat.completions.create.__func__  # type: ignore[attr-defined]

    async def _patched_create(self, *args, **kwargs):
        t0 = time.perf_counter()
        response = await original_create(self, *args, **kwargs)
        duration_ms = (time.perf_counter() - t0) * 1000

        bb = BlackBox.current()
        if bb is not None and response.usage:
            model = kwargs.get("model", getattr(response, "model", "unknown"))
            messages = kwargs.get("messages", [])
            input_text = " | ".join(
                m.get("content", "") for m in messages if isinstance(m.get("content"), str)
            )
            output_text = ""
            if response.choices:
                c = response.choices[0].message
                output_text = c.content or ""

            bb.record_llm_call(
                model=model,
                input_text=input_text,
                output_text=output_text,
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                latency_ms=duration_ms,
            )
        return response

    try:
        import openai
        openai.AsyncOpenAI.chat = property(  # type: ignore[assignment]
            lambda self: _PatchedCompletions(self, _patched_create)
        )
    except Exception:
        pass  # best-effort patching


class _AsyncBlackBoxWrapper:
    def __init__(self, agent_name: str) -> None:
        self._agent_name = agent_name
        self._bb = None

    async def __aenter__(self):
        from ..recorder import BlackBox
        self._bb = BlackBox.session(self._agent_name)
        self._bb.start()
        return self._bb

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._bb is not None:
            if exc_val is not None:
                self._bb.record_error(exc_val)
            self._bb.stop(success=exc_type is None)
        return False


class _PatchedCompletions:
    def __init__(self, client, patched_fn):
        self._client = client
        self._fn = patched_fn

    @property
    def completions(self):
        return self

    async def create(self, *args, **kwargs):
        return await self._fn(self._client, *args, **kwargs)
