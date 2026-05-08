"""LangChain callback handler for agentblackbox."""
from __future__ import annotations

import time
import uuid
from typing import Any, Optional, Union
from uuid import UUID


class BlackBoxCallbackHandler:
    """LangChain callback handler that records LLM calls and tool use.

    Usage::

        from agentblackbox.integrations.langchain import BlackBoxCallbackHandler
        from agentblackbox import BlackBox

        handler = BlackBoxCallbackHandler()

        with BlackBox.session("my-agent") as bb:
            handler.bind(bb)
            chain.invoke({"input": "..."}, config={"callbacks": [handler]})
    """

    def __init__(self) -> None:
        self._bb: Any = None
        self._llm_start_times: dict[str, float] = {}
        self._tool_start_times: dict[str, float] = {}
        self._tool_inputs: dict[str, dict] = {}

    def bind(self, bb: Any) -> "BlackBoxCallbackHandler":
        self._bb = bb
        return self

    # ── LLM ──────────────────────────────────────────────────────────────

    def on_llm_start(
        self,
        serialized: dict,
        prompts: list[str],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        self._llm_start_times[str(run_id)] = time.perf_counter()

    def on_llm_end(self, response: Any, *, run_id: UUID, **kwargs: Any) -> None:
        if self._bb is None:
            return
        t0 = self._llm_start_times.pop(str(run_id), None)
        duration_ms = (time.perf_counter() - t0) * 1000 if t0 else 0.0

        try:
            gen = response.generations[0][0] if response.generations else None
            output_text = getattr(gen, "text", "") if gen else ""
            llm_output = response.llm_output or {}
            usage = llm_output.get("token_usage", {})
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)
            model = llm_output.get("model_name", "unknown")

            self._bb.record_llm_call(
                model=model,
                input_text=" | ".join(response.generations[0][0].generation_info.get("finish_reason", "") if response.generations else []),
                output_text=output_text,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                duration_ms=duration_ms,
            )
        except Exception:
            pass

    def on_llm_error(self, error: Exception, *, run_id: UUID, **kwargs: Any) -> None:
        self._llm_start_times.pop(str(run_id), None)
        if self._bb is not None:
            self._bb.record_error(error)

    # ── Tool ─────────────────────────────────────────────────────────────

    def on_tool_start(
        self,
        serialized: dict,
        input_str: str,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        self._tool_start_times[str(run_id)] = time.perf_counter()
        tool_name = serialized.get("name", "unknown_tool")
        self._tool_inputs[str(run_id)] = {"name": tool_name, "input": input_str}

    def on_tool_end(self, output: str, *, run_id: UUID, **kwargs: Any) -> None:
        if self._bb is None:
            return
        t0 = self._tool_start_times.pop(str(run_id), None)
        duration_ms = (time.perf_counter() - t0) * 1000 if t0 else 0.0
        info = self._tool_inputs.pop(str(run_id), {})

        self._bb.record_tool_call(
            tool_name=info.get("name", "unknown_tool"),
            arguments={"input": info.get("input", "")},
            result=output,
            duration_ms=duration_ms,
        )

    def on_tool_error(self, error: Exception, *, run_id: UUID, **kwargs: Any) -> None:
        t0 = self._tool_start_times.pop(str(run_id), None)
        duration_ms = (time.perf_counter() - t0) * 1000 if t0 else 0.0
        info = self._tool_inputs.pop(str(run_id), {})

        if self._bb is not None:
            self._bb.record_tool_call(
                tool_name=info.get("name", "unknown_tool"),
                arguments={"input": info.get("input", "")},
                result=None,
                duration_ms=duration_ms,
                error=str(error),
            )

    # ── Chain (no-op, required by some LangChain versions) ───────────────

    def on_chain_start(self, *args: Any, **kwargs: Any) -> None:
        pass

    def on_chain_end(self, *args: Any, **kwargs: Any) -> None:
        pass

    def on_chain_error(self, error: Exception, **kwargs: Any) -> None:
        if self._bb is not None:
            self._bb.record_error(error)
