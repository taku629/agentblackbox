"""LangChain callback handler for agentblackbox."""
from __future__ import annotations

import time
from typing import Any, Optional


class BlackBoxCallbackHandler:
    """LangChain callback handler that records LLM and tool calls to BlackBox.

    Does not require importing LangChain at install time — duck-typed to the
    LangChain BaseCallbackHandler interface.

    Usage:
        from agentblackbox.integrations.langchain_cb import BlackBoxCallbackHandler

        handler = BlackBoxCallbackHandler()
        llm = ChatOpenAI(callbacks=[handler])

        with BlackBox.session("my_agent") as bb:
            result = llm.invoke("Hello!")
    """

    def __init__(self) -> None:
        self._llm_starts: dict[str, tuple[float, list[str]]] = {}
        self._tool_starts: dict[str, tuple[float, str, str]] = {}

    # ── LLM callbacks ─────────────────────────────────────────────────────

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        self._llm_starts[str(run_id)] = (time.perf_counter(), prompts)

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        from ..recorder import BlackBox

        key = str(run_id)
        if key not in self._llm_starts:
            return
        t0, prompts = self._llm_starts.pop(key)
        duration_ms = (time.perf_counter() - t0) * 1000

        bb = BlackBox.current()
        if bb is None:
            return

        model = "unknown"
        input_tokens = 0
        output_tokens = 0
        output_text = ""

        if hasattr(response, "llm_output") and response.llm_output:
            llm_out = response.llm_output
            model = llm_out.get("model_name", model)
            usage = llm_out.get("token_usage", {})
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)

        if hasattr(response, "generations") and response.generations:
            gen = response.generations[0]
            if gen and hasattr(gen[0], "text"):
                output_text = gen[0].text

        bb.record_llm_call(
            model=model,
            input_text=" | ".join(prompts),
            output_text=output_text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            duration_ms=duration_ms,
        )

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        from ..recorder import BlackBox

        self._llm_starts.pop(str(run_id), None)
        bb = BlackBox.current()
        if bb is not None:
            bb.record_error(error)

    # ── Tool callbacks ─────────────────────────────────────────────────────

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        tool_name = serialized.get("name", "unknown_tool")
        self._tool_starts[str(run_id)] = (time.perf_counter(), tool_name, input_str)

    def on_tool_end(
        self,
        output: str,
        *,
        run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        from ..recorder import BlackBox

        key = str(run_id)
        if key not in self._tool_starts:
            return
        t0, tool_name, input_str = self._tool_starts.pop(key)
        duration_ms = (time.perf_counter() - t0) * 1000

        bb = BlackBox.current()
        if bb is not None:
            bb.record_tool_call(
                tool_name=tool_name,
                arguments={"input": input_str},
                result=output,
                duration_ms=duration_ms,
            )

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        from ..recorder import BlackBox

        key = str(run_id)
        if key not in self._tool_starts:
            return
        t0, tool_name, input_str = self._tool_starts.pop(key)
        duration_ms = (time.perf_counter() - t0) * 1000

        bb = BlackBox.current()
        if bb is not None:
            bb.record_tool_call(
                tool_name=tool_name,
                arguments={"input": input_str},
                result=None,
                duration_ms=duration_ms,
                error=str(error),
            )

    # ── Chain / Agent callbacks (no-ops — satisfy LangChain interface) ─────

    def on_chain_start(self, *args: Any, **kwargs: Any) -> None:
        pass

    def on_chain_end(self, *args: Any, **kwargs: Any) -> None:
        pass

    def on_chain_error(self, *args: Any, **kwargs: Any) -> None:
        pass

    def on_text(self, *args: Any, **kwargs: Any) -> None:
        pass

    def on_agent_action(self, *args: Any, **kwargs: Any) -> None:
        pass

    def on_agent_finish(self, *args: Any, **kwargs: Any) -> None:
        pass
