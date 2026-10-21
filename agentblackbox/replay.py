"""Rendering and export helpers for recorded sessions."""
from __future__ import annotations

import json
from datetime import datetime

from .models import ErrorRecord, LLMCall, Session, ToolCall


def render_replay(session: Session, events) -> str:
    """Render a recorded session as human-readable text."""
    out = [
        f"\n{'='*70}",
        f"  Session: {session.session_id}",
        f"  Agent:   {session.agent_name}",
        f"  Status:  {session.status}",
        f"  Start:   {datetime.fromtimestamp(session.start_time / 1e9).strftime('%Y-%m-%d %H:%M:%S.%f')}",
    ]
    if session.end_time:
        end_dt = datetime.fromtimestamp(session.end_time / 1e9)
        elapsed_ms = (session.end_time - session.start_time) / 1e6  # type: ignore[operator]
        out.append(f"  End:     {end_dt.strftime('%Y-%m-%d %H:%M:%S.%f')}  ({elapsed_ms:.1f}ms)")
    out.append(f"  Cost:    ${session.total_cost_usd:.6f}")
    out.append(f"{'='*70}")

    for ts, kind, ev in events:
        dt = datetime.fromtimestamp(ts / 1e9)
        prefix = dt.strftime("%H:%M:%S.%f")[:-3]
        if kind == "llm":
            out.append(f"\n[{prefix}] LLM  model={ev.model}  tokens={ev.input_tokens}→{ev.output_tokens}  ${ev.cost_usd:.6f}  {ev.duration_ms:.1f}ms")
            out.append(_truncated("  IN ", ev.input_text, 200))
            out.append(_truncated("  OUT", ev.output_text, 200))
        elif kind == "tool":
            status_mark = "✗" if ev.error else "✓"
            out.append(f"\n[{prefix}] TOOL [{status_mark}] {ev.tool_name}  {ev.duration_ms:.1f}ms")
            out.append(_truncated("  ARGS  ", json.dumps(ev.arguments), 200))
            if ev.error:
                out.append(_truncated("  ERROR ", ev.error, 200))
            else:
                out.append(_truncated("  RESULT", str(ev.result), 200))
        elif kind == "error":
            out.append(f"\n[{prefix}] ERROR {ev.error_type}: {ev.message}")
            out.append("  " + ev.traceback.replace("\n", "\n  ").rstrip())

    out.append(f"\n{'='*70}\n")
    return "\n".join(out)


def export_session_dict(
    session: Session,
    llm_calls: list[LLMCall],
    tool_calls: list[ToolCall],
    errors: list[ErrorRecord],
) -> dict:
    """Build the JSON-serializable export payload for a session."""
    return {
        "session": {
            "session_id": session.session_id,
            "agent_name": session.agent_name,
            "start_time": session.start_time,
            "end_time": session.end_time,
            "status": session.status,
            "total_cost_usd": session.total_cost_usd,
            "metadata": session.metadata,
        },
        "llm_calls": [
            {
                "id": c.id,
                "timestamp": c.timestamp,
                "model": c.model,
                "input_tokens": c.input_tokens,
                "output_tokens": c.output_tokens,
                "input_text": c.input_text,
                "output_text": c.output_text,
                "duration_ms": c.duration_ms,
                "cost_usd": c.cost_usd,
            }
            for c in llm_calls
        ],
        "tool_calls": [
            {
                "id": c.id,
                "timestamp": c.timestamp,
                "tool_name": c.tool_name,
                "arguments": c.arguments,
                "result": c.result,
                "duration_ms": c.duration_ms,
                "error": c.error,
            }
            for c in tool_calls
        ],
        "errors": [
            {
                "id": e.id,
                "timestamp": e.timestamp,
                "error_type": e.error_type,
                "message": e.message,
                "traceback": e.traceback,
            }
            for e in errors
        ],
    }


def _truncated(label: str, text: str, max_len: int) -> str:
    if len(text) > max_len:
        text = text[:max_len] + f"... [{len(text) - max_len} more chars]"
    return f"  {label}: {text}"
