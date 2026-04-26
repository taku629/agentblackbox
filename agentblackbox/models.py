from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Session:
    session_id: str
    agent_name: str
    start_time: int  # nanoseconds
    end_time: Optional[int] = None
    status: str = "running"
    total_cost_usd: float = 0.0
    metadata: dict = field(default_factory=dict)


@dataclass
class LLMCall:
    id: str
    session_id: str
    timestamp: int  # nanoseconds
    model: str
    input_tokens: int
    output_tokens: int
    input_text: str
    output_text: str
    duration_ms: float
    cost_usd: float
    metadata: dict = field(default_factory=dict)


@dataclass
class ToolCall:
    id: str
    session_id: str
    timestamp: int  # nanoseconds
    tool_name: str
    arguments: dict
    result: Any
    duration_ms: float
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class ErrorRecord:
    id: str
    session_id: str
    timestamp: int  # nanoseconds
    error_type: str
    message: str
    traceback: str
    metadata: dict = field(default_factory=dict)
