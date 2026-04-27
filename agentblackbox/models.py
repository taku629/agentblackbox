from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class LLMCall:
    call_id: str
    session_id: str
    timestamp: float
    model: str
    input_text: str
    output_text: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float


@dataclass
class ToolCall:
    call_id: str
    session_id: str
    timestamp: float
    tool_name: str
    args: dict
    result: Any
    latency_ms: float


@dataclass
class ErrorRecord:
    error_id: str
    session_id: str
    timestamp: float
    error_type: str
    message: str
    traceback: str


@dataclass
class Session:
    session_id: str
    agent_name: str
    start_time: float
    end_time: Optional[float] = None
    status: str = "running"
    llm_calls: list = field(default_factory=list)
    tool_calls: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    total_cost_usd: float = 0.0
    metadata: dict = field(default_factory=dict)
