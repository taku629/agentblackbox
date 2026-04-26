"""agentblackbox — zero-dependency black-box recorder for AI agents."""
from .recorder import BlackBox
from .storage import SQLiteStorage, DEFAULT_DB_PATH
from .cost import calculate_cost, list_supported_models
from .models import Session, LLMCall, ToolCall, ErrorRecord

__version__ = "0.1.0"
__all__ = [
    "BlackBox",
    "SQLiteStorage",
    "DEFAULT_DB_PATH",
    "calculate_cost",
    "list_supported_models",
    "Session",
    "LLMCall",
    "ToolCall",
    "ErrorRecord",
]
