"""agentblackbox — zero-dependency black-box recorder for AI agents."""
from .recorder import BlackBox, MockBlackBox
from .storage import SQLiteStorage, DEFAULT_DB_PATH
from .cost import calculate_cost, list_supported_models
from .models import Session, LLMCall, ToolCall, ErrorRecord
from .integrations import patch_anthropic, patch_openai_agents, BlackBoxCallbackHandler
from .sharing import DEFAULT_BASE_URL as SHARE_BASE_URL, ShareLink, decode_share_payload, share_session

__version__ = "0.2.2"
__all__ = [
    "BlackBox",
    "MockBlackBox",
    "SQLiteStorage",
    "DEFAULT_DB_PATH",
    "calculate_cost",
    "list_supported_models",
    "Session",
    "LLMCall",
    "ToolCall",
    "ErrorRecord",
    "patch_anthropic",
    "patch_openai_agents",
    "BlackBoxCallbackHandler",
    "SHARE_BASE_URL",
    "ShareLink",
    "decode_share_payload",
    "share_session",
]
