"""Pricing tables and cost calculation for LLM providers."""
from __future__ import annotations

# USD per 1M tokens  {model: (input_price, output_price)}
_PRICING: dict[str, tuple[float, float]] = {
    # OpenAI
    "gpt-4o": (5.0, 15.0),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o-mini-2024-07-18": (0.15, 0.60),
    "gpt-4-turbo": (10.0, 30.0),
    "gpt-4-turbo-preview": (10.0, 30.0),
    "gpt-3.5-turbo": (0.50, 1.50),
    "gpt-3.5-turbo-0125": (0.50, 1.50),
    "o1": (15.0, 60.0),
    "o1-mini": (3.0, 12.0),
    "o3-mini": (1.10, 4.40),
    # Anthropic
    "claude-3-5-sonnet-20241022": (3.0, 15.0),
    "claude-3-5-sonnet-20240620": (3.0, 15.0),
    "claude-3-5-haiku-20241022": (0.80, 4.0),
    "claude-3-opus-20240229": (15.0, 75.0),
    "claude-3-sonnet-20240229": (3.0, 15.0),
    "claude-3-haiku-20240307": (0.25, 1.25),
    "claude-opus-4-5": (15.0, 75.0),
    "claude-opus-4-7": (15.0, 75.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5-20251001": (0.80, 4.0),
    # Google
    "gemini-1.5-pro": (3.5, 10.5),
    "gemini-1.5-flash": (0.075, 0.30),
    "gemini-2.0-flash": (0.10, 0.40),
    # Meta / Llama
    "llama-3.1-70b": (0.88, 0.88),
    "llama-3.1-8b": (0.20, 0.20),
    # Mistral
    "mistral-large": (3.0, 9.0),
    "mistral-medium": (2.7, 8.1),
    # DeepSeek
    "deepseek-chat": (0.27, 1.10),
    # xAI Grok
    "grok-2": (2.0, 10.0),
    "grok-2-mini": (0.20, 2.0),
}

# Prefix-based fallback for model families
_FAMILY_PRICING: list[tuple[str, tuple[float, float]]] = [
    ("claude-opus", (15.0, 75.0)),
    ("claude-sonnet", (3.0, 15.0)),
    ("claude-haiku", (0.80, 4.0)),
    ("gpt-4o-mini", (0.15, 0.60)),
    ("gpt-4o", (5.0, 15.0)),
    ("gpt-4", (10.0, 30.0)),
    ("gpt-3.5", (0.50, 1.50)),
    ("o1-mini", (3.0, 12.0)),
    ("o1", (15.0, 60.0)),
    ("o3-mini", (1.10, 4.40)),
    ("gemini-1.5-pro", (3.5, 10.5)),
    ("gemini-1.5-flash", (0.075, 0.30)),
    ("gemini-2.0-flash", (0.10, 0.40)),
    ("llama-3.1-70b", (0.88, 0.88)),
    ("mistral-large", (3.0, 9.0)),
    ("deepseek", (0.27, 1.10)),
    ("grok-2", (2.0, 10.0)),
]


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    model_lower = model.lower()
    prices = _PRICING.get(model_lower)
    if prices is None:
        for prefix, p in _FAMILY_PRICING:
            if model_lower.startswith(prefix):
                prices = p
                break
    if prices is None:
        return 0.0
    input_price, output_price = prices
    return (input_tokens * input_price + output_tokens * output_price) / 1_000_000


def list_supported_models() -> list[str]:
    return sorted(_PRICING.keys())
