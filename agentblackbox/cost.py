"""Pricing tables and cost calculation for LLM providers."""
from __future__ import annotations

# USD per 1M tokens  {model: (input_price, output_price)}
_PRICING: dict[str, tuple[float, float]] = {
    # OpenAI
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o-mini-2024-07-18": (0.15, 0.60),
    "gpt-4-turbo": (10.00, 30.00),
    "gpt-4-turbo-preview": (10.00, 30.00),
    "gpt-4": (30.00, 60.00),
    "gpt-3.5-turbo": (0.50, 1.50),
    "gpt-3.5-turbo-0125": (0.50, 1.50),
    "o1": (15.00, 60.00),
    "o1-mini": (3.00, 12.00),
    "o3-mini": (1.10, 4.40),
    # Anthropic
    "claude-opus-4-7": (15.00, 75.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5-20251001": (0.80, 4.00),
    "claude-3-5-sonnet-20241022": (3.00, 15.00),
    "claude-3-5-sonnet-20240620": (3.00, 15.00),
    "claude-3-5-haiku-20241022": (0.80, 4.00),
    "claude-3-opus-20240229": (15.00, 75.00),
    "claude-3-sonnet-20240229": (3.00, 15.00),
    "claude-3-haiku-20240307": (0.25, 1.25),
}

# Prefix-based fallback for model families
_FAMILY_PRICING: list[tuple[str, tuple[float, float]]] = [
    ("claude-opus", (15.00, 75.00)),
    ("claude-sonnet", (3.00, 15.00)),
    ("claude-haiku", (0.80, 4.00)),
    ("gpt-4o-mini", (0.15, 0.60)),
    ("gpt-4o", (2.50, 10.00)),
    ("gpt-4", (10.00, 30.00)),
    ("gpt-3.5", (0.50, 1.50)),
    ("o1-mini", (3.00, 12.00)),
    ("o1", (15.00, 60.00)),
    ("o3-mini", (1.10, 4.40)),
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
