"""PII masking for agentblackbox."""
from __future__ import annotations

import copy
import re
from typing import Any, Optional


class PIIMasker:
    BUILTIN_PATTERNS: dict[str, str] = {
        "credit_card": r"\b(?:\d{4}[- ]?){3}\d{4}\b",
        "email": r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b",
        "phone_jp": r"\b0\d{1,4}[-.\s]?\d{2,4}[-.\s]?\d{4}\b",
        "phone_us": r"\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
        "ipv4": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
        "api_key": r"\b(sk|pk|api|key|token|secret)[_-]?[A-Za-z0-9]{20,}\b",
        "aws_key": r"\bAKIA[0-9A-Z]{16}\b",
        "jwt": r"\beyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\b",
        "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    }

    def __init__(
        self,
        patterns: Optional[list[str]] = None,
        custom_patterns: Optional[dict[str, str]] = None,
        enabled: bool = True,
    ) -> None:
        self.enabled = enabled
        # Determine which builtin patterns to activate
        active_names = set(patterns) if patterns is not None else set(self.BUILTIN_PATTERNS)
        self._compiled: dict[str, re.Pattern] = {}
        for name in active_names:
            if name in self.BUILTIN_PATTERNS:
                self._compiled[name] = re.compile(self.BUILTIN_PATTERNS[name], re.IGNORECASE)
        if custom_patterns:
            for name, pattern in custom_patterns.items():
                self._compiled[name] = re.compile(pattern)

    def mask(self, text: str) -> str:
        if not self.enabled or not text:
            return text
        for name, pattern in self._compiled.items():
            replacement = f"[MASKED_{name.upper()}]"
            text = pattern.sub(replacement, text)
        return text

    def mask_dict(self, data: dict) -> dict:
        """Recursively mask string values in a dict."""
        if not self.enabled:
            return data
        return self._mask_value(data)

    def _mask_value(self, value: Any) -> Any:
        if isinstance(value, str):
            return self.mask(value)
        if isinstance(value, dict):
            return {k: self._mask_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._mask_value(item) for item in value]
        return value
