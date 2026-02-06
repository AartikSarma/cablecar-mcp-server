"""Computation cache to avoid redundant re-runs."""
from __future__ import annotations
import hashlib
import json
from typing import Any

class ComputationCache:
    """Cache analysis results keyed by parameters."""

    def __init__(self):
        self._cache: dict[str, Any] = {}

    def _make_key(self, analysis_type: str, params: dict) -> str:
        """Create a deterministic cache key from analysis type and parameters."""
        key_data = {"type": analysis_type, "params": params}
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.sha256(key_str.encode()).hexdigest()[:16]

    def get(self, analysis_type: str, params: dict) -> Any | None:
        key = self._make_key(analysis_type, params)
        return self._cache.get(key)

    def set(self, analysis_type: str, params: dict, result: Any):
        key = self._make_key(analysis_type, params)
        self._cache[key] = result

    def has(self, analysis_type: str, params: dict) -> bool:
        key = self._make_key(analysis_type, params)
        return key in self._cache

    def clear(self):
        self._cache.clear()

    def size(self) -> int:
        return len(self._cache)
