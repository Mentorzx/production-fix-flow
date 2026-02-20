"""Runtime state storage for Search Space Advisor."""

from __future__ import annotations

import time
from typing import Any

from pff.shared.core.cache import CacheManager

from .models import TrustState


class AdvisorRuntimeStateStore:
    """Centralized runtime state for trust, self-audit, and adaptive perf signals."""

    def __init__(self, *, max_memory_items: int = 1024) -> None:
        self._cache = CacheManager(max_memory_items=max_memory_items)
        self._version_tag = "search_space_advisor_runtime_v1"

    def _study_key(self, study_key: str) -> str:
        return study_key or "__default__"

    def _key(self, namespace: str, study_key: str) -> str:
        return f"{self._version_tag}:{namespace}:{self._study_key(study_key)}"

    def _tag(self, study_key: str) -> str:
        return f"study:{self._study_key(study_key)}"

    def get_trust_bucket(self, study_key: str) -> dict[str, TrustState]:
        key = self._key("trust", study_key)
        value = self._cache.get(key)
        if isinstance(value, dict):
            return value
        bucket: dict[str, TrustState] = {}
        self._cache.set(key, bucket, tags=[self._tag(study_key)])
        return bucket

    def get_trust_state(self, study_key: str, param_name: str) -> TrustState:
        bucket = self.get_trust_bucket(study_key)
        current = bucket.get(param_name)
        if isinstance(current, TrustState):
            return current
        state = TrustState()
        bucket[param_name] = state
        self._cache.set(self._key("trust", study_key), bucket, tags=[self._tag(study_key)])
        return state

    def set_trust_state(self, study_key: str, param_name: str, state: TrustState) -> None:
        bucket = self.get_trust_bucket(study_key)
        bucket[param_name] = state
        self._cache.set(self._key("trust", study_key), bucket, tags=[self._tag(study_key)])

    def get_self_audit_snapshot(self, study_key: str) -> dict[str, Any] | None:
        key = self._key("self_audit", study_key)
        value = self._cache.get(key)
        if isinstance(value, dict):
            return dict(value)
        return None

    def set_self_audit_snapshot(self, study_key: str, snapshot: dict[str, Any]) -> None:
        payload = dict(snapshot)
        payload.setdefault("updated_at_ns", time.time_ns())
        self._cache.set(
            self._key("self_audit", study_key),
            payload,
            tags=[self._tag(study_key)],
        )

    def get_adaptive_state(self, study_key: str) -> dict[str, Any]:
        key = self._key("adaptive_perf", study_key)
        value = self._cache.get(key)
        if isinstance(value, dict):
            return dict(value)
        return {}

    def set_adaptive_state(self, study_key: str, state: dict[str, Any]) -> None:
        payload = dict(state)
        payload.setdefault("updated_at_ns", time.time_ns())
        self._cache.set(
            self._key("adaptive_perf", study_key),
            payload,
            tags=[self._tag(study_key)],
        )

    def clear_study(self, study_key: str) -> int:
        return int(self._cache.invalidate(tags=[self._tag(study_key)]))

