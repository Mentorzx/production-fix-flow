"""L1/L2 cache for Search Space Advisor recommendations."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from pff.infrastructure.hpo.trials.postgres_store import HpoPostgresStore
from pff.shared import logger, stable_hash
from pff.shared.core.cache import CacheManager
from pff.shared.core.file_manager import FileManager


def _hash_hex(payload: str) -> str:
    digest_int = int(stable_hash(payload, truncate=64))
    return f"{digest_int & ((1 << 64) - 1):016x}"


def build_search_space_hash(search_space: dict[str, Any]) -> str:
    """Build deterministic search-space hash from canonical JSON."""
    raw = FileManager.json_dumps(search_space or {}, sort_keys=True)
    return _hash_hex(raw)


def build_objective_schema_hash(objective_directions: list[str] | None) -> str:
    """Build deterministic objective schema hash."""
    raw = FileManager.json_dumps(
        {"objective_directions": objective_directions or []}, sort_keys=True
    )
    return _hash_hex(raw)


def compute_cache_key(
    *,
    study_name: str,
    last_trial_number: int,
    dataset_fingerprint: str | None,
    direction: str = "maximize",
    advisor_version: str = "2.3.0",
) -> str:
    """Build deterministic short cache key compatible with legacy advisor tests."""
    norm_direction = str(direction).strip().lower()
    if norm_direction not in {"maximize", "minimize"}:
        norm_direction = "maximize"
    raw = (
        f"{study_name}:{last_trial_number}:{dataset_fingerprint or 'none'}:"
        f"{norm_direction}:{advisor_version}"
    )
    return f"{int(stable_hash(raw, truncate=96)) & ((1 << 96) - 1):024x}"[:24]


@dataclass(frozen=True)
class AdvisorCacheSpec:
    """Cache identity tuple for advisor recommendations."""

    study_name: str
    dataset_fingerprint: str
    direction: str
    advisor_version: str
    last_trial: int
    search_space_hash: str
    objective_schema_hash: str

    def cache_key(self) -> str:
        raw = (
            f"{self.study_name}:{self.dataset_fingerprint}:{self.direction}:{self.advisor_version}:"
            f"{self.last_trial}:{self.search_space_hash}:{self.objective_schema_hash}"
        )
        return _hash_hex(raw)


@dataclass(frozen=True)
class AdvisorCacheGetResult:
    """Structured cache read result with observability metadata."""

    payload: dict[str, Any] | None
    layer_hit: str
    status: str
    error_code: str | None = None


@dataclass(frozen=True)
class AdvisorCacheWriteResult:
    """Structured cache write result with observability metadata."""

    status: str
    error_code: str | None = None


class AdvisorCache:
    """Two-level cache: in-memory L1 + PostgreSQL-backed L2."""

    def __init__(
        self,
        *,
        max_memory_items: int = 128,
        ttl_seconds: int = 900,
        enable_persistent_l2: bool = True,
    ) -> None:
        self.ttl_seconds = max(1, int(ttl_seconds))
        self._l1 = CacheManager(max_memory_items=max_memory_items)
        self._enable_l2 = bool(enable_persistent_l2)
        self._store: HpoPostgresStore | None = (
            HpoPostgresStore() if self._enable_l2 else None
        )

    @staticmethod
    def _run_async(coro: Any) -> tuple[Any | None, str | None]:
        if not asyncio.iscoroutine(coro):
            return None, "invalid_coroutine"
        try:
            asyncio.get_running_loop()
            # Running loop in current thread: close coroutine explicitly to avoid
            # "coroutine was never awaited" RuntimeWarning.
            coro.close()
            return None, "event_loop_running"
        except RuntimeError:
            pass
        try:
            return asyncio.run(coro), None
        except Exception as exc:
            error_code = exc.__class__.__name__.lower()
            logger.warning(
                f"Search-space advisor L2 cache degraded: error_code={error_code} error={exc}"
            )
            return None, error_code

    def get(self, spec: AdvisorCacheSpec) -> tuple[dict[str, Any] | None, str]:
        """Return cached payload and hit layer marker (l1|l2|none)."""
        result = self.get_with_status(spec)
        return result.payload, result.layer_hit

    def get_with_status(self, spec: AdvisorCacheSpec) -> AdvisorCacheGetResult:
        """Return cache payload with observability status."""
        key = spec.cache_key()
        l1_payload = self._l1.get(key)
        if isinstance(l1_payload, dict):
            return AdvisorCacheGetResult(
                payload=l1_payload, layer_hit="l1", status="ok"
            )

        if self._store is None:
            return AdvisorCacheGetResult(
                payload=None, layer_hit="none", status="disabled"
            )

        l2_payload, error_code = self._run_async(
            self._store.load_advisor_cache(
                study_name=spec.study_name,
                dataset_fingerprint=spec.dataset_fingerprint,
                direction=spec.direction,
                advisor_version=spec.advisor_version,
                last_trial=spec.last_trial,
                search_space_hash=spec.search_space_hash,
                objective_schema_hash=spec.objective_schema_hash,
            )
        )
        if error_code is not None:
            if error_code == "event_loop_running":
                return AdvisorCacheGetResult(
                    payload=None,
                    layer_hit="none",
                    status="disabled",
                    error_code=error_code,
                )
            logger.warning(
                "Search-space advisor L2 cache read degraded: "
                f"error_code={error_code} study_name={spec.study_name}"
            )
            return AdvisorCacheGetResult(
                payload=None,
                layer_hit="none",
                status="degraded",
                error_code=error_code,
            )
        if isinstance(l2_payload, dict):
            self._l1.set(
                key, l2_payload, ttl=self.ttl_seconds, tags=[f"study:{spec.study_name}"]
            )
            return AdvisorCacheGetResult(
                payload=l2_payload, layer_hit="l2", status="ok"
            )

        return AdvisorCacheGetResult(payload=None, layer_hit="none", status="ok")

    def set(self, spec: AdvisorCacheSpec, payload: dict[str, Any]) -> None:
        """Store payload on L1 and best-effort L2."""
        self.set_with_status(spec, payload)

    def set_with_status(
        self, spec: AdvisorCacheSpec, payload: dict[str, Any]
    ) -> AdvisorCacheWriteResult:
        """Store payload on L1/L2 and return write status."""
        key = spec.cache_key()
        self._l1.set(
            key, payload, ttl=self.ttl_seconds, tags=[f"study:{spec.study_name}"]
        )

        if self._store is None:
            return AdvisorCacheWriteResult(status="disabled")

        _, error_code = self._run_async(
            self._store.upsert_advisor_cache(
                study_name=spec.study_name,
                dataset_fingerprint=spec.dataset_fingerprint,
                direction=spec.direction,
                advisor_version=spec.advisor_version,
                last_trial=spec.last_trial,
                search_space_hash=spec.search_space_hash,
                objective_schema_hash=spec.objective_schema_hash,
                payload=payload,
                ttl_seconds=self.ttl_seconds,
            )
        )
        if error_code is not None:
            if error_code == "event_loop_running":
                return AdvisorCacheWriteResult(status="disabled", error_code=error_code)
            logger.warning(
                "Search-space advisor L2 cache write degraded: "
                f"error_code={error_code} study_name={spec.study_name}"
            )
            return AdvisorCacheWriteResult(status="degraded", error_code=error_code)
        return AdvisorCacheWriteResult(status="ok")


__all__ = [
    "AdvisorCache",
    "AdvisorCacheGetResult",
    "AdvisorCacheSpec",
    "AdvisorCacheWriteResult",
    "compute_cache_key",
    "build_objective_schema_hash",
    "build_search_space_hash",
]
