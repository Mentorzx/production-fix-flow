"""
Embedding Cache Module for HPO Trials.

Provides caching utilities for KGE embeddings to avoid redundant computation
when embedding hyperparameters are identical between trials.

Uses the existing pff/utils/core/cache.py infrastructure rather than
reimplementing cache logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pff.shared import logger
from pff.shared.core.cache import DiskCache
from pff.shared.core.config import CACHE_CONFIG_PATH, settings
from pff.shared.core.config_loader import load_config
from pff.shared.core.file_manager import FileManager, ParquetBundle
from pff.shared.hash import stable_hash


def _resolve_embedding_cache_purge_seconds(
    file_manager: FileManager | None = None,
) -> int | None:
    """Resolve purge policy for the embedding cache from `config/infra/cache.yaml`.

    Args:
        file_manager: Optional FileManager dependency.

    Returns:
        Purge age in seconds, or None to use DiskCache defaults.
    """
    cfg = load_config(CACHE_CONFIG_PATH)
    if not cfg:
        return None
    ttl_days = cfg.get("template_ttl_days")
    if ttl_days is None:
        return None
    try:
        ttl_days_int = int(ttl_days)
    except Exception as exc:
        logger.warning(f"Invalid cache TTL days: value={ttl_days!r} error={exc}")
        return None
    return max(0, ttl_days_int) * 24 * 3600


_embedding_disk_cache = DiskCache(
    root=settings.CACHE_DIR / "embeddings",
    purge_older_than=_resolve_embedding_cache_purge_seconds(),
)


@dataclass(frozen=True)
class EmbeddingCacheKey:
    """Immutable key for embedding cache lookup.

    Contains all hyperparameters that affect embedding computation.
    """

    embedding_dim: int
    dslfm_epochs: int
    batch_size: int
    learning_rate: float
    negative_sample_size: int
    self_adversarial: bool
    adversarial_temperature: float
    data_hash: str

    def to_hash(self) -> str:
        """Generate stable hash for cache key."""
        key_dict = {
            "embedding_dim": self.embedding_dim,
            "dslfm_epochs": self.dslfm_epochs,
            "batch_size": self.batch_size,
            "learning_rate": round(self.learning_rate, 10),
            "negative_sample_size": self.negative_sample_size,
            "self_adversarial": self.self_adversarial,
            "adversarial_temperature": round(self.adversarial_temperature, 6),
            "data_hash": self.data_hash,
        }
        return str(stable_hash(key_dict))


def create_cache_key_from_params(
    params: dict[str, Any],
    data_hash: str,
) -> EmbeddingCacheKey:
    """Create cache key from trial parameters.

    Args:
        params: Trial hyperparameters dictionary.
        data_hash: Hash of training data for cache invalidation.

    Returns:
        Immutable cache key.
    """
    return EmbeddingCacheKey(
        embedding_dim=int(params["embedding_dim"]),
        dslfm_epochs=int(params["dslfm_epochs"]),
        batch_size=int(params["batch_size"]),
        learning_rate=float(params["learning_rate"]),
        negative_sample_size=int(params["negative_sample_size"]),
        self_adversarial=bool(params.get("self_adversarial", False)),
        adversarial_temperature=float(params["adversarial_temperature"]),
        data_hash=data_hash,
    )


def compute_data_hash(train_df: Any, valid_df: Any) -> str:
    """Compute hash of training data for cache invalidation.

    Args:
        train_df: Training DataFrame (Polars).
        valid_df: Validation DataFrame (Polars).

    Returns:
        Stable hash string.
    """
    train_shape = train_df.shape if hasattr(train_df, "shape") else (0, 0)
    valid_shape = valid_df.shape if hasattr(valid_df, "shape") else (0, 0)

    data_info = {
        "train_shape": train_shape,
        "valid_shape": valid_shape,
        "train_rows": len(train_df) if hasattr(train_df, "__len__") else 0,
        "valid_rows": len(valid_df) if hasattr(valid_df, "__len__") else 0,
    }

    return str(stable_hash(data_info))


class EmbeddingCache:
    """
    Embedding cache wrapper using DiskCache from pff/utils.

    Provides a simplified interface for caching DSLFM embeddings.
    """

    _instance: EmbeddingCache | None = None

    def __init__(self) -> None:
        self._disk_cache = _embedding_disk_cache
        self._hits = 0
        self._misses = 0

    @classmethod
    def get_instance(cls, **kwargs: Any) -> EmbeddingCache:
        """Get or create singleton cache instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance (for testing)."""
        cls._instance = None

    def get_cache_path(self, key: EmbeddingCacheKey) -> Path:
        """Get disk path for a cache key."""
        key_hash = key.to_hash()
        return self._disk_cache.root / f"emb_{key_hash}.pkl"

    def has(self, key: EmbeddingCacheKey) -> bool:
        """Check if embeddings exist in cache."""
        return FileManager.exists(self.get_cache_path(key))

    def get(self, key: EmbeddingCacheKey) -> dict[str, Any] | None:
        """Get cached embeddings."""
        cache_path = self.get_cache_path(key)
        if not FileManager.exists(cache_path):
            self._misses += 1
            return None

        try:
            payload = FileManager.read(cache_path)
            data = payload.to_native() if isinstance(payload, ParquetBundle) else payload
            self._hits += 1
            logger.info(
                f"Cache de embeddings HIT: dim={key.embedding_dim}, epochs={key.dslfm_epochs}"
            )
            return data
        except Exception as exc:
            logger.warning(f"Failed to load embedding cache: {exc}")
            self._misses += 1
            return None

    def put(self, key: EmbeddingCacheKey, data: dict[str, Any]) -> None:
        """Store embeddings in cache."""
        cache_path = self.get_cache_path(key)
        try:
            FileManager.save(data, cache_path)
            logger.info(f"Embeddings cacheados: dim={key.embedding_dim}, epochs={key.dslfm_epochs}")
        except Exception as exc:
            logger.warning(f"Failed to save embedding cache: {exc}")

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0.0,
        }

    def clear(self) -> None:
        """Clear all cache entries."""
        FileManager.delete_directory(self._disk_cache.root, ignore_errors=True)
        FileManager.ensure_dir(self._disk_cache.root)
        logger.info("Cache de embeddings limpo")


__all__ = [
    "EmbeddingCache",
    "EmbeddingCacheKey",
    "create_cache_key_from_params",
    "compute_data_hash",
]
