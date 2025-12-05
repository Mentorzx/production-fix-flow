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

from pff import settings
from pff.utils import logger
from pff.utils.core.cache import DiskCache
from pff.utils.core.file_manager import FileManager
from pff.utils.hash import stable_hash


_embedding_disk_cache = DiskCache(
    root=settings.OUTPUTS_DIR / "cache" / "embeddings",
    purge_older_than=7 * 24 * 3600,
)


@dataclass(frozen=True)
class EmbeddingCacheKey:
    """Immutable key for embedding cache lookup.
    
    Contains all hyperparameters that affect embedding computation.
    """
    embedding_dim: int
    gamma: float
    epsilon: float
    regularization_weight: float
    rotate_epochs: int
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
            "gamma": round(self.gamma, 6),
            "epsilon": round(self.epsilon, 6),
            "regularization_weight": round(self.regularization_weight, 10),
            "rotate_epochs": self.rotate_epochs,
            "batch_size": self.batch_size,
            "learning_rate": round(self.learning_rate, 10),
            "negative_sample_size": self.negative_sample_size,
            "self_adversarial": self.self_adversarial,
            "adversarial_temperature": round(self.adversarial_temperature, 6),
            "data_hash": self.data_hash,
        }
        return stable_hash(key_dict)


def create_cache_key_from_params(
    params: dict[str, Any],
    data_hash: str,
) -> EmbeddingCacheKey:
    """
    Create cache key from trial parameters.
    
    Args:
        params: Trial hyperparameters dictionary
        data_hash: Hash of training data for cache invalidation
        
    Returns:
        Immutable cache key
    """
    return EmbeddingCacheKey(
        embedding_dim=int(params.get("embedding_dim", 256)),
        gamma=float(params.get("gamma", 12.0)),
        epsilon=float(params.get("epsilon", 2.0)),
        regularization_weight=float(params.get("regularization_weight", 1e-5)),
        rotate_epochs=int(params.get("rotate_epochs", 100)),
        batch_size=int(params.get("batch_size", 512)),
        learning_rate=float(params.get("meta_learning_rate", 0.0001)),
        negative_sample_size=int(params.get("negative_sample_size", 256)),
        self_adversarial=bool(params.get("self_adversarial", True)),
        adversarial_temperature=float(params.get("adversarial_temperature", 1.0)),
        data_hash=data_hash,
    )


def compute_data_hash(train_df: Any, valid_df: Any) -> str:
    """
    Compute hash of training data for cache invalidation.
    
    Args:
        train_df: Training DataFrame (Polars)
        valid_df: Validation DataFrame (Polars)
        
    Returns:
        Stable hash string
    """
    train_shape = train_df.shape if hasattr(train_df, "shape") else (0, 0)
    valid_shape = valid_df.shape if hasattr(valid_df, "shape") else (0, 0)
    
    data_info = {
        "train_shape": train_shape,
        "valid_shape": valid_shape,
        "train_rows": len(train_df) if hasattr(train_df, "__len__") else 0,
        "valid_rows": len(valid_df) if hasattr(valid_df, "__len__") else 0,
    }
    
    return stable_hash(data_info)


class EmbeddingCache:
    """
    Embedding cache wrapper using DiskCache from pff/utils.
    
    Provides a simplified interface for caching RotatE embeddings.
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
        return self.get_cache_path(key).exists()
    
    def get(self, key: EmbeddingCacheKey) -> dict[str, Any] | None:
        """Get cached embeddings."""
        cache_path = self.get_cache_path(key)
        if not cache_path.exists():
            self._misses += 1
            return None
        
        try:
            data = FileManager.read(cache_path)
            self._hits += 1
            logger.info(
                f"Cache de embeddings HIT: dim={key.embedding_dim}, "
                f"epochs={key.rotate_epochs}"
            )
            return data
        except Exception as e:
            logger.warning(f"Failed to load embedding cache: {e}")
            self._misses += 1
            return None
    
    def put(self, key: EmbeddingCacheKey, data: dict[str, Any]) -> None:
        """Store embeddings in cache."""
        cache_path = self.get_cache_path(key)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            FileManager.save(data, cache_path)
            logger.info(
                f"Embeddings cacheados: dim={key.embedding_dim}, "
                f"epochs={key.rotate_epochs}"
            )
        except Exception as e:
            logger.warning(f"Failed to save embedding cache: {e}")
    
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
        for path in self._disk_cache.root.glob("emb_*.pkl"):
            try:
                path.unlink()
            except Exception:
                pass
        logger.info("Cache de embeddings limpo")


__all__ = [
    "EmbeddingCache",
    "EmbeddingCacheKey", 
    "create_cache_key_from_params",
    "compute_data_hash",
]
