"""FAISS-based Approximate Nearest Neighbor evaluator for fast MRR/Hits@K.

Provides 10-100x speedup over exact ranking for large KGs (>50k entities)
by using FAISS indexing for approximate nearest neighbor search.

Design Patterns:
    - Strategy: Supports multiple FAISS index types (Flat, IVF, HNSW)
    - Facade: Simplified interface for link prediction evaluation

Author: PFF Team
Date: 2025-12
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from pff.shared import logger
from pff.shared.acceleration.faiss_utils import import_faiss
from pff.shared.core.config import DSLFM_CONFIG_PATH
from pff.shared.core.config_loader import load_config

faiss = None
FAISS_AVAILABLE = False


def _try_enable_faiss() -> bool:
    global faiss, FAISS_AVAILABLE
    if faiss is None:
        faiss_module, available = import_faiss()
        faiss = faiss_module
        FAISS_AVAILABLE = available
    return bool(FAISS_AVAILABLE)


def _ensure_faiss_available() -> None:
    if not _try_enable_faiss():
        raise ImportError("FAISS not available. Install with: pip install faiss-cpu")


def _load_ann_defaults() -> dict[str, Any]:
    config = load_config(DSLFM_CONFIG_PATH)
    if not isinstance(config, Mapping):
        raise ValueError(
            f"ANN config payload must be a mapping (path={DSLFM_CONFIG_PATH})"
        )
    ann_cfg = config.get("ann", {})
    if ann_cfg is None:
        return {}
    if not isinstance(ann_cfg, Mapping):
        raise ValueError("ANN config section must be a mapping")
    return dict(ann_cfg)


def _ann_defaults() -> dict[str, Any]:
    return _load_ann_defaults()


@dataclass
class ANNConfig:
    """Configuration for ANN evaluation."""

    index_type: str = field(
        default_factory=lambda: str(_ann_defaults().get("index_type", "flat"))
    )
    nlist: int = field(default_factory=lambda: int(_ann_defaults().get("nlist", 100)))
    nprobe: int = field(default_factory=lambda: int(_ann_defaults().get("nprobe", 10)))
    ef_search: int = field(
        default_factory=lambda: int(_ann_defaults().get("ef_search", 64))
    )
    ef_construction: int = field(
        default_factory=lambda: int(_ann_defaults().get("ef_construction", 200))
    )
    M: int = field(default_factory=lambda: int(_ann_defaults().get("m", 32)))
    use_gpu: bool = field(
        default_factory=lambda: bool(_ann_defaults().get("use_gpu", False))
    )
    threshold_entities: int = field(
        default_factory=lambda: int(_ann_defaults().get("threshold_entities", 50000))
    )

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> ANNConfig:
        """Build ANNConfig from mapping with config defaults."""
        defaults = _ann_defaults()
        return cls(
            index_type=str(data.get("index_type", defaults.get("index_type", "flat"))),
            nlist=int(data.get("nlist", defaults.get("nlist", 100))),
            nprobe=int(data.get("nprobe", defaults.get("nprobe", 10))),
            ef_search=int(data.get("ef_search", defaults.get("ef_search", 64))),
            ef_construction=int(
                data.get("ef_construction", defaults.get("ef_construction", 200))
            ),
            M=int(data.get("M", data.get("m", defaults.get("m", 32)))),
            use_gpu=bool(data.get("use_gpu", defaults.get("use_gpu", False))),
            threshold_entities=int(
                data.get(
                    "threshold_entities", defaults.get("threshold_entities", 50000)
                ),
            ),
        )

    @classmethod
    def from_defaults(cls) -> ANNConfig:
        """Build ANNConfig from config defaults."""
        return cls.from_mapping(_ann_defaults())


def should_use_ann(num_entities: int, config: ANNConfig | None = None) -> bool:
    """Check if ANN should be used based on entity count.

    Args:
        num_entities: Number of entities in the KG.
        config: Optional ANN config.

    Returns:
        True if ANN should be used for faster evaluation.
    """
    if not _try_enable_faiss():
        return False
    cfg = config or ANNConfig.from_defaults()
    threshold = cfg.threshold_entities
    return num_entities >= threshold


class ANNEvaluator:
    """FAISS-based ANN evaluator for link prediction.

    Builds an index of entity embeddings and uses approximate
    nearest neighbor search for fast ranking.
    """

    def __init__(
        self,
        embedding_dim: int,
        config: ANNConfig | None = None,
    ) -> None:
        """Initialize the ANN evaluator.

        Args:
            embedding_dim: Dimension of entity embeddings.
            config: ANN configuration.
        """
        _ensure_faiss_available()

        self.embedding_dim = embedding_dim
        self.config = config or ANNConfig.from_defaults()
        self.index: Any | None = None
        self._num_entities = 0

    def build_index(self, embeddings: torch.Tensor | np.ndarray) -> None:
        """Build FAISS index from entity embeddings.

        Args:
            embeddings: Entity embeddings of shape (num_entities, dim).
        """
        _ensure_faiss_available()
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().numpy().astype(np.float32)
        else:
            embeddings = embeddings.astype(np.float32)

        self._num_entities = embeddings.shape[0]
        d = embeddings.shape[1]

        assert faiss is not None
        use_gpu = self.config.use_gpu and faiss.get_num_gpus() > 0

        if self.config.index_type == "flat":
            cpu_index = faiss.IndexFlatL2(d)
        elif self.config.index_type == "ivf":
            quantizer = faiss.IndexFlatL2(d)
            nlist = min(self.config.nlist, self._num_entities // 10)
            cpu_index = faiss.IndexIVFFlat(quantizer, d, max(1, nlist))
            cpu_index.train(embeddings)
            cpu_index.nprobe = self.config.nprobe
        elif self.config.index_type == "hnsw":
            cpu_index = faiss.IndexHNSWFlat(d, self.config.M)
            cpu_index.hnsw.efConstruction = self.config.ef_construction
            cpu_index.hnsw.efSearch = self.config.ef_search
            use_gpu = False
        else:
            raise ValueError(f"Unknown index type: {self.config.index_type}")

        if use_gpu:
            try:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
                logger.debug("FAISS index moved to GPU 0")
            except Exception as e:
                logger.warning(f"Failed to move FAISS to GPU: {e}. Using CPU.")
                self.index = cpu_index
        else:
            self.index = cpu_index

        assert self.index is not None
        self.index.add(embeddings)
        gpu_str = " (GPU)" if use_gpu else ""
        logger.debug(
            f"Built FAISS {self.config.index_type} index{gpu_str}: "
            f"{self._num_entities:,} entities, dim={d}"
        )

    def compute_ranks(
        self,
        query_embeddings: torch.Tensor | np.ndarray,
        target_indices: torch.Tensor | np.ndarray,
        k: int = 100,
    ) -> np.ndarray:
        """Compute ranks of target entities using ANN search.

        Args:
            query_embeddings: Query embeddings of shape (batch, dim).
            target_indices: True target entity indices of shape (batch,).
            k: Number of neighbors to retrieve.

        Returns:
            Array of ranks (1-indexed) for each query.
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index first.")

        if isinstance(query_embeddings, torch.Tensor):
            query_embeddings = (
                query_embeddings.detach().cpu().numpy().astype(np.float32)
            )
        if isinstance(target_indices, torch.Tensor):
            target_indices = target_indices.detach().cpu().numpy()

        k_search = min(k, self._num_entities)
        distances, indices = self.index.search(query_embeddings, k_search)

        matches = indices == target_indices[:, None]
        has_match = matches.any(axis=1)
        pos = matches.argmax(axis=1)
        ranks = np.where(has_match, pos + 1, k_search + 1).astype(np.int32)

        return ranks

    def compute_mrr_hits(
        self,
        query_embeddings: torch.Tensor | np.ndarray,
        target_indices: torch.Tensor | np.ndarray,
        k_values: list[int] = [1, 3, 10],
    ) -> dict[str, float]:
        """Compute MRR and Hits@K metrics using ANN.

        Args:
            query_embeddings: Query embeddings.
            target_indices: True target indices.
            k_values: K values for Hits@K.

        Returns:
            Dict with MRR and Hits@K metrics.
        """
        max_k = max(k_values)
        ranks = self.compute_ranks(query_embeddings, target_indices, k=max_k)

        mrr = np.mean(1.0 / ranks.astype(np.float32))

        metrics = {"mrr": float(mrr)}
        for k in k_values:
            hits_k = np.mean(ranks <= k)
            metrics[f"hits@{k}"] = float(hits_k)

        return metrics


def create_ann_evaluator(
    embeddings: torch.Tensor,
    config: ANNConfig | None = None,
) -> ANNEvaluator | None:
    """Factory function to create an ANN evaluator.

    Returns None if FAISS unavailable or entity count below threshold.

    Args:
        embeddings: Entity embeddings.
        config: Optional ANN config.

    Returns:
        ANNEvaluator or None.
    """
    if not FAISS_AVAILABLE:
        logger.debug("FAISS not available, using exact ranking")
        return None

    config = config or ANNConfig.from_defaults()
    num_entities = embeddings.shape[0]

    if not should_use_ann(num_entities, config):
        logger.debug(f"Only {num_entities:,} entities, using exact ranking")
        return None

    evaluator = ANNEvaluator(embeddings.shape[1], config)
    evaluator.build_index(embeddings)

    return evaluator


__all__ = [
    "ANNEvaluator",
    "ANNConfig",
    "should_use_ann",
    "create_ann_evaluator",
    "FAISS_AVAILABLE",
]
