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

import ctypes
import importlib
import math
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pff.shared import logger
from pff.shared.acceleration.faiss_utils import import_faiss
from pff.shared.core.config import DSLFM_CONFIG_PATH
from pff.shared.core.config_loader import load_config

faiss = None
FAISS_AVAILABLE = False
SCANN_AVAILABLE = False
CUVS_AVAILABLE = False
_CUVS_RUNTIME_PREPARED = False


def _prepare_cuvs_runtime() -> None:
    """Load cuVS/CUDA shared libs for Python runtime on Linux."""
    global _CUVS_RUNTIME_PREPARED
    if _CUVS_RUNTIME_PREPARED or sys.platform != "linux":
        return

    pyver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    site_paths = [
        Path(sys.prefix) / "lib" / pyver / "site-packages",
        Path(sys.prefix) / "lib64" / pyver / "site-packages",
    ]
    shared_libs: list[Path] = []
    for base in site_paths:
        shared_libs.extend(
            [
                base / "nvidia" / "cublas" / "lib" / "libcublas.so.12",
                base / "nvidia" / "curand" / "lib" / "libcurand.so.10",
                base / "nvidia" / "cusolver" / "lib" / "libcusolver.so.11",
                base / "nvidia" / "cusparse" / "lib" / "libcusparse.so.12",
                base / "nvidia" / "nvjitlink" / "lib" / "libnvJitLink.so.12",
                base / "nvidia" / "nccl" / "lib" / "libnccl.so.2",
                base / "rapids_logger" / "lib64" / "librapids_logger.so",
                base / "librmm" / "lib64" / "librmm.so",
                base / "libraft" / "lib64" / "libraft.so",
                base / "libcuvs" / "lib64" / "libcuvs_c.so",
            ]
        )
    existing = [path for path in shared_libs if path.exists()]
    for so_file in existing:
        ctypes.CDLL(str(so_file), mode=ctypes.RTLD_GLOBAL)

    existing_dirs = sorted({str(path.parent) for path in existing})
    if existing_dirs:
        old = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = ":".join(existing_dirs + ([old] if old else []))

    _CUVS_RUNTIME_PREPARED = True


def _try_enable_faiss() -> bool:
    """Execute try enable faiss.



    Returns:

        Return value produced by the callable.

    """

    global faiss, FAISS_AVAILABLE
    if faiss is None:
        faiss_module, available = import_faiss()
        faiss = faiss_module
        FAISS_AVAILABLE = available
    return bool(FAISS_AVAILABLE)


def _ensure_faiss_available() -> None:
    """Execute ensure faiss available.



    Raises:

        Exception: Propagates domain-specific failures with context.

    """

    if not _try_enable_faiss():
        raise ImportError("FAISS not available. Install with: pip install faiss-cpu")


def _try_enable_scann() -> bool:
    """Return whether ScaNN is importable."""
    global SCANN_AVAILABLE
    if SCANN_AVAILABLE:
        return True
    try:
        importlib.import_module("scann")
        SCANN_AVAILABLE = True
    except Exception:
        SCANN_AVAILABLE = False
    return SCANN_AVAILABLE


def _try_enable_cuvs() -> bool:
    """Return whether cuVS Python bindings are importable."""
    global CUVS_AVAILABLE
    if CUVS_AVAILABLE:
        return True
    try:
        _prepare_cuvs_runtime()
        importlib.import_module("cuvs.neighbors")
        CUVS_AVAILABLE = True
    except Exception:
        CUVS_AVAILABLE = False
    return CUVS_AVAILABLE


def ann_backend_available(backend: str) -> bool:
    """Check whether a specific ANN backend is available."""
    backend_norm = backend.lower().strip()
    if backend_norm == "faiss":
        return _try_enable_faiss()
    if backend_norm == "scann":
        return _try_enable_scann()
    if backend_norm == "cuvs":
        return _try_enable_cuvs()
    return False


def _load_ann_defaults() -> dict[str, Any]:
    """Execute load ann defaults.



    Returns:

        Return value produced by the callable.



    Raises:

        Exception: Propagates domain-specific failures with context.

    """

    config = load_config(DSLFM_CONFIG_PATH)
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

    backend: str = field(default_factory=lambda: str(_ann_defaults().get("backend", "faiss")))
    index_type: str = field(default_factory=lambda: str(_ann_defaults().get("index_type", "flat")))
    metric: str = field(default_factory=lambda: str(_ann_defaults().get("metric", "ip")))
    nlist: int = field(default_factory=lambda: int(_ann_defaults().get("nlist", 100)))
    nprobe: int = field(default_factory=lambda: int(_ann_defaults().get("nprobe", 10)))
    ef_search: int = field(default_factory=lambda: int(_ann_defaults().get("ef_search", 64)))
    ef_construction: int = field(
        default_factory=lambda: int(_ann_defaults().get("ef_construction", 200))
    )
    M: int = field(default_factory=lambda: int(_ann_defaults().get("m", 32)))
    pq_bits: int = field(default_factory=lambda: int(_ann_defaults().get("pq_bits", 8)))
    scann_num_leaves: int = field(
        default_factory=lambda: int(_ann_defaults().get("scann_num_leaves", 0))
    )
    scann_num_leaves_to_search: int = field(
        default_factory=lambda: int(_ann_defaults().get("scann_num_leaves_to_search", 0))
    )
    scann_reorder_k: int = field(
        default_factory=lambda: int(_ann_defaults().get("scann_reorder_k", 0))
    )
    cagra_graph_degree: int = field(
        default_factory=lambda: int(_ann_defaults().get("cagra_graph_degree", 32))
    )
    cagra_itopk_size: int = field(
        default_factory=lambda: int(_ann_defaults().get("cagra_itopk_size", 64))
    )
    cagra_search_width: int = field(
        default_factory=lambda: int(_ann_defaults().get("cagra_search_width", 1))
    )
    cagra_algo: str = field(default_factory=lambda: str(_ann_defaults().get("cagra_algo", "auto")))
    use_gpu: bool = field(default_factory=lambda: bool(_ann_defaults().get("use_gpu", False)))
    threshold_entities: int = field(
        default_factory=lambda: int(_ann_defaults().get("threshold_entities", 50000))
    )

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> ANNConfig:
        """Build ANNConfig from mapping with config defaults."""
        defaults = _ann_defaults()
        return cls(
            backend=str(data.get("backend", defaults.get("backend", "faiss"))),
            index_type=str(data.get("index_type", defaults.get("index_type", "flat"))),
            metric=str(data.get("metric", defaults.get("metric", "ip"))),
            nlist=int(data.get("nlist", defaults.get("nlist", 100))),
            nprobe=int(data.get("nprobe", defaults.get("nprobe", 10))),
            ef_search=int(data.get("ef_search", defaults.get("ef_search", 64))),
            ef_construction=int(data.get("ef_construction", defaults.get("ef_construction", 200))),
            M=int(data.get("M", data.get("m", defaults.get("m", 32)))),
            pq_bits=int(data.get("pq_bits", defaults.get("pq_bits", 8))),
            scann_num_leaves=int(data.get("scann_num_leaves", defaults.get("scann_num_leaves", 0))),
            scann_num_leaves_to_search=int(
                data.get(
                    "scann_num_leaves_to_search",
                    defaults.get("scann_num_leaves_to_search", 0),
                )
            ),
            scann_reorder_k=int(data.get("scann_reorder_k", defaults.get("scann_reorder_k", 0))),
            cagra_graph_degree=int(
                data.get("cagra_graph_degree", defaults.get("cagra_graph_degree", 32))
            ),
            cagra_itopk_size=int(
                data.get("cagra_itopk_size", defaults.get("cagra_itopk_size", 64))
            ),
            cagra_search_width=int(
                data.get("cagra_search_width", defaults.get("cagra_search_width", 1))
            ),
            cagra_algo=str(data.get("cagra_algo", defaults.get("cagra_algo", "auto"))),
            use_gpu=bool(data.get("use_gpu", defaults.get("use_gpu", False))),
            threshold_entities=int(
                data.get("threshold_entities", defaults.get("threshold_entities", 50000)),
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
    cfg = config or ANNConfig.from_defaults()
    if not ann_backend_available(cfg.backend):
        return False
    threshold = cfg.threshold_entities
    return num_entities >= threshold


def build_faiss_index(
    embeddings: np.ndarray,
    config: ANNConfig,
    *,
    metric: str | None = None,
) -> tuple[Any, bool]:
    """Build and populate a FAISS index from float32 embeddings."""
    _ensure_faiss_available()
    assert faiss is not None

    num_entities, dim = embeddings.shape
    metric_name = (metric or config.metric).lower()
    index_type = config.index_type.lower()

    if index_type == "flat":
        cpu_index = faiss.IndexFlatIP(dim) if metric_name == "ip" else faiss.IndexFlatL2(dim)
    elif index_type == "ivf":
        quantizer = faiss.IndexFlatL2(dim)
        # FAISS recommends at least 39 * nlist training points for IVF
        max_nlist = max(1, num_entities // 39)
        nlist = max(1, min(config.nlist, max_nlist))
        if nlist < config.nlist:
            logger.info(
                f"Reduzindo IVF nlist de {config.nlist} para {nlist} (num_entities={num_entities})"
            )
        cpu_index = faiss.IndexIVFFlat(quantizer, dim, nlist)
        cpu_index.train(embeddings)
        cpu_index.nprobe = min(config.nprobe, nlist)
    elif index_type == "ivfpq":
        quantizer = faiss.IndexFlatL2(dim)
        # FAISS recommends at least 39 * nlist training points for IVF
        max_nlist = max(1, num_entities // 39)
        nlist = max(1, min(config.nlist, max_nlist))
        if nlist < config.nlist:
            logger.debug(
                f"Reducing IVF-PQ nlist from {config.nlist} to {nlist} (num_entities={num_entities})"
            )
        # PQ codebook training needs enough samples for 2**pq_bits centroids.
        # Bound pq_bits to avoid FAISS KMeans warnings on small datasets.
        max_centroids = max(2, num_entities // 39)
        safe_pq_bits = max(1, int(math.floor(math.log2(max_centroids))))
        pq_bits = min(int(config.pq_bits), safe_pq_bits)
        if pq_bits < int(config.pq_bits):
            logger.info(
                f"Reduzindo IVF-PQ pq_bits de {config.pq_bits} para {pq_bits} "
                f"(num_entities={num_entities})"
            )
        cpu_index = faiss.IndexIVFPQ(quantizer, dim, nlist, config.M, pq_bits)
        cpu_index.train(embeddings)
        cpu_index.nprobe = min(config.nprobe, nlist)
    elif index_type == "hnsw":
        cpu_index = faiss.IndexHNSWFlat(dim, config.M)
        cpu_index.hnsw.efConstruction = config.ef_construction
        cpu_index.hnsw.efSearch = config.ef_search
    else:
        raise ValueError(f"Unknown index type: {config.index_type}")

    use_gpu = config.use_gpu and faiss.get_num_gpus() > 0 and index_type != "hnsw"
    index = cpu_index
    if use_gpu:
        try:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        except Exception as e:
            logger.warning(f"Failed to move FAISS to GPU: {e}. Using CPU.")
            use_gpu = False
            index = cpu_index

    index.add(embeddings)
    return index, use_gpu


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
        self.embedding_dim = embedding_dim
        self.config = config or ANNConfig.from_defaults()
        if not ann_backend_available(self.config.backend):
            raise ImportError(
                f"ANN backend '{self.config.backend}' is not available in this environment."
            )
        self.index: Any | None = None
        self._num_entities = 0

    def build_index(self, embeddings: torch.Tensor | np.ndarray) -> None:
        """Build FAISS index from entity embeddings.

        Args:
            embeddings: Entity embeddings of shape (num_entities, dim).
        """
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().numpy().astype(np.float32)
        else:
            embeddings = embeddings.astype(np.float32)

        self._num_entities = embeddings.shape[0]
        d = embeddings.shape[1]

        backend = self.config.backend.lower()
        if backend != "faiss":
            raise ValueError(
                "ANNEvaluator currently supports only backend='faiss'. "
                "DSLFM runtime ANN path supports additional backends."
            )
        metric = self.config.metric.lower()
        normalize = metric == "ip" and self.config.index_type.lower() in {"ivf", "ivfpq", "hnsw"}
        if normalize:
            _ensure_faiss_available()
            assert faiss is not None
            faiss.normalize_L2(embeddings)
        self.index, use_gpu = build_faiss_index(embeddings, self.config, metric=metric)
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
            query_embeddings = query_embeddings.detach().cpu().numpy().astype(np.float32)
        if isinstance(target_indices, torch.Tensor):
            target_indices = target_indices.detach().cpu().numpy()

        if self.config.metric.lower() == "ip" and self.config.index_type.lower() in {
            "ivf",
            "ivfpq",
            "hnsw",
        }:
            if faiss is None:
                raise RuntimeError("FAISS backend unavailable for L2 normalization")
            faiss.normalize_L2(query_embeddings)

        k_search = min(k, self._num_entities)
        _, indices = self.index.search(query_embeddings, k_search)

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
    config = config or ANNConfig.from_defaults()
    if not ann_backend_available(config.backend):
        logger.debug(f"ANN backend '{config.backend}' not available, using exact ranking")
        return None
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
    "ann_backend_available",
    "_prepare_cuvs_runtime",
    "should_use_ann",
    "create_ann_evaluator",
    "build_faiss_index",
    "FAISS_AVAILABLE",
]
