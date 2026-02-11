"""Approximate evaluation using FAISS for fast KG link prediction.

This module provides approximate nearest neighbor (ANN) based evaluation
for accelerating link prediction metrics (MRR, Hits@K) on large KGs.

Design Patterns:
    - Strategy: Different index types (IVF, HNSW) can be swapped
    - Facade: Simple interface hiding FAISS complexity

References:
    - FAISS: https://github.com/facebookresearch/faiss
    - Evaluation protocols: OGB leaderboard methodology
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from pff.shared.acceleration.faiss_utils import import_faiss
from pff.shared.core.logging import logger


class IndexType(str, Enum):
    """Available FAISS index types."""

    FLAT = "flat"
    IVF = "ivf"
    HNSW = "hnsw"


@dataclass
class EvaluatorConfig:
    """Configuration for approximate evaluator.

    Attributes:
        index_type: Type of FAISS index to use.
        top_k: Number of candidates to retrieve for refinement.
        nlist: Number of clusters for IVF index.
        nprobe: Number of clusters to search for IVF.
        ef_search: Search parameter for HNSW.
        use_gpu: Whether to use GPU for FAISS (requires faiss-gpu).
    """

    index_type: IndexType = IndexType.IVF
    top_k: int = 1024
    nlist: int = 100
    nprobe: int = 10
    ef_search: int = 64
    use_gpu: bool = False
    batch_size: int = 256


class BaseEvaluator(ABC):
    """Base class for link prediction evaluators."""

    @abstractmethod
    def evaluate(
        self,
        model: Any,
        test_triples: np.ndarray,
        all_entity_embeddings: np.ndarray,
    ) -> dict[str, float]:
        """Evaluate model on test triples.

        Args:
            model: Model to evaluate.
            test_triples: Test triples [N, 3] as (head, relation, tail).
            all_entity_embeddings: Entity embeddings [num_entities, dim].

        Returns:
            Dictionary with metrics (mrr, hits@1, hits@3, hits@10).
        """
        ...


class ExactEvaluator(BaseEvaluator):
    """Exact evaluation by ranking all entities.

    Slower but accurate. Use for small KGs or final validation.
    """

    def __init__(self, config: EvaluatorConfig | None = None) -> None:
        self.config = config or EvaluatorConfig()

    def evaluate(
        self,
        model: Any,
        test_triples: np.ndarray,
        all_entity_embeddings: np.ndarray,
    ) -> dict[str, float]:
        """Full ranking evaluation."""
        logger.info(f"Avaliacao exata iniciada: {len(test_triples)} triples")

        ranks_list = []

        for i in range(0, len(test_triples), self.config.batch_size):
            batch = test_triples[i : i + self.config.batch_size]
            heads = batch[:, 0]
            rels = batch[:, 1]
            tails = batch[:, 2]

            scores = self._score_all_tails_batch(model, heads, rels, all_entity_embeddings)

            import torch

            if isinstance(scores, torch.Tensor):
                tails_tensor = torch.as_tensor(tails, device=scores.device)

                true_scores = scores.gather(1, tails_tensor.unsqueeze(1)).squeeze(1)
                batch_ranks = (scores > true_scores.unsqueeze(1)).sum(dim=1) + 1
                ranks_list.append(batch_ranks)
            else:
                true_scores = scores[np.arange(scores.shape[0]), tails]
                batch_ranks = (scores > true_scores[:, None]).sum(axis=1) + 1
                ranks_list.append(batch_ranks)

        if not ranks_list:
            return self._compute_metrics(np.array([], dtype=np.int64))

        import torch

        if isinstance(ranks_list[0], torch.Tensor):
            ranks_tensor = torch.cat(ranks_list)
            return self._compute_metrics(ranks_tensor)

        ranks_arr = np.concatenate(ranks_list, axis=0)
        return self._compute_metrics(ranks_arr)

    def _score_all_tails_batch(
        self,
        model: Any,
        heads: np.ndarray,
        relations: np.ndarray,
        all_embeddings: np.ndarray,
    ) -> np.ndarray:
        """Score a batch of head-relation pairs against all tails."""
        scorer = getattr(model, "score_all_tails", None)
        if callable(scorer):
            return scorer(heads, relations, all_embeddings)

        raise NotImplementedError(
            f"Model {type(model).__name__} does not implement score_all_tails(). "
            "ExactEvaluator requires a model with this method for link prediction. "
            "Use DSLFMKGCModel.evaluate() directly or ensure your model exposes score_all_tails."
        )

    def _compute_metrics(self, ranks: np.ndarray | Any) -> dict[str, float]:
        """Compute standard link prediction metrics."""
        import torch

        if isinstance(ranks, torch.Tensor):
            mrr = torch.mean(1.0 / ranks.float()).item()
            hits1 = torch.mean((ranks <= 1).float()).item()
            hits3 = torch.mean((ranks <= 3).float()).item()
            hits10 = torch.mean((ranks <= 10).float()).item()
            return {
                "mrr": mrr,
                "hits@1": hits1,
                "hits@3": hits3,
                "hits@10": hits10,
            }

        return {
            "mrr": float(np.mean(1.0 / ranks)),
            "hits@1": float(np.mean(ranks <= 1)),
            "hits@3": float(np.mean(ranks <= 3)),
            "hits@10": float(np.mean(ranks <= 10)),
        }


class ApproximateEvaluator(BaseEvaluator):
    """Approximate evaluation using FAISS ANN search.

    Retrieves top-k candidates using FAISS index, then performs
    exact scoring only on candidates. Much faster for large KGs.

    Args:
        config: Evaluator configuration.
    """

    def __init__(self, config: EvaluatorConfig | None = None) -> None:
        self.config = config or EvaluatorConfig()
        self._index = None
        self._faiss = None
        self._index_built = False

    def _ensure_faiss(self) -> None:
        """Lazy import FAISS to avoid dependency issues."""
        if self._faiss is None:
            try:
                faiss, available = import_faiss()
                if not available:
                    raise ImportError("faiss import failed")
                self._faiss = faiss
                logger.info("FAISS carregado com sucesso")
            except ImportError as e:
                msg = "faiss-cpu not installed. Run: poetry add faiss-cpu"
                raise ImportError(msg) from e

    def build_index(
        self,
        embeddings: np.ndarray,
        normalize: bool = True,
    ) -> None:
        """Build FAISS index from entity embeddings.

        Args:
            embeddings: Entity embeddings [num_entities, dim].
            normalize: Whether to L2-normalize embeddings.
        """
        self._ensure_faiss()
        faiss = self._faiss
        assert faiss is not None, "FAISS should be loaded after _ensure_faiss()"

        num_entities, dim = embeddings.shape

        if normalize:
            embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)

        embeddings = embeddings.astype(np.float32)

        if self.config.index_type == IndexType.FLAT:
            self._index = faiss.IndexFlatIP(dim)

        elif self.config.index_type == IndexType.IVF:
            quantizer = faiss.IndexFlatIP(dim)
            nlist = min(self.config.nlist, num_entities // 10)
            self._index = faiss.IndexIVFFlat(quantizer, dim, nlist)
            self._index.train(embeddings)
            self._index.nprobe = self.config.nprobe

        elif self.config.index_type == IndexType.HNSW:
            self._index = faiss.IndexHNSWFlat(dim, 32)
            self._index.hnsw.efSearch = self.config.ef_search

        assert self._index is not None, "Index should be created"
        self._index.add(embeddings)
        self._index_built = True

        logger.info(
            f"Indice FAISS construido: {num_entities:,} entidades, "
            f"tipo={self.config.index_type.value}"
        )

    def search(
        self,
        queries: np.ndarray,
        k: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Search for nearest neighbors.

        Args:
            queries: Query embeddings [batch, dim].
            k: Number of neighbors to retrieve.

        Returns:
            Tuple of (distances [batch, k], indices [batch, k]).
        """
        if not self._index_built:
            msg = "Index not built. Call build_index() first."
            raise RuntimeError(msg)

        k = k or self.config.top_k
        queries = queries.astype(np.float32)

        assert self._index is not None
        distances, indices = self._index.search(queries, k)

        return distances, indices

    def evaluate(
        self,
        model: Any,
        test_triples: np.ndarray,
        all_entity_embeddings: np.ndarray,
    ) -> dict[str, float]:
        """Approximate evaluation with top-k refinement.

        Args:
            model: Model to evaluate (must have score_triples method).
            test_triples: Test triples [N, 3].
            all_entity_embeddings: Entity embeddings for index.

        Returns:
            Approximate metrics.
        """
        if not self._index_built:
            self.build_index(all_entity_embeddings)

        logger.info(
            f"Avaliacao aproximada iniciada: {len(test_triples)} triples, top-{self.config.top_k}"
        )

        ranks = []

        for i in range(0, len(test_triples), self.config.batch_size):
            batch = test_triples[i : i + self.config.batch_size]
            batch_ranks = self._evaluate_batch(model, batch, all_entity_embeddings)
            ranks.extend(batch_ranks)

        ranks_array = np.array(ranks)

        metrics = self._compute_metrics(ranks_array)
        logger.info(
            f"Avaliacao concluida: MRR={metrics['mrr']:.4f}, Hits@10={metrics['hits@10']:.4f}"
        )

        return metrics

    def _evaluate_batch(
        self,
        model: Any,
        batch: np.ndarray,
        all_embeddings: np.ndarray,
    ) -> list[int]:
        """Evaluate a batch of triples."""
        heads = batch[:, 0]
        tails = batch[:, 2]
        queries = all_embeddings[heads]
        _, candidates = self.search(queries, self.config.top_k)

        matches = candidates == tails[:, None]
        has_match = matches.any(axis=1)
        pos = matches.argmax(axis=1)
        ranks = np.where(has_match, pos + 1, self.config.top_k + 1).astype(np.int64)
        return ranks.tolist()

    def _compute_metrics(self, ranks: np.ndarray) -> dict[str, float]:
        """Compute standard link prediction metrics."""
        return {
            "mrr": float(np.mean(1.0 / ranks)),
            "hits@1": float(np.mean(ranks <= 1)),
            "hits@3": float(np.mean(ranks <= 3)),
            "hits@10": float(np.mean(ranks <= 10)),
        }


def get_evaluator(
    evaluator_type: str = "approximate",
    config: EvaluatorConfig | None = None,
) -> BaseEvaluator:
    """Factory function to create evaluators.

    Args:
        evaluator_type: Either "exact" or "approximate".
        config: Evaluator configuration.

    Returns:
        Configured evaluator instance.
    """
    if evaluator_type == "exact":
        return ExactEvaluator(config)
    elif evaluator_type == "approximate":
        return ApproximateEvaluator(config)
    else:
        msg = f"Unknown evaluator type: {evaluator_type}"
        raise ValueError(msg)
