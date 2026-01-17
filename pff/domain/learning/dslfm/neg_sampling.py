"""Negative sampling strategies for DSLFM-KGC contrastive learning.

This module provides pluggable negative sampling strategies following the
Strategy pattern. Samplers can be swapped via configuration without
modifying the model or decoder.

Design Patterns:
    - Strategy: Interchangeable sampling algorithms
    - Factory: Creation via get_negative_sampler()
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

import torch
import torch.nn.functional as F

from pff.shared.core.cache import CacheManager

NUMBA_AVAILABLE = False


class SamplerType(str, Enum):
    DEGREE_BASED = "degree_based"
    NSCACHING = "nscaching"
    UNIFORM = "uniform"


@dataclass
class SamplerConfig:
    """Configuration for negative samplers.

    Attributes:
        sampler_type: Type of sampler to use.
        temperature: Temperature for score weighting.
        alpha: Smoothing exponent for degree-based sampling.
        num_entities: Total number of entities (required for some samplers).
        num_triples: Total number of triples (for NSCaching tensor allocation).
        cache_size: Number of negative candidates per triple (NSCaching).
        sample_ratio: Ratio of cached vs random negatives (NSCaching).
    """

    sampler_type: SamplerType = SamplerType.DEGREE_BASED
    temperature: float = 1.0
    alpha: float = 0.75
    num_entities: int = 0
    num_triples: int = 0
    cache_size: int = 64
    sample_ratio: float = 0.5


class NegativeSamplerProtocol(Protocol):
    """Protocol for negative samplers."""

    def get_positive_negative_scores(
        self,
        all_scores: torch.Tensor,
        tails: torch.Tensor,
        known_positive_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract positive and negative scores AND negative IDs.

        Returns:
            Tuple of (pos_scores [batch], neg_scores [batch, num_neg], neg_ids [batch, num_neg]).
        """
        ...

    def sample_negatives(
        self,
        heads: torch.Tensor,
        relations: torch.Tensor,
        tails: torch.Tensor,
        num_negatives: int,
        triple_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample negative tail ids."""
        ...

    def update_cache(
        self,
        heads: torch.Tensor,
        relations: torch.Tensor,
        tails: torch.Tensor,
        neg_ids: torch.Tensor,
        neg_scores: torch.Tensor,
        triple_indices: torch.Tensor | None = None,
    ) -> None:
        """Update internal cache (if any) with sampled negatives and their scores."""
        ...

    def weight_negatives(
        self,
        neg_scores: torch.Tensor,
    ) -> torch.Tensor:
        """Compute importance weights for negative samples."""
        ...

    def contrastive_loss(
        self,
        pos_scores: torch.Tensor,
        neg_scores: torch.Tensor,
        temperature: float | torch.Tensor = 0.07,
    ) -> torch.Tensor:
        """Compute weighted contrastive loss."""
        ...


class BaseNegativeSampler(ABC):
    """Base class for negative samplers."""

    def __init__(self, config: SamplerConfig | None = None) -> None:
        self.config = config or SamplerConfig()

    def get_positive_negative_scores(
        self,
        all_scores: torch.Tensor,
        tails: torch.Tensor,
        known_positive_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract positive (diagonal) and negative (off-diagonal) scores and ids."""
        batch_size = all_scores.shape[0]
        pos_scores = all_scores.diag()
        neg_scores = all_scores.clone()
        neg_scores.fill_diagonal_(float("-inf"))

        neg_ids_all = tails.unsqueeze(0).expand(batch_size, batch_size)

        if known_positive_mask is not None:
            neg_scores = neg_scores.masked_fill(known_positive_mask, float("-inf"))

        flat_scores = neg_scores.reshape(-1)
        flat_ids = neg_ids_all.reshape(-1)

        keep_mask = torch.ones(
            batch_size * batch_size, dtype=torch.bool, device=neg_scores.device
        )
        diag_idx = torch.arange(
            0, batch_size * batch_size, batch_size + 1, device=neg_scores.device
        )
        keep_mask[diag_idx] = False

        neg_scores = flat_scores[keep_mask].view(batch_size, batch_size - 1)
        neg_ids = flat_ids[keep_mask].view(batch_size, batch_size - 1)

        return pos_scores, neg_scores, neg_ids

    def sample_negatives(
        self,
        heads: torch.Tensor,
        relations: torch.Tensor,
        tails: torch.Tensor,
        num_negatives: int,
        triple_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample negative tail ids (default: random)."""
        num_entities = self.config.num_entities
        if num_entities <= 0:
            return torch.empty((len(heads), 0), device=heads.device, dtype=torch.long)

        return torch.randint(
            0,
            num_entities,
            (len(heads), num_negatives),
            device=heads.device,
            dtype=torch.long,
        )

    def update_cache(
        self,
        heads: torch.Tensor,
        relations: torch.Tensor,
        tails: torch.Tensor,
        neg_ids: torch.Tensor,
        neg_scores: torch.Tensor,
        triple_indices: torch.Tensor | None = None,
    ) -> None:
        """Update internal cache (no-op by default)."""
        pass

    def save_persistence(self) -> None:
        """Save state to persistence (no-op by default)."""
        pass

    @abstractmethod
    def weight_negatives(self, neg_scores: torch.Tensor) -> torch.Tensor:
        """Compute importance weights for negative samples."""
        ...

    def contrastive_loss(
        self,
        pos_scores: torch.Tensor,
        neg_scores: torch.Tensor,
        temperature: float | torch.Tensor = 0.07,
    ) -> torch.Tensor:
        """Compute InfoNCE contrastive loss."""
        pos_scaled = pos_scores / temperature
        neg_is_inf = torch.isinf(neg_scores)
        if neg_is_inf.any():
            neg_scores_safe = neg_scores.masked_fill(neg_is_inf, 0.0)
            neg_scaled = neg_scores_safe / temperature
            neg_scaled = neg_scaled.masked_fill(neg_is_inf, float("-inf"))
        else:
            neg_scaled = neg_scores / temperature

        logits = torch.cat([pos_scaled.unsqueeze(1), neg_scaled], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        return F.cross_entropy(logits, labels)


class UniformSampler(BaseNegativeSampler):
    """Uniform random negative sampling."""

    def weight_negatives(self, neg_scores: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(neg_scores)


class DegreeBasedSampler(BaseNegativeSampler):
    """Degree-based negative sampling."""

    def __init__(
        self,
        config: SamplerConfig | None = None,
        entity_degrees: torch.Tensor | None = None,
    ) -> None:
        super().__init__(config)
        self._entity_degrees = entity_degrees
        self._degree_weights: torch.Tensor | None = None

    def set_entity_degrees(self, degrees: torch.Tensor) -> None:
        self._entity_degrees = degrees
        alpha = self.config.alpha
        self._degree_weights = degrees.float() ** alpha
        self._degree_weights = self._degree_weights / self._degree_weights.sum()

    def weight_negatives(self, neg_scores: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(neg_scores)


class NSCachingSampler(BaseNegativeSampler):
    """NSCaching implementation using vectorized GPU Tensor storage."""

    def __init__(self, config: SamplerConfig | None = None) -> None:
        super().__init__(config)
        self.cache_manager = CacheManager()
        self._cache_tensor: torch.Tensor | None = None
        self._num_entities = self.config.num_entities
        self._num_triples = self.config.num_triples
        self._cache_size = self.config.cache_size

    def _ensure_cache_tensor(self, device: torch.device) -> torch.Tensor:
        if self._cache_tensor is not None:
            return self._cache_tensor
        cache_key = f"nsc_tensor_{self._num_triples}_{self._cache_size}"
        cached = self.cache_manager.get(cache_key)
        if cached is not None and isinstance(cached, torch.Tensor):
            if cached.shape == (self._num_triples, self._cache_size):
                self._cache_tensor = cached.to(device)
                return self._cache_tensor
        self._cache_tensor = torch.randint(
            0,
            self._num_entities,
            (self._num_triples, self._cache_size),
            device=device,
            dtype=torch.long,
        )
        return self._cache_tensor

    def sample_negatives(
        self,
        heads: torch.Tensor,
        relations: torch.Tensor,
        tails: torch.Tensor,
        num_negatives: int,
        triple_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if triple_indices is None:
            return torch.randint(
                0,
                self._num_entities,
                (len(heads), num_negatives),
                device=heads.device,
                dtype=torch.long,
            )
        cache_tensor = self._ensure_cache_tensor(heads.device)
        num_from_cache = int(num_negatives * self.config.sample_ratio)
        num_random = num_negatives - num_from_cache
        cached_samples = cache_tensor[triple_indices]
        s_c = cached_samples[:, :num_from_cache]
        s_r = torch.randint(
            0,
            self._num_entities,
            (len(heads), num_random),
            device=heads.device,
            dtype=torch.long,
        )
        return torch.cat([s_c, s_r], dim=1)

    def update_cache(
        self,
        heads: torch.Tensor,
        relations: torch.Tensor,
        tails: torch.Tensor,
        neg_ids: torch.Tensor,
        neg_scores: torch.Tensor,
        triple_indices: torch.Tensor | None = None,
    ) -> None:
        if triple_indices is None or self._cache_tensor is None:
            return
        _, top_idx = torch.topk(
            neg_scores, min(self._cache_size, neg_ids.shape[1]), dim=1
        )
        new_cache_vals = torch.gather(neg_ids, 1, top_idx)
        self._cache_tensor[triple_indices] = new_cache_vals

    def save_persistence(self) -> None:
        if self._cache_tensor is not None:
            cache_key = f"nsc_tensor_{self._num_triples}_{self._cache_size}"
            self.cache_manager.set(cache_key, self._cache_tensor.cpu())

    def weight_negatives(self, neg_scores: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(neg_scores)


def get_negative_sampler(
    sampler_type: SamplerType | str = SamplerType.DEGREE_BASED,
    config: SamplerConfig | None = None,
    **kwargs,
) -> BaseNegativeSampler:
    if isinstance(sampler_type, str):
        sampler_type = SamplerType(sampler_type)
    if config is None:
        config = SamplerConfig(sampler_type=sampler_type)
    samplers = {
        SamplerType.DEGREE_BASED: DegreeBasedSampler,
        SamplerType.NSCACHING: NSCachingSampler,
        SamplerType.UNIFORM: UniformSampler,
    }
    sampler_cls = samplers.get(sampler_type)
    if sampler_cls is None:
        raise ValueError(f"Unknown sampler type: {sampler_type}")
    return sampler_cls(config=config, **kwargs)
