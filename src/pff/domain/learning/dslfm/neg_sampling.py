"""Negative sampling strategies for DSLFM-KGC contrastive learning.

This module provides pluggable negative sampling strategies following the
Strategy pattern. Samplers can be swapped via configuration without
modifying the model or decoder.

Includes a Lance-based implementation for out-of-core scaling.

Design Patterns:
    - Strategy: Interchangeable sampling algorithms
    - Factory: Creation via get_negative_sampler()
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import importlib
from typing import Any, Protocol

import numpy as np
import torch
import torch.nn.functional as F

try:
    lance = importlib.import_module("lance")
    lancedb = importlib.import_module("lancedb")
    pa = importlib.import_module("pyarrow")

    LANCE_AVAILABLE = True
except ImportError:
    LANCE_AVAILABLE = False

from pff.shared.core.cache import CacheManager
from pff.shared.core.logging import logger

try:
    from pff_rust import (
        degree_weighted_negative_sampling as rust_degree_weighted_negative_sampling,
    )
except Exception:  # pragma: no cover - optional acceleration path
    rust_degree_weighted_negative_sampling = None


class SamplerType(str, Enum):
    """Represent SamplerType."""

    DEGREE_BASED = "degree_based"
    NSCACHING = "nscaching"
    UNIFORM = "uniform"
    SELF_ADVERSARIAL = "self_adversarial"
    LANCE_DISK = "lance_disk"


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

    lance_db_path: str = "/tmp/pff_lance_db"
    lance_table_name: str = "negative_cache"
    rebuild_lance: bool = False


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
        """Execute init.



        Args:

            config: Optional input value.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        self.config = config or SamplerConfig()
        self._mask_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

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

        if (
            batch_size in self._mask_cache
            and self._mask_cache[batch_size][0].device == neg_scores.device
        ):
            keep_mask, _ = self._mask_cache[batch_size]
        else:
            keep_mask = torch.ones(
                batch_size * batch_size, dtype=torch.bool, device=neg_scores.device
            )
            diag_idx = torch.arange(
                0, batch_size * batch_size, batch_size + 1, device=neg_scores.device
            )
            keep_mask[diag_idx] = False
            self._mask_cache[batch_size] = (keep_mask, diag_idx)

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
        """Execute weight negatives.



        Args:

            neg_scores: Input value used by this callable.



        Returns:

            Return value produced by the callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        return torch.ones_like(neg_scores)


class SelfAdversarialSampler(BaseNegativeSampler):
    """Self-adversarial negative sampling.

    Weights negatives by their current score using softmax with temperature.
    """

    def weight_negatives(self, neg_scores: torch.Tensor) -> torch.Tensor:
        """Execute weight negatives.



        Args:

            neg_scores: Input value used by this callable.



        Returns:

            Return value produced by the callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        temp = self.config.temperature
        return F.softmax(neg_scores / temp, dim=1)


class DegreeBasedSampler(BaseNegativeSampler):
    """Degree-based negative sampling."""

    def __init__(
        self,
        config: SamplerConfig | None = None,
        entity_degrees: torch.Tensor | None = None,
    ) -> None:
        """Execute init.



        Args:

            config: Optional input value.

            entity_degrees: Optional input value.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        super().__init__(config)
        self._entity_degrees = entity_degrees
        self._degree_weights: torch.Tensor | None = None
        if entity_degrees is not None:
            self.set_entity_degrees(entity_degrees)

    def set_entity_degrees(self, degrees: torch.Tensor) -> None:
        """Execute set entity degrees.



        Args:

            degrees: Input value used by this callable.

        """

        self._entity_degrees = degrees
        alpha = self.config.alpha
        weights = degrees.float() ** alpha
        self._degree_weights = weights / weights.sum()
        logger.info("Pesos de amostragem por grau atualizados: alpha=%.2f", alpha)

    def sample_negatives(
        self,
        heads: torch.Tensor,
        relations: torch.Tensor,
        tails: torch.Tensor,
        num_negatives: int,
        triple_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Execute sample negatives.



        Args:

            heads: Input value used by this callable.

            relations: Input value used by this callable.

            tails: Input value used by this callable.

            num_negatives: Input value used by this callable.

            triple_indices: Optional input value.



        Returns:

            Return value produced by the callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        if self._degree_weights is None:
            return super().sample_negatives(
                heads, relations, tails, num_negatives, triple_indices
            )

        device = heads.device
        if self._degree_weights.device != device:
            self._degree_weights = self._degree_weights.to(device)

        if rust_degree_weighted_negative_sampling is not None and device.type == "cpu":
            seed = int(torch.randint(0, 2**31 - 1, (1,), device=device).item())
            sampled_triples = rust_degree_weighted_negative_sampling(
                heads.detach().cpu().numpy().astype(np.int64),
                relations.detach().cpu().numpy().astype(np.int64),
                tails.detach().cpu().numpy().astype(np.int64),
                self._degree_weights.detach().cpu().numpy().astype(np.float64),
                int(self.config.num_entities),
                int(num_negatives),
                seed,
            )
            sampled_np = np.asarray(sampled_triples, dtype=np.int64).reshape(
                heads.shape[0], num_negatives, 3
            )
            return torch.as_tensor(sampled_np[:, :, 2], device=device, dtype=torch.long)

        batch_size = heads.shape[0]
        samples = torch.multinomial(
            self._degree_weights,
            num_samples=batch_size * num_negatives,
            replacement=True,
        )
        return samples.view(batch_size, num_negatives)

    def weight_negatives(self, neg_scores: torch.Tensor) -> torch.Tensor:
        """Execute weight negatives.



        Args:

            neg_scores: Input value used by this callable.



        Returns:

            Return value produced by the callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        return torch.ones_like(neg_scores)


class NSCachingSampler(BaseNegativeSampler):
    """NSCaching implementation using vectorized GPU Tensor storage."""

    def __init__(self, config: SamplerConfig | None = None) -> None:
        """Execute init.



        Args:

            config: Optional input value.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        super().__init__(config)
        self.cache_manager = CacheManager()
        self._cache_tensor: torch.Tensor | None = None
        self._num_entities = self.config.num_entities
        self._num_triples = self.config.num_triples
        self._cache_size = self.config.cache_size

    def _ensure_cache_tensor(self, device: torch.device) -> torch.Tensor:
        """Execute ensure cache tensor.



        Args:

            device: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        if self._cache_tensor is not None:
            return self._cache_tensor
        cache_key = (
            f"nsc_tensor_{self._num_triples}_{self._cache_size}_{self._num_entities}"
        )
        legacy_cache_key = f"nsc_tensor_{self._num_triples}_{self._cache_size}"
        cached = self.cache_manager.get(cache_key)
        if cached is None:
            cached = self.cache_manager.get(legacy_cache_key)
        if cached is not None and isinstance(cached, torch.Tensor):
            if cached.shape == (self._num_triples, self._cache_size):
                cached_tensor = cached.to(device=device, dtype=torch.long)
                if self._num_entities > 0:
                    # Keep backward compatibility with legacy cache entries while ensuring valid IDs.
                    cached_tensor = torch.remainder(cached_tensor, self._num_entities)
                self._cache_tensor = cached_tensor
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
        """Execute sample negatives.



        Args:

            heads: Input value used by this callable.

            relations: Input value used by this callable.

            tails: Input value used by this callable.

            num_negatives: Input value used by this callable.

            triple_indices: Optional input value.



        Returns:

            Return value produced by the callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

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
        """Execute update cache.



        Args:

            heads: Input value used by this callable.

            relations: Input value used by this callable.

            tails: Input value used by this callable.

            neg_ids: Input value used by this callable.

            neg_scores: Input value used by this callable.

            triple_indices: Optional input value.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        if triple_indices is None or self._cache_tensor is None:
            return
        _, top_idx = torch.topk(
            neg_scores, min(self._cache_size, neg_ids.shape[1]), dim=1
        )
        new_cache_vals = torch.gather(neg_ids, 1, top_idx)
        self._cache_tensor[triple_indices] = new_cache_vals

    def save_persistence(self) -> None:
        """Execute save persistence.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        if self._cache_tensor is not None:
            cache_key = f"nsc_tensor_{self._num_triples}_{self._cache_size}_{self._num_entities}"
            self.cache_manager.set(cache_key, self._cache_tensor.cpu())

    def weight_negatives(self, neg_scores: torch.Tensor) -> torch.Tensor:
        """Execute weight negatives.



        Args:

            neg_scores: Input value used by this callable.



        Returns:

            Return value produced by the callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        return torch.ones_like(neg_scores)


class LanceDiskSampler(BaseNegativeSampler):
    """Lance-optimized sampler for out-of-core scaling.

    Instead of holding a [num_triples, cache_size] tensor in RAM/GPU (NSCaching),
    it reads pre-computed or cached negatives from a Lance dataset on disk.

    Ideal for:
    1. Massive datasets (>100M triples) where the NSCaching tensor won't fit in RAM.
    2. Using static 'Hard Negatives' mined offline (e.g. by approximate nearest neighbors).

    Note: Dynamic updates (update_cache) are ignored because Lance is append-only.
    """

    def __init__(self, config: SamplerConfig | None = None) -> None:
        """Execute init.



        Args:

            config: Optional input value.



        Raises:

            Exception: Propagates domain-specific failures with context.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        super().__init__(config)
        if not LANCE_AVAILABLE:
            raise RuntimeError("LanceDB not installed. pip install lancedb")

        cfg = self.config
        self.db_path = cfg.lance_db_path
        self.table_name = cfg.lance_table_name
        self._table: Any = None
        self._dataset: Any = None
        self._init_dataset()

    def _init_dataset(self) -> None:
        """Initialize or create the Lance dataset."""
        db = lancedb.connect(self.db_path)

        if self.config.rebuild_lance and self.table_name in db.list_tables().tables:
            db.drop_table(self.table_name)

        if self.table_name in db.list_tables().tables:
            self._table = db.open_table(self.table_name)

            try:
                self._dataset = self._table.to_lance()
            except Exception:
                if hasattr(lance, "dataset"):
                    self._dataset = lance.dataset(
                        f"{self.db_path}/{self.table_name}.lance"
                    )
                else:
                    logger.warning(
                        "Could not access low-level lance.dataset. Performance may suffer."
                    )
                    self._dataset = None
            return

        logger.info(
            f"Inicializando cache de negativos Lance em {self.db_path}/{self.table_name}"
        )
        if self.config.num_triples > 0:
            cache_size = self.config.cache_size

            dummy_data = pa.Table.from_pydict(
                {
                    "id": np.arange(self.config.num_triples),
                    "negatives": np.random.randint(
                        0,
                        self.config.num_entities,
                        (self.config.num_triples, cache_size),
                    ).tolist(),
                }
            )

            self._table = db.create_table(self.table_name, dummy_data)
            try:
                self._dataset = self._table.to_lance()
            except Exception:
                self._dataset = None

    def sample_negatives(
        self,
        heads: torch.Tensor,
        relations: torch.Tensor,
        tails: torch.Tensor,
        num_negatives: int,
        triple_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample using disk-based Lance random access."""
        if triple_indices is None:
            return torch.randint(
                0,
                self.config.num_entities,
                (len(heads), num_negatives),
                device=heads.device,
                dtype=torch.long,
            )

        indices_list = triple_indices.cpu().tolist()

        flat_values = None

        if self._dataset is not None:
            try:
                table = self._dataset.take(indices_list, columns=["negatives"])
                flat_values = table["negatives"].combine_chunks().values.to_numpy()
            except Exception as e:
                logger.warning(f"Lance take failed: {e}")

        if flat_values is None:
            logger.warning("Falling back to random sampling (Lance read failed)")
            return torch.randint(
                0,
                self.config.num_entities,
                (len(heads), num_negatives),
                device=heads.device,
                dtype=torch.long,
            )

        expected_size = len(indices_list) * self.config.cache_size
        if flat_values.size != expected_size:
            return torch.randint(
                0,
                self.config.num_entities,
                (len(heads), num_negatives),
                device=heads.device,
                dtype=torch.long,
            )

        cached_samples = torch.from_numpy(
            flat_values.reshape(len(indices_list), -1)
        ).to(heads.device)

        num_from_cache = int(num_negatives * self.config.sample_ratio)
        num_random = num_negatives - num_from_cache

        s_c = cached_samples[:, :num_from_cache]

        if num_random > 0:
            s_r = torch.randint(
                0,
                self.config.num_entities,
                (len(heads), num_random),
                device=heads.device,
                dtype=torch.long,
            )
            return torch.cat([s_c, s_r], dim=1)

        return s_c

    def update_cache(self, *args: Any, **kwargs: Any) -> None:
        """No-op: Lance is append-only."""

    def weight_negatives(self, neg_scores: torch.Tensor) -> torch.Tensor:
        """Execute weight negatives.



        Args:

            neg_scores: Input value used by this callable.



        Returns:

            Return value produced by the callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        return torch.ones_like(neg_scores)


def get_negative_sampler(
    sampler_type: SamplerType | str = SamplerType.DEGREE_BASED,
    config: SamplerConfig | None = None,
    **kwargs,
) -> BaseNegativeSampler:
    """Execute get negative sampler.



    Args:

        sampler_type: Optional input value.

        config: Optional input value.

        **kwargs: Additional keyword arguments.



    Returns:

        Return value produced by the callable.



    Raises:

        Exception: Propagates domain-specific failures with context.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    if isinstance(sampler_type, SamplerType):
        resolved_sampler_type = sampler_type
    else:
        sampler_key = sampler_type
        try:
            resolved_sampler_type = SamplerType(sampler_key.lower())
        except ValueError:
            if "adversarial" in sampler_key.lower():
                resolved_sampler_type = SamplerType.SELF_ADVERSARIAL
            elif "lance" in sampler_key.lower():
                resolved_sampler_type = SamplerType.LANCE_DISK
            else:
                raise

    if config is None:
        config = SamplerConfig(sampler_type=resolved_sampler_type)

    samplers = {
        SamplerType.DEGREE_BASED: DegreeBasedSampler,
        SamplerType.NSCACHING: NSCachingSampler,
        SamplerType.UNIFORM: UniformSampler,
        SamplerType.SELF_ADVERSARIAL: SelfAdversarialSampler,
        SamplerType.LANCE_DISK: LanceDiskSampler,
    }
    sampler_cls = samplers.get(resolved_sampler_type)
    if sampler_cls is None:
        raise ValueError(f"Unknown sampler type: {resolved_sampler_type}")
    return sampler_cls(config=config, **kwargs)  # type: ignore[no-any-return]
