"""Negative sampling strategies for DSLFM-KGC contrastive learning (Lance Optimized).

This module provides pluggable negative sampling strategies, including a
Lance-based implementation for out-of-core scaling.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

import numpy as np
import torch
import torch.nn.functional as F

try:
    import lance
    import lancedb
    import pyarrow as pa

    LANCE_AVAILABLE = True
except ImportError:
    LANCE_AVAILABLE = False

from pff.shared.core.cache import CacheManager
from pff.shared.core.logging import logger

NUMBA_AVAILABLE = False


class SamplerType(str, Enum):
    DEGREE_BASED = "degree_based"
    NSCACHING = "nscaching"
    UNIFORM = "uniform"
    SELF_ADVERSARIAL = "self_adversarial"
    LANCE_DISK = "lance_disk"


@dataclass
class SamplerConfig:
    """Configuration for negative samplers."""

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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...

    def sample_negatives(
        self,
        heads: torch.Tensor,
        relations: torch.Tensor,
        tails: torch.Tensor,
        num_negatives: int,
        triple_indices: torch.Tensor | None = None,
    ) -> torch.Tensor: ...

    def update_cache(
        self,
        heads: torch.Tensor,
        relations: torch.Tensor,
        tails: torch.Tensor,
        neg_ids: torch.Tensor,
        neg_scores: torch.Tensor,
        triple_indices: torch.Tensor | None = None,
    ) -> None: ...

    def weight_negatives(
        self,
        neg_scores: torch.Tensor,
    ) -> torch.Tensor: ...

    def contrastive_loss(
        self,
        pos_scores: torch.Tensor,
        neg_scores: torch.Tensor,
        temperature: float | torch.Tensor = 0.07,
    ) -> torch.Tensor: ...


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

        keep_mask = torch.ones(batch_size * batch_size, dtype=torch.bool, device=neg_scores.device)
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


class SelfAdversarialSampler(BaseNegativeSampler):
    """Self-adversarial negative sampling."""

    def weight_negatives(self, neg_scores: torch.Tensor) -> torch.Tensor:
        temp = self.config.temperature
        return F.softmax(neg_scores / temp, dim=1)


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
        _, top_idx = torch.topk(neg_scores, min(self._cache_size, neg_ids.shape[1]), dim=1)
        new_cache_vals = torch.gather(neg_ids, 1, top_idx)
        self._cache_tensor[triple_indices] = new_cache_vals

    def save_persistence(self) -> None:
        if self._cache_tensor is not None:
            cache_key = f"nsc_tensor_{self._num_triples}_{self._cache_size}"
            self.cache_manager.set(cache_key, self._cache_tensor.cpu())

    def weight_negatives(self, neg_scores: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(neg_scores)


class LanceDiskSampler(BaseNegativeSampler):
    """
    Lance-optimized sampler for out-of-core scaling.

    Instead of holding a [num_triples, cache_size] tensor in RAM/GPU (NSCaching),
    it reads pre-computed or cached negatives from a Lance dataset on disk.

    Ideal for:
    1. Massive datasets (>100M triples) where the NSCaching tensor won't fit in RAM.
    2. Using static 'Hard Negatives' mined offline (e.g. by approximate nearest neighbors).

    Note: Dynamic updates (update_cache) are ignored or buffered because Lance is append-only.
    """

    def __init__(self, config: SamplerConfig | None = None) -> None:
        super().__init__(config)
        if not LANCE_AVAILABLE:
            raise RuntimeError("LanceDB not installed. pip install lancedb")

        cfg = cast(SamplerConfig, self.config)
        self.db_path = cfg.lance_db_path
        self.table_name = cfg.lance_table_name
        self._table: Any = None
        self._dataset: Any = None
        self._init_dataset()

    def _init_dataset(self):
        """Initialize or create the Lance dataset."""
        db = lancedb.connect(self.db_path)

        if self.config.rebuild_lance and self.table_name in db.table_names():
            db.drop_table(self.table_name)

        if self.table_name in db.table_names():
            self._table = db.open_table(self.table_name)

            try:
                self._dataset = self._table.to_lance()
            except Exception:
                if hasattr(lance, "dataset"):
                    self._dataset = lance.dataset(f"{self.db_path}/{self.table_name}.lance")
                else:
                    logger.warning(
                        "Could not access low-level lance.dataset. Performance may suffer."
                    )
                    self._dataset = None
            return

        logger.info(f"Inicializando cache de negativos Lance em {self.db_path}/{self.table_name}")
        if self.config.num_triples > 0:
            cache_size = self.config.cache_size

            dummy_data = pa.Table.from_pydict(
                {
                    "id": np.arange(self.config.num_triples),
                    "negatives": np.random.randint(
                        0, self.config.num_entities, (self.config.num_triples, cache_size)
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

        if flat_values is None and self._table is not None:
            logger.warning("Falling back to random sampling (Lance read failed)")
            return torch.randint(
                0,
                self.config.num_entities,
                (len(heads), num_negatives),
                device=heads.device,
                dtype=torch.long,
            )

        if flat_values is None:
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

        cached_samples = torch.from_numpy(flat_values.reshape(len(indices_list), -1)).to(
            heads.device
        )

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

    def update_cache(self, *args, **kwargs) -> None:
        pass

    def weight_negatives(self, neg_scores: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(neg_scores)


def get_negative_sampler(
    sampler_type: SamplerType | str = SamplerType.DEGREE_BASED,
    config: SamplerConfig | None = None,
    **kwargs,
) -> BaseNegativeSampler:
    if isinstance(sampler_type, str):
        try:
            sampler_type = SamplerType(sampler_type.lower())
        except ValueError:
            if "adversarial" in sampler_type.lower():
                sampler_type = SamplerType.SELF_ADVERSARIAL
            elif "lance" in sampler_type.lower():
                sampler_type = SamplerType.LANCE_DISK
            else:
                raise

    if config is None:
        config = SamplerConfig(sampler_type=sampler_type)

    samplers = {
        SamplerType.DEGREE_BASED: DegreeBasedSampler,
        SamplerType.NSCACHING: NSCachingSampler,
        SamplerType.UNIFORM: UniformSampler,
        SamplerType.SELF_ADVERSARIAL: SelfAdversarialSampler,
        SamplerType.LANCE_DISK: LanceDiskSampler,
    }
    sampler_cls = samplers.get(sampler_type)
    if sampler_cls is None:
        raise ValueError(f"Unknown sampler type: {sampler_type}")
    return sampler_cls(config=config, **kwargs)
