"""Adaptive training configuration calculator.

This module provides dynamic calculation of training hyperparameters
(epochs, early stopping, validation frequency) based on dataset characteristics.

Design Patterns:
    - Strategy Pattern: Different calculation strategies for different dataset sizes
    - Builder Pattern: Fluent configuration building
    - Factory Pattern: Create configs from dataset statistics

References:
    - DSLFM-KGC architecture
    - Self-adversarial negatives inspired by Sun et al. (2019)
    - FB15k-237: 310k triples, ~500 epochs
    - WN18RR: 86k triples, ~200 epochs
    - YAGO3-10: 1M triples, ~50-60 epochs

Author: PFF Team
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pff.shared.core.config import DSLFM_CONFIG_PATH
from pff.shared.core.config_loader import load_config
from pff.shared.core.logging import logger

_resource_manager = None
_adaptive_settings: dict[str, Any] | None = None


def _get_resource_manager():
    """Lazy load ResourceManager to avoid circular imports."""
    global _resource_manager
    if _resource_manager is None:
        from pff.shared.system.resource_manager import get_resource_manager

        _resource_manager = get_resource_manager()
    return _resource_manager


def _load_adaptive_training_settings() -> dict[str, Any]:
    """Load adaptive training settings from DSLFM config."""
    global _adaptive_settings
    if _adaptive_settings is not None:
        return _adaptive_settings
    try:
        cfg = load_config(DSLFM_CONFIG_PATH) or {}

        adaptive_cfg = cfg.get("adaptive_training", {})
        _adaptive_settings = adaptive_cfg if isinstance(adaptive_cfg, dict) else {}
        return _adaptive_settings
    except Exception as exc:
        logger.warning(
            f"Failed to load adaptive_training config from {DSLFM_CONFIG_PATH}: {exc}",
        )
        _adaptive_settings = {}
        return _adaptive_settings


class DatasetScale(Enum):
    """Dataset size categories for training strategy selection."""

    TINY = "tiny"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    HUGE = "huge"


@dataclass
class DatasetStats:
    """Statistics extracted from a knowledge graph dataset.

    Attributes:
        num_train_triples: Number of triples in training set.
        num_valid_triples: Number of triples in validation set.
        num_test_triples: Number of triples in test set.
        num_entities: Number of unique entities.
        num_relations: Number of unique relations.
    """

    num_train_triples: int
    num_valid_triples: int
    num_test_triples: int = 0
    num_entities: int = 0
    num_relations: int = 0

    @property
    def total_triples(self) -> int:
        """Total number of triples across all splits."""
        return self.num_train_triples + self.num_valid_triples + self.num_test_triples

    @property
    def scale(self) -> DatasetScale:
        """Determine dataset scale category."""
        n = self.num_train_triples
        if n < 10_000:
            return DatasetScale.TINY
        elif n < 100_000:
            return DatasetScale.SMALL
        elif n < 1_000_000:
            return DatasetScale.MEDIUM
        elif n < 10_000_000:
            return DatasetScale.LARGE
        else:
            return DatasetScale.HUGE

    @property
    def triples_per_entity(self) -> float:
        """Average number of triples per entity (graph density indicator)."""
        if self.num_entities == 0:
            return 0.0
        return self.total_triples / self.num_entities

    @property
    def avg_degree(self) -> float:
        """Average node degree in the graph."""
        if self.num_entities == 0:
            return 0.0
        return self.total_triples * 2 / self.num_entities

    @property
    def validation_ratio(self) -> float:
        """Ratio of validation to training triples."""
        if self.num_train_triples == 0:
            return 0.0
        return self.num_valid_triples / self.num_train_triples


@dataclass
class AdaptiveTrainingConfig:
    """Computed training configuration based on dataset characteristics.

    Attributes:
        epochs: Maximum number of training epochs.
        early_stopping_patience: Epochs without improvement before stopping.
        validate_every: Frequency of validation (every N epochs).
        min_delta: Minimum improvement threshold for early stopping.
        batch_size: Recommended batch size.
        num_neg: Number of negative samples per positive.
        learning_rate: Initial learning rate.
    """

    epochs: int
    early_stopping_patience: int
    validate_every: int
    min_delta: float
    batch_size: int = 1024
    num_neg: int = 128
    learning_rate: float = 1e-4

    dataset_scale: DatasetScale = field(default=DatasetScale.MEDIUM)
    computation_details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for YAML serialization."""
        return {
            "epochs": self.epochs,
            "early_stopping_patience": self.early_stopping_patience,
            "validate_every": self.validate_every,
            "min_delta": self.min_delta,
            "batch_size": self.batch_size,
            "num_neg": self.num_neg,
            "learning_rate": self.learning_rate,
        }


class AdaptiveTrainingCalculator:
    """Calculator for adaptive training hyperparameters.

    Implements dynamic calculation of epochs, early stopping, and other
    training parameters based on dataset statistics.

    Design Pattern: Strategy + Builder

    Example:
        >>> stats = DatasetStats(
        ...     num_train_triples=4_898_391,
        ...     num_valid_triples=612_298,
        ...     num_entities=269_889,
        ...     num_relations=44
        ... )
        >>> calculator = AdaptiveTrainingCalculator(stats)
        >>> config = calculator.compute()
        >>> config.epochs
        >>> config.early_stopping_patience
    """

    _CONFIG = _load_adaptive_training_settings()

    REFERENCE_TRIPLES = int(_CONFIG.get("reference_triples", 310_116))
    REFERENCE_EPOCHS = int(_CONFIG.get("reference_epochs", 100))

    _BASE_EPOCHS_CFG = _CONFIG.get("base_epochs", {}) or {}
    BASE_EPOCHS: dict[DatasetScale, int] = {
        DatasetScale.TINY: int(_BASE_EPOCHS_CFG.get("tiny", 120)),
        DatasetScale.SMALL: int(_BASE_EPOCHS_CFG.get("small", 80)),
        DatasetScale.MEDIUM: int(_BASE_EPOCHS_CFG.get("medium", 50)),
        DatasetScale.LARGE: int(_BASE_EPOCHS_CFG.get("large", 40)),
        DatasetScale.HUGE: int(_BASE_EPOCHS_CFG.get("huge", 30)),
    }

    _BASE_PATIENCE_CFG = _CONFIG.get("base_patience", {}) or {}
    BASE_PATIENCE: dict[DatasetScale, int] = {
        DatasetScale.TINY: int(_BASE_PATIENCE_CFG.get("tiny", 12)),
        DatasetScale.SMALL: int(_BASE_PATIENCE_CFG.get("small", 10)),
        DatasetScale.MEDIUM: int(_BASE_PATIENCE_CFG.get("medium", 8)),
        DatasetScale.LARGE: int(_BASE_PATIENCE_CFG.get("large", 6)),
        DatasetScale.HUGE: int(_BASE_PATIENCE_CFG.get("huge", 5)),
    }

    def __init__(
        self,
        stats: DatasetStats,
        *,
        is_dslfm: bool = True,
        use_contrastive: bool = False,
        embedding_dim: int = 128,
    ) -> None:
        """Initialize calculator with dataset statistics and hardware detection.

        Args:
            stats: Dataset statistics for computation.
            is_dslfm: Whether training DSLFM (adds overhead factor).
            use_contrastive: Whether using contrastive learning.
            embedding_dim: Embedding dimension for memory estimation.
        """
        self.stats = stats
        self.is_dslfm = is_dslfm
        self.use_contrastive = use_contrastive
        self.embedding_dim = embedding_dim
        self._details: dict[str, Any] = {}

        self._resource_manager = _get_resource_manager()
        self._hardware = self._resource_manager.hardware

    def compute(self) -> AdaptiveTrainingConfig:
        """Compute all adaptive training parameters.

        Returns:
            AdaptiveTrainingConfig with computed values.
        """
        epochs = self._compute_epochs()
        patience = self._compute_patience()
        validate_every = self._compute_validate_every()
        min_delta = self._compute_min_delta()
        batch_size = self._compute_batch_size()
        num_neg = self._compute_num_neg()
        learning_rate = self._compute_learning_rate()

        config = AdaptiveTrainingConfig(
            epochs=epochs,
            early_stopping_patience=patience,
            validate_every=validate_every,
            min_delta=min_delta,
            batch_size=batch_size,
            num_neg=num_neg,
            learning_rate=learning_rate,
            dataset_scale=self.stats.scale,
            computation_details=self._details,
        )

        logger.info(
            f"Configuracao adaptativa calculada: epochs={epochs}, "
            f"patience={patience}, validate_every={validate_every}"
        )
        logger.debug(f"computation_details={self._details}")

        return config

    def _compute_epochs(self) -> int:
        """Compute optimal number of epochs.

        Formula:
            epochs = base_epochs * entity_factor * relation_factor * model_factor

        Where:
            - base_epochs: From scale lookup (50-150)
            - entity_factor: log10(entities) / 4, clamped to [1.0, 1.5]
            - relation_factor: 1.0 + (relations - 20) / 100 if relations > 20
            - model_factor: 1.2 for DSLFM (joint training overhead)

        Returns:
            Computed epochs, clamped to [30, 200].
        """
        base = self.BASE_EPOCHS[self.stats.scale]

        if self.stats.num_entities > 0:
            entity_factor = min(1.5, max(1.0, math.log10(self.stats.num_entities) / 4))
        else:
            entity_factor = 1.0

        if self.stats.num_relations > 20:
            relation_factor = 1.0 + (self.stats.num_relations - 20) / 100
        else:
            relation_factor = 1.0
        relation_factor = min(relation_factor, 1.5)

        model_factor = 1.0
        if self.is_dslfm:
            model_factor *= 1.2
        if self.use_contrastive:
            model_factor *= 1.1

        if self.stats.triples_per_entity > 15:
            coverage_factor = 0.9
        elif self.stats.triples_per_entity < 5:
            coverage_factor = 1.2
        else:
            coverage_factor = 1.0

        epochs = int(
            base * entity_factor * relation_factor * model_factor * coverage_factor
        )

        if self.stats.scale == DatasetScale.SMALL:
            epochs = min(epochs, base)

        epochs = max(30, min(200, epochs))

        self._details["epochs"] = {
            "base": base,
            "entity_factor": round(entity_factor, 3),
            "relation_factor": round(relation_factor, 3),
            "model_factor": round(model_factor, 3),
            "coverage_factor": round(coverage_factor, 3),
            "raw": epochs,
        }

        return epochs

    def _compute_patience(self) -> int:
        """Compute early stopping patience.

        Formula:
            patience = base_patience * validation_stability_factor

        Where:
            - base_patience: From scale lookup (5-12)
            - validation_stability_factor: Based on validation set size
                - > 500k valid triples: 0.8 (very stable metrics)
                - > 100k: 0.9
                - > 10k: 1.0
                - < 10k: 1.2 (noisy metrics, need more patience)

        Returns:
            Computed patience, clamped to [3, 15].
        """
        base = self.BASE_PATIENCE[self.stats.scale]

        valid = self.stats.num_valid_triples
        if valid > 500_000:
            stability_factor = 0.8
        elif valid > 100_000:
            stability_factor = 0.9
        elif valid > 10_000:
            stability_factor = 1.0
        elif valid > 1_000:
            stability_factor = 1.1
        else:
            stability_factor = 1.3

        patience = int(base * stability_factor)
        patience = max(3, min(15, patience))

        self._details["patience"] = {
            "base": base,
            "stability_factor": round(stability_factor, 3),
            "valid_triples": valid,
        }

        return patience

    def _compute_validate_every(self) -> int:
        """Compute validation frequency.

        Formula:
            validate_every = base_frequency * scale_factor

        Where:
            - base_frequency: 3-5 depending on scale
            - scale_factor: Adjusted for validation cost

        Returns:
            Validation frequency in epochs (2-10).
        """

        if self.stats.scale in (DatasetScale.TINY, DatasetScale.SMALL):
            base = 3
        elif self.stats.scale == DatasetScale.MEDIUM:
            base = 4
        else:
            base = 5

        if self.stats.num_valid_triples > 1_000_000:
            base = min(base + 2, 10)

        self._details["validate_every"] = {
            "base": base,
            "scale": self.stats.scale.value,
        }

        return base

    def _compute_min_delta(self) -> float:
        """Compute minimum improvement threshold for early stopping.

        Formula:
            min_delta = base_delta / sqrt(num_valid_triples / 10000)

        Intuition:
            - Larger validation sets have more stable metrics
            - Smaller deltas are meaningful with stable metrics
            - Smaller datasets need larger deltas to avoid noise

        Returns:
            Minimum delta, clamped to [0.0001, 0.01].
        """
        base_delta = 0.001

        valid = max(self.stats.num_valid_triples, 1000)
        scale_factor = math.sqrt(10_000 / valid)

        min_delta = base_delta * scale_factor
        min_delta = max(0.0001, min(0.01, min_delta))

        self._details["min_delta"] = {
            "base": base_delta,
            "scale_factor": round(scale_factor, 4),
            "valid_triples": valid,
        }

        return round(min_delta, 6)

    def _compute_batch_size(self) -> int:
        """Compute batch size with hardware-aware memory scaling (SOTA).

        Uses 90% of available resources as target (ResourceManager default).
        Considers:
        - Available RAM for validation tensors [batch × entities × 4 bytes]
        - GPU memory if available
        - Entity count for embedding memory pressure

        Returns:
            Batch size that maximizes throughput without OOM.
        """
        scale_batches = {
            DatasetScale.TINY: 256,
            DatasetScale.SMALL: 512,
            DatasetScale.MEDIUM: 1024,
            DatasetScale.LARGE: 2048,
            DatasetScale.HUGE: 4096,
        }
        base_batch = scale_batches[self.stats.scale]

        available_gb = self._hardware.available_ram_gb
        memory_usage_percent = self._resource_manager.memory_usage_percent / 100
        safe_memory_gb = available_gb * memory_usage_percent

        entities = self.stats.num_entities
        memory_safe_batch = base_batch

        if entities > 0:
            max_val_memory_bytes = int(safe_memory_gb * 1024**3 * 0.15)
            bytes_per_score = 4
            memory_safe_batch = max(
                16, max_val_memory_bytes // (entities * bytes_per_score)
            )

        gpu_safe_batch = base_batch
        if self._hardware.has_gpu and self._hardware.gpu_memory_gb:
            gpu_mem_gb = self._hardware.gpu_memory_gb

            embedding_mem_gb = (entities * self.embedding_dim * 4 * 2 * 3) / (1024**3)

            overhead_factor = 0.4
            available_gpu = gpu_mem_gb * (1 - overhead_factor)

            gpu_usage_factor = self._resource_manager.memory_usage_percent / 100
            remaining_gpu_gb = max(
                0.5, (available_gpu - embedding_mem_gb) * gpu_usage_factor
            )

            bytes_per_sample = 256 * self.embedding_dim * 8
            gpu_safe_batch = max(16, int(remaining_gpu_gb * 1024**3 / bytes_per_sample))

        batch_size = min(base_batch, memory_safe_batch, gpu_safe_batch)

        batch_size = 2 ** int(math.log2(max(16, batch_size)))

        self._details["batch_size"] = {
            "scale": self.stats.scale.value,
            "base": base_batch,
            "memory_safe": memory_safe_batch,
            "gpu_safe": gpu_safe_batch,
            "entities": entities,
            "available_ram_gb": round(available_gb, 2),
            "gpu_memory_gb": self._hardware.gpu_memory_gb,
            "value": batch_size,
        }

        return batch_size  # type: ignore[no-any-return]

    def _compute_num_neg(self) -> int:
        """Compute negative samples with hardware-aware memory scaling (SOTA).

        Uses 90% of available resources as target.
        Reduces num_neg proportionally as embedding memory increases.
        Formula ensures batch × num_neg × 12 bytes fits in remaining memory.

        Returns:
            Number of negative samples that maximizes learning without OOM.
        """
        scale_neg = {
            DatasetScale.TINY: 64,
            DatasetScale.SMALL: 128,
            DatasetScale.MEDIUM: 256,
            DatasetScale.LARGE: 256,
            DatasetScale.HUGE: 512,
        }
        base_neg = scale_neg[self.stats.scale]

        entities = self.stats.num_entities
        available_gb = self._hardware.available_ram_gb
        memory_usage_percent = self._resource_manager.memory_usage_percent / 100
        safe_memory_gb = available_gb * memory_usage_percent

        embedding_mem_gb = (entities * self.embedding_dim * 4 * 2) / (1024**3)

        embedding_fraction = (
            embedding_mem_gb / safe_memory_gb if safe_memory_gb > 0 else 0
        )

        if embedding_fraction > 0.5:
            num_neg = min(base_neg, 16)
        elif embedding_fraction > 0.3:
            num_neg = min(base_neg, 32)
        elif embedding_fraction > 0.2:
            num_neg = min(base_neg, 64)
        elif embedding_fraction > 0.1:
            num_neg = min(base_neg, 128)
        else:
            num_neg = base_neg

        if self._hardware.has_gpu and self._hardware.gpu_memory_gb:
            if self._hardware.gpu_memory_gb < 4:
                num_neg = min(num_neg, 32)
            elif self._hardware.gpu_memory_gb < 8:
                num_neg = min(num_neg, 64)

        num_neg = 2 ** int(math.log2(max(16, num_neg)))

        self._details["num_neg"] = {
            "scale": self.stats.scale.value,
            "base": base_neg,
            "entities": entities,
            "embedding_mem_gb": round(embedding_mem_gb, 3),
            "embedding_fraction": round(embedding_fraction, 4),
            "available_ram_gb": round(available_gb, 2),
            "gpu_memory_gb": self._hardware.gpu_memory_gb,
            "value": num_neg,
        }

        return num_neg  # type: ignore[no-any-return]

    def _compute_learning_rate(self) -> float:
        """Compute initial learning rate.

        Formula:
            lr = base_lr * scale_factor

        Where:
            - base_lr: 1e-4 (standard for Adam)
            - scale_factor: Slight adjustment for dataset size

        Returns:
            Learning rate (1e-5 to 1e-3).
        """
        base_lr = 1e-4

        if self.stats.scale in (DatasetScale.LARGE, DatasetScale.HUGE):
            lr = base_lr * 2
        elif self.stats.scale == DatasetScale.TINY:
            lr = base_lr / 2
        else:
            lr = base_lr

        self._details["learning_rate"] = {
            "base": base_lr,
            "value": lr,
        }

        return lr


def compute_adaptive_config(
    num_train_triples: int,
    num_valid_triples: int,
    num_entities: int = 0,
    num_relations: int = 0,
    *,
    is_dslfm: bool = True,
    embedding_dim: int = 128,
) -> AdaptiveTrainingConfig:
    """Convenience function to compute adaptive training config.

    Args:
        num_train_triples: Number of training triples.
        num_valid_triples: Number of validation triples.
        num_entities: Number of unique entities (for memory estimation).
        num_relations: Number of unique relations (optional).
        is_dslfm: Whether training DSLFM model.
        embedding_dim: Embedding dimension for memory estimation.

    Returns:
        Computed AdaptiveTrainingConfig with hardware-aware parameters.

    Example:
        >>> config = compute_adaptive_config(
        ...     num_train_triples=100_000,
        ...     num_valid_triples=10_000,
        ...     num_entities=50_000,
        ...     num_relations=90,
        ... )
        >>> f"batch={config.batch_size}, neg={config.num_neg}"
    """
    stats = DatasetStats(
        num_train_triples=num_train_triples,
        num_valid_triples=num_valid_triples,
        num_entities=num_entities,
        num_relations=num_relations,
    )
    calculator = AdaptiveTrainingCalculator(
        stats, is_dslfm=is_dslfm, embedding_dim=embedding_dim
    )
    return calculator.compute()
