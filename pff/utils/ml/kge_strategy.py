"""KGE Model Strategy Pattern Implementation.

Provides a Strategy abstraction for Knowledge Graph Embedding models,
enabling interchangeable model implementations. RotatE is the primary
and recommended model (SOTA for sparse KGs).

Design Patterns Applied:
    - **Strategy Pattern:** Each KGE model implements the same interface,
      allowing runtime swapping.
    - **Template Method:** Common scoring/training flows are defined in the
      abstract class with customizable steps.

Example:
    >>> from pff.utils.ml import KGEModelStrategy, RotatEStrategy
    >>> strategy = RotatEStrategy(config)
    >>> model = strategy.create_model(num_entities=1000, num_relations=50)
    >>> scores = strategy.score_batch(model, triples)

Author: PFF Team
Date: 2025-11-25
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from pff.utils import logger


@dataclass
class KGEConfig:
    """Configuration for KGE models.

    Attributes:
        embedding_dim: Dimension of entity/relation embeddings.
        margin: Margin for ranking loss.
        norm: L-norm for distance calculation (1 or 2).
        learning_rate: Initial learning rate.
        batch_size: Training batch size.
        num_negatives: Negative samples per positive triple.
        use_self_adversarial: Enable self-adversarial negative sampling.
        adversarial_temperature: Temperature for self-adversarial weighting.
        extra: Additional model-specific parameters.
    """

    embedding_dim: int = 128
    margin: float = 2.0
    norm: int = 2
    learning_rate: float = 0.001
    batch_size: int = 1024
    num_negatives: int = 5
    use_self_adversarial: bool = True
    adversarial_temperature: float = 1.0
    extra: dict[str, Any] = field(default_factory=dict)


class KGEModelStrategy(ABC):
    """Abstract Strategy for Knowledge Graph Embedding models.

    This interface defines the contract for all KGE model implementations.
    RotatE is the primary implementation (SOTA for sparse KGs).

    Design Pattern: Strategy
        - Encapsulates each algorithm (RotatE) in a separate class.
        - Allows the algorithm to vary independently from clients.
        - Enables runtime model selection without code changes.
    """

    def __init__(self, config: KGEConfig | None = None) -> None:
        """Initialize strategy with configuration.

        Args:
            config: KGE configuration. Uses defaults if None.
        """
        self.config = config or KGEConfig()

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the model name identifier."""
        pass

    @abstractmethod
    def create_model(
        self,
        num_entities: int,
        num_relations: int,
        device: torch.device | None = None,
    ) -> nn.Module:
        """Create and initialize the KGE model.

        Args:
            num_entities: Number of entities in the knowledge graph.
            num_relations: Number of relations in the knowledge graph.
            device: Target device for the model.

        Returns:
            Initialized PyTorch model.
        """
        pass

    @abstractmethod
    def score_triple(
        self,
        model: nn.Module,
        head: int,
        relation: int,
        tail: int,
    ) -> float:
        """Score a single triple.

        Args:
            model: The KGE model.
            head: Head entity index.
            relation: Relation index.
            tail: Tail entity index.

        Returns:
            Score for the triple (higher = more plausible).
        """
        pass

    @abstractmethod
    def score_batch(
        self,
        model: nn.Module,
        triples: np.ndarray | torch.Tensor,
    ) -> np.ndarray:
        """Score multiple triples efficiently.

        Args:
            model: The KGE model.
            triples: Array of shape (n_triples, 3) with [head, rel, tail].

        Returns:
            Array of scores with shape (n_triples,).
        """
        pass

    @abstractmethod
    def compute_loss(
        self,
        model: nn.Module,
        positive_triples: torch.Tensor,
        negative_triples: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the training loss.

        Args:
            model: The KGE model.
            positive_triples: Positive samples [batch, 3].
            negative_triples: Negative samples [batch, num_neg, 3].

        Returns:
            Scalar loss tensor.
        """
        pass

    def normalize_embeddings(self, model: nn.Module) -> None:
        """Normalize entity embeddings (optional operation).

        Args:
            model: The KGE model to normalize.
        """
        pass


class TransEStrategy(KGEModelStrategy):
    """DEPRECATED: TransE has been replaced by RotatE.

    This class is kept for backward compatibility only and will raise
    NotImplementedError when used. Use RotatEStrategy instead.

    Note:
        TransE was removed in v12.0.0 in favor of RotatE, which provides
        better performance on sparse knowledge graphs (>99% sparsity).
    """

    @property
    def name(self) -> str:
        """Return model name."""
        return "TransE"

    def create_model(
        self,
        num_entities: int,
        num_relations: int,
        device: torch.device | None = None,
    ) -> nn.Module:
        """DEPRECATED: Use RotatEStrategy instead.

        Raises:
            NotImplementedError: TransE has been removed. Use RotatE.
        """
        raise NotImplementedError(
            "TransE has been removed. Use RotatEStrategy instead. "
            "RotatE provides better performance on sparse KGs."
        )

    def score_triple(
        self,
        model: nn.Module,
        head: int,
        relation: int,
        tail: int,
    ) -> float:
        """DEPRECATED: Use RotatEStrategy instead."""
        raise NotImplementedError("TransE has been removed. Use RotatE.")

    def score_batch(
        self,
        model: nn.Module,
        triples: np.ndarray | torch.Tensor,
    ) -> np.ndarray:
        """DEPRECATED: Use RotatEStrategy instead."""
        raise NotImplementedError("TransE has been removed. Use RotatE.")

    def compute_loss(
        self,
        model: nn.Module,
        positive_triples: torch.Tensor,
        negative_triples: torch.Tensor,
    ) -> torch.Tensor:
        """DEPRECATED: Use RotatEStrategy instead."""
        raise NotImplementedError("TransE has been removed. Use RotatE.")

    def normalize_embeddings(self, model: nn.Module) -> None:
        """DEPRECATED: Use RotatEStrategy instead."""
        raise NotImplementedError("TransE has been removed. Use RotatE.")


class RotatEStrategy(KGEModelStrategy):
    """Strategy implementation for RotatE model.

    RotatE models relations as rotations in complex space: h ∘ r = t.
    This captures anti-symmetric relations and is effective for hierarchical KGs.

    Design Pattern: Concrete Strategy
        - Implements KGEModelStrategy interface for RotatE.
        - Provides RotatE-specific scoring with complex embeddings.
        - Uses self-adversarial negative sampling (SOTA from Sun et al. 2019).

    Mathematical Foundation:
        - Entities: h, t ∈ ℂ^d (complex vectors)
        - Relations: r = e^(iθ) (rotation by phase θ)
        - Score: γ - ||h ∘ r - t||

    Example:
        >>> strategy = RotatEStrategy(KGEConfig(embedding_dim=256, gamma=12.0))
        >>> model = strategy.create_model(num_entities=5000, num_relations=50)
        >>> scores = strategy.score_batch(model, triples)
    """

    @property
    def name(self) -> str:
        """Return model name."""
        return "RotatE"

    def create_model(
        self,
        num_entities: int,
        num_relations: int,
        device: torch.device | None = None,
    ) -> nn.Module:
        """Create RotatE model with complex embeddings.

        Args:
            num_entities: Number of entities.
            num_relations: Number of relations.
            device: Target device.

        Returns:
            RotatE model instance.
        """
        from pff.validators.rotate.core import RotatEModel
        from pff.validators.rotate.config import RotatEConfig

        rotate_config = RotatEConfig(
            embedding_dim=self.config.embedding_dim,
            gamma=self.config.extra.get("gamma", 12.0),
            epsilon=self.config.extra.get("epsilon", 2.0),
            learning_rate=self.config.learning_rate,
            batch_size=self.config.batch_size,
            num_negatives=self.config.num_negatives,
            use_self_adversarial=self.config.use_self_adversarial,
            adversarial_temperature=self.config.adversarial_temperature,
        )

        model = RotatEModel(
            num_entities=num_entities,
            num_relations=num_relations,
            embedding_dim=rotate_config.embedding_dim,
            gamma=rotate_config.gamma,
            epsilon=rotate_config.epsilon,
            config=rotate_config,
        )

        if device is not None:
            model = model.to(device)

        logger.info(
            f"RotatE criado: {num_entities:,} entidades, "
            f"{num_relations} relacoes, dim={rotate_config.embedding_dim}, "
            f"gamma={rotate_config.gamma}"
        )

        return model

    def score_triple(
        self,
        model: nn.Module,
        head: int,
        relation: int,
        tail: int,
    ) -> float:
        """Score single triple using RotatE rotation distance.

        Args:
            model: RotatE model.
            head: Head entity index.
            relation: Relation index.
            tail: Tail entity index.

        Returns:
            Score for the triple (higher = more plausible).
        """
        return model.score_triple(head, relation, tail)

    def score_batch(
        self,
        model: nn.Module,
        triples: np.ndarray | torch.Tensor,
    ) -> np.ndarray:
        """Score batch using vectorized RotatE.

        Args:
            model: RotatE model.
            triples: Triples array [n, 3].

        Returns:
            Scores array [n].
        """
        return model.score_triples_batch(triples)

    def compute_loss(
        self,
        model: nn.Module,
        positive_triples: torch.Tensor,
        negative_triples: torch.Tensor,
    ) -> torch.Tensor:
        """Compute self-adversarial negative sampling loss.

        Uses the loss function from RotatE paper:
        Loss = -log σ(γ - d_r(h,t)) - Σ p_i * log σ(d_r(h'_i,t'_i) - γ)

        Args:
            model: RotatE model.
            positive_triples: Positive samples [batch, 3].
            negative_triples: Negative samples [batch, num_neg, 3].

        Returns:
            Scalar loss.
        """
        return model.compute_loss(positive_triples, negative_triples)

    def normalize_embeddings(self, model: nn.Module) -> None:
        """Normalize embeddings (no-op for RotatE).

        RotatE doesn't require embedding normalization like TransE.
        The rotation mechanism naturally maintains embedding scale.

        Args:
            model: RotatE model.
        """
        pass
