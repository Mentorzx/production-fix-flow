"""Contrastive Learning Losses for KG Embeddings.

Implements SOTA contrastive losses for knowledge graph embedding training:
- InfoNCE Loss (used in SimCLR, MoCo)
- NTXent Loss (Normalized Temperature-scaled Cross Entropy)
- Triplet Loss with margin
- Self-Adversarial Negative Sampling Loss (RotatE paper)

Design Patterns Applied:
    - **Strategy Pattern:** LossStrategy ABC with interchangeable implementations.
    - **Factory Pattern:** ContrastiveLossFactory for creating loss functions.
    - **Template Method:** Base loss provides structure, subclasses customize.

References:
    - Sun et al. 2019 "RotatE: Knowledge Graph Embedding by Relational Rotation"
    - Chen et al. 2020 "A Simple Framework for Contrastive Learning (SimCLR)"
    - Schroff et al. 2015 "FaceNet: A Unified Embedding for Face Recognition"

Author: PFF Team
Date: 2025-11-26
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class LossType(Enum):
    """Enumeration of supported loss types."""

    SELF_ADVERSARIAL = auto()
    MARGIN_RANKING = auto()
    INFONCE = auto()
    NTXENT = auto()
    TRIPLET = auto()


class LossStrategy(ABC):
    """Abstract base class for loss strategies.

    Pattern: Strategy Pattern

    Defines the interface for all contrastive loss implementations.
    Subclasses must implement the `compute` method.
    """

    @abstractmethod
    def compute(
        self,
        positive_scores: torch.Tensor,
        negative_scores: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute the loss value.

        Args:
            positive_scores: Scores for positive triples.
            negative_scores: Scores for negative triples.
            **kwargs: Additional loss-specific parameters.

        Returns:
            Scalar loss tensor.
        """
        pass


class SelfAdversarialLoss(LossStrategy):
    """Self-Adversarial Negative Sampling Loss from RotatE paper.

    Weights negative samples by their probability under the current model,
    focusing training on hard negatives.

    loss = -log σ(γ - d(h,r,t)) - Σ p(h',r,t') log σ(d(h',r,t') - γ)

    Where p(h',r,t') ∝ exp(α * score(h',r,t'))
    """

    def __init__(
        self,
        gamma: float = 12.0,
        adversarial_temperature: float = 1.0,
        reduction: str = "mean",
    ) -> None:
        """Initialize self-adversarial loss.

        Args:
            gamma: Margin hyperparameter (higher = stricter separation).
            adversarial_temperature: Temperature for adversarial weighting.
            reduction: Reduction method ('mean', 'sum', 'none').
        """
        self.gamma = gamma
        self.adversarial_temperature = adversarial_temperature
        self.reduction = reduction

    def compute(
        self,
        positive_scores: torch.Tensor,
        negative_scores: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute self-adversarial negative sampling loss.

        Args:
            positive_scores: Shape (batch_size,) - scores for positive triples.
            negative_scores: Shape (batch_size, num_negatives) - scores for negatives.

        Returns:
            Scalar loss tensor.
        """
        # Positive loss: -log σ(γ - score)
        positive_loss = F.logsigmoid(self.gamma - positive_scores)

        # Adversarial weights: softmax over negative scores
        with torch.no_grad():
            weights = F.softmax(
                self.adversarial_temperature * negative_scores, dim=-1
            ).detach()

        # Negative loss: -Σ w_i * log σ(score_i - γ)
        negative_loss = (weights * F.logsigmoid(negative_scores - self.gamma)).sum(dim=-1)

        loss = -(positive_loss + negative_loss)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class MarginRankingLoss(LossStrategy):
    """Margin-based ranking loss for KG embeddings.

    Ensures positive scores are higher than negative scores by at least margin.
    Similar to TransE's original loss function.
    """

    def __init__(self, margin: float = 1.0, reduction: str = "mean") -> None:
        """Initialize margin ranking loss.

        Args:
            margin: Minimum margin between positive and negative scores.
            reduction: Reduction method ('mean', 'sum', 'none').
        """
        self.margin = margin
        self.reduction = reduction

    def compute(
        self,
        positive_scores: torch.Tensor,
        negative_scores: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute margin ranking loss.

        Args:
            positive_scores: Shape (batch_size,) - scores for positive triples.
            negative_scores: Shape (batch_size, num_negatives) or (batch_size,).

        Returns:
            Scalar loss tensor.
        """
        if negative_scores.dim() == 2:
            # Multiple negatives per positive: average over negatives
            positive_expanded = positive_scores.unsqueeze(1)
            loss = F.relu(negative_expanded - positive_scores + self.margin)
            loss = loss.mean(dim=-1)
        else:
            loss = F.relu(negative_scores - positive_scores + self.margin)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class InfoNCELoss(LossStrategy):
    """InfoNCE Loss (Noise Contrastive Estimation).

    Used in contrastive representation learning (SimCLR, MoCo).
    Treats positives and negatives as a classification problem.
    """

    def __init__(self, temperature: float = 0.07, reduction: str = "mean") -> None:
        """Initialize InfoNCE loss.

        Args:
            temperature: Temperature parameter for softmax scaling.
            reduction: Reduction method ('mean', 'sum', 'none').
        """
        self.temperature = temperature
        self.reduction = reduction

    def compute(
        self,
        positive_scores: torch.Tensor,
        negative_scores: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute InfoNCE loss.

        Args:
            positive_scores: Shape (batch_size,) - scores for positive triples.
            negative_scores: Shape (batch_size, num_negatives) - scores for negatives.

        Returns:
            Scalar loss tensor.
        """
        # Concatenate positive and negatives
        positive_scaled = positive_scores.unsqueeze(1) / self.temperature
        negative_scaled = negative_scores / self.temperature

        # All scores: (batch_size, 1 + num_negatives)
        all_scores = torch.cat([positive_scaled, negative_scaled], dim=1)

        # Target: positive is always at index 0
        targets = torch.zeros(all_scores.size(0), dtype=torch.long, device=all_scores.device)

        loss = F.cross_entropy(all_scores, targets, reduction=self.reduction)
        return loss


class NTXentLoss(LossStrategy):
    """Normalized Temperature-scaled Cross Entropy Loss.

    Similar to InfoNCE but with normalized embeddings.
    """

    def __init__(self, temperature: float = 0.5, reduction: str = "mean") -> None:
        """Initialize NTXent loss.

        Args:
            temperature: Temperature parameter for softmax scaling.
            reduction: Reduction method ('mean', 'sum', 'none').
        """
        self.temperature = temperature
        self.reduction = reduction

    def compute(
        self,
        positive_scores: torch.Tensor,
        negative_scores: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute NTXent loss.

        Args:
            positive_scores: Shape (batch_size,) - similarity for positive pairs.
            negative_scores: Shape (batch_size, num_negatives) - similarity for negatives.

        Returns:
            Scalar loss tensor.
        """
        # Numerator: exp(sim(pos) / τ)
        numerator = torch.exp(positive_scores / self.temperature)

        # Denominator: exp(sim(pos) / τ) + Σ exp(sim(neg) / τ)
        denominator = numerator + torch.exp(negative_scores / self.temperature).sum(dim=-1)

        loss = -torch.log(numerator / denominator + 1e-8)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class TripletLoss(LossStrategy):
    """Triplet Loss with semi-hard mining.

    Learns embeddings where anchor-positive distance < anchor-negative distance.
    """

    def __init__(self, margin: float = 1.0, reduction: str = "mean") -> None:
        """Initialize triplet loss.

        Args:
            margin: Margin between positive and negative distances.
            reduction: Reduction method ('mean', 'sum', 'none').
        """
        self.margin = margin
        self.reduction = reduction

    def compute(
        self,
        positive_scores: torch.Tensor,
        negative_scores: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute triplet loss.

        Note: For distance-based scoring (lower = better), use scores directly.
              For similarity-based (higher = better), negate before passing.

        Args:
            positive_scores: Shape (batch_size,) - distances to positives.
            negative_scores: Shape (batch_size, num_negatives) - distances to negatives.

        Returns:
            Scalar loss tensor.
        """
        if negative_scores.dim() == 2:
            # Use hardest negative (smallest distance = highest score)
            negative_scores = negative_scores.min(dim=-1).values

        loss = F.relu(positive_scores - negative_scores + self.margin)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class ContrastiveLossFactory:
    """Factory for creating contrastive loss functions.

    Design Pattern: Factory
        - Centralizes loss creation logic.
        - Decouples client code from concrete loss classes.
    """

    _registry: dict[LossType, type[LossStrategy]] = {
        LossType.SELF_ADVERSARIAL: SelfAdversarialLoss,
        LossType.MARGIN_RANKING: MarginRankingLoss,
        LossType.INFONCE: InfoNCELoss,
        LossType.NTXENT: NTXentLoss,
        LossType.TRIPLET: TripletLoss,
    }

    @classmethod
    def create(cls, loss_type: LossType, **kwargs: Any) -> LossStrategy:
        """Create a loss function by type.

        Args:
            loss_type: Type of loss to create.
            **kwargs: Loss-specific parameters.

        Returns:
            Instantiated loss strategy.

        Raises:
            ValueError: If loss type is not supported.
        """
        if loss_type not in cls._registry:
            raise ValueError(f"Unsupported loss type: {loss_type}")

        return cls._registry[loss_type](**kwargs)

    @classmethod
    def register(cls, loss_type: LossType, loss_class: type[LossStrategy]) -> None:
        """Register a custom loss implementation.

        Args:
            loss_type: Enum value for the loss type.
            loss_class: Class implementing LossStrategy.
        """
        cls._registry[loss_type] = loss_class


class ContrastiveLearner(nn.Module):
    """High-level contrastive learner wrapper.

    Combines a loss strategy with model scoring for convenient training.
    """

    def __init__(
        self,
        loss_strategy: LossStrategy | None = None,
        loss_type: LossType = LossType.SELF_ADVERSARIAL,
        **loss_kwargs: Any,
    ) -> None:
        """Initialize contrastive learner.

        Args:
            loss_strategy: Pre-configured loss strategy (overrides loss_type).
            loss_type: Type of loss to use if loss_strategy not provided.
            **loss_kwargs: Parameters for loss construction.
        """
        super().__init__()
        if loss_strategy is not None:
            self.loss_strategy = loss_strategy
        else:
            self.loss_strategy = ContrastiveLossFactory.create(loss_type, **loss_kwargs)

    def forward(
        self,
        positive_scores: torch.Tensor,
        negative_scores: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute loss.

        Args:
            positive_scores: Scores for positive samples.
            negative_scores: Scores for negative samples.
            **kwargs: Additional parameters passed to loss strategy.

        Returns:
            Scalar loss tensor.
        """
        return self.loss_strategy.compute(positive_scores, negative_scores, **kwargs)
