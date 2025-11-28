"""Ensemble Feature Balancer.

Extracts feature balance validation from AdvancedEnsembleTrainer,
following the Single Responsibility Principle (SRP).

Design Patterns Applied:
    - **Strategy Pattern:** Different balancing strategies can be plugged in.
    - **Guard Pattern:** Validates balance constraints before training proceeds.

Author: PFF Team
Date: 2025-11-25
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from pff.utils import logger


class SymbolicBalanceError(RuntimeError):
    """Raised when symbolic features dominate beyond configured limits."""
    pass


@dataclass
class BalanceConfig:
    """Configuration for feature balance validation.

    Attributes:
        max_symbolic_ratio: Maximum allowed symbolic contribution (0.0-1.0).
        min_hybrid_ratio: Minimum required hybrid contribution (0.0-1.0).
        min_symbolic_rules: Minimum number of active symbolic rules.
        warn_threshold: Threshold for warning (below max_symbolic_ratio).
    """
    max_symbolic_ratio: float = 0.70
    min_hybrid_ratio: float = 0.30
    min_symbolic_rules: int = 10
    warn_threshold: float = 0.60


class EnsembleFeatureBalancer:
    """Validates and monitors feature balance in ensemble models.

    Ensures that symbolic features don't dominate the ensemble predictions,
    which would defeat the purpose of the hybrid neuro-symbolic approach.

    Design Pattern: Guard
        - Validates constraints before allowing training to proceed.
        - Raises SymbolicBalanceError when limits are exceeded.

    Attributes:
        config: Balance configuration.
        last_balance: Most recent computed balance.
    """

    def __init__(self, config: BalanceConfig | None = None) -> None:
        """Initialize balancer with configuration.

        Args:
            config: Optional balance configuration.
        """
        self.config = config or BalanceConfig()
        self.last_balance: dict[str, float] | None = None

    def compute_balance(
        self,
        feature_importances: np.ndarray,
        feature_names: list[str],
        symbolic_prefix: str = "rule_",
    ) -> dict[str, float]:
        """Compute feature contribution balance.

        Args:
            feature_importances: Array of feature importance values.
            feature_names: List of feature names.
            symbolic_prefix: Prefix identifying symbolic features.

        Returns:
            Dictionary with balance metrics.
        """
        total = np.sum(np.abs(feature_importances))
        if total == 0:
            self.last_balance = {
                "symbolic_contribution": 0.0,
                "hybrid_contribution": 0.0,
                "symbolic_rules_count": 0,
                "total_features": len(feature_names),
            }
            return self.last_balance

        symbolic_mask = np.array([
            name.startswith(symbolic_prefix) for name in feature_names
        ])
        symbolic_importance = np.sum(np.abs(feature_importances[symbolic_mask]))

        symbolic_ratio = symbolic_importance / total
        hybrid_ratio = 1.0 - symbolic_ratio

        self.last_balance = {
            "symbolic_contribution": round(symbolic_ratio * 100, 2),
            "hybrid_contribution": round(hybrid_ratio * 100, 2),
            "symbolic_rules_count": int(np.sum(symbolic_mask)),
            "total_features": len(feature_names),
        }

        return self.last_balance

    def validate(
        self,
        feature_importances: np.ndarray,
        feature_names: list[str],
        symbolic_prefix: str = "rule_",
        raise_on_violation: bool = True,
    ) -> bool:
        """Validate feature balance against configured limits.

        Args:
            feature_importances: Array of feature importance values.
            feature_names: List of feature names.
            symbolic_prefix: Prefix for symbolic features.
            raise_on_violation: If True, raises SymbolicBalanceError.

        Returns:
            True if balance is within limits.

        Raises:
            SymbolicBalanceError: If symbolic dominance exceeds limit.
        """
        balance = self.compute_balance(
            feature_importances, feature_names, symbolic_prefix
        )

        symbolic_ratio = balance["symbolic_contribution"] / 100.0

        if symbolic_ratio > self.config.warn_threshold:
            logger.warning(
                f"Symbolic contribution is high: {balance['symbolic_contribution']:.2f}% "
                f"(threshold: {self.config.warn_threshold * 100:.0f}%)"
            )

        if symbolic_ratio > self.config.max_symbolic_ratio:
            msg = (
                f"Symbolic contribution ({balance['symbolic_contribution']:.2f}%) "
                f"exceeds maximum allowed ({self.config.max_symbolic_ratio * 100:.0f}%)"
            )
            logger.error(msg)

            if raise_on_violation:
                raise SymbolicBalanceError(msg)
            return False

        hybrid_ratio = balance["hybrid_contribution"] / 100.0
        if hybrid_ratio < self.config.min_hybrid_ratio:
            msg = (
                f"Hybrid contribution ({balance['hybrid_contribution']:.2f}%) "
                f"below minimum required ({self.config.min_hybrid_ratio * 100:.0f}%)"
            )
            logger.error(msg)

            if raise_on_violation:
                raise SymbolicBalanceError(msg)
            return False

        if balance["symbolic_rules_count"] < self.config.min_symbolic_rules:
            logger.warning(
                f"Low symbolic rules count: {balance['symbolic_rules_count']} "
                f"(minimum: {self.config.min_symbolic_rules})"
            )

        logger.info(
            f"Balanço de features validado: "
            f"híbrido={balance['hybrid_contribution']:.2f}%, "
            f"simbólico={balance['symbolic_contribution']:.2f}%"
        )

        return True

    def get_balance_summary(self) -> dict[str, Any]:
        """Get summary of last computed balance.

        Returns:
            Balance summary dictionary.
        """
        if self.last_balance is None:
            return {
                "status": "not_computed",
                "symbolic_contribution": 0.0,
                "hybrid_contribution": 0.0,
            }

        symbolic_ratio = self.last_balance["symbolic_contribution"] / 100.0

        if symbolic_ratio > self.config.max_symbolic_ratio:
            status = "violation"
        elif symbolic_ratio > self.config.warn_threshold:
            status = "warning"
        else:
            status = "healthy"

        return {
            "status": status,
            **self.last_balance,
            "config": {
                "max_symbolic_ratio": self.config.max_symbolic_ratio,
                "min_hybrid_ratio": self.config.min_hybrid_ratio,
            },
        }

    def suggest_adjustments(self) -> list[str]:
        """Suggest adjustments to improve balance.

        Returns:
            List of suggested adjustments.
        """
        if self.last_balance is None:
            return ["Run validation first to compute balance"]

        suggestions = []
        symbolic_ratio = self.last_balance["symbolic_contribution"] / 100.0

        if symbolic_ratio > self.config.max_symbolic_ratio:
            suggestions.extend([
                "Reduce max_rules in ensemble config",
                "Increase min_confidence_threshold for rules",
                "Lower rules_weight in ensemble weights",
                "Enable stricter activation pruning",
            ])

        if self.last_balance["symbolic_rules_count"] < self.config.min_symbolic_rules:
            suggestions.extend([
                "Lower min_confidence_threshold to include more rules",
                "Check if rules file exists and is properly formatted",
                "Increase max_rules_per_predicate",
            ])

        return suggestions
