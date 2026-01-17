"""
Violation Penalty Calculator - Extracted for Single Responsibility Principle.

This module computes score adjustments based on rule violations, following
the Strategy pattern for penalty calculation.

Design Patterns Applied:
    - **Strategy Pattern:** Different penalty strategies can be configured
      via the config file (bonus/penalty thresholds).
    - **Single Responsibility:** Extracted from ModelIntegration to handle
      only penalty/bonus computation.

Configuration:
    All thresholds are loaded from `config/models/validator.yaml` under the
    `violation_scoring` section.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pff.config import VALIDATOR_CONFIG_PATH
from pff.shared import FileManager


@dataclass(frozen=True)
class PenaltyConfig:
    """
    Immutable configuration for violation penalty calculation.

    Attributes:
        rate_floor: Minimum violations per 1K rules to trigger penalty (default: 5.0)
        penalty_multiplier: Multiplier for violations per 1K rules (default: 0.05)
        max_penalty: Maximum penalty cap (default: 0.65)
        no_violations_bonus: Bonus for zero violations (default: 0.35)
        below_threshold_bonus: Bonus for below-floor violations (default: 0.15)
        confidence_anchor: Confidence threshold for extra penalty (default: 0.5)
    """

    rate_floor: float = 5.0  # Violations per 1K rules threshold
    penalty_multiplier: float = 0.05  # Penalty per violation per 1K rules
    max_penalty: float = 0.65
    no_violations_bonus: float = 0.35
    below_threshold_bonus: float = 0.15
    confidence_anchor: float = 0.5

    @classmethod
    def from_config(cls, config: dict[str, Any] | None = None) -> PenaltyConfig:
        """
        Create PenaltyConfig from a config dictionary.

        Args:
            config: Dictionary with violation_scoring keys. If None, loads
                from config/validator/validator.yaml.

        Returns:
            PenaltyConfig instance with values from config or defaults.
        """
        if config is None:
            file_manager = FileManager()
            full_config = file_manager.read(VALIDATOR_CONFIG_PATH, return_native=True)
            config = full_config.get("violation_scoring", {})

        return cls(
            rate_floor=config.get("rate_floor", cls.rate_floor),
            penalty_multiplier=config.get("penalty_multiplier", cls.penalty_multiplier),
            max_penalty=config.get("max_penalty", cls.max_penalty),
            no_violations_bonus=config.get(
                "no_violations_bonus", cls.no_violations_bonus
            ),
            below_threshold_bonus=config.get(
                "below_threshold_bonus", cls.below_threshold_bonus
            ),
            confidence_anchor=config.get("confidence_anchor", cls.confidence_anchor),
        )


class ViolationPenaltyCalculator:
    """
    Calculator for violation-based score adjustments.

    Computes penalties or bonuses based on rule violation features.
    Extracted from ModelIntegration for Single Responsibility Principle.

    Strategy:
        - No violations → Strong bonus (raises score toward 0.85+)
        - Few violations (below rate_floor per 1K rules) → Small bonus
        - Many violations → Penalty proportional to violations per 1K rules

    The penalty uses violations_per_k_rules instead of raw violation_rate
    to handle large rule sets properly. With 18K+ rules, even 100+ violations
    have a small violation_rate (< 1%), but violations_per_k_rules (5-15)
    properly reflects the severity.

    Example:
        >>> calc = ViolationPenaltyCalculator()
        >>> features = {"num_violations": 0, "total_rules": 18000}
        >>> penalty, metadata = calc.compute(features)
        >>> print(penalty)  # -0.35 (bonus)
    """

    def __init__(self, config: PenaltyConfig | None = None):
        """
        Initialize the calculator.

        Args:
            config: PenaltyConfig instance. If None, loads from config file.
        """
        self.config = config or PenaltyConfig.from_config()

    def compute(
        self, violation_features: dict[str, Any]
    ) -> tuple[float, dict[str, Any]]:
        """
        Compute score adjustment based on violations.

        Uses violations_per_k_rules (violations per 1000 rules) as the base
        metric instead of violation_rate, which is too small for large rule sets.

        Args:
            violation_features: Dictionary containing:
                - num_violations: Number of violations detected
                - total_rules: Total rules validated
                - violation_rate: Violations / total_rules ratio
                - violations_per_k_rules: Violations per 1000 rules
                - avg_confidence: Average confidence of violations

        Returns:
            Tuple of (adjustment, metadata):
                - adjustment: Positive = penalty, Negative = bonus
                - metadata: Dict with computation details
        """
        num_violations = int(violation_features.get("num_violations", 0))
        total_rules = int(violation_features.get("total_rules", 0))
        violation_rate = float(violation_features.get("violation_rate", 0.0))
        avg_conf = float(violation_features.get("avg_confidence", 0.0))

        # Use violations_per_k_rules for penalty calculation
        # This handles large rule sets better than violation_rate
        violations_per_k = float(
            violation_features.get("violations_per_k_rules", violation_rate * 1000)
        )

        # BONUS for no violations
        if num_violations == 0:
            return -self.config.no_violations_bonus, {
                "penalty_reason": "no_violations_bonus",
                "violation_rate": 0.0,
                "violations_per_k_rules": 0.0,
                "num_violations": 0,
                "total_rules": total_rules,
                "applied_bonus": self.config.no_violations_bonus,
            }

        # BONUS for very few violations (below threshold per 1K rules)
        if violations_per_k <= self.config.rate_floor:
            return -self.config.below_threshold_bonus, {
                "penalty_reason": "below_threshold_bonus",
                "violation_rate": violation_rate,
                "violations_per_k_rules": violations_per_k,
                "num_violations": num_violations,
                "total_rules": total_rules,
                "applied_bonus": self.config.below_threshold_bonus,
            }

        # PENALTY for violations above threshold
        # Base penalty: violations_per_k * multiplier
        penalty = violations_per_k * self.config.penalty_multiplier

        # Extra penalty for high-confidence violations
        confidence_penalty = max(0.0, avg_conf - self.config.confidence_anchor) * 0.1
        penalty += confidence_penalty

        # Cap at max_penalty
        penalty = min(self.config.max_penalty, penalty)

        return penalty, {
            "penalty_reason": "violation_density",
            "violation_rate": violation_rate,
            "violations_per_k_rules": violations_per_k,
            "avg_confidence": avg_conf,
            "num_violations": num_violations,
            "total_rules": total_rules,
            "confidence_penalty": confidence_penalty,
            "applied_penalty": penalty,
        }
