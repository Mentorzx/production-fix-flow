"""
Decision Router for Hierarchical Ensemble.

This module implements the routing logic that decides how to combine
symbolic and neural predictions based on confidence thresholds.

Design Patterns Applied:
    - **Strategy Pattern:** Different routing behaviors per decision.
    - **State Pattern:** Decision state determines final score calculation.
    - **Observer Pattern:** Route decisions can be observed for metrics.

Routing Decisions:
    1. SYMBOLIC_DECIDES: High symbolic confidence → use symbolic directly
    2. NEURAL_FALLBACK: Low symbolic, adequate neural → use neural
    3. BLEND: Medium confidence → weighted combination

Flow:
    symbolic_score ──┬──▶ [Router] ──▶ final_score
    neural_score   ──┘        │
                              └──▶ routing_decision (for metrics)

Reference:
    - Hierarchical ensemble per SOTA 2024-2025 neuro-symbolic research
    - Confidence-based routing inspired by mixture of experts
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pff.utils.logger import logger
from pff.validators.ensembles.hierarchical.config_loader import (
    DecisionRouterConfig,
    load_hierarchical_config,
)


class RoutingDecision(str, Enum):
    """Possible routing decisions."""

    SYMBOLIC_DECIDES = "symbolic_decides"
    NEURAL_FALLBACK = "neural_fallback"
    BLEND = "blend"


@dataclass
class RoutingResult:
    """Result of routing decision for a single triple.

    Attributes:
        decision: The routing decision made.
        final_score: Combined score after routing [0, 1].
        symbolic_score: Input symbolic confidence.
        neural_score: Input neural score.
        blend_weights: Actual weights used if blending.
        metadata: Additional decision metadata.
    """

    decision: RoutingDecision
    final_score: float
    symbolic_score: float
    neural_score: float
    blend_weights: tuple[float, float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RoutingStatistics:
    """Aggregate statistics from batch routing.

    Attributes:
        total_decisions: Total number of routing decisions.
        symbolic_decides_count: Times SYMBOLIC_DECIDES was chosen.
        neural_fallback_count: Times NEURAL_FALLBACK was chosen.
        blend_count: Times BLEND was chosen.
        avg_final_score: Average final score across all decisions.
        avg_symbolic_score: Average symbolic input score.
        avg_neural_score: Average neural input score.
    """

    total_decisions: int = 0
    symbolic_decides_count: int = 0
    neural_fallback_count: int = 0
    blend_count: int = 0
    avg_final_score: float = 0.0
    avg_symbolic_score: float = 0.0
    avg_neural_score: float = 0.0

    @property
    def symbolic_decides_rate(self) -> float:
        """Fraction of decisions that were SYMBOLIC_DECIDES."""
        if self.total_decisions == 0:
            return 0.0
        return self.symbolic_decides_count / self.total_decisions

    @property
    def neural_fallback_rate(self) -> float:
        """Fraction of decisions that were NEURAL_FALLBACK."""
        if self.total_decisions == 0:
            return 0.0
        return self.neural_fallback_count / self.total_decisions

    @property
    def blend_rate(self) -> float:
        """Fraction of decisions that were BLEND."""
        if self.total_decisions == 0:
            return 0.0
        return self.blend_count / self.total_decisions

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging/metrics."""
        return {
            "total_decisions": self.total_decisions,
            "symbolic_decides_count": self.symbolic_decides_count,
            "neural_fallback_count": self.neural_fallback_count,
            "blend_count": self.blend_count,
            "symbolic_decides_rate": self.symbolic_decides_rate,
            "neural_fallback_rate": self.neural_fallback_rate,
            "blend_rate": self.blend_rate,
            "avg_final_score": self.avg_final_score,
            "avg_symbolic_score": self.avg_symbolic_score,
            "avg_neural_score": self.avg_neural_score,
        }


class DecisionRouter:
    """Routes between symbolic and neural predictions.

    Implements confidence-based routing:
    - High symbolic confidence → trust symbolic (rules are certain)
    - Low symbolic, good neural → fall back to neural
    - Otherwise → blend both signals

    Attributes:
        symbolic_threshold: Min confidence for SYMBOLIC_DECIDES.
        neural_threshold: Min score for NEURAL_FALLBACK.
        blend_weight_symbolic: Symbolic weight in BLEND mode.
        blend_weight_neural: Neural weight in BLEND mode.

    Usage:
        router = DecisionRouter.from_config()
        result = router.route(symbolic_score=0.9, neural_score=0.6)
        print(result.decision)  # RoutingDecision.SYMBOLIC_DECIDES
    """

    def __init__(
        self,
        symbolic_threshold: float = 0.85,
        symbolic_low_threshold: float = 0.30,
        neural_threshold: float = 0.70,
        blend_weight_symbolic: float = 0.6,
        blend_weight_neural: float = 0.4,
    ):
        """Initialize the decision router.

        Args:
            symbolic_threshold: Min symbolic confidence for SYMBOLIC_DECIDES.
            symbolic_low_threshold: Max symbolic confidence treated as weak; if
                symbolic is below this and neural is forte, fall back to neural.
            neural_threshold: Min neural score for NEURAL_FALLBACK.
            blend_weight_symbolic: Symbolic weight when blending.
            blend_weight_neural: Neural weight when blending.
        """
        self.symbolic_threshold = symbolic_threshold
        self.symbolic_low_threshold = symbolic_low_threshold
        self.neural_threshold = neural_threshold
        self.blend_weight_symbolic = blend_weight_symbolic
        self.blend_weight_neural = blend_weight_neural

        total_weight = blend_weight_symbolic + blend_weight_neural
        if abs(total_weight - 1.0) > 0.001:
            self._blend_symbolic_norm = blend_weight_symbolic / total_weight
            self._blend_neural_norm = blend_weight_neural / total_weight
            logger.debug(
                f"Blend weights normalized: symbolic={self._blend_symbolic_norm:.3f}, "
                f"neural={self._blend_neural_norm:.3f}"
            )
        else:
            self._blend_symbolic_norm = blend_weight_symbolic
            self._blend_neural_norm = blend_weight_neural

        logger.debug(
            f"DecisionRouter initialized: symbolic_threshold={symbolic_threshold}, "
            f"neural_threshold={neural_threshold}"
        )

    @classmethod
    def from_config(cls, config: DecisionRouterConfig | None = None) -> DecisionRouter:
        """Create router from configuration.

        Args:
            config: Router configuration. Loads from file if None.

        Returns:
            DecisionRouter: Configured router instance.
        """
        if config is None:
            hier_config = load_hierarchical_config()
            config = hier_config.decision_router

        return cls(
            symbolic_threshold=config.symbolic_confidence_threshold,
            symbolic_low_threshold=config.symbolic_low_threshold,
            neural_threshold=config.neural_confidence_threshold,
            blend_weight_symbolic=config.blend_weight_symbolic,
            blend_weight_neural=config.blend_weight_neural,
        )

    def _decide(
        self,
        symbolic_score: float,
        neural_score: float,
    ) -> RoutingDecision:
        """Determine routing decision based on scores.

        Decision tree:
        1. symbolic_score >= symbolic_threshold → SYMBOLIC_DECIDES
        2. symbolic_score < symbolic_low_threshold AND neural_score >= neural_threshold → NEURAL_FALLBACK
        3. symbolic_score < symbolic_threshold AND neural_score >= neural_threshold → NEURAL_FALLBACK
        3. Otherwise → BLEND

        Args:
            symbolic_score: Aggregated symbolic confidence [0, 1].
            neural_score: Aggregated neural score [0, 1].

        Returns:
            RoutingDecision: The routing decision.
        """
        if symbolic_score >= self.symbolic_threshold:
            return RoutingDecision.SYMBOLIC_DECIDES

        if symbolic_score < self.symbolic_low_threshold and neural_score >= self.neural_threshold:
            return RoutingDecision.NEURAL_FALLBACK

        if neural_score >= self.neural_threshold:
            return RoutingDecision.NEURAL_FALLBACK

        return RoutingDecision.BLEND

    def _compute_final_score(
        self,
        decision: RoutingDecision,
        symbolic_score: float,
        neural_score: float,
    ) -> tuple[float, tuple[float, float] | None]:
        """Compute final score based on routing decision.

        Args:
            decision: The routing decision.
            symbolic_score: Aggregated symbolic confidence.
            neural_score: Aggregated neural score.

        Returns:
            tuple[float, tuple[float, float] | None]:
                (final_score, blend_weights if BLEND else None)
        """
        if decision == RoutingDecision.SYMBOLIC_DECIDES:
            return symbolic_score, None

        if decision == RoutingDecision.NEURAL_FALLBACK:
            return neural_score, None

        final = (
            self._blend_symbolic_norm * symbolic_score
            + self._blend_neural_norm * neural_score
        )
        return final, (self._blend_symbolic_norm, self._blend_neural_norm)

    def route(
        self,
        symbolic_score: float,
        neural_score: float,
    ) -> RoutingResult:
        """Route a single triple's scores.

        Args:
            symbolic_score: Aggregated symbolic confidence [0, 1].
            neural_score: Aggregated neural score [0, 1].

        Returns:
            RoutingResult: Complete routing result with decision and score.
        """
        decision = self._decide(symbolic_score, neural_score)
        final_score, blend_weights = self._compute_final_score(
            decision, symbolic_score, neural_score
        )

        return RoutingResult(
            decision=decision,
            final_score=final_score,
            symbolic_score=symbolic_score,
            neural_score=neural_score,
            blend_weights=blend_weights,
            metadata={
                "symbolic_threshold": self.symbolic_threshold,
                "neural_threshold": self.neural_threshold,
            },
        )

    def route_batch(
        self,
        symbolic_scores: NDArray[np.float64] | list[float],
        neural_scores: NDArray[np.float64] | list[float],
    ) -> tuple[list[RoutingResult], RoutingStatistics]:
        """Route multiple triples' scores.

        Args:
            symbolic_scores: Array of symbolic confidences.
            neural_scores: Array of neural scores.

        Returns:
            tuple[list[RoutingResult], RoutingStatistics]:
                (results per triple, aggregate statistics)
        """
        sym_arr = np.asarray(symbolic_scores, dtype=np.float64)
        neu_arr = np.asarray(neural_scores, dtype=np.float64)

        if len(sym_arr) != len(neu_arr):
            raise ValueError(
                f"Score arrays must have same length: "
                f"symbolic={len(sym_arr)}, neural={len(neu_arr)}"
            )

        results = []
        stats = RoutingStatistics(total_decisions=len(sym_arr))

        final_scores = []

        for sym, neu in zip(sym_arr, neu_arr, strict=True):
            result = self.route(sym, neu)
            results.append(result)
            final_scores.append(result.final_score)

            if result.decision == RoutingDecision.SYMBOLIC_DECIDES:
                stats.symbolic_decides_count += 1
            elif result.decision == RoutingDecision.NEURAL_FALLBACK:
                stats.neural_fallback_count += 1
            else:
                stats.blend_count += 1

        if len(final_scores) > 0:
            stats.avg_final_score = float(np.mean(final_scores))
            stats.avg_symbolic_score = float(np.mean(sym_arr))
            stats.avg_neural_score = float(np.mean(neu_arr))

        return results, stats

    def route_vectorized(
        self,
        symbolic_scores: NDArray[np.float64],
        neural_scores: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.int32]]:
        """Vectorized routing for large batches.

        More efficient than route_batch for very large arrays.
        Returns only final scores and decision codes.

        Args:
            symbolic_scores: Array of symbolic confidences.
            neural_scores: Array of neural scores.

        Returns:
            tuple[NDArray[np.float64], NDArray[np.int32]]:
                (final_scores, decision_codes)
                Decision codes: 0=SYMBOLIC_DECIDES, 1=NEURAL_FALLBACK, 2=BLEND
        """
        sym_decides = symbolic_scores >= self.symbolic_threshold
        sym_weak = symbolic_scores < self.symbolic_low_threshold
        neu_conf = neural_scores >= self.neural_threshold

        neu_fallback = (~sym_decides) & neu_conf & sym_weak
        neu_fallback |= (~sym_decides) & neu_conf & (~sym_weak)
        blend = (~sym_decides) & (~neu_fallback)

        decisions = np.zeros(len(symbolic_scores), dtype=np.int32)
        decisions[neu_fallback] = 1
        decisions[blend] = 2

        final_scores = np.zeros_like(symbolic_scores)
        final_scores[sym_decides] = symbolic_scores[sym_decides]
        final_scores[neu_fallback] = neural_scores[neu_fallback]
        final_scores[blend] = (
            self._blend_symbolic_norm * symbolic_scores[blend]
            + self._blend_neural_norm * neural_scores[blend]
        )

        return final_scores, decisions

    @staticmethod
    def decision_code_to_enum(code: int) -> RoutingDecision:
        """Convert decision code to enum.

        Args:
            code: 0=SYMBOLIC_DECIDES, 1=NEURAL_FALLBACK, 2=BLEND

        Returns:
            RoutingDecision: Corresponding enum value.
        """
        mapping = {
            0: RoutingDecision.SYMBOLIC_DECIDES,
            1: RoutingDecision.NEURAL_FALLBACK,
            2: RoutingDecision.BLEND,
        }
        return mapping.get(code, RoutingDecision.BLEND)

    def compute_statistics_from_codes(
        self,
        decision_codes: NDArray[np.int32],
        final_scores: NDArray[np.float64],
        symbolic_scores: NDArray[np.float64],
        neural_scores: NDArray[np.float64],
    ) -> RoutingStatistics:
        """Compute statistics from vectorized results.

        Args:
            decision_codes: Array of decision codes from route_vectorized.
            final_scores: Array of final scores.
            symbolic_scores: Array of symbolic scores.
            neural_scores: Array of neural scores.

        Returns:
            RoutingStatistics: Aggregate statistics.
        """
        total = len(decision_codes)
        return RoutingStatistics(
            total_decisions=total,
            symbolic_decides_count=int(np.sum(decision_codes == 0)),
            neural_fallback_count=int(np.sum(decision_codes == 1)),
            blend_count=int(np.sum(decision_codes == 2)),
            avg_final_score=float(np.mean(final_scores)),
            avg_symbolic_score=float(np.mean(symbolic_scores)),
            avg_neural_score=float(np.mean(neural_scores)),
        )
