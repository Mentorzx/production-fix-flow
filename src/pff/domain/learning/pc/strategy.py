"""
Probabilistic Circuit aggregation strategy.

This strategy is intentionally conservative: it compiles lightweight
metadata for the current rule cardinality, runs a deterministic forward
pass, and falls back to Noisy-OR under strict guards (timeouts, limits,
or compilation failures).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from pff.domain.learning.pc.compiler import (
    CircuitCompilationError,
    RuleToCircuitCompiler,
)
from pff.domain.learning.pc.inference import PCInferenceEngine
from pff.shared.core.logging import logger


class ProbabilisticCircuitStrategy:
    """Probabilistic Circuit-based aggregation with safe fallbacks."""

    def __init__(
        self,
        compilation_timeout_ms: int = 500,
        max_rules_per_circuit: int = 1000,
        cache_compiled_circuits: bool = True,
        fallback_to_noisy_or: bool = True,
        log_rule_hash: bool = True,
        normalize_weights: bool = True,
        base_confidence: float = 0.01,
    ) -> None:
        """Execute init.



        Args:

            compilation_timeout_ms: Optional input value.

            max_rules_per_circuit: Optional input value.

            cache_compiled_circuits: Optional input value.

            fallback_to_noisy_or: Optional input value.

            log_rule_hash: Optional input value.

            normalize_weights: Optional input value.

            base_confidence: Optional input value.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        self.fallback_to_noisy_or = fallback_to_noisy_or
        self.base_confidence = base_confidence
        self.compiler = RuleToCircuitCompiler(
            max_rules_per_circuit=max_rules_per_circuit,
            compilation_timeout_ms=compilation_timeout_ms,
            cache_compiled_circuits=cache_compiled_circuits,
            log_rule_hash=log_rule_hash,
            normalize_weights=normalize_weights,
        )
        self.inference = PCInferenceEngine(normalize_weights=normalize_weights)
        self._fallback_strategy = None

    @property
    def name(self) -> str:
        """Execute name.



        Returns:

            Return value produced by the callable.

        """

        return "pc"

    def aggregate(
        self,
        confidences: NDArray[np.float64],
        weights: NDArray[np.float64] | None = None,
    ) -> float:
        """Aggregate confidences using a compiled circuit."""
        if len(confidences) == 0:
            return self.base_confidence

        rule_count = len(confidences)
        try:
            circuit = self.compiler.compile(rule_count)
            return self.inference.infer(circuit, confidences, weights)
        except TimeoutError as exc:
            logger.warning(
                f"PC compilation exceeded timeout (rules={rule_count}, timeout_ms={self.compiler.compilation_timeout_ms}): {exc}"
            )
        except CircuitCompilationError as exc:
            logger.warning(
                f"PC compilation failed (rules={rule_count}, max={self.compiler.max_rules_per_circuit}): {exc}"
            )
        except Exception as exc:
            logger.warning(f"PC aggregation unexpected failure (rules={rule_count}): {exc}")
            if not self.fallback_to_noisy_or:
                raise RuntimeError(
                    f"PC aggregation failed without fallback (rules={rule_count})"
                ) from exc

        return self._get_fallback().aggregate(confidences, weights)  # type: ignore[no-any-return]

    def _get_fallback(self):
        """Execute get fallback.



        Returns:

            Return value produced by the callable.

        """

        if self._fallback_strategy is None:
            from pff.domain.learning.ml.aggregation_strategies import NoisyOrStrategy

            self._fallback_strategy = NoisyOrStrategy(base_confidence=self.base_confidence)
        return self._fallback_strategy


def build_pc_params_from_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Normalize PC params ensuring only known keys are kept."""
    if not config:
        return {}

    allowed = {
        "compilation_timeout_ms",
        "max_rules_per_circuit",
        "cache_compiled_circuits",
        "fallback_to_noisy_or",
        "log_rule_hash",
        "normalize_weights",
        "base_confidence",
    }
    return {k: config[k] for k in config if k in allowed}
