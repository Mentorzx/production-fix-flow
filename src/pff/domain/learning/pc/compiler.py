"""
Probabilistic Circuit compiler (minimal).

Design Patterns:
    - Factory/Builder (compilation with cached artifacts)
    - Fail-fast with fallback to Noisy-OR when constraints are violated
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

from pff.shared.core.logging import logger
from pff_rust import stable_hash


class CircuitCompilationError(RuntimeError):
    """Raised when a circuit cannot be compiled."""


@dataclass
class CompiledCircuit:
    """Lightweight compiled circuit placeholder."""

    rule_count: int
    rule_hash: str
    metadata: dict[str, Any]


class RuleToCircuitCompiler:
    """Minimal compiler that prepares circuit metadata for inference.

    The current implementation deliberately keeps the structure simple:
    it builds a deterministic hash for the rule set and caches it, while
    enforcing safeguards such as maximum rule counts and timeouts. This
    is a safe stepping stone toward a richer sum-product compiler.
    """

    def __init__(
        self,
        max_rules_per_circuit: int = 1000,
        compilation_timeout_ms: int = 500,
        cache_compiled_circuits: bool = True,
        log_rule_hash: bool = True,
        normalize_weights: bool = True,
    ) -> None:
        """Execute init.



        Args:

            max_rules_per_circuit: Optional input value.

            compilation_timeout_ms: Optional input value.

            cache_compiled_circuits: Optional input value.

            log_rule_hash: Optional input value.

            normalize_weights: Optional input value.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        self.max_rules_per_circuit = int(max_rules_per_circuit)
        self.compilation_timeout_ms = int(compilation_timeout_ms)
        self.cache_compiled_circuits = cache_compiled_circuits
        self.log_rule_hash = log_rule_hash
        self.normalize_weights = normalize_weights
        self._cache: dict[tuple[int, bool], CompiledCircuit] = {}

    def _cache_key(self, rule_count: int) -> tuple[int, bool]:
        return (rule_count, self.normalize_weights)

    def compile(self, rule_count: int) -> CompiledCircuit:
        """Compile (or fetch) a circuit for a given rule cardinality."""
        if rule_count <= 0:
            return CompiledCircuit(rule_count=0, rule_hash="empty", metadata={"compiled_ms": 0.0})

        if rule_count > self.max_rules_per_circuit:
            raise CircuitCompilationError(
                f"Rule count {rule_count} exceeds max_rules_per_circuit={self.max_rules_per_circuit}"
            )

        key = self._cache_key(rule_count)
        if self.cache_compiled_circuits and key in self._cache:
            return self._cache[key]

        start = perf_counter()
        rule_hash = str(stable_hash({"rules": rule_count, "normalize": self.normalize_weights}))
        elapsed_ms = (perf_counter() - start) * 1000.0

        if elapsed_ms > self.compilation_timeout_ms:
            raise TimeoutError(
                f"Compilation exceeded {self.compilation_timeout_ms}ms (elapsed={elapsed_ms:.2f}ms)"
            )

        compiled = CompiledCircuit(
            rule_count=rule_count,
            rule_hash=rule_hash,
            metadata={
                "compiled_ms": elapsed_ms,
                "normalize_weights": self.normalize_weights,
            },
        )

        if self.log_rule_hash:
            logger.debug(
                f"PC compiler generated hash={rule_hash} for rule_count={rule_count}, elapsed={elapsed_ms:.2f}ms"
            )

        if self.cache_compiled_circuits:
            self._cache[key] = compiled

        return compiled
