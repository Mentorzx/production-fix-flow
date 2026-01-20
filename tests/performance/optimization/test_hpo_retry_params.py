"""Tests for symbolic retry params derivation and weight normalization.

Fast unit tests targeting:
- _derive_symbolic_retry_params logic
- Weight normalization invariants
- Parameter clipping behavior
"""

from __future__ import annotations

import numpy as np
import pytest


def _derive_symbolic_retry_params(current_params: dict[str, float]) -> dict[str, float] | None:
    """Generate fallback params biased toward higher symbolic coverage.

    Mirrors logic from pff/domain/hpo/selection.py (adapted for DSLFM-KGC).
    """
    if not current_params:
        return None

    fallback = dict(current_params)
    fallback["feature_selection_threshold"] = float(
        np.clip(current_params.get("feature_selection_threshold", 0.15) * 0.5, 0.03, 0.2)
    )
    fallback["target_symbolic_ratio"] = float(
        np.clip(current_params.get("target_symbolic_ratio", 0.38), 0.3, 0.42)
    )
    fallback["rules_threshold"] = float(
        np.clip(current_params.get("rules_threshold", 0.3) * 0.8, 0.15, 0.45)
    )
    fallback["rules_weight"] = float(np.clip(current_params.get("rules_weight", 0.18), 0.12, 0.40))

    # Neural weight takes the remainder
    fallback["neural_weight"] = float(
        np.clip(
            1.0 - fallback["rules_weight"],
            0.15,
            0.88,
        )
    )
    total_weight = fallback["neural_weight"] + fallback["rules_weight"]
    if total_weight <= 0:
        return None
    scale = 1.0 / total_weight
    fallback["neural_weight"] *= scale
    fallback["rules_weight"] *= scale
    return fallback


class TestDeriveSymbolicRetryParams:
    """Test symbolic retry param derivation."""

    def test_empty_params_returns_none(self):
        """Empty dict should return None."""
        assert _derive_symbolic_retry_params({}) is None

    def test_default_values_used_for_missing_keys(self):
        """Missing keys should use defaults."""
        result = _derive_symbolic_retry_params({"some_other_key": 1.0})
        assert result is not None
        # Defaults: feature_selection_threshold=0.15, etc.
        assert "feature_selection_threshold" in result

    def test_weights_sum_to_one_after_normalization(self):
        """Weights must sum to 1.0 after normalization."""
        params = {
            "neural_weight": 0.7,
            "rules_weight": 0.3,
        }
        result = _derive_symbolic_retry_params(params)
        assert result is not None

        total = result["neural_weight"] + result["rules_weight"]
        assert abs(total - 1.0) < 1e-9

    def test_feature_selection_threshold_halved(self):
        """Feature selection threshold should be halved."""
        params = {"feature_selection_threshold": 0.3}
        result = _derive_symbolic_retry_params(params)
        # 0.3 * 0.5 = 0.15, within [0.03, 0.2]
        assert result["feature_selection_threshold"] == 0.15

    def test_rules_threshold_reduced(self):
        """Rules threshold should be reduced by 20%."""
        params = {"rules_threshold": 0.5}
        result = _derive_symbolic_retry_params(params)
        # 0.5 * 0.8 = 0.4, within [0.15, 0.45]
        assert result["rules_threshold"] == 0.4

    def test_extreme_weights_still_normalize(self):
        """Extreme weight values should still normalize to 1.0."""
        params = {
            "neural_weight": 0.01,
            "rules_weight": 0.01,
        }
        result = _derive_symbolic_retry_params(params)
        assert result is not None

        total = result["neural_weight"] + result["rules_weight"]
        assert abs(total - 1.0) < 1e-9


class TestWeightClippingBehavior:
    """Test weight clipping edge cases."""

    def test_neural_weight_derivation(self):
        """Neural weight is derived from 1 - rules."""
        params = {
            "rules_weight": 0.2,
        }
        result = _derive_symbolic_retry_params(params)

        # rules clamped to [0.12, 0.40] -> 0.2
        # neural = 1.0 - 0.2 = 0.8
        assert result["neural_weight"] > 0

    def test_zero_weight_handling(self):
        """Zero weights should be clamped to minimums."""
        params = {
            "neural_weight": 0.0,
            "rules_weight": 0.0,
        }
        result = _derive_symbolic_retry_params(params)
        assert result is not None
        # All should be > 0 after clipping
        assert result["neural_weight"] > 0
        assert result["rules_weight"] > 0


class TestSymbolicRetryInvariants:
    """Test invariants that should always hold."""

    @pytest.mark.parametrize(
        "neural,rules",
        [
            (0.5, 0.5),
            (0.8, 0.2),
            (0.1, 0.9),
        ],
    )
    def test_output_weights_always_sum_to_one(self, neural: float, rules: float):
        """Output weights must always sum to 1.0."""
        params = {
            "neural_weight": neural,
            "rules_weight": rules,
        }
        result = _derive_symbolic_retry_params(params)
        assert result is not None

        total = result["neural_weight"] + result["rules_weight"]
        assert abs(total - 1.0) < 1e-9, f"Weights sum to {total}, not 1.0"
