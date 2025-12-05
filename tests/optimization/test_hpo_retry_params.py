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
    
    Mirrors logic from scripts/optimization/core.py.
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
    fallback["rules_weight"] = float(
        np.clip(current_params.get("rules_weight", 0.18), 0.12, 0.25)
    )
    fallback["lightgbm_weight"] = float(
        np.clip(current_params.get("lightgbm_weight", 0.55), 0.5, 0.65)
    )
    fallback["neural_weight"] = float(
        np.clip(
            1.0 - fallback["rules_weight"] - fallback["lightgbm_weight"],
            0.15,
            0.45,
        )
    )
    total_weight = (
        fallback["neural_weight"]
        + fallback["rules_weight"]
        + fallback["lightgbm_weight"]
    )
    if total_weight <= 0:
        return None
    scale = 1.0 / total_weight
    fallback["neural_weight"] *= scale
    fallback["rules_weight"] *= scale
    fallback["lightgbm_weight"] *= scale
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
            "neural_weight": 0.3,
            "rules_weight": 0.2,
            "lightgbm_weight": 0.5,
        }
        result = _derive_symbolic_retry_params(params)
        assert result is not None
        
        total = result["neural_weight"] + result["rules_weight"] + result["lightgbm_weight"]
        assert abs(total - 1.0) < 1e-9

    def test_feature_selection_threshold_halved(self):
        """Feature selection threshold should be halved."""
        params = {"feature_selection_threshold": 0.3}
        result = _derive_symbolic_retry_params(params)
        # 0.3 * 0.5 = 0.15, within [0.03, 0.2]
        assert result["feature_selection_threshold"] == 0.15

    def test_feature_selection_threshold_clamped_low(self):
        """Feature selection threshold should be clamped to min 0.03."""
        params = {"feature_selection_threshold": 0.04}
        result = _derive_symbolic_retry_params(params)
        # 0.04 * 0.5 = 0.02, clamped to 0.03
        assert result["feature_selection_threshold"] == 0.03

    def test_feature_selection_threshold_clamped_high(self):
        """Feature selection threshold should be clamped to max 0.2."""
        params = {"feature_selection_threshold": 0.6}
        result = _derive_symbolic_retry_params(params)
        # 0.6 * 0.5 = 0.3, clamped to 0.2
        assert result["feature_selection_threshold"] == 0.2

    def test_rules_threshold_reduced(self):
        """Rules threshold should be reduced by 20%."""
        params = {"rules_threshold": 0.5}
        result = _derive_symbolic_retry_params(params)
        # 0.5 * 0.8 = 0.4, within [0.15, 0.45]
        assert result["rules_threshold"] == 0.4

    def test_target_symbolic_ratio_clamped(self):
        """Target symbolic ratio should be clamped to [0.3, 0.42]."""
        # Test below range
        result_low = _derive_symbolic_retry_params({"target_symbolic_ratio": 0.1})
        assert result_low["target_symbolic_ratio"] == 0.3
        
        # Test above range
        result_high = _derive_symbolic_retry_params({"target_symbolic_ratio": 0.9})
        assert result_high["target_symbolic_ratio"] == 0.42

    def test_extreme_weights_still_normalize(self):
        """Extreme weight values should still normalize to 1.0."""
        params = {
            "neural_weight": 0.01,
            "rules_weight": 0.01,
            "lightgbm_weight": 0.98,
        }
        result = _derive_symbolic_retry_params(params)
        assert result is not None
        
        total = result["neural_weight"] + result["rules_weight"] + result["lightgbm_weight"]
        assert abs(total - 1.0) < 1e-9


class TestWeightClippingBehavior:
    """Test weight clipping edge cases."""

    def test_neural_weight_derivation(self):
        """Neural weight is derived from 1 - rules - lgbm."""
        params = {
            "rules_weight": 0.2,
            "lightgbm_weight": 0.6,
        }
        result = _derive_symbolic_retry_params(params)
        
        # Before normalization: neural = 1.0 - 0.2 - 0.6 = 0.2
        # Clamped to [0.15, 0.45] = 0.2
        # Then normalized
        assert result["neural_weight"] > 0

    def test_all_weights_at_minimum(self):
        """When all weights at min, normalization should work."""
        params = {
            "rules_weight": 0.05,  # Will be clamped to 0.12
            "lightgbm_weight": 0.3,  # Will be clamped to 0.5
        }
        result = _derive_symbolic_retry_params(params)
        
        # After clipping: rules=0.12, lgbm=0.5
        # neural = 1.0 - 0.12 - 0.5 = 0.38, clamped to [0.15, 0.45] = 0.38
        # Total before norm: 0.12 + 0.5 + 0.38 = 1.0 (already normalized)
        total = result["neural_weight"] + result["rules_weight"] + result["lightgbm_weight"]
        assert abs(total - 1.0) < 1e-9

    def test_zero_weight_handling(self):
        """Zero weights should be clamped to minimums."""
        params = {
            "neural_weight": 0.0,
            "rules_weight": 0.0,
            "lightgbm_weight": 0.0,
        }
        result = _derive_symbolic_retry_params(params)
        assert result is not None
        # All should be > 0 after clipping
        assert result["neural_weight"] > 0
        assert result["rules_weight"] > 0
        assert result["lightgbm_weight"] > 0


class TestSymbolicRetryInvariants:
    """Test invariants that should always hold."""

    @pytest.mark.parametrize("neural,rules,lgbm", [
        (0.1, 0.1, 0.8),
        (0.4, 0.3, 0.3),
        (0.33, 0.33, 0.34),
        (0.5, 0.25, 0.25),
        (0.2, 0.2, 0.6),
    ])
    def test_output_weights_always_sum_to_one(self, neural: float, rules: float, lgbm: float):
        """Output weights must always sum to 1.0."""
        params = {
            "neural_weight": neural,
            "rules_weight": rules,
            "lightgbm_weight": lgbm,
        }
        result = _derive_symbolic_retry_params(params)
        assert result is not None
        
        total = result["neural_weight"] + result["rules_weight"] + result["lightgbm_weight"]
        assert abs(total - 1.0) < 1e-9, f"Weights sum to {total}, not 1.0"

    @pytest.mark.parametrize("fst", [0.01, 0.1, 0.2, 0.5, 1.0])
    def test_feature_selection_always_in_bounds(self, fst: float):
        """Feature selection threshold must be in [0.03, 0.2]."""
        params = {"feature_selection_threshold": fst}
        result = _derive_symbolic_retry_params(params)
        
        assert 0.03 <= result["feature_selection_threshold"] <= 0.2

    @pytest.mark.parametrize("tsr", [0.0, 0.25, 0.35, 0.5, 1.0])
    def test_target_symbolic_ratio_always_in_bounds(self, tsr: float):
        """Target symbolic ratio must be in [0.3, 0.42]."""
        params = {"target_symbolic_ratio": tsr}
        result = _derive_symbolic_retry_params(params)
        
        assert 0.3 <= result["target_symbolic_ratio"] <= 0.42

    def test_no_nan_or_inf_in_output(self):
        """Output should never contain NaN or Inf."""
        params = {
            "neural_weight": 0.3,
            "rules_weight": 0.2,
            "lightgbm_weight": 0.5,
            "feature_selection_threshold": 0.15,
            "target_symbolic_ratio": 0.35,
            "rules_threshold": 0.3,
        }
        result = _derive_symbolic_retry_params(params)
        
        for key, value in result.items():
            assert not np.isnan(value), f"{key} is NaN"
            assert not np.isinf(value), f"{key} is Inf"
