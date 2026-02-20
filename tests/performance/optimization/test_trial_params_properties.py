"""Property tests for trial parameter validation.

Tests that trial parameters satisfy constraints:
(1) Weight pairs (Neural + Rules) must sum to ~1.0
(2) Thresholds must be in valid ranges [0, 1]
(3) Parameter dependencies are satisfied
(4) Invalid parameters are rejected early
"""

from __future__ import annotations

import math

import pytest

# ============================================================================
# Parameter validation functions (matching production logic)
# ============================================================================


def validate_weight_pair(
    neural: float,
    rules: float,
    tolerance: float = 0.05,
) -> tuple[bool, str]:
    """Validate that ensemble weights sum to ~1.0."""
    total = neural + rules
    if abs(total - 1.0) > tolerance:
        return False, f"Weights sum to {total:.3f}, expected ~1.0"
    if any(w < 0 for w in [neural, rules]):
        return False, "Negative weight detected"
    if any(w > 1 for w in [neural, rules]):
        return False, "Weight > 1.0 detected"
    return True, "OK"


def validate_threshold(value: float, name: str) -> tuple[bool, str]:
    """Validate threshold is in [0, 1]."""
    if math.isnan(value):
        return False, f"{name} is NaN"
    if value < 0:
        return False, f"{name} < 0: {value}"
    if value > 1:
        return False, f"{name} > 1: {value}"
    return True, "OK"


def validate_trial_params(params: dict) -> tuple[bool, list[str]]:
    """Validate complete trial parameters."""
    errors = []

    # Validate weights if present
    if all(k in params for k in ["neural_weight", "rules_weight"]):
        valid, msg = validate_weight_pair(
            params["neural_weight"],
            params["rules_weight"],
        )
        if not valid:
            errors.append(msg)

    # Validate thresholds
    for thresh_name in ["neural_threshold", "rules_threshold"]:
        if thresh_name in params:
            valid, msg = validate_threshold(params[thresh_name], thresh_name)
            if not valid:
                errors.append(msg)

    # Validate target_symbolic_ratio
    if "target_symbolic_ratio" in params:
        ratio = params["target_symbolic_ratio"]
        if ratio < 0 or ratio > 1:
            errors.append(f"target_symbolic_ratio out of [0, 1]: {ratio}")

    return len(errors) == 0, errors


# ============================================================================
# Tests: Weight validation
# ============================================================================


class TestWeightPairValidation:
    """Test weight pair validation."""

    @pytest.mark.parametrize(
        "neural,rules",
        [
            (0.4, 0.6),
            (0.5, 0.5),
            (0.1, 0.9),
            (0.8, 0.2),
            (0.0, 1.0),
        ],
    )
    def test_valid_weight_pairs(self, neural: float, rules: float):
        """Property: weights summing to 1.0 should be valid."""
        valid, _ = validate_weight_pair(neural, rules)
        assert valid, f"Valid weights rejected: {neural}, {rules}"

    @pytest.mark.parametrize(
        "neural,rules",
        [
            (0.6, 0.6),
            (0.1, 0.1),
            (0.0, 0.0),
        ],
    )
    def test_invalid_weight_sums(self, neural: float, rules: float):
        """Property: weights not summing to ~1.0 should be invalid."""
        valid, msg = validate_weight_pair(neural, rules)
        assert not valid, f"Invalid weights accepted: {neural}, {rules}"
        assert "sum" in msg.lower()

    def test_negative_weight_rejected(self):
        """Property: negative weights should be rejected."""
        valid, msg = validate_weight_pair(-0.1, 1.1)
        assert not valid
        assert "negative" in msg.lower()

    def test_weight_over_one_rejected(self):
        """Property: weight > 1.0 should be rejected."""
        valid, msg = validate_weight_pair(1.5, -0.5)
        assert not valid

    def test_tolerance_respected(self):
        """Property: small deviations within tolerance should pass."""
        # 0.99 sum with 0.05 tolerance should pass
        valid, _ = validate_weight_pair(0.49, 0.50, tolerance=0.05)
        assert valid

        # 0.90 sum with 0.05 tolerance should fail
        valid, _ = validate_weight_pair(0.45, 0.45, tolerance=0.05)
        assert not valid


class TestThresholdValidation:
    """Test threshold validation."""

    @pytest.mark.parametrize("value", [0.0, 0.3, 0.5, 0.7, 1.0])
    def test_valid_thresholds(self, value: float):
        """Property: thresholds in [0, 1] should be valid."""
        valid, _ = validate_threshold(value, "test_threshold")
        assert valid

    @pytest.mark.parametrize("value", [-0.1, -1.0, 1.1, 2.0, 100.0])
    def test_invalid_thresholds(self, value: float):
        """Property: thresholds outside [0, 1] should be invalid."""
        valid, msg = validate_threshold(value, "test_threshold")
        assert not valid
        assert "test_threshold" in msg

    def test_nan_threshold_rejected(self):
        """Property: NaN threshold should be rejected."""
        valid, msg = validate_threshold(float("nan"), "test_threshold")
        assert not valid
        assert "nan" in msg.lower()


class TestTrialParamsValidation:
    """Test complete trial parameter validation."""

    def test_valid_params_accepted(self):
        """Property: valid parameter set should pass validation."""
        params = {
            "neural_weight": 0.4,
            "rules_weight": 0.6,
            "neural_threshold": 0.5,
            "rules_threshold": 0.4,
            "target_symbolic_ratio": 0.35,
        }
        valid, errors = validate_trial_params(params)
        assert valid, f"Valid params rejected: {errors}"

    def test_missing_params_partial_validation(self):
        """Property: missing params should not cause failure, only validate present ones."""
        params = {
            "neural_threshold": 0.5,
        }
        valid, errors = validate_trial_params(params)
        assert valid, f"Partial params rejected: {errors}"

    def test_invalid_weight_detected(self):
        """Property: invalid weights should be detected."""
        params = {
            "neural_weight": 0.5,
            "rules_weight": 0.8,
        }
        valid, errors = validate_trial_params(params)
        assert not valid
        assert any("sum" in e.lower() for e in errors)

    def test_invalid_threshold_detected(self):
        """Property: invalid threshold should be detected."""
        params = {
            "neural_threshold": 1.5,
        }
        valid, errors = validate_trial_params(params)
        assert not valid
        assert any("neural_threshold" in e for e in errors)

    def test_invalid_symbolic_ratio_detected(self):
        """Property: invalid target_symbolic_ratio should be detected."""
        params = {
            "target_symbolic_ratio": 1.5,
        }
        valid, errors = validate_trial_params(params)
        assert not valid
        assert any("symbolic_ratio" in e for e in errors)

    def test_multiple_errors_collected(self):
        """Property: multiple errors should all be reported."""
        params = {
            "neural_weight": 0.5,
            "rules_weight": 0.8,
            "neural_threshold": 1.5,
            "target_symbolic_ratio": -0.1,
        }
        valid, errors = validate_trial_params(params)
        assert not valid
        assert len(errors) >= 3, f"Expected 3+ errors, got {len(errors)}: {errors}"


# ============================================================================
# Tests: Parameter generation properties
# ============================================================================


class TestParameterGeneration:
    """Test properties of generated parameters."""

    @staticmethod
    def generate_weight_pair(seed: int = 42) -> tuple[float, float]:
        """Generate valid weight pair using Dirichlet-like distribution."""
        import numpy as np

        rng = np.random.RandomState(seed)
        # Generate 2 positive values and normalize
        raw = rng.exponential(1.0, 2)
        normalized = raw / raw.sum()
        return tuple(normalized)

    def test_generated_weights_sum_to_one(self):
        """Property: generated weights should always sum to 1.0."""
        for seed in range(100):
            weights = self.generate_weight_pair(seed)
            total = sum(weights)
            assert abs(total - 1.0) < 1e-9, f"Seed {seed}: sum = {total}"

    def test_generated_weights_all_positive(self):
        """Property: generated weights should all be positive."""
        for seed in range(100):
            weights = self.generate_weight_pair(seed)
            assert all(w > 0 for w in weights), f"Seed {seed}: {weights}"

    def test_generated_weights_valid(self):
        """Property: generated weights should pass validation."""
        for seed in range(100):
            neural, rules = self.generate_weight_pair(seed)
            valid, msg = validate_weight_pair(neural, rules)
            assert valid, f"Seed {seed}: {msg}"
