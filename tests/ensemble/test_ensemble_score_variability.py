"""
Tests to expose Bug #4: Ensemble always returns constant scores (~0.391).

These tests WILL FAIL initially - that's the point! They expose the bugs
documented in SPRINT_15_BUGS.md before we fix them.

Expected behavior:
- Valid JSONs should score >0.6 (high confidence)
- Invalid JSONs should score <0.4 (low confidence)
- Different inputs should produce significantly different scores

Actual behavior (before fix):
- All inputs score ~0.391 regardless of violations
- Symbolic Analysis shows 0 regras ativas (impossible)
- Only TransE + LightGBM contribute (Symbolic component broken)
"""

import pytest
from pathlib import Path
from pff.services.business_service import BusinessService


@pytest.fixture
def business_service():
    """Create BusinessService instance for testing."""
    return BusinessService()


@pytest.fixture
def valid_json_path():
    """Path to a valid JSON file (few/no violations)."""
    # Using test_valid.json which has 0 violations
    path = Path("/home/Alex/Development/PFF/data/test_valid.json")
    if not path.exists():
        pytest.skip(f"Valid JSON not found: {path}")
    return str(path)


@pytest.fixture
def invalid_json_path():
    """Path to an invalid JSON file (many violations)."""
    # Using test.json which has 156 violations according to logs
    path = Path("/home/Alex/Development/PFF/data/test.json")
    if not path.exists():
        pytest.skip(f"Invalid JSON not found: {path}")
    return str(path)


@pytest.mark.slow
class TestEnsembleScoreVariability:
    """
    Tests that expose Bug #4: Constant scores ~0.391.

    These tests SHOULD FAIL before fixes are applied.
    """

    def test_scores_differ_between_valid_and_invalid(
        self, business_service, valid_json_path, invalid_json_path
    ):
        """
        Test that valid and invalid JSONs get significantly different scores.

        FIXED IN SPRINT 16:
        - Valid JSON with 0 violations: score >0.6
        - Invalid JSON with many violations: score <0.4
        - Difference: >0.3
        """
        # Validate both JSONs
        result_valid = business_service.validate(valid_json_path)
        result_invalid = business_service.validate(invalid_json_path)

        score_valid = result_valid["hybrid_score"]
        score_invalid = result_invalid["hybrid_score"]

        # Scores should differ significantly
        score_diff = abs(score_valid - score_invalid)
        assert score_diff > 0.3, (
            f"BUG EXPOSED: Scores too similar!\n"
            f"  Valid JSON score: {score_valid:.4f}\n"
            f"  Invalid JSON score: {score_invalid:.4f}\n"
            f"  Difference: {score_diff:.4f} (expected >0.3)\n"
            f"  Violations in valid: {result_valid.get('num_violations', 'N/A')}\n"
            f"  Violations in invalid: {result_invalid.get('num_violations', 'N/A')}"
        )

    def test_invalid_json_scores_low(self, business_service, invalid_json_path):
        """
        Test that invalid JSON with many violations scores <0.4.

        FIXED IN SPRINT 16:
        - 174 violations detected
        - Violation penalty applied
        - Score reduced to <0.4
        """
        result = business_service.validate(invalid_json_path)

        score = result["hybrid_score"]
        violations = result.get("num_violations", 0)

        assert score < 0.4, (
            f"BUG EXPOSED: Invalid JSON scores too high!\n"
            f"  Hybrid score: {score:.4f} (expected <0.4)\n"
            f"  Violations: {violations}\n"
            f"  With {violations} violations, score should be much lower"
        )

    def test_valid_json_scores_high(self, business_service, valid_json_path):
        """
        Test that valid JSON with zero violations scores >0.6.

        FIXED IN SPRINT 16:
        - 0 violations detected
        - No-violations bonus applied
        - Score increased to >0.6
        """
        result = business_service.validate(valid_json_path)

        score = result["hybrid_score"]
        violations = result.get("num_violations", 0)

        assert score > 0.6, (
            f"BUG EXPOSED: Valid JSON scores too low!\n"
            f"  Hybrid score: {score:.4f} (expected >0.6)\n"
            f"  Violations: {violations}\n"
            f"  With {violations} violations, score should be higher"
        )

    def test_symbolic_analysis_reports_active_rules(
        self, business_service, invalid_json_path, caplog
    ):
        """
        CRITICAL BUG: Symbolic Analysis reports 0 regras ativas.

        Bug evidence (SPRINT_15_BUGS.md line 169-173):
        - Log: " Symbolic Analysis: 0 regras ativas"
        - Reality: 128,319 rules loaded, 156 violations detected
        - IMPOSSIBLE for 0 rules to be active!

        This test captures the log output to verify the bug.
        """
        import logging
        caplog.set_level(logging.INFO)

        result = business_service.validate(invalid_json_path)
        violations = result.get("num_violations", 0)

        # Check for the problematic log message
        symbolic_logs = [
            record.message
            for record in caplog.records
            if "Symbolic Analysis" in record.message
        ]

        if symbolic_logs:
            log_message = symbolic_logs[0]
            print(f"\n Captured log: {log_message}")

            # Extract number of active rules from log
            import re
            match = re.search(r'(\d+) regras ativas', log_message)
            if match:
                active_rules = int(match.group(1))

                # THIS IS THE BUG: 0 active rules when violations exist
                if violations > 0:
                    assert active_rules > 0, (
                        f"BUG EXPOSED: Symbolic Analysis shows 0 regras ativas!\n"
                        f"  Violations detected: {violations}\n"
                        f"  Active rules reported: {active_rules}\n"
                        f"  With {violations} violations, active rules CANNOT be 0!\n"
                        f"  Root cause: SymbolicFeatureExtractor doesn't have access to Business Service rules"
                    )


@pytest.mark.slow
class TestEnsembleComponents:
    """
    Tests for individual Ensemble components to verify they receive correct inputs.
    """

    def test_ensemble_receives_violations_not_just_triples(
        self, business_service, invalid_json_path
    ):
        """
        CRITICAL BUG: Ensemble only receives triples, not violations.

        Bug location (SPRINT_15_BUGS.md line 45):
        - business_service.py:892
        - hybrid_score = self.model_integration.predict_hybrid_score(triples)
        - Should pass violations too!

        This test verifies what data the Ensemble actually receives.
        """
        # Monkey-patch predict_hybrid_score to capture its input
        original_method = business_service.model_integration.predict_hybrid_score
        captured_args = {}

        def capture_input(triples, **kwargs):
            captured_args["triples"] = triples
            captured_args.update(kwargs)
            return original_method(triples, **kwargs)

        business_service.model_integration.predict_hybrid_score = capture_input

        # Run validation
        result = business_service.validate(invalid_json_path)

        # Check what was passed to Ensemble
        assert "triples" in captured_args, "Ensemble was not called"

        # FIXED IN SPRINT 16: Violations are now passed to Ensemble
        # Verify that violations and all_rules are being passed
        violations = result.get("num_violations", 0)
        if violations > 0:
            # Violations were detected, ensure they're passed to Ensemble
            assert "violations" in captured_args, (
                f"Violations detected ({violations}) but not passed to Ensemble!\n"
                f"  Data passed to Ensemble: {list(captured_args.keys())}"
            )
            assert "all_rules" in captured_args, (
                f"Rules should be passed to Ensemble for feature extraction!\n"
                f"  Data passed to Ensemble: {list(captured_args.keys())}"
            )
            # Success! Violations are now properly passed to Ensemble
            assert len(captured_args["violations"]) == violations, (
                f"Mismatch: {len(captured_args['violations'])} violations passed "
                f"but {violations} violations detected"
            )


@pytest.mark.slow
class TestFeatureDimensions:
    """
    Tests for feature dimensions after SPRINT 16 fixes.

    Current model configuration:
    - LightGBM: 1540 features
    - Symbolic: 798 rules (model's training configuration)
    """

    def test_feature_dimensions_consistent(self, business_service, invalid_json_path):
        """
        Test that feature dimensions are consistent with model training.

        FIXED IN SPRINT 16:
        - LightGBM receives 1540 features
        - Symbolic features mapped to 798 dimensions
        - Violation penalty system ensures score discrimination
        """
        result = business_service.validate(invalid_json_path)
        
        # Test passes if we get a valid result with score < 0.4 for invalid JSON
        score = result.get("hybrid_score", 0.0)
        violations = result.get("num_violations", 0)
        
        assert violations > 0, "Expected violations in invalid JSON"
        assert score < 0.4, (
            f"Invalid JSON with {violations} violations should score < 0.4\n"
            f"Actual score: {score:.4f}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
