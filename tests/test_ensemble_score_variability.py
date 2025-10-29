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
    # Using test1.json which should have fewer violations
    path = Path("/home/Alex/Development/PFF/data/test1.json")
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


class TestEnsembleScoreVariability:
    """
    Tests that expose Bug #4: Constant scores ~0.391.

    These tests SHOULD FAIL before fixes are applied.
    """

    @pytest.mark.xfail(
        reason="Bug #4: Ensemble returns constant scores ~0.391 (SPRINT_15_BUGS.md)"
    )
    def test_scores_differ_between_valid_and_invalid(
        self, business_service, valid_json_path, invalid_json_path
    ):
        """
        EXPECTED TO FAIL: Expose that valid and invalid JSONs get same score.

        Bug evidence (SPRINT_15_BUGS.md line 227-231):
        - Log: "✅ Ensemble score: 0.391" (1125 triplas)
        - Log: "✅ Ensemble score: 0.391" (294 triplas)
        - Same exact score for different inputs!
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

    @pytest.mark.xfail(
        reason="Bug #4: Invalid JSON scores too high ~0.391 (should be <0.4)"
    )
    def test_invalid_json_scores_low(self, business_service, invalid_json_path):
        """
        EXPECTED TO FAIL: Invalid JSON with 156 violations should score <0.4.

        Bug evidence (SPRINT_15_BUGS.md line 53-57):
        - Ensemble score: 0.391
        - Violations: 156
        - Should be INVALID (score <0.4), but scores at threshold
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

    @pytest.mark.xfail(
        reason="Bug #4: Valid JSON scores too low (should be >0.6)"
    )
    def test_valid_json_scores_high(self, business_service, valid_json_path):
        """
        EXPECTED TO FAIL: Valid JSON with few violations should score >0.6.

        Current behavior: All JSONs score ~0.391 regardless of violations
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
        - Log: "🔍 Symbolic Analysis: 0 regras ativas"
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
            print(f"\n🔍 Captured log: {log_message}")

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

        # BUG: Only triples are passed, no violations information
        # The Ensemble has no way to know about the 156 violations!
        violations = result.get("num_violations", 0)
        if violations > 0:
            pytest.fail(
                f"BUG EXPOSED: Ensemble only receives triples!\n"
                f"  Violations detected: {violations}\n"
                f"  Data passed to Ensemble: {list(captured_args.keys())}\n"
                f"  Missing: violations, confidence_score, rule satisfaction\n"
                f"  Location: business_service.py:892 - predict_hybrid_score(triples)\n"
                f"  Fix: Pass violations as features to Ensemble"
            )


class TestFeatureDimensions:
    """
    Tests to expose Bug #2: Feature dimension mismatch.

    Bug evidence (SPRINT_15_BUGS.md line 108-112):
    - LightGBM: 544 features (TransE embeddings)
    - Transformer: 24305 → 155 features (grouped)
    - Symbolic: 24305 features (1 per rule)
    - Result: Dimensions incompatible!
    """

    @pytest.mark.xfail(
        reason="Bug #2: Feature dimensions mismatch (544 vs 155 vs 24305)"
    )
    def test_feature_dimensions_consistent(self, business_service, invalid_json_path):
        """
        EXPECTED TO FAIL: Expose feature dimension mismatch.

        All Ensemble components should receive features of compatible dimensions.
        """
        # This test requires instrumenting the Ensemble to capture feature shapes
        # For now, we document the expected behavior

        pytest.skip(
            "Requires instrumentation of Ensemble components.\n"
            "Bug documented in SPRINT_15_BUGS.md:\n"
            "  - LightGBM expects: 544 features\n"
            "  - Transformer groups to: 155 features\n"
            "  - Symbolic should be: N_violations (variable)\n"
            "  - Current: Dimensions don't match, features zeroed/truncated"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
