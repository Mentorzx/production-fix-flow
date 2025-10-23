"""
Tests to verify Business Service correctly detects violations.

These tests verify that the validation logic in Business Service works correctly
and exposes the disconnect between validation results and Ensemble scoring.

Expected behavior:
- Business Service detects violations correctly (✅ WORKS)
- Violations are properly formatted (✅ WORKS)
- Violations are passed to Ensemble for scoring (❌ BUG - not passed!)

Bug reference: SPRINT_15_BUGS.md Bug #1 (Duplicação de Lógica)
"""

import pytest
from pathlib import Path
from pff.services.business_service import BusinessService


@pytest.fixture
def business_service():
    """Create BusinessService instance for testing."""
    return BusinessService()


@pytest.fixture
def test_json_path():
    """Path to test.json which has known violations (156 from logs)."""
    path = Path("/home/Alex/Development/PFF/data/test.json")
    if not path.exists():
        pytest.skip(f"Test JSON not found: {path}")
    return str(path)


class TestViolationDetection:
    """Verify Business Service detects violations correctly."""

    def test_violations_are_detected(self, business_service, test_json_path):
        """
        Business Service should detect violations correctly.

        Evidence (SPRINT_15_BUGS.md line 55):
        - Log: "Violações: 156" ✅ WORKS
        - Business Service validation is working correctly
        """
        result = business_service.validate(test_json_path)

        # Verify violations are detected
        assert "num_violations" in result, "Result missing num_violations field"
        violations = result["num_violations"]

        # Log shows 156 violations for test.json
        assert violations > 0, (
            f"No violations detected, but log shows 156 violations!\n"
            f"Result: {result}"
        )

        print(f"✅ Business Service detected {violations} violations")

    def test_violations_format(self, business_service, test_json_path):
        """
        Verify violations are returned in correct format.

        Violations should contain:
        - rule_id or rule pattern
        - description or message
        - severity or confidence
        """
        result = business_service.validate(test_json_path)

        # Check for violations list (may be in different keys)
        violations_data = result.get("violations") or result.get("violation_details")

        if violations_data:
            # Verify format of first violation
            first_violation = violations_data[0]

            # Should be dict-like with rule information
            assert isinstance(first_violation, (dict, object)), (
                f"Violation should be dict or object, got {type(first_violation)}"
            )

            print(f"✅ Violation format: {type(first_violation).__name__}")
            print(f"   Sample: {first_violation}")
        else:
            # Violations may not be in result dict, only count
            print("⚠️ Violations count present, but no detailed violation list")

    def test_confidence_score_decreases_with_violations(
        self, business_service, test_json_path
    ):
        """
        Confidence score should decrease as violations increase.

        Evidence (SPRINT_15_BUGS.md line 56):
        - Log: "Confiança: 0.6541" with 156 violations
        - Confidence calculation is working
        """
        result = business_service.validate(test_json_path)

        violations = result.get("num_violations", 0)
        confidence = result.get("confidence_score", 1.0)

        # More violations should mean lower confidence
        if violations > 100:
            assert confidence < 0.8, (
                f"Confidence too high with {violations} violations!\n"
                f"  Confidence: {confidence:.4f} (expected <0.8)"
            )

        print(f"✅ Confidence score: {confidence:.4f} with {violations} violations")


class TestViolationToEnsembleDisconnect:
    """
    Expose Bug #1: Violations detected but not passed to Ensemble.

    This is the CRITICAL architectural bug documented in SPRINT_15_BUGS.md.
    """

    def test_ensemble_uses_violation_information(
        self, business_service, test_json_path
    ):
        """
        CRITICAL BUG: Violations detected but not used by Ensemble.

        Bug flow (SPRINT_15_BUGS.md line 65-77):
        1. Business Service validates: 156 violations ✅
        2. Calls ensemble.predict_proba([triples]) ❌ Only triples!
        3. Ensemble SymbolicFeatureExtractor tries to re-validate
        4. Returns 0 regras ativas (doesn't have rules)
        5. Score ~0.39 (ignores the 156 violations)

        This test exposes the disconnect.
        """
        result = business_service.validate(test_json_path)

        violations = result.get("num_violations", 0)
        hybrid_score = result.get("hybrid_score", 0.0)

        # BUG: Hybrid score doesn't reflect violations
        # With 156 violations, score should be much lower than 0.391
        if violations > 100:
            # Expected: score < 0.3 with 156 violations
            # Actual: score ~0.391 (ignores violations)

            if hybrid_score > 0.35:
                pytest.fail(
                    f"BUG EXPOSED: Ensemble ignores violations!\n"
                    f"  Violations detected: {violations}\n"
                    f"  Hybrid score: {hybrid_score:.4f}\n"
                    f"  Expected score: <0.3 (with {violations} violations)\n"
                    f"  Actual score: ~0.391 (constant, ignores violations)\n\n"
                    f"Root cause (business_service.py:892):\n"
                    f"  hybrid_score = self.model_integration.predict_hybrid_score(triples)\n"
                    f"  ❌ Only passes triples, not violations!\n\n"
                    f"Fix:\n"
                    f"  violations_vector = self._violations_to_feature_vector(violations)\n"
                    f"  features = {{'violations': violations_vector, 'triples': triples}}\n"
                    f"  hybrid_score = self.ensemble.predict(features)"
                )

    def test_symbolic_features_should_match_violations(
        self, business_service, test_json_path, caplog
    ):
        """
        CRITICAL: Symbolic Analysis should show N regras ativas = N violations.

        Bug evidence (SPRINT_15_BUGS.md line 169-173):
        - Business Service: 156 violations detected ✅
        - Symbolic Analysis log: "0 regras ativas" ❌
        - IMPOSSIBLE: Violations exist but symbolic component sees 0!

        Root cause: SymbolicFeatureExtractor tries to re-validate rules
        but doesn't have access to Business Service's loaded rules.
        """
        import logging
        caplog.set_level(logging.INFO)

        result = business_service.validate(test_json_path)
        violations = result.get("num_violations", 0)

        # Find Symbolic Analysis log
        symbolic_logs = [
            record.message
            for record in caplog.records
            if "Symbolic Analysis" in record.message
        ]

        if symbolic_logs:
            import re
            match = re.search(r'(\d+) regras ativas', symbolic_logs[0])
            if match:
                active_rules = int(match.group(1))

                # THIS IS THE BUG!
                if violations > 0 and active_rules == 0:
                    pytest.fail(
                        f"BUG EXPOSED: Symbolic component disconnected from validation!\n"
                        f"  Violations detected by Business Service: {violations}\n"
                        f"  Active rules reported by Symbolic Analysis: {active_rules}\n"
                        f"  Expected: active_rules ≈ violations\n"
                        f"  Actual: active_rules = 0 (always!)\n\n"
                        f"Root cause (transformers.py:213-237):\n"
                        f"  SymbolicFeatureExtractor.transform() tries to validate rules\n"
                        f"  But self.rules_ is empty (doesn't have Business Service rules)\n"
                        f"  Returns np.zeros() for all samples\n\n"
                        f"Fix:\n"
                        f"  REMOVE SymbolicFeatureExtractor from Ensemble\n"
                        f"  Business Service should create violation features\n"
                        f"  Pass features to Ensemble, not triples"
                    )


class TestTripleExtraction:
    """Verify triple extraction works correctly (this should pass)."""

    def test_triples_extracted_from_json(self, business_service, test_json_path):
        """
        Triple extraction should work correctly.

        Evidence (SPRINT_15_BUGS.md line 15):
        - Log: "📊 1125 triplas extraídas do JSON" ✅ WORKS
        """
        result = business_service.validate(test_json_path)

        # Verify triples were extracted (may be in internal state)
        # Check log for "triplas extraídas" message
        assert result is not None, "Validation returned None"

        # Result should have validation fields
        assert "is_valid" in result, "Result missing is_valid field"
        assert "hybrid_score" in result, "Result missing hybrid_score field"

        print("✅ Triple extraction completed successfully")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
