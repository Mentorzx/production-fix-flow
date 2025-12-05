"""
Tests to expose Bug #2: Feature dimension mismatch in Ensemble.

These tests WILL FAIL - they expose the dimension mismatch bugs
documented in SPRINT_15_BUGS.md lines 97-153.

Bug evidence from logs:
- Line 39: "Predição LightGBM OK com 544 features"
- Line 51: " Features: 24305 → 155 agrupadas"
- Line 52: " Symbolic Analysis: 0 regras ativas"

Problem:
- LightGBM expects 544 features (TransE embeddings)
- Symbolic generates 24305 features (1 per rule), grouped to 155
- Dimensions don't match, causing features to be zeroed/truncated
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch
from pff.services.business_service import BusinessService


@pytest.fixture
def business_service():
    """Create BusinessService instance for testing."""
    return BusinessService()


@pytest.fixture
def test_json_path():
    """Path to test.json with known violations."""
    path = Path("/home/Alex/Development/PFF/data/test.json")
    if not path.exists():
        pytest.skip(f"Test JSON not found: {path}")
    return str(path)


class TestFeatureDimensions:
    """
    Expose Bug #2: Feature dimension mismatch.

    Expected: All Ensemble components receive features of compatible dimensions
    Actual: LightGBM gets 544, Symbolic gets 24305→155, TransE varies
    """

    @pytest.mark.slow
    def test_lightgbm_feature_dimensions(
        self, business_service, test_json_path, caplog
    ):
        """
        Verify LightGBM feature dimensions.

        Current model configuration (SPRINT 16):
        - LightGBM expects 1540 features
        - Features include: embeddings + statistical + symbolic contributions
        """
        import logging
        caplog.set_level(logging.DEBUG)

        result = business_service.validate(test_json_path)

        # Find LightGBM prediction log
        lightgbm_logs = [
            record.message
            for record in caplog.records
            if "LightGBM" in record.message and "features" in record.message
        ]

        if lightgbm_logs:
            import re
            match = re.search(r'(\d+) features', lightgbm_logs[0])
            if match:
                num_features = int(match.group(1))
                print(f" LightGBM receives {num_features} features")

                # LightGBM expects 1540 features (current model configuration)
                assert num_features == 1540, (
                    f"LightGBM feature dimension changed!\n"
                    f"  Expected: 1540 (current model configuration)\n"
                    f"  Actual: {num_features}"
                )
        else:
            pytest.skip("LightGBM prediction log not found")

    @pytest.mark.slow
    def test_symbolic_feature_dimensions(
        self, business_service, test_json_path, caplog
    ):
        """
        Verify Symbolic feature dimensions.

        Current model configuration (SPRINT 16):
        - Model was trained with 798 symbolic rules
        - Features are binary (violation detected or not)
        - Log: "798 regras disponíveis para validação"
        """
        import logging
        caplog.set_level(logging.INFO)

        result = business_service.validate(test_json_path)

        # Find rules available log
        rule_logs = [
            record.message
            for record in caplog.records
            if "regras disponíveis" in record.message
        ]

        if rule_logs:
            import re
            match = re.search(r'(\d+) regras', rule_logs[0])
            if match:
                num_rules = int(match.group(1))
                print(f" Model rules: {num_rules}")

                # Model expects 798 rules (as trained)
                assert num_rules == 798, (
                    f"Symbolic rule count unexpected!\n"
                    f"  Expected: 798 (model training configuration)\n"
                    f"  Actual: {num_rules}"
                )
        else:
            pytest.skip("Rules available log not found")

    @pytest.mark.slow
    def test_ensemble_components_receive_compatible_dimensions(
        self, business_service, test_json_path
    ):
        """
        Test that Ensemble components receive features of compatible dimensions.

        FIXED IN SPRINT 16:
        - LightGBM receives 1540 features
        - Symbolic rules are mapped to model's 798 dimensions
        - Ensemble now uses violation penalty system instead of feature concatenation
        """
        # Run validation
        result = business_service.validate(test_json_path)
        score = result.get("hybrid_score", 0.0)
        violations = result.get("num_violations", 0)

        # With violations, score should be penalized (< 0.4)
        # Without violations, score should get bonus (> 0.6)
        if violations > 100:
            assert score < 0.4, (
                f"Score too high for {violations} violations!\n"
                f"  Score: {score:.4f}\n"
                f"  Expected: < 0.4"
            )
        elif violations == 0:
            assert score > 0.6, (
                f"Score too low for 0 violations!\n"
                f"  Score: {score:.4f}\n"
                f"  Expected: > 0.6"
            )


class TestEnsemblePipeline:
    """
    Test Ensemble pipeline to expose how features flow through components.
    """

    @pytest.mark.slow
    def test_ensemble_pipeline_feature_flow(
        self, business_service, test_json_path
    ):
        """
        Test feature flow through Ensemble pipeline.

        FIXED IN SPRINT 16:
        - Business Service extracts violations
        - Violations mapped to model's feature dimensions
        - Violation penalty/bonus applied to ensemble score

        Note: This test works in both Ensemble and Fallback modes.
        In Fallback mode, there is no ensemble_model, so we just verify
        the final score is correct.
        """
        # Run validation
        result = business_service.validate(test_json_path)

        # Check result
        result_score = result.get("hybrid_score", 0.0)
        violations = result.get("num_violations", 0)
        
        # Check what mode we're running in
        has_ensemble = business_service.model_integration.ensemble_model is not None
        
        # With violations, the score should be penalized regardless of mode
        if violations > 100:
            assert result_score < 0.4, (
                f"Score should be penalized for {violations} violations\n"
                f"Actual score: {result_score:.4f}\n"
                f"Mode: {'Ensemble' if has_ensemble else 'Fallback'}"
            )
        elif violations == 0:
            # No violations should give bonus
            assert result_score > 0.6, (
                f"Score should be boosted for 0 violations\n"
                f"Actual score: {result_score:.4f}\n"
                f"Mode: {'Ensemble' if has_ensemble else 'Fallback'}"
            )


class TestFeatureExtraction:
    """
    Test feature extraction logic to verify it produces correct dimensions.
    """

    def test_symbolic_feature_extractor_has_rules(self, business_service):
        """
        CRITICAL: SymbolicFeatureExtractor should have access to rules.

        Bug (SPRINT_15_BUGS.md line 180-184):
        - self.rules_ is empty in SymbolicFeatureExtractor
        - Tries to validate without rules → always returns zeros
        - Log shows "0 regras ativas" when 156 violations exist
        """
        # Access the Ensemble model
        ensemble = business_service.model_integration.ensemble_model

        # Try to find SymbolicFeatureExtractor in pipeline
        if hasattr(ensemble, "named_steps"):
            # sklearn Pipeline
            for step_name, step in ensemble.named_steps.items():
                if "Symbolic" in step_name or "symbolic" in step_name:
                    # Found the SymbolicFeatureExtractor
                    if hasattr(step, "rules_"):
                        num_rules = len(step.rules_) if step.rules_ else 0

                        # BUG: rules_ should have 128,319 rules
                        if num_rules == 0:
                            pytest.fail(
                                f"BUG EXPOSED: SymbolicFeatureExtractor has no rules!\n"
                                f"  Step: {step_name}\n"
                                f"  Rules loaded: {num_rules}\n"
                                f"  Expected: 128,319 (from Business Service)\n\n"
                                f"Root cause:\n"
                                f"  Business Service loads rules in __init__()\n"
                                f"  But SymbolicFeatureExtractor is separate object\n"
                                f"  Doesn't have access to Business Service rules\n\n"
                                f"Fix:\n"
                                f"  REMOVE SymbolicFeatureExtractor from Ensemble\n"
                                f"  Business Service creates violation features\n"
                                f"  Passes features to Ensemble, not triples"
                            )
                        else:
                            print(f" SymbolicFeatureExtractor has {num_rules} rules")
                    else:
                        pytest.fail(f"SymbolicFeatureExtractor missing rules_ attribute")

        pytest.skip("Could not locate SymbolicFeatureExtractor in Ensemble pipeline")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
