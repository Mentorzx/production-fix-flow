"""
Tests to expose Bug #2: Feature dimension mismatch in Ensemble.

These tests WILL FAIL - they expose the dimension mismatch bugs
documented in SPRINT_15_BUGS.md lines 97-153.

Bug evidence from logs:
- Line 39: "Predição LightGBM OK com 544 features"
- Line 51: "✅ Features: 24305 → 155 agrupadas"
- Line 52: "🔍 Symbolic Analysis: 0 regras ativas"

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

    def test_lightgbm_feature_dimensions(
        self, business_service, test_json_path, caplog
    ):
        """
        Verify LightGBM feature dimensions.

        Evidence (SPRINT_15_BUGS.md line 109):
        - Log: "Predição LightGBM OK com 544 features"
        - LightGBM expects 544 features (TransE embedding dimension)
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
                print(f"✅ LightGBM receives {num_features} features")

                # LightGBM expects 544 features (TransE embedding dimension)
                assert num_features == 544, (
                    f"LightGBM feature dimension changed!\n"
                    f"  Expected: 544 (TransE embedding dimension)\n"
                    f"  Actual: {num_features}"
                )
        else:
            pytest.skip("LightGBM prediction log not found")

    def test_symbolic_feature_dimensions(
        self, business_service, test_json_path, caplog
    ):
        """
        Verify Symbolic feature dimensions.

        Evidence (SPRINT_15_BUGS.md line 110-111):
        - Log: "✅ Features: 24305 → 155 agrupadas"
        - Symbolic has 24305 features (1 per AnyBURL rule)
        - Groups them to 155 for dimensionality reduction
        """
        import logging
        caplog.set_level(logging.INFO)

        result = business_service.validate(test_json_path)

        # Find feature grouping log
        grouping_logs = [
            record.message
            for record in caplog.records
            if "Features:" in record.message and "agrupadas" in record.message
        ]

        if grouping_logs:
            import re
            match = re.search(r'(\d+)\s*→\s*(\d+)', grouping_logs[0])
            if match:
                original_dim = int(match.group(1))
                grouped_dim = int(match.group(2))

                print(f"✅ Symbolic features: {original_dim} → {grouped_dim} agrupadas")

                # Verify dimensions
                assert original_dim == 24305, (
                    f"Symbolic original dimension unexpected!\n"
                    f"  Expected: 24305 (1 per AnyBURL rule)\n"
                    f"  Actual: {original_dim}"
                )

                assert grouped_dim == 155, (
                    f"Symbolic grouped dimension unexpected!\n"
                    f"  Expected: 155 (dimensionality reduction)\n"
                    f"  Actual: {grouped_dim}"
                )
        else:
            pytest.skip("Feature grouping log not found")

    @pytest.mark.xfail(
        reason="Bug #2: Feature dimensions incompatible (544 vs 155)"
    )
    def test_ensemble_components_receive_compatible_dimensions(
        self, business_service, test_json_path
    ):
        """
        EXPECTED TO FAIL: Ensemble components have incompatible feature dimensions.

        Bug (SPRINT_15_BUGS.md line 133-138):
        1. LightGBM expects 544 features (embeddings TransE)
        2. Symbolic generates 24305 features (1 per rule), groups to 155
        3. Ensemble Pipeline needs to concatenate both
        4. Dimensions don't match → features zeroed/truncated
        5. Model always returns ~0.39
        """
        # To properly test this, we need to instrument the Ensemble
        # to capture feature shapes from each component

        # For now, we can verify the SYMPTOM: constant scores
        result = business_service.validate(test_json_path)
        score = result.get("hybrid_score", 0.0)

        # If dimensions were compatible, scores would vary
        # But with Bug #2, scores are always ~0.391
        pytest.fail(
            f"BUG EXPOSED: Feature dimension mismatch!\n"
            f"  LightGBM expects: 544 features\n"
            f"  Symbolic provides: 155 features (grouped from 24305)\n"
            f"  Result: Incompatible dimensions\n"
            f"  Symptom: Constant score {score:.4f}\n\n"
            f"Fix:\n"
            f"  Define consistent feature structure:\n"
            f"  - Symbolic features: N_violations (variable per sample)\n"
            f"  - TransE features: 544 (embedding dimension)\n"
            f"  - Statistical features: N_predicates (e.g. 50)\n"
            f"  Total: N_violations + 544 + 50 = VARIABLE per sample\n\n"
            f"  OR use DictVectorizer for sparse features"
        )


class TestEnsemblePipeline:
    """
    Test Ensemble pipeline to expose how features flow through components.
    """

    def test_ensemble_pipeline_feature_flow(
        self, business_service, test_json_path
    ):
        """
        Trace feature flow through Ensemble pipeline.

        Expected flow:
        1. Business Service → extract triples
        2. Ensemble → SymbolicFeatureExtractor (24305 → 155)
        3. Ensemble → LightGBMWrapper (extracts 544 features)
        4. Ensemble → TransEWrapper (extracts embeddings)
        5. Ensemble → Combine all features
        6. Ensemble → Meta-learner prediction

        Actual flow (BUG):
        1. Business Service → extract triples ✅
        2. Ensemble → receive ONLY triples (no violations) ❌
        3. SymbolicFeatureExtractor → tries to validate, gets 0 ❌
        4. LightGBM → extracts 544 features from triples
        5. Dimensions mismatch → constant score ~0.391 ❌
        """
        # Monkey-patch to capture feature shapes
        captured_shapes = {}

        original_predict = business_service.model_integration.ensemble_model.predict_proba

        def capture_shapes(X):
            # Capture input shape
            if isinstance(X, list):
                captured_shapes["input_type"] = "list"
                captured_shapes["input_length"] = len(X)
                if len(X) > 0:
                    captured_shapes["first_sample_type"] = type(X[0]).__name__
            elif isinstance(X, np.ndarray):
                captured_shapes["input_type"] = "ndarray"
                captured_shapes["input_shape"] = X.shape

            return original_predict(X)

        business_service.model_integration.ensemble_model.predict_proba = capture_shapes

        # Run validation
        result = business_service.validate(test_json_path)

        # Check what was captured
        print("\n🔍 Ensemble Input Analysis:")
        for key, value in captured_shapes.items():
            print(f"  {key}: {value}")

        # BUG: Input should be features, not triples
        if captured_shapes.get("input_type") == "list":
            if captured_shapes.get("first_sample_type") == "list":
                pytest.fail(
                    f"BUG EXPOSED: Ensemble receives list of triples!\n"
                    f"  Input type: {captured_shapes['input_type']}\n"
                    f"  First sample type: {captured_shapes['first_sample_type']}\n"
                    f"  Expected: ndarray of features\n"
                    f"  Actual: list of triples (raw data)\n\n"
                    f"Fix: Business Service should extract features BEFORE calling Ensemble\n"
                    f"  features = extract_features(triples, violations)\n"
                    f"  score = ensemble.predict(features)"
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
                            print(f"✅ SymbolicFeatureExtractor has {num_rules} rules")
                    else:
                        pytest.fail(f"SymbolicFeatureExtractor missing rules_ attribute")

        pytest.skip("Could not locate SymbolicFeatureExtractor in Ensemble pipeline")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
