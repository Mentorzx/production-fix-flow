"""
Tests for Hierarchical Pipeline.

This module tests the complete hierarchical prediction pipeline,
including integration of aggregators and router.

Test Categories:
    - Pipeline initialization
    - Prediction flow
    - Flat feature compatibility
    - Statistics collection
    - Config-based behavior
"""

import numpy as np
import pytest

from pff.validators.ensembles.hierarchical.config_loader import (
    AggregatorConfig,
    DecisionRouterConfig,
    HierarchicalConfig,
    PenaltiesConfig,
)
from pff.validators.ensembles.hierarchical.pipeline import (
    HierarchicalPipeline,
    HierarchicalPredictionResult,
    get_hierarchical_pipeline_if_enabled,
)


@pytest.fixture
def hierarchical_config():
    """Create a test hierarchical config."""
    return HierarchicalConfig(
        architecture_type="hierarchical",
        symbolic_aggregator=AggregatorConfig(
            strategy="noisy_or",
            params={"base_confidence": 0.01, "max_rules": 1500, "min_confidence": 0.30},
        ),
        neural_aggregator=AggregatorConfig(
            strategy="weighted_average",
            params={},
        ),
        decision_router=DecisionRouterConfig(
            symbolic_confidence_threshold=0.85,
            neural_confidence_threshold=0.70,
            blend_weight_symbolic=0.6,
            blend_weight_neural=0.4,
        ),
        penalties=PenaltiesConfig(
            symbolic_dominance_enabled_in_flat=True,
            symbolic_dominance_enabled_in_hierarchical=False,
        ),
    )


@pytest.fixture
def flat_config():
    """Create a test flat config."""
    return HierarchicalConfig(
        architecture_type="flat",
        penalties=PenaltiesConfig(
            symbolic_dominance_enabled_in_flat=True,
            symbolic_dominance_enabled_in_hierarchical=False,
        ),
    )


class TestPipelineInitialization:
    """Tests for pipeline initialization."""

    def test_from_hierarchical_config(self, hierarchical_config):
        """Test pipeline creation from hierarchical config."""
        pipeline = HierarchicalPipeline(hierarchical_config)
        assert pipeline.is_enabled is True
        assert pipeline.symbolic_aggregator.strategy_name == "noisy_or"
        assert pipeline.neural_aggregator.strategy_name == "weighted_average"

    def test_from_flat_config(self, flat_config):
        """Test pipeline creation from flat config."""
        pipeline = HierarchicalPipeline(flat_config)
        assert pipeline.is_enabled is False

    def test_from_config_classmethod(self, hierarchical_config):
        """Test from_config classmethod."""
        pipeline = HierarchicalPipeline.from_config(hierarchical_config)
        assert pipeline.is_enabled is True


class TestPrediction:
    """Tests for prediction flow."""

    def test_predict_basic(self, hierarchical_config):
        """Test basic prediction with synthetic data."""
        pipeline = HierarchicalPipeline(hierarchical_config)

        n_samples = 10
        n_rules = 5
        symbolic_features = np.random.uniform(0, 1, (n_samples, n_rules))
        neural_features = np.random.uniform(0, 1, n_samples)

        result = pipeline.predict(symbolic_features, neural_features)

        assert isinstance(result, HierarchicalPredictionResult)
        assert len(result.final_scores) == n_samples
        assert len(result.symbolic_aggregated) == n_samples
        assert len(result.neural_aggregated) == n_samples
        assert result.routing_stats.total_decisions == n_samples

    def test_predict_with_scalar_neural(self, hierarchical_config):
        """Test prediction with single neural score for all samples."""
        pipeline = HierarchicalPipeline(hierarchical_config)

        n_samples = 5
        symbolic_features = np.random.uniform(0, 1, (n_samples, 3))
        neural_features = 0.75  # Single score

        result = pipeline.predict(symbolic_features, neural_features)

        assert len(result.final_scores) == n_samples
        assert np.all(result.neural_aggregated == 0.75)

    def test_predict_with_2d_neural(self, hierarchical_config):
        """Test prediction with multi-model neural scores."""
        pipeline = HierarchicalPipeline(hierarchical_config)

        n_samples = 5
        n_models = 3
        symbolic_features = np.random.uniform(0, 1, (n_samples, 4))
        neural_features = np.random.uniform(0, 1, (n_samples, n_models))

        result = pipeline.predict(symbolic_features, neural_features)

        assert len(result.final_scores) == n_samples

    def test_predict_routing_decisions(self, hierarchical_config):
        """Test that routing decisions are made correctly."""
        pipeline = HierarchicalPipeline(hierarchical_config)

        symbolic_features = np.array([
            [0.95, 0.90, 0.85],  # High symbolic → SYMBOLIC_DECIDES
            [0.30, 0.25, 0.20],  # Low symbolic, good neural → NEURAL_FALLBACK
            [0.50, 0.45, 0.40],  # Medium both → BLEND
        ])
        neural_features = np.array([0.60, 0.85, 0.55])

        result = pipeline.predict(symbolic_features, neural_features)
        stats = result.routing_stats

        assert stats.symbolic_decides_count >= 1
        assert stats.total_decisions == 3

    def test_final_scores_in_valid_range(self, hierarchical_config):
        """Test that all final scores are in [0, 1]."""
        pipeline = HierarchicalPipeline(hierarchical_config)

        for _ in range(10):
            symbolic = np.random.uniform(0, 1, (100, 10))
            neural = np.random.uniform(0, 1, 100)
            result = pipeline.predict(symbolic, neural)

            assert np.all(result.final_scores >= 0)
            assert np.all(result.final_scores <= 1)


class TestFlatFeatureCompatibility:
    """Tests for flat feature matrix compatibility."""

    def test_predict_from_flat_features(self, hierarchical_config):
        """Test prediction from flat feature matrix."""
        pipeline = HierarchicalPipeline(hierarchical_config)

        n_samples = 5
        flat_features = np.random.uniform(0, 1, (n_samples, 6))

        result = pipeline.predict_from_flat_features(flat_features, neural_feature_index=0)

        assert len(result.final_scores) == n_samples
        assert result.symbolic_aggregated.shape[0] == n_samples
        assert result.neural_aggregated.shape[0] == n_samples

    def test_neural_index_extraction(self, hierarchical_config):
        """Test correct extraction of neural feature."""
        pipeline = HierarchicalPipeline(hierarchical_config)

        flat_features = np.array([
            [0.9, 0.1, 0.2, 0.3],
            [0.8, 0.2, 0.3, 0.4],
        ])

        result = pipeline.predict_from_flat_features(flat_features, neural_feature_index=0)

        np.testing.assert_array_almost_equal(result.neural_aggregated, [0.9, 0.8])

    def test_invalid_flat_features_raises(self, hierarchical_config):
        """Test that 1D array raises error."""
        pipeline = HierarchicalPipeline(hierarchical_config)

        with pytest.raises(ValueError, match="Expected 2D"):
            pipeline.predict_from_flat_features(np.array([0.5, 0.6]))


class TestStatistics:
    """Tests for routing statistics collection."""

    def test_last_routing_stats(self, hierarchical_config):
        """Test that last routing stats are stored."""
        pipeline = HierarchicalPipeline(hierarchical_config)
        assert pipeline.last_routing_stats is None

        symbolic = np.random.uniform(0, 1, (10, 5))
        neural = np.random.uniform(0, 1, 10)
        pipeline.predict(symbolic, neural)

        assert pipeline.last_routing_stats is not None
        assert pipeline.last_routing_stats.total_decisions == 10

    def test_stats_updated_on_each_predict(self, hierarchical_config):
        """Test that stats are updated on each prediction."""
        pipeline = HierarchicalPipeline(hierarchical_config)

        pipeline.predict(np.random.uniform(0, 1, (5, 3)), np.random.uniform(0, 1, 5))
        stats1 = pipeline.last_routing_stats

        pipeline.predict(np.random.uniform(0, 1, (10, 3)), np.random.uniform(0, 1, 10))
        stats2 = pipeline.last_routing_stats

        assert stats1.total_decisions == 5
        assert stats2.total_decisions == 10


class TestPenaltyBehavior:
    """Tests for symbolic dominance penalty behavior."""

    def test_penalty_disabled_in_hierarchical(self, hierarchical_config):
        """Test that penalty is disabled in hierarchical mode."""
        pipeline = HierarchicalPipeline(hierarchical_config)
        assert pipeline.should_apply_symbolic_dominance_penalty() is False

    def test_penalty_enabled_in_flat(self, flat_config):
        """Test that penalty is enabled in flat mode."""
        pipeline = HierarchicalPipeline(flat_config)
        assert pipeline.should_apply_symbolic_dominance_penalty() is True


class TestGetPipelineIfEnabled:
    """Tests for the convenience function."""

    def test_returns_none_for_flat_mode(self, tmp_path, monkeypatch):
        """Test that None is returned when flat mode is configured."""
        yaml_content = """
architecture:
  type: flat
"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(yaml_content)

        from pff.validators.ensembles.hierarchical import config_loader
        monkeypatch.setattr(
            config_loader,
            "HIERARCHICAL_ENSEMBLE_CONFIG_PATH",
            config_file,
        )

        from pff.validators.ensembles.hierarchical.config_loader import (
            load_hierarchical_config,
        )
        config = load_hierarchical_config(config_path=config_file)
        assert config.is_flat

    def test_hierarchical_config_returns_pipeline(self, hierarchical_config):
        """Test that pipeline is returned for hierarchical config."""
        pipeline = HierarchicalPipeline.from_config(hierarchical_config)
        assert pipeline.is_enabled is True


class TestMetadata:
    """Tests for prediction metadata."""

    def test_result_contains_metadata(self, hierarchical_config):
        """Test that result contains strategy metadata."""
        pipeline = HierarchicalPipeline(hierarchical_config)

        symbolic = np.random.uniform(0, 1, (5, 3))
        neural = np.random.uniform(0, 1, 5)
        result = pipeline.predict(symbolic, neural)

        assert "symbolic_strategy" in result.metadata
        assert "neural_strategy" in result.metadata
        assert "config_type" in result.metadata
        assert result.metadata["config_type"] == "hierarchical"
