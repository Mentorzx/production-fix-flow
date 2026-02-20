"""Tests for adaptive training configuration calculator.

Tests the dynamic hyperparameter computation based on dataset characteristics.
"""

from pff.domain.learning.ml.adaptive_training import (
    AdaptiveTrainingCalculator,
    AdaptiveTrainingConfig,
    DatasetScale,
    DatasetStats,
    compute_adaptive_config,
)


class TestDatasetStats:
    """Tests for DatasetStats dataclass."""

    def test_scale_tiny(self) -> None:
        """Test TINY scale classification."""
        stats = DatasetStats(num_train_triples=5_000, num_valid_triples=500)
        assert stats.scale == DatasetScale.TINY

    def test_scale_small(self) -> None:
        """Test SMALL scale classification."""
        stats = DatasetStats(num_train_triples=50_000, num_valid_triples=5_000)
        assert stats.scale == DatasetScale.SMALL

    def test_scale_medium(self) -> None:
        """Test MEDIUM scale classification."""
        stats = DatasetStats(num_train_triples=500_000, num_valid_triples=50_000)
        assert stats.scale == DatasetScale.MEDIUM

    def test_scale_large(self) -> None:
        """Test LARGE scale classification."""
        stats = DatasetStats(num_train_triples=5_000_000, num_valid_triples=500_000)
        assert stats.scale == DatasetScale.LARGE

    def test_scale_huge(self) -> None:
        """Test HUGE scale classification."""
        stats = DatasetStats(num_train_triples=50_000_000, num_valid_triples=5_000_000)
        assert stats.scale == DatasetScale.HUGE

    def test_total_triples(self) -> None:
        """Test total triples calculation."""
        stats = DatasetStats(
            num_train_triples=1000,
            num_valid_triples=100,
            num_test_triples=100,
        )
        assert stats.total_triples == 1200

    def test_triples_per_entity(self) -> None:
        """Test triples per entity calculation."""
        stats = DatasetStats(
            num_train_triples=8000,
            num_valid_triples=1000,
            num_test_triples=1000,
            num_entities=1000,
        )
        assert stats.triples_per_entity == 10.0

    def test_triples_per_entity_zero_entities(self) -> None:
        """Test triples per entity with zero entities."""
        stats = DatasetStats(num_train_triples=1000, num_valid_triples=100)
        assert stats.triples_per_entity == 0.0

    def test_validation_ratio(self) -> None:
        """Test validation ratio calculation."""
        stats = DatasetStats(num_train_triples=1000, num_valid_triples=100)
        assert stats.validation_ratio == 0.1


class TestAdaptiveTrainingCalculator:
    """Tests for AdaptiveTrainingCalculator."""

    def test_tiny_dataset_epochs(self) -> None:
        """Test epochs for tiny dataset (should be high)."""
        stats = DatasetStats(
            num_train_triples=5_000,
            num_valid_triples=500,
            num_entities=500,
            num_relations=10,
        )
        calc = AdaptiveTrainingCalculator(stats, is_dslfm=False)
        config = calc.compute()

        # Tiny datasets need many epochs
        assert config.epochs >= 100
        assert config.epochs <= 200

    def test_large_dataset_epochs(self) -> None:
        """Test epochs for large dataset (should be lower than tiny)."""
        stats = DatasetStats(
            num_train_triples=5_000_000,
            num_valid_triples=500_000,
            num_entities=300_000,
            num_relations=50,
        )
        calc = AdaptiveTrainingCalculator(stats, is_dslfm=False)
        config = calc.compute()

        # Large datasets converge faster per epoch, but still need reasonable epochs
        # With 300k entities and 50 relations, entity_factor ~1.37, relation_factor ~1.3
        assert config.epochs >= 30
        assert config.epochs <= 120

    def test_dslfm_adds_epoch_overhead(self) -> None:
        """Test that DSLFM training adds epoch overhead."""
        stats = DatasetStats(
            num_train_triples=100_000,
            num_valid_triples=10_000,
            num_entities=10_000,
            num_relations=20,
        )

        calc_base = AdaptiveTrainingCalculator(stats, is_dslfm=False)
        calc_dslfm = AdaptiveTrainingCalculator(stats, is_dslfm=True)

        config_base = calc_base.compute()
        config_dslfm = calc_dslfm.compute()

        # DSLFM should require more epochs
        assert config_dslfm.epochs > config_base.epochs

    def test_patience_scales_with_validation_size(self) -> None:
        """Test that patience decreases with larger validation sets."""
        stats_small_valid = DatasetStats(
            num_train_triples=100_000,
            num_valid_triples=1_000,
            num_entities=10_000,
        )
        stats_large_valid = DatasetStats(
            num_train_triples=100_000,
            num_valid_triples=100_000,
            num_entities=10_000,
        )

        calc_small = AdaptiveTrainingCalculator(stats_small_valid)
        calc_large = AdaptiveTrainingCalculator(stats_large_valid)

        config_small = calc_small.compute()
        config_large = calc_large.compute()

        # Large validation = more stable metrics = shorter patience
        assert config_large.early_stopping_patience <= config_small.early_stopping_patience

    def test_min_delta_scales_with_validation_size(self) -> None:
        """Test that min_delta decreases with larger validation sets."""
        stats_small = DatasetStats(
            num_train_triples=100_000,
            num_valid_triples=1_000,
        )
        stats_large = DatasetStats(
            num_train_triples=100_000,
            num_valid_triples=1_000_000,
        )

        calc_small = AdaptiveTrainingCalculator(stats_small)
        calc_large = AdaptiveTrainingCalculator(stats_large)

        config_small = calc_small.compute()
        config_large = calc_large.compute()

        # Large validation = smaller meaningful deltas
        assert config_large.min_delta < config_small.min_delta

    def test_batch_size_scales_with_dataset(self) -> None:
        """Test batch size scales with dataset size."""
        stats_small = DatasetStats(num_train_triples=5_000, num_valid_triples=500)
        stats_large = DatasetStats(num_train_triples=5_000_000, num_valid_triples=500_000)

        config_small = AdaptiveTrainingCalculator(stats_small).compute()
        config_large = AdaptiveTrainingCalculator(stats_large).compute()

        assert config_large.batch_size > config_small.batch_size

    def test_config_to_dict(self) -> None:
        """Test config serialization to dict."""
        stats = DatasetStats(num_train_triples=100_000, num_valid_triples=10_000)
        config = AdaptiveTrainingCalculator(stats).compute()

        d = config.to_dict()

        assert "epochs" in d
        assert "early_stopping_patience" in d
        assert "validate_every" in d
        assert "min_delta" in d
        assert "batch_size" in d
        assert "num_neg" in d
        assert "learning_rate" in d

    def test_computation_details_populated(self) -> None:
        """Test that computation details are populated."""
        stats = DatasetStats(
            num_train_triples=100_000,
            num_valid_triples=10_000,
            num_entities=10_000,
            num_relations=30,
        )
        config = AdaptiveTrainingCalculator(stats).compute()

        assert "epochs" in config.computation_details
        assert "patience" in config.computation_details
        assert "min_delta" in config.computation_details


class TestComputeAdaptiveConfig:
    """Tests for convenience function."""

    def test_basic_usage(self) -> None:
        """Test basic convenience function usage."""
        config = compute_adaptive_config(
            num_train_triples=100_000,
            num_valid_triples=10_000,
            num_entities=10_000,
            num_relations=20,
        )

        assert isinstance(config, AdaptiveTrainingConfig)
        assert config.epochs > 0
        assert config.early_stopping_patience > 0

    def test_pff_sample_dataset(self) -> None:
        """Test with PFF sample dataset characteristics."""
        # Current sample: 18k train, 2.3k valid, ~5k entities, 46 relations
        # SMALL scale with sparse graph (3.8 triples/entity) + DSLFM overhead
        config = compute_adaptive_config(
            num_train_triples=18_622,
            num_valid_triples=2_335,
            num_entities=4_948,
            num_relations=46,
        )

        # base=100, entity_factor=1.0, relation_factor=1.26, model_factor=1.2, coverage=1.2
        # Raw calculation: ~181 epochs (clamped to 200 max)
        assert 100 <= config.epochs <= 200
        assert 8 <= config.early_stopping_patience <= 15

    def test_pff_full_dataset(self) -> None:
        """Test with PFF full dataset characteristics (correct.parquet)."""
        # Full dataset: ~4.9M train, ~612k valid, ~270k entities, 44 relations
        # LARGE scale with DSLFM overhead
        config = compute_adaptive_config(
            num_train_triples=4_898_391,
            num_valid_triples=612_298,
            num_entities=269_889,
            num_relations=44,
        )

        # base=60, entity_factor=1.36, relation_factor=1.24, model_factor=1.2, coverage=0.9
        # Raw calculation: ~109 epochs
        assert 80 <= config.epochs <= 130
        assert 3 <= config.early_stopping_patience <= 6


class TestRegressionBenchmarks:
    """Regression tests against known benchmark datasets."""

    def test_fb15k237_like(self) -> None:
        """Test FB15k-237 like dataset."""
        config = compute_adaptive_config(
            num_train_triples=310_116,
            num_valid_triples=17_535,
            num_entities=14_541,
            num_relations=237,
            is_dslfm=False,
        )

        # FB15k-237 typically uses 50-100 epochs
        assert 50 <= config.epochs <= 120

    def test_wn18rr_like(self) -> None:
        """Test WN18RR like dataset."""
        config = compute_adaptive_config(
            num_train_triples=86_835,
            num_valid_triples=3_034,
            num_entities=40_943,
            num_relations=11,
            is_dslfm=False,
        )

        # WN18RR is sparse, needs more epochs
        assert 70 <= config.epochs <= 150

    def test_yago3_10_like(self) -> None:
        """Test YAGO3-10 like dataset."""
        config = compute_adaptive_config(
            num_train_triples=1_079_040,
            num_valid_triples=5_000,
            num_entities=123_182,
            num_relations=37,
            is_dslfm=False,
        )

        # LARGE scale with high entity count
        # base=60, entity_factor=1.27, relation_factor=1.17
        assert 60 <= config.epochs <= 100
