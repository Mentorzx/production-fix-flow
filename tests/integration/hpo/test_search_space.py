"""Tests for HPO search space and tuning configuration."""

from pff.domain.hpo.search_space import TuningConfig, TuningConfigBuilder
from pff.infrastructure.hpo.config_loader import load_hpo_defaults


def _defaults() -> dict[str, object]:
    return load_hpo_defaults()


class TestTuningConfig:
    """Tests for TuningConfig dataclass."""

    def test_tuning_config_has_embedding_dim_choices(self):
        """Verify TuningConfig has embedding_dim_choices field."""
        config = TuningConfigBuilder(_defaults()).build()
        assert hasattr(config, "embedding_dim_choices")
        assert isinstance(config.embedding_dim_choices, (tuple, list))

    def test_tuning_config_has_batch_size_bounds(self):
        """Verify TuningConfig has batch size bounds."""
        config = TuningConfigBuilder(_defaults()).build()
        assert hasattr(config, "batch_size_low")
        assert hasattr(config, "batch_size_high")
        assert config.batch_size_low <= config.batch_size_high

    def test_tuning_config_has_learning_rate_bounds(self):
        """Verify TuningConfig has learning rate bounds."""
        config = TuningConfigBuilder(_defaults()).build()
        assert hasattr(config, "learning_rate_low")
        assert hasattr(config, "learning_rate_high")
        assert config.learning_rate_low < config.learning_rate_high

    def test_tuning_config_has_n_trials(self):
        """Verify TuningConfig has n_trials field."""
        config = TuningConfigBuilder(_defaults()).build()
        assert hasattr(config, "n_trials")
        assert config.n_trials > 0

    def test_tuning_config_has_timeout(self):
        """Verify TuningConfig has timeout_seconds field."""
        config = TuningConfigBuilder(_defaults()).build()
        assert hasattr(config, "timeout_seconds")
        assert config.timeout_seconds > 0


class TestTuningConfigBuilder:
    """Tests for TuningConfigBuilder fluent API."""

    def test_tuning_config_builder_creation(self):
        """Verify builder can be created."""
        builder = TuningConfigBuilder(_defaults())
        assert builder is not None

    def test_tuning_config_builder_build_returns_tuning_config(self):
        """Verify build returns a TuningConfig."""
        builder = TuningConfigBuilder(_defaults())
        config = builder.build()
        assert isinstance(config, TuningConfig)

    def test_tuning_config_builder_with_defaults(self):
        """Verify builder accepts defaults dict."""
        defaults = {**_defaults(), "n_trials": 50, "batch_size_low": 64}
        builder = TuningConfigBuilder(defaults)
        config = builder.build()
        assert config.n_trials == 50
        assert config.batch_size_low == 64

    def test_tuning_config_builder_uses_kl_weight_defaults(self):
        """Verify builder uses kl_weight defaults from config."""
        defaults = {**_defaults(), "kl_weight_low": 1e-5, "kl_weight_high": 1e-2}
        config = TuningConfigBuilder(defaults).build()
        assert config.kl_weight_low == 1e-5
        assert config.kl_weight_high == 1e-2

    def test_tuning_config_builder_with_batch_size(self):
        """Verify batch size can be customized."""
        builder = TuningConfigBuilder(_defaults())
        config = builder.with_batch_size(32, 256).build()
        assert config.batch_size_low == 32
        assert config.batch_size_high == 256

    def test_tuning_config_builder_with_embedding_dim_choices(self):
        """Verify embedding dim choices can be customized."""
        builder = TuningConfigBuilder(_defaults())
        config = builder.with_embedding_dim_choices([64, 128, 256]).build()
        assert 64 in config.embedding_dim_choices
        assert 128 in config.embedding_dim_choices
        assert 256 in config.embedding_dim_choices

    def test_tuning_config_builder_with_negative_ratio(self):
        """Verify negative ratio can be customized."""
        builder = TuningConfigBuilder(_defaults())
        config = builder.with_negative_ratio(0.3, 0.9).build()
        assert config.negative_ratio_low == 0.3
        assert config.negative_ratio_high == 0.9

    def test_tuning_config_builder_chained(self):
        """Verify builder methods can be chained."""
        config = (
            TuningConfigBuilder(_defaults())
            .with_batch_size(64, 512)
            .with_embedding_dim_choices([128, 256])
            .with_negative_ratio(0.5, 0.8)
            .build()
        )
        assert config.batch_size_low == 64
        assert 128 in config.embedding_dim_choices
        assert config.negative_ratio_low == 0.5


class TestTuningConfigDefaults:
    """Tests for default TuningConfig values."""

    def test_default_embedding_dim_choices(self):
        """Verify default embedding dim choices."""
        config = TuningConfigBuilder(_defaults()).build()
        # Should have reasonable default dimensions
        assert len(config.embedding_dim_choices) >= 1
        assert all(d > 0 for d in config.embedding_dim_choices)

    def test_default_negative_ratio_bounds(self):
        """Verify default negative ratio bounds."""
        config = TuningConfigBuilder(_defaults()).build()
        assert 0 < config.negative_ratio_low < 1
        assert 0 < config.negative_ratio_high <= 1
        assert config.negative_ratio_low <= config.negative_ratio_high

    def test_default_temperature_bounds(self):
        """Verify default temperature bounds."""
        config = TuningConfigBuilder(_defaults()).build()
        assert config.adversarial_temperature_low > 0
        assert config.contrastive_temperature_low > 0

    def test_default_lambda_bounds(self):
        """Verify default lambda bounds for logic component."""
        config = TuningConfigBuilder(_defaults()).build()
        assert hasattr(config, "lambda_logic_low")
        assert hasattr(config, "lambda_logic_high")
        assert config.lambda_logic_low >= 0

    def test_default_t_norm_choices(self):
        """Verify default t-norm choices."""
        config = TuningConfigBuilder(_defaults()).build()
        assert hasattr(config, "t_norm_choices")
        assert len(config.t_norm_choices) >= 1


class TestTuningConfigValidRanges:
    """Tests for TuningConfig value range validity."""

    def test_learning_rate_in_valid_range(self):
        """Verify learning rate is in valid range for neural networks."""
        config = TuningConfigBuilder(_defaults()).build()
        assert 1e-7 <= config.learning_rate_low <= 1e-1
        assert 1e-6 <= config.learning_rate_high <= 1.0

    def test_batch_size_reasonable(self):
        """Verify batch sizes are reasonable for GPU efficiency."""
        config = TuningConfigBuilder(_defaults()).build()
        assert config.batch_size_low >= 8
        assert config.batch_size_high <= 4096

    def test_embedding_dims_reasonable(self):
        """Verify embedding dimensions are reasonable."""
        config = TuningConfigBuilder(_defaults()).build()
        for dim in config.embedding_dim_choices:
            assert 16 <= dim <= 2048
            # Commonly divisible by 8 for GPU efficiency
            assert dim % 8 == 0

    def test_timeout_reasonable(self):
        """Verify timeout is reasonable (not too short, not too long)."""
        config = TuningConfigBuilder(_defaults()).build()
        assert config.timeout_seconds >= 60
        assert config.timeout_seconds <= 86400 * 7

    def test_n_trials_positive(self):
        """Verify n_trials is positive."""
        config = TuningConfigBuilder(_defaults()).build()
        assert config.n_trials > 0

    def test_lambda_sum_cap_valid(self):
        """Verify lambda_sum_cap is in valid range."""
        config = TuningConfigBuilder(_defaults()).build()
        assert 0 <= config.lambda_sum_cap <= 1.0
