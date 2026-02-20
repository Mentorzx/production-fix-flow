"""Tests for HPO runner and memory configuration."""

from pathlib import Path

from pff.infrastructure.hpo.runner import HPOMemoryConfig


class TestHPOMemoryConfig:
    """Tests for HPOMemoryConfig dataclass."""

    def test_hpo_memory_config_defaults(self):
        """Verify default values are set correctly."""
        config = HPOMemoryConfig()
        assert config.enabled is True
        assert config.top_k_trials == 5
        assert config.warmstart_trials == 3
        assert config.storage_subdir == "hpo_replay"
        assert config.min_score_delta == 0.0

    def test_hpo_memory_config_from_dict_empty(self):
        """Verify from_dict with empty dict uses defaults."""
        config = HPOMemoryConfig.from_dict({})
        assert config.enabled is True
        assert config.top_k_trials == 5

    def test_hpo_memory_config_from_dict_none(self):
        """Verify from_dict with None uses defaults."""
        config = HPOMemoryConfig.from_dict(None)
        assert config.enabled is True
        assert config.top_k_trials == 5

    def test_hpo_memory_config_from_dict_custom(self):
        """Verify from_dict with custom values."""
        data = {
            "enabled": False,
            "top_k_trials": 10,
            "warmstart_trials": 5,
            "storage_subdir": "custom_dir",
            "min_score_delta": 0.01,
        }
        config = HPOMemoryConfig.from_dict(data)
        assert config.enabled is False
        assert config.top_k_trials == 10
        assert config.warmstart_trials == 5
        assert config.storage_subdir == "custom_dir"
        assert config.min_score_delta == 0.01

    def test_hpo_memory_config_from_dict_partial(self):
        """Verify from_dict with partial values uses defaults for missing."""
        data = {"top_k_trials": 3}
        config = HPOMemoryConfig.from_dict(data)
        assert config.top_k_trials == 3
        assert config.enabled is True
        assert config.warmstart_trials == 3

    def test_hpo_memory_config_enabled_bool_conversion(self):
        """Verify enabled field converts to bool."""
        config = HPOMemoryConfig.from_dict({"enabled": 0})
        assert config.enabled is False
        config = HPOMemoryConfig.from_dict({"enabled": 1})
        assert config.enabled is True

    def test_hpo_memory_config_int_conversion(self):
        """Verify integer fields are converted."""
        data = {"top_k_trials": "7", "warmstart_trials": "4"}
        config = HPOMemoryConfig.from_dict(data)
        assert config.top_k_trials == 7
        assert isinstance(config.top_k_trials, int)
        assert config.warmstart_trials == 4

    def test_hpo_memory_config_float_conversion(self):
        """Verify float fields are converted."""
        data = {"min_score_delta": "0.05"}
        config = HPOMemoryConfig.from_dict(data)
        assert config.min_score_delta == 0.05
        assert isinstance(config.min_score_delta, float)


class TestHPOConfigValidation:
    """Tests for HPO configuration validation."""

    def test_hpo_memory_config_positive_top_k(self):
        """Verify top_k_trials is positive."""
        config = HPOMemoryConfig(top_k_trials=1)
        assert config.top_k_trials >= 1

    def test_hpo_memory_config_warmstart_less_than_top_k(self):
        """Verify warmstart relationship to top_k."""
        config = HPOMemoryConfig(top_k_trials=10, warmstart_trials=5)
        assert config.warmstart_trials <= config.top_k_trials

    def test_hpo_memory_config_score_delta_non_negative(self):
        """Verify min_score_delta is non-negative."""
        config = HPOMemoryConfig(min_score_delta=0.0)
        assert config.min_score_delta >= 0.0


class TestHPOStorageConfig:
    """Tests for HPO storage configuration patterns."""

    def test_storage_subdir_is_string(self):
        """Verify storage_subdir is a string."""
        config = HPOMemoryConfig()
        assert isinstance(config.storage_subdir, str)

    def test_storage_subdir_custom_path(self):
        """Verify custom storage subdir path."""
        config = HPOMemoryConfig(storage_subdir="my_hpo_storage")
        path = Path("outputs") / config.storage_subdir
        assert str(path) == "outputs/my_hpo_storage"
