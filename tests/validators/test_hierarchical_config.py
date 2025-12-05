"""
Tests for Hierarchical Ensemble Configuration Loader.

This module tests the config_loader for the hierarchical ensemble,
including defaults, deep merge, and property behavior.

Test Categories:
    - Default loading without YAML file
    - Deep merge with partial overrides
    - HierarchicalConfig properties
    - Penalty application logic per architecture type
"""

import tempfile
from pathlib import Path

import pytest

from pff.validators.ensembles.hierarchical.config_loader import (
    AggregatorConfig,
    DecisionRouterConfig,
    HierarchicalConfig,
    PenaltiesConfig,
    _deep_merge,
    _get_defaults,
    _parse_config,
    load_hierarchical_config,
)


class TestDeepMerge:
    """Tests for the deep merge utility function."""

    def test_deep_merge_simple_override(self):
        """Test that simple values are overridden."""
        base = {"a": 1, "b": 2}
        override = {"b": 3}
        result = _deep_merge(base, override)
        assert result == {"a": 1, "b": 3}

    def test_deep_merge_nested_dict(self):
        """Test that nested dicts are merged recursively."""
        base = {"outer": {"inner1": 1, "inner2": 2}}
        override = {"outer": {"inner2": 3}}
        result = _deep_merge(base, override)
        assert result == {"outer": {"inner1": 1, "inner2": 3}}

    def test_deep_merge_add_new_key(self):
        """Test that new keys in override are added."""
        base = {"a": 1}
        override = {"b": 2}
        result = _deep_merge(base, override)
        assert result == {"a": 1, "b": 2}

    def test_deep_merge_original_unchanged(self):
        """Test that original dicts are not mutated."""
        base = {"a": {"b": 1}}
        override = {"a": {"b": 2}}
        result = _deep_merge(base, override)
        assert base == {"a": {"b": 1}}
        assert result == {"a": {"b": 2}}


class TestDefaults:
    """Tests for default configuration values."""

    def test_defaults_has_architecture(self):
        """Test that defaults include architecture section."""
        defaults = _get_defaults()
        assert "architecture" in defaults
        assert defaults["architecture"]["type"] == "flat"

    def test_defaults_has_aggregators(self):
        """Test that defaults include aggregator configs."""
        defaults = _get_defaults()
        assert "aggregators" in defaults
        assert defaults["aggregators"]["symbolic"]["strategy"] == "noisy_or"
        assert defaults["aggregators"]["neural"]["strategy"] == "weighted_average"

    def test_defaults_has_decision_router(self):
        """Test that defaults include decision router config."""
        defaults = _get_defaults()
        assert "decision_router" in defaults
        assert "thresholds" in defaults["decision_router"]
        assert defaults["decision_router"]["thresholds"]["symbolic_confidence"] == 0.70

    def test_defaults_has_penalties(self):
        """Test that defaults include penalty config."""
        defaults = _get_defaults()
        assert "penalties" in defaults
        assert defaults["penalties"]["symbolic_dominance"]["enabled_in_flat"] is True
        assert defaults["penalties"]["symbolic_dominance"]["enabled_in_hierarchical"] is False


class TestAggregatorConfig:
    """Tests for AggregatorConfig dataclass."""

    def test_default_strategy_noisy_or(self):
        """Test that default strategy is noisy_or."""
        config = AggregatorConfig()
        assert config.strategy == "noisy_or"

    def test_custom_strategy(self):
        """Test custom strategy assignment."""
        config = AggregatorConfig(strategy="max_confidence", params={"k": 5})
        assert config.strategy == "max_confidence"
        assert config.params == {"k": 5}


class TestDecisionRouterConfig:
    """Tests for DecisionRouterConfig dataclass."""

    def test_default_thresholds(self):
        """Test default threshold values."""
        config = DecisionRouterConfig()
        assert config.symbolic_confidence_threshold == 0.70
        assert config.neural_confidence_threshold == 0.50

    def test_default_blend_weights(self):
        """Test default blend weights."""
        config = DecisionRouterConfig()
        assert config.blend_weight_symbolic == 0.6
        assert config.blend_weight_neural == 0.4


class TestPenaltiesConfig:
    """Tests for PenaltiesConfig dataclass."""

    def test_penalty_enabled_in_flat_by_default(self):
        """Test that penalty is enabled in flat mode by default."""
        config = PenaltiesConfig()
        assert config.symbolic_dominance_enabled_in_flat is True

    def test_penalty_disabled_in_hierarchical_by_default(self):
        """Test that penalty is disabled in hierarchical mode by default."""
        config = PenaltiesConfig()
        assert config.symbolic_dominance_enabled_in_hierarchical is False


class TestHierarchicalConfig:
    """Tests for HierarchicalConfig dataclass."""

    def test_default_is_flat(self):
        """Test that default architecture is flat."""
        config = HierarchicalConfig()
        assert config.is_flat is True
        assert config.is_hierarchical is False

    def test_hierarchical_mode(self):
        """Test hierarchical mode detection."""
        config = HierarchicalConfig(architecture_type="hierarchical")
        assert config.is_hierarchical is True
        assert config.is_flat is False

    def test_should_apply_penalty_in_flat_mode(self):
        """Test that penalty is applied in flat mode."""
        config = HierarchicalConfig(
            architecture_type="flat",
            penalties=PenaltiesConfig(
                symbolic_dominance_enabled_in_flat=True,
                symbolic_dominance_enabled_in_hierarchical=False,
            ),
        )
        assert config.should_apply_symbolic_dominance_penalty is True

    def test_should_not_apply_penalty_in_hierarchical_mode(self):
        """Test that penalty is NOT applied in hierarchical mode."""
        config = HierarchicalConfig(
            architecture_type="hierarchical",
            penalties=PenaltiesConfig(
                symbolic_dominance_enabled_in_flat=True,
                symbolic_dominance_enabled_in_hierarchical=False,
            ),
        )
        assert config.should_apply_symbolic_dominance_penalty is False

    def test_penalty_can_be_enabled_in_hierarchical_if_configured(self):
        """Test that penalty can be explicitly enabled in hierarchical mode."""
        config = HierarchicalConfig(
            architecture_type="hierarchical",
            penalties=PenaltiesConfig(
                symbolic_dominance_enabled_in_hierarchical=True,
            ),
        )
        assert config.should_apply_symbolic_dominance_penalty is True


class TestParseConfig:
    """Tests for _parse_config function."""

    def test_parse_minimal_config(self):
        """Test parsing minimal config dict."""
        raw = {"architecture": {"type": "hierarchical"}}
        config = _parse_config(raw)
        assert config.architecture_type == "hierarchical"
        assert config.symbolic_aggregator.strategy == "noisy_or"

    def test_parse_full_config(self):
        """Test parsing complete config dict."""
        raw = {
            "architecture": {"type": "hierarchical"},
            "aggregators": {
                "symbolic": {"strategy": "max_confidence", "params": {"k": 10}},
                "neural": {"strategy": "softmax", "params": {"temp": 0.5}},
            },
            "decision_router": {
                "thresholds": {"symbolic_confidence": 0.9, "neural_confidence": 0.8},
                "blend_weights": {"symbolic": 0.7, "neural": 0.3},
            },
            "penalties": {
                "symbolic_dominance": {
                    "enabled_in_flat": False,
                    "enabled_in_hierarchical": True,
                    "threshold": 0.99,
                    "penalty_factor": 0.1,
                },
            },
        }
        config = _parse_config(raw)
        assert config.architecture_type == "hierarchical"
        assert config.symbolic_aggregator.strategy == "max_confidence"
        assert config.symbolic_aggregator.params == {"k": 10}
        assert config.neural_aggregator.strategy == "softmax"
        assert config.decision_router.symbolic_confidence_threshold == 0.9
        assert config.decision_router.blend_weight_symbolic == 0.7
        assert config.penalties.symbolic_dominance_enabled_in_flat is False
        assert config.penalties.symbolic_dominance_threshold == 0.99


class TestLoadHierarchicalConfig:
    """Tests for load_hierarchical_config function."""

    def test_load_with_nonexistent_file_returns_defaults(self):
        """Test that missing file returns defaults (flat mode)."""
        config = load_hierarchical_config(config_path=Path("/nonexistent/path.yaml"))
        assert config.is_flat is True
        assert config.symbolic_aggregator.strategy == "noisy_or"

    def test_load_with_valid_yaml_file(self, tmp_path):
        """Test loading from valid YAML file."""
        yaml_content = """
architecture:
  type: hierarchical

aggregators:
  symbolic:
    strategy: voting
"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(yaml_content)

        config = load_hierarchical_config(config_path=config_file)
        assert config.is_hierarchical is True
        assert config.symbolic_aggregator.strategy == "voting"
        assert config.neural_aggregator.strategy == "weighted_average"

    def test_load_with_overrides(self, tmp_path):
        """Test that runtime overrides are applied."""
        yaml_content = """
architecture:
  type: flat
"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(yaml_content)

        overrides = {"architecture": {"type": "hierarchical"}}
        config = load_hierarchical_config(config_path=config_file, overrides=overrides)
        assert config.is_hierarchical is True

    def test_load_with_partial_yaml_gets_defaults(self, tmp_path):
        """Test that partial YAML is merged with defaults."""
        yaml_content = """
decision_router:
  thresholds:
    symbolic_confidence: 0.95
"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(yaml_content)

        config = load_hierarchical_config(config_path=config_file)
        assert config.is_flat is True
        assert config.decision_router.symbolic_confidence_threshold == 0.95
        # Defaults to 0.50 now (was 0.70)
        assert config.decision_router.neural_confidence_threshold == 0.50

    def test_raw_config_preserved(self, tmp_path):
        """Test that raw config dict is available."""
        yaml_content = """
architecture:
  type: hierarchical
custom_field: custom_value
"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(yaml_content)

        config = load_hierarchical_config(config_path=config_file)
        assert "custom_field" in config.raw_config


class TestNoisyOrDefault:
    """Tests ensuring Noisy-OR is the default symbolic strategy."""

    def test_noisy_or_is_default_in_defaults(self):
        """Test that noisy_or is default in _get_defaults."""
        defaults = _get_defaults()
        assert defaults["aggregators"]["symbolic"]["strategy"] == "noisy_or"

    def test_noisy_or_is_default_in_dataclass(self):
        """Test that noisy_or is default in AggregatorConfig."""
        config = AggregatorConfig()
        assert config.strategy == "noisy_or"

    def test_noisy_or_preserved_when_not_overridden(self, tmp_path):
        """Test that noisy_or is preserved when YAML doesn't override it."""
        yaml_content = """
architecture:
  type: hierarchical
"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(yaml_content)

        config = load_hierarchical_config(config_path=config_file)
        assert config.symbolic_aggregator.strategy == "noisy_or"
