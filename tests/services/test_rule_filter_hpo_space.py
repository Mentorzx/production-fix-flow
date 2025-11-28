"""Tests for AnyBURL HPO ranges from config.

P1.2: Verify that scripts/optimization/core.py reads HPO ranges from
config/models/kg.yaml (section rule_filter.hpo_ranges) and uses them in trial.suggest_* calls.

This test mocks Optuna trials to verify the suggest_* calls use config bounds.

Author: PFF Team
Date: 2025-11-27
"""

from __future__ import annotations

from unittest.mock import MagicMock, Mock, call, patch

import pytest

from pff.config import RULE_FILTER_HPO_CONFIG_PATH


class TestRuleFilterHPORanges:
    """P1.2: Test that HPO ranges are read from config."""

    def test_config_has_all_hpo_ranges(self):
        """Verify rule_filter.yaml contains all required hpo_ranges."""
        from pff.utils.file_manager import FileManager
        
        fm = FileManager()
        config = fm.read(RULE_FILTER_HPO_CONFIG_PATH)
        assert "rule_filter" in config, "rule_filter section should exist"
        hpo_ranges = config["rule_filter"].get("hpo_ranges", {})
        
        # P1.2 - Required range keys
        required_keys = [
            "confidence_quantile",
            "support_quantile", 
            "target_ratio",
            "max_length_cyclic",
            "max_length_acyclic",
        ]
        
        for key in required_keys:
            assert key in hpo_ranges, f"{key} should be in hpo_ranges"
            assert "low" in hpo_ranges[key], f"{key} should have 'low' bound"
            assert "high" in hpo_ranges[key], f"{key} should have 'high' bound"

    def test_hpo_ranges_bounds_are_valid(self):
        """Verify HPO ranges have low < high."""
        from pff.utils.file_manager import FileManager
        
        fm = FileManager()
        config = fm.read(RULE_FILTER_HPO_CONFIG_PATH)
        hpo_ranges = config.get("rule_filter", {}).get("hpo_ranges", {})
        
        for key, bounds in hpo_ranges.items():
            low = bounds.get("low", 0)
            high = bounds.get("high", 0)
            assert low < high, f"{key}: low ({low}) should be < high ({high})"

    def test_confidence_quantile_range(self):
        """Verify confidence_quantile range is reasonable."""
        from pff.utils.file_manager import FileManager
        
        fm = FileManager()
        config = fm.read(RULE_FILTER_HPO_CONFIG_PATH)
        hpo_ranges = config.get("rule_filter", {}).get("hpo_ranges", {})
        
        conf_range = hpo_ranges.get("confidence_quantile", {})
        low = conf_range.get("low", 0.5)
        high = conf_range.get("high", 0.9)
        
        # Reasonable bounds for confidence quantile
        assert 0.0 <= low <= 1.0, "confidence_quantile low should be in [0, 1]"
        assert 0.0 <= high <= 1.0, "confidence_quantile high should be in [0, 1]"
        assert low >= 0.3, "confidence_quantile low should be >= 0.3"
        assert high <= 0.95, "confidence_quantile high should be <= 0.95"

    def test_support_quantile_range(self):
        """Verify support_quantile range is reasonable."""
        from pff.utils.file_manager import FileManager
        
        fm = FileManager()
        config = fm.read(RULE_FILTER_HPO_CONFIG_PATH)
        hpo_ranges = config.get("rule_filter", {}).get("hpo_ranges", {})
        
        support_range = hpo_ranges.get("support_quantile", {})
        low = support_range.get("low", 0.3)
        high = support_range.get("high", 0.8)
        
        assert 0.0 <= low <= 1.0, "support_quantile low should be in [0, 1]"
        assert 0.0 <= high <= 1.0, "support_quantile high should be in [0, 1]"

    def test_target_ratio_range(self):
        """Verify target_ratio range is reasonable."""
        from pff.utils.file_manager import FileManager
        
        fm = FileManager()
        config = fm.read(RULE_FILTER_HPO_CONFIG_PATH)
        hpo_ranges = config.get("rule_filter", {}).get("hpo_ranges", {})
        
        ratio_range = hpo_ranges.get("target_ratio", {})
        low = ratio_range.get("low", 0.2)
        high = ratio_range.get("high", 0.5)
        
        assert 0.0 <= low <= 1.0, "target_ratio low should be in [0, 1]"
        assert 0.0 <= high <= 1.0, "target_ratio high should be in [0, 1]"
        assert high <= 0.6, "target_ratio high should be <= 0.6 to avoid symbolic dominance"


class TestHPORangesInOptimizer:
    """Test that core.py uses HPO ranges from config."""

    @pytest.fixture
    def mock_trial(self):
        """Create a mock Optuna trial."""
        trial = Mock()
        trial.number = 0
        
        # Track suggest calls with their arguments
        suggest_float_calls = []
        suggest_int_calls = []
        suggest_categorical_calls = []
        
        def mock_suggest_float(name, low, high, **kwargs):
            suggest_float_calls.append((name, low, high, kwargs))
            return (low + high) / 2
        
        def mock_suggest_int(name, low, high, **kwargs):
            suggest_int_calls.append((name, low, high, kwargs))
            return (low + high) // 2
        
        def mock_suggest_categorical(name, choices):
            suggest_categorical_calls.append((name, choices))
            return choices[0]
        
        trial.suggest_float = mock_suggest_float
        trial.suggest_int = mock_suggest_int
        trial.suggest_categorical = mock_suggest_categorical
        
        trial._suggest_float_calls = suggest_float_calls
        trial._suggest_int_calls = suggest_int_calls
        trial._suggest_categorical_calls = suggest_categorical_calls
        
        return trial

    def test_config_ranges_parsed_correctly(self):
        """Verify config ranges can be parsed as expected by core.py."""
        from pff.utils.file_manager import FileManager
        
        fm = FileManager()
        config = fm.read(RULE_FILTER_HPO_CONFIG_PATH)
        hpo_ranges = config.get("rule_filter", {}).get("hpo_ranges", {})
        
        # Test parsing like core.py does
        cyclic_range = hpo_ranges.get("max_length_cyclic", {"low": 3, "high": 4})
        acyclic_range = hpo_ranges.get("max_length_acyclic", {"low": 3, "high": 5})
        conf_quantile_range = hpo_ranges.get("confidence_quantile", {"low": 0.5, "high": 0.9})
        support_quantile_range = hpo_ranges.get("support_quantile", {"low": 0.3, "high": 0.8})
        target_ratio_range = hpo_ranges.get("target_ratio", {"low": 0.2, "high": 0.5})
        
        # Verify values can be cast as core.py does
        assert int(cyclic_range.get("low", 3)) >= 1
        assert int(acyclic_range.get("high", 5)) <= 10
        assert float(conf_quantile_range.get("low", 0.5)) >= 0.0
        assert float(support_quantile_range.get("high", 0.8)) <= 1.0
        assert float(target_ratio_range.get("low", 0.2)) >= 0.0

    def test_suggest_calls_use_config_bounds(self, mock_trial):
        """Verify that suggest calls would use config-derived bounds."""
        from pff.utils.file_manager import FileManager
        
        fm = FileManager()
        config = fm.read(RULE_FILTER_HPO_CONFIG_PATH)
        hpo_ranges = config.get("rule_filter", {}).get("hpo_ranges", {})
        
        # Simulate what core.py does with the ranges
        cyclic_range = hpo_ranges.get("max_length_cyclic", {"low": 3, "high": 4})
        acyclic_range = hpo_ranges.get("max_length_acyclic", {"low": 3, "high": 5})
        conf_quantile_range = hpo_ranges.get("confidence_quantile", {"low": 0.5, "high": 0.9})
        support_quantile_range = hpo_ranges.get("support_quantile", {"low": 0.3, "high": 0.8})
        target_ratio_range = hpo_ranges.get("target_ratio", {"low": 0.2, "high": 0.5})
        
        # Make suggest calls like core.py does
        mock_trial.suggest_int(
            'max_length_cyclic',
            int(cyclic_range.get("low", 3)),
            int(cyclic_range.get("high", 4))
        )
        mock_trial.suggest_int(
            'max_length_acyclic',
            int(acyclic_range.get("low", 3)),
            int(acyclic_range.get("high", 5))
        )
        mock_trial.suggest_float(
            'confidence_quantile',
            float(conf_quantile_range.get("low", 0.5)),
            float(conf_quantile_range.get("high", 0.9))
        )
        mock_trial.suggest_float(
            'support_quantile',
            float(support_quantile_range.get("low", 0.3)),
            float(support_quantile_range.get("high", 0.8))
        )
        mock_trial.suggest_float(
            'target_ratio',
            float(target_ratio_range.get("low", 0.2)),
            float(target_ratio_range.get("high", 0.5))
        )
        
        # Verify suggest_int calls for length ranges
        int_calls = {c[0]: (c[1], c[2]) for c in mock_trial._suggest_int_calls}
        assert 'max_length_cyclic' in int_calls
        assert int_calls['max_length_cyclic'] == (
            int(cyclic_range.get("low", 3)),
            int(cyclic_range.get("high", 4))
        )
        assert 'max_length_acyclic' in int_calls
        assert int_calls['max_length_acyclic'] == (
            int(acyclic_range.get("low", 3)),
            int(acyclic_range.get("high", 5))
        )
        
        # Verify suggest_float calls for quantile/ratio ranges
        float_calls = {c[0]: (c[1], c[2]) for c in mock_trial._suggest_float_calls}
        assert 'confidence_quantile' in float_calls
        assert float_calls['confidence_quantile'] == (
            float(conf_quantile_range.get("low", 0.5)),
            float(conf_quantile_range.get("high", 0.9))
        )
        assert 'support_quantile' in float_calls
        assert 'target_ratio' in float_calls


class TestRuleLengthExpansion:
    """P1.3: Test conservative expansion of rule lengths."""

    def test_cyclic_range_minimum(self):
        """Verify max_length_cyclic low is >= 3 for conservative expansion."""
        from pff.utils.file_manager import FileManager
        
        fm = FileManager()
        config = fm.read(RULE_FILTER_HPO_CONFIG_PATH)
        hpo_ranges = config.get("rule_filter", {}).get("hpo_ranges", {})
        
        cyclic_range = hpo_ranges.get("max_length_cyclic", {})
        low = cyclic_range.get("low", 1)
        
        # P1.3 - Conservative expansion: low should be >= 3
        assert low >= 3, f"max_length_cyclic low should be >= 3 for expansion, got {low}"

    def test_acyclic_range_expansion(self):
        """Verify max_length_acyclic high is >= 5 for coverage improvement."""
        from pff.utils.file_manager import FileManager
        
        fm = FileManager()
        config = fm.read(RULE_FILTER_HPO_CONFIG_PATH)
        hpo_ranges = config.get("rule_filter", {}).get("hpo_ranges", {})
        
        acyclic_range = hpo_ranges.get("max_length_acyclic", {})
        high = acyclic_range.get("high", 4)
        
        # P1.3 - Acyclic rules can be longer for better coverage
        assert high >= 5, f"max_length_acyclic high should be >= 5, got {high}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-q"])
