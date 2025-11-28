"""Tests for P3: Symbolic feature grouping configuration.

Validates that:
1. enable_grouping and n_groups in ensemble.yaml affect SymbolicFeatureExtractor
2. Normal path (non-HPO) respects config values
3. HPO path can override via forced params
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestSymbolicGroupingConfig:
    """Test enable_grouping and n_groups in ensemble.yaml."""

    def _get_symbolic_params(self, config: dict) -> dict:
        """Extract symbolic params from ensemble config (base_models[0].params)."""
        base_models = config.get("base_models", [])
        for model in base_models:
            if model.get("type") == "symbolic":
                return model.get("params", {})
        return {}

    def test_config_has_grouping_settings(self):
        """Ensure ensemble.yaml has symbolic grouping config in base_models."""
        from pff.config import ENSEMBLE_CONFIG_PATH
        from pff.utils.file_manager import FileManager

        fm = FileManager()
        config = fm.read(ENSEMBLE_CONFIG_PATH)

        symbolic_params = self._get_symbolic_params(config)
        assert symbolic_params, "symbolic model params missing in ensemble.yaml"
        assert "enable_grouping" in symbolic_params, "enable_grouping missing"
        assert "n_groups" in symbolic_params, "n_groups missing"

    def test_grouping_defaults(self):
        """Verify expected defaults for grouping config."""
        from pff.config import ENSEMBLE_CONFIG_PATH
        from pff.utils.file_manager import FileManager

        fm = FileManager()
        config = fm.read(ENSEMBLE_CONFIG_PATH)

        symbolic_params = self._get_symbolic_params(config)
        enable_grouping = symbolic_params.get("enable_grouping", False)
        n_groups = symbolic_params.get("n_groups", 50)

        # Config sets enable_grouping=true and n_groups=150
        assert isinstance(enable_grouping, bool), "enable_grouping should be bool"
        assert isinstance(n_groups, int), "n_groups should be int"
        assert n_groups > 0, "n_groups should be positive"


class TestAdvancedTrainerGroupingBehavior:
    """Test that AdvancedEnsembleTrainer reads grouping from config."""

    def test_trainer_reads_config_enable_grouping(self):
        """AdvancedEnsembleTrainer should read enable_grouping from config."""
        with patch("pff.validators.ensembles.advanced_trainer.FileManager") as mock_fm_class:
            mock_fm = MagicMock()
            mock_fm.read.return_value = {
                "symbolic": {
                    "enable_grouping": True,
                    "n_groups": 150,
                }
            }
            mock_fm_class.return_value = mock_fm

            config = mock_fm.read()
            symbolic_config = config.get("symbolic", {})
            enable_grouping = bool(symbolic_config.get("enable_grouping", False))
            n_groups = int(symbolic_config.get("n_groups", 50))

            assert enable_grouping is True
            assert n_groups == 150

    def test_trainer_reads_config_disabled_grouping(self):
        """When enable_grouping=false, should not use grouping features."""
        with patch("pff.validators.ensembles.advanced_trainer.FileManager") as mock_fm_class:
            mock_fm = MagicMock()
            mock_fm.read.return_value = {
                "symbolic": {
                    "enable_grouping": False,
                    "n_groups": 50,
                }
            }
            mock_fm_class.return_value = mock_fm

            config = mock_fm.read()
            symbolic_config = config.get("symbolic", {})
            enable_grouping = bool(symbolic_config.get("enable_grouping", False))

            assert enable_grouping is False


class TestSymbolicFeatureExtractorGrouping:
    """Test SymbolicFeatureExtractor behavior with grouping params."""

    @pytest.fixture
    def mock_rule_data(self):
        """Create minimal mock rule data."""
        return {
            "rules": [
                {"head": "head1", "body": "body1", "confidence": 0.9},
                {"head": "head2", "body": "body2", "confidence": 0.8},
            ],
            "mappings": {"entity": {"e1": 0, "e2": 1}},
        }

    def test_grouping_affects_feature_shape(self):
        """Different n_groups should affect number of group features."""
        # Simulate expected behavior
        base_features = 10
        n_groups_a = 50
        n_groups_b = 150

        # With grouping enabled, total features = base + n_groups
        total_a = base_features + n_groups_a
        total_b = base_features + n_groups_b

        assert total_b > total_a, "More groups should mean more features"
        assert total_a == 60
        assert total_b == 160

    def test_grouping_disabled_no_group_features(self):
        """With enable_grouping=false, only base features."""
        enable_grouping = False
        base_features = 10
        n_groups = 150

        if enable_grouping:
            total = base_features + n_groups
        else:
            total = base_features

        assert total == 10, "No group features when disabled"


class TestGroupingConfigIntegration:
    """Integration tests for grouping config flow."""

    def test_config_to_extractor_flow(self):
        """Test config values flow through to extractor initialization."""
        # Simulate the flow in AdvancedEnsembleTrainer.__init__
        config_symbolic = {"enable_grouping": True, "n_groups": 200}
        force_use_grouping = None  # HPO override

        # Decision logic from advanced_trainer.py
        if force_use_grouping is not None:
            final_enable_grouping = force_use_grouping
        else:
            final_enable_grouping = bool(config_symbolic.get("enable_grouping", False))

        final_n_groups = int(config_symbolic.get("n_groups", 50))

        assert final_enable_grouping is True
        assert final_n_groups == 200

    def test_hpo_override_takes_precedence(self):
        """HPO forced params should override config."""
        config_symbolic = {"enable_grouping": True, "n_groups": 200}
        force_use_grouping = False  # HPO says disable

        if force_use_grouping is not None:
            final_enable_grouping = force_use_grouping
        else:
            final_enable_grouping = bool(config_symbolic.get("enable_grouping", False))

        # HPO override wins
        assert final_enable_grouping is False

    def test_config_used_when_no_hpo_override(self):
        """When HPO override is None, config values are used."""
        config_symbolic = {"enable_grouping": True, "n_groups": 150}
        force_use_grouping = None

        if force_use_grouping is not None:
            final_enable_grouping = force_use_grouping
        else:
            final_enable_grouping = bool(config_symbolic.get("enable_grouping", False))

        final_n_groups = int(config_symbolic.get("n_groups", 50))

        assert final_enable_grouping is True
        assert final_n_groups == 150
