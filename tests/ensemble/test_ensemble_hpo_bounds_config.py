"""Tests for config-driven ensemble HPO bounds (weights/thresholds).

Ensures scripts/optimization/core.py reads bounds from config/models/ensemble.yaml
and falls back to defaults when missing.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from pff.config import ENSEMBLE_HPO_CONFIG_PATH, ENSEMBLE_CONFIG_PATH


class TestEnsembleHPOBoundsConfig:
    """Validate hpo_bounds presence and structure."""

    def test_config_has_hpo_bounds(self):
        """Ensure ensemble.yaml declares hpo_bounds with required keys."""
        from pff.utils.file_manager import FileManager

        fm = FileManager()
        # Prefer dedicated HPO file; fall back to legacy if needed
        config = fm.read(ENSEMBLE_HPO_CONFIG_PATH)

        assert "hpo_bounds" in config, "hpo_bounds section missing in ensemble.yaml"
        bounds = config["hpo_bounds"]
        for key in ("weights", "thresholds", "target_symbolic_ratio", "feature_selection_threshold"):
            assert key in bounds, f"{key} missing in hpo_bounds"


class TestEnsembleHPOBoundsLoading:
    """Test helper that loads and reads HPO bounds."""

    def test_loads_custom_bounds(self):
        """Verify _load_ensemble_hpo_bounds returns custom config values."""
        custom_bounds = {
            "weights": {
                "neural_weight": {"low": 0.25, "high": 0.5},
                "rules_weight": {"low": 0.15, "high": 0.3},
                "lightgbm_weight": {"low": 0.4, "high": 0.65},
            },
            "thresholds": {
                "neural_threshold": {"low": 0.25, "high": 0.6},
                "rules_threshold": {"low": 0.15, "high": 0.65},
                "lightgbm_threshold": {"low": 0.25, "high": 0.6},
            },
            "target_symbolic_ratio": {"low": 0.25, "high": 0.5},
            "feature_selection_threshold": {"low": 0.25, "high": 0.6},
        }

        with patch("scripts.optimization.core.FileManager") as mock_fm_class:
            mock_fm = MagicMock()
            mock_fm.read.return_value = {"hpo_bounds": custom_bounds}
            mock_fm_class.return_value = mock_fm

            import importlib
            import scripts.optimization.core as core_module
            importlib.reload(core_module)

            with patch.object(core_module, "FileManager", mock_fm_class):
                bounds = core_module._load_ensemble_hpo_bounds()
                nw_low, nw_high = core_module._get_range(bounds, ["weights", "neural_weight"], 0.2, 0.45)
                nt_low, nt_high = core_module._get_range(bounds, ["thresholds", "neural_threshold"], 0.3, 0.7)

        assert bounds == custom_bounds
        assert (nw_low, nw_high) == (0.25, 0.5)
        assert (nt_low, nt_high) == (0.25, 0.6)

    def test_defaults_when_missing(self):
        """Verify defaults are used when hpo_bounds is absent."""
        with patch("scripts.optimization.core.FileManager") as mock_fm_class:
            mock_fm = MagicMock()
            mock_fm.read.return_value = {}  # No hpo_bounds
            mock_fm_class.return_value = mock_fm

            import importlib
            import scripts.optimization.core as core_module
            importlib.reload(core_module)

            with patch.object(core_module, "FileManager", mock_fm_class):
                bounds = core_module._load_ensemble_hpo_bounds()
                rw_low, rw_high = core_module._get_range(bounds, ["weights", "rules_weight"], 0.1, 0.25)

        # Defaults mirror legacy literals
        assert rw_low == 0.1
        assert rw_high == 0.25
