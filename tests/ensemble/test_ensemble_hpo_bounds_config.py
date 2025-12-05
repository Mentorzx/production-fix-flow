"""Tests for config-driven ensemble HPO bounds (weights/thresholds).

Ensures scripts/optimization/trials/config_loader.py reads bounds from 
config/hpo/ensemble_hpo.yaml and falls back to defaults when missing.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from pff.config import ENSEMBLE_HPO_CONFIG_PATH
from scripts.optimization.trials.config_loader import clear_config_cache


@pytest.fixture(autouse=True)
def clear_cache():
    """Clear config cache before each test."""
    clear_config_cache()
    yield
    clear_config_cache()


class TestEnsembleHPOBoundsConfig:
    """Validate hpo_bounds presence and structure."""

    def test_config_has_hpo_bounds(self):
        """Ensure ensemble_hpo.yaml declares hpo_bounds with required keys."""
        from pff.utils.file_manager import FileManager

        fm = FileManager()
        # Prefer dedicated HPO file
        config = fm.read(ENSEMBLE_HPO_CONFIG_PATH)

        assert "hpo_bounds" in config, "hpo_bounds section missing in ensemble_hpo.yaml"
        bounds = config["hpo_bounds"]
        for key in ("weights", "thresholds", "target_symbolic_ratio", "feature_selection_threshold", "kge"):
            assert key in bounds, f"{key} missing in hpo_bounds"
        assert "batch_size" in bounds["kge"]


class TestEnsembleHPOBoundsLoading:
    """Test helper that loads and reads HPO bounds."""

    def test_loads_custom_bounds(self):
        """Verify load_ensemble_hpo_bounds returns custom config values."""
        from scripts.optimization.trials.config_loader import load_ensemble_hpo_bounds
        from scripts.optimization.trials.bounds import get_range

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
            "kge": {
                "negative_ratio": {"low": 0.4, "high": 0.8},
                "embedding_dim": {"choices": [128]},
                "self_adversarial": {"choices": [False]},
                "batch_size": {"low": 256, "high": 640},
            },
        }

        mock_fm = MagicMock()
        mock_fm.read.return_value = {"hpo_bounds": custom_bounds}

        bounds = load_ensemble_hpo_bounds(mock_fm)
        nw_low, nw_high = get_range(bounds, ["weights", "neural_weight"], 0.2, 0.45)
        nt_low, nt_high = get_range(bounds, ["thresholds", "neural_threshold"], 0.3, 0.7)

        assert bounds == custom_bounds
        assert (nw_low, nw_high) == (0.25, 0.5)
        assert (nt_low, nt_high) == (0.25, 0.6)

    def test_defaults_when_missing(self):
        """Verify defaults are used when hpo_bounds is absent."""
        from scripts.optimization.trials.config_loader import load_ensemble_hpo_bounds
        from scripts.optimization.trials.bounds import get_range

        mock_fm = MagicMock()
        mock_fm.read.return_value = {}  # No hpo_bounds

        bounds = load_ensemble_hpo_bounds(mock_fm)
        # When config is missing, load_ensemble_hpo_bounds returns built-in defaults
        # which have rules_weight: {"low": 0.1, "high": 0.25}
        rw_low, rw_high = get_range(bounds, ["weights", "rules_weight"], 0.1, 0.25)

        # Should get the built-in defaults from load_ensemble_hpo_bounds
        assert rw_low == 0.1
        assert rw_high == 0.25


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-q"])
