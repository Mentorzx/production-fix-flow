"""Tests for config-driven metric normalization bounds."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from pff.config import ENSEMBLE_HPO_CONFIG_PATH


class TestMetricBoundsConfig:
    """Verify metrics_bounds exist in ensemble.yaml."""

    def test_config_has_metrics_bounds(self):
        from pff.utils.file_manager import FileManager

        fm = FileManager()
        cfg = fm.read(ENSEMBLE_HPO_CONFIG_PATH)
        assert "metrics_bounds" in cfg, "metrics_bounds missing in ensemble.yaml"


class TestMetricBoundsHelpers:
    """Test helpers that load and read metric bounds."""

    def test_load_metric_bounds_custom(self):
        custom = {
            "kge": {"mrr": {"low": 0.2, "high": 0.8}},
            "rules": {"coverage": {"low": 0.1, "high": 0.6}},
            "learner": {"lgbm_auc": {"low": 0.55, "high": 0.95}},
        }
        with patch("scripts.optimization.core.FileManager") as mock_fm_class:
            mock_fm = MagicMock()
            mock_fm.read.return_value = {"metrics_bounds": custom}
            mock_fm_class.return_value = mock_fm

            import importlib
            import scripts.optimization.core as core_module
            importlib.reload(core_module)

            with patch.object(core_module, "FileManager", mock_fm_class):
                bounds = core_module._load_metric_bounds()
                mrr_low, mrr_high = core_module._get_range(bounds, ["kge", "mrr"], 0.15, 0.75)
                cov_low, cov_high = core_module._get_range(bounds, ["rules", "coverage"], 0.05, 0.5)

        assert bounds["kge"]["mrr"]["low"] == 0.2
        assert (mrr_low, mrr_high) == (0.2, 0.8)
        assert (cov_low, cov_high) == (0.1, 0.6)

    def test_metric_bounds_defaults_when_missing(self):
        with patch("scripts.optimization.core.FileManager") as mock_fm_class:
            mock_fm = MagicMock()
            mock_fm.read.return_value = {}  # missing metrics_bounds
            mock_fm_class.return_value = mock_fm

            import importlib
            import scripts.optimization.core as core_module
            importlib.reload(core_module)

            with patch.object(core_module, "FileManager", mock_fm_class):
                bounds = core_module._load_metric_bounds()
                lgb_low, lgb_high = core_module._get_range(bounds, ["learner", "lgbm_auc"], 0.6, 0.99)

        assert lgb_low == 0.6
        assert lgb_high == 0.99
