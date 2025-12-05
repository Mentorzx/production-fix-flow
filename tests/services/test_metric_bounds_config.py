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

    def test_metrics_bounds_include_p2_keys(self):
        """Ensure P2 metrics (relation/xgb agreement) are present in config bounds."""
        from pff.utils.file_manager import FileManager

        fm = FileManager()
        cfg = fm.read(ENSEMBLE_HPO_CONFIG_PATH)
        bounds = cfg.get("metrics_bounds", {})

        assert "relation_coverage" in bounds.get("rules", {})
        assert "rules_per_relation" in bounds.get("rules", {})
        assert "base_learner_agreement" in bounds.get("learner", {})
        assert "xgb_test_auc" in bounds.get("learner", {})
        assert "lightgbm_ece" in bounds.get("learner", {})
        assert "lightgbm_entropy" in bounds.get("learner", {})
        assert "ensemble_ece" in bounds.get("ensemble", {})
        assert "ensemble_entropy" in bounds.get("ensemble", {})


class TestMetricBoundsHelpers:
    """Test helpers that load and read metric bounds."""

    def test_load_metric_bounds_custom(self):
        custom = {
            "kge": {"mrr": {"low": 0.2, "high": 0.8}},
            "rules": {"coverage": {"low": 0.1, "high": 0.6}},
            "learner": {"lgbm_auc": {"low": 0.55, "high": 0.95}},
        }
        from scripts.optimization.trials import config_loader
        config_loader._CONFIG_CACHE.clear()

        with patch("scripts.optimization.trials.bounds.get_cached_config") as mock_config:
            mock_config.return_value = {"metrics_bounds": custom}

            from scripts.optimization.trials.bounds import load_metric_bounds, get_range
            bounds = load_metric_bounds()
            mrr_low, mrr_high = get_range(bounds, ["kge", "mrr"], 0.15, 0.75)
            cov_low, cov_high = get_range(bounds, ["rules", "coverage"], 0.05, 0.5)

        assert bounds["kge"]["mrr"]["low"] == 0.2
        assert (mrr_low, mrr_high) == (0.2, 0.8)
        assert (cov_low, cov_high) == (0.1, 0.6)

    def test_metric_bounds_defaults_when_missing(self):
        """Test that defaults are returned when metrics_bounds is missing from config."""
        from scripts.optimization.trials.bounds import load_metric_bounds, get_range

        # Clear cache to ensure fresh load
        from scripts.optimization.trials import config_loader
        config_loader._CONFIG_CACHE.clear()

        with patch("scripts.optimization.trials.bounds.get_cached_config") as mock_config:
            mock_config.return_value = {}  # missing metrics_bounds

            bounds = load_metric_bounds()
            # Check defaults are used
            lgb_low, lgb_high = get_range(bounds, ["learner", "lgbm_auc"], 0.6, 0.99)

        # When bounds is missing, get_range should use the function default args
        assert lgb_low == 0.6
        assert lgb_high == 0.99
