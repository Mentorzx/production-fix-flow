"""Tests for config-driven metric normalization bounds."""

from __future__ import annotations

from unittest.mock import patch

from pff.shared.core.config import ENSEMBLE_HPO_CONFIG_PATH


class TestMetricBoundsConfig:
    """Verify metrics_bounds exist in ensemble.yaml."""

    def test_config_has_metrics_bounds(self):
        from pff.shared.core.file_manager import FileManager

        fm = FileManager()
        cfg = fm.read(ENSEMBLE_HPO_CONFIG_PATH, return_native=True)
        assert "metrics_bounds" in cfg, "metrics_bounds missing in ensemble.yaml"

    def test_metrics_bounds_include_p2_keys(self):
        """Ensure rules + learner metric bounds are present in config."""
        from pff.shared.core.file_manager import FileManager

        fm = FileManager()
        cfg = fm.read(ENSEMBLE_HPO_CONFIG_PATH, return_native=True)
        bounds = cfg.get("metrics_bounds", {})

        assert "relation_coverage" in bounds.get("rules", {})
        assert "rules_per_relation" in bounds.get("rules", {})
        assert "auc" in bounds.get("learner", {})
        assert "pr_auc" in bounds.get("learner", {})
        assert "precision" in bounds.get("learner", {})
        assert "recall" in bounds.get("learner", {})
        assert "ensemble_ece" in bounds.get("ensemble", {})
        assert "ensemble_entropy" in bounds.get("ensemble", {})


class TestMetricBoundsHelpers:
    """Test helpers that load and read metric bounds."""

    def test_load_metric_bounds_custom(self):
        custom = {
            "kge": {"mrr": {"low": 0.2, "high": 0.8}},
            "rules": {"coverage": {"low": 0.1, "high": 0.6}},
            "learner": {"auc": {"low": 0.55, "high": 0.95}},
        }
        with patch(
            "pff.infrastructure.hpo.config_loader.load_optimization_config"
        ) as mock_config:
            mock_config.return_value = {"metrics_bounds": custom}

            from pff.infrastructure.hpo.config_loader import load_metric_bounds
            from pff.domain.hpo.bounds import get_range

            bounds = load_metric_bounds()
            mrr_low, mrr_high = get_range(bounds, ["kge", "mrr"], 0.15, 0.75)
            cov_low, cov_high = get_range(bounds, ["rules", "coverage"], 0.05, 0.5)

        assert bounds["kge"]["mrr"]["low"] == 0.2
        assert (mrr_low, mrr_high) == (0.2, 0.8)
        assert (cov_low, cov_high) == (0.1, 0.6)

    def test_metric_bounds_defaults_when_missing(self):
        """Test that defaults are returned when metrics_bounds is missing from config."""
        from pff.infrastructure.hpo.config_loader import load_metric_bounds
        from pff.domain.hpo.bounds import get_range

        with patch(
            "pff.infrastructure.hpo.config_loader.load_optimization_config"
        ) as mock_config:
            mock_config.return_value = {}  # missing metrics_bounds

            bounds = load_metric_bounds()
            # Check defaults are used
            auc_low, auc_high = get_range(bounds, ["learner", "auc"], 0.6, 0.99)

        # When bounds is missing, get_range should use the function default args
        assert auc_low == 0.5
        assert auc_high == 0.99
