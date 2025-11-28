"""Tests for P2.3 - Top-k symbolic feature importance logging.

Verifies that the ensemble trainer extracts and logs top-k symbolic features
with their importance scores.

Author: PFF Team
Date: 2025-11-27
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Test the extraction method
class TestTopSymbolicFeaturesExtraction:
    """Test _extract_top_symbolic_features method."""

    def test_extract_top_k_features(self):
        """Test extraction of top-k symbolic features."""
        from pff.validators.ensembles.advanced_trainer import AdvancedEnsembleTrainer

        trainer = AdvancedEnsembleTrainer.__new__(AdvancedEnsembleTrainer)
        trainer.logger = MagicMock()

        # First element is hybrid_probability (skipped)
        feature_names = [
            "hybrid_probability",
            "rule_conf_max",
            "rule_coverage",
            "rule_head_coverage",
            "embedding_sim",
        ]
        importances = np.array([0.10, 0.35, 0.25, 0.20, 0.10])

        result = trainer._extract_top_symbolic_features(
            importances=importances,
            feature_names=feature_names,
            top_k=3,
        )

        assert len(result) == 3
        # Results are dicts with 'name' and 'importance'
        assert result[0]["name"] == "rule_conf_max"
        assert result[0]["importance"] == 0.35
        assert result[1]["name"] == "rule_coverage"
        assert result[1]["importance"] == 0.25

    def test_extract_features_returns_sorted(self):
        """Test that features are sorted by importance (descending)."""
        from pff.validators.ensembles.advanced_trainer import AdvancedEnsembleTrainer

        trainer = AdvancedEnsembleTrainer.__new__(AdvancedEnsembleTrainer)
        trainer.logger = MagicMock()

        # First element is hybrid_probability (skipped), rest NOT in order
        feature_names = ["hybrid", "feature_a", "feature_b", "feature_c", "feature_d"]
        importances = np.array([0.05, 0.10, 0.40, 0.30, 0.15])

        result = trainer._extract_top_symbolic_features(
            importances=importances,
            feature_names=feature_names,
            top_k=4,
        )

        # Should be sorted descending by importance (excluding hybrid)
        assert result[0]["name"] == "feature_b"  # 0.40
        assert result[1]["name"] == "feature_c"  # 0.30
        assert result[2]["name"] == "feature_d"  # 0.15
        assert result[3]["name"] == "feature_a"  # 0.10

    def test_extract_features_with_fewer_than_k(self):
        """Test extraction when fewer features than k exist."""
        from pff.validators.ensembles.advanced_trainer import AdvancedEnsembleTrainer

        trainer = AdvancedEnsembleTrainer.__new__(AdvancedEnsembleTrainer)
        trainer.logger = MagicMock()

        feature_names = ["hybrid", "feature_a", "feature_b"]
        importances = np.array([0.1, 0.5, 0.4])

        result = trainer._extract_top_symbolic_features(
            importances=importances,
            feature_names=feature_names,
            top_k=10,  # Requesting more than available
        )

        assert len(result) == 2  # Only 2 symbolic features (hybrid skipped)
        assert result[0]["name"] == "feature_a"
        assert result[1]["name"] == "feature_b"

    def test_extract_features_insufficient_input(self):
        """Test extraction with insufficient feature lists."""
        from pff.validators.ensembles.advanced_trainer import AdvancedEnsembleTrainer

        trainer = AdvancedEnsembleTrainer.__new__(AdvancedEnsembleTrainer)
        trainer.logger = MagicMock()

        # Only 1 element, not enough for extraction (need at least 2)
        result = trainer._extract_top_symbolic_features(
            importances=np.array([0.5]),
            feature_names=["hybrid"],
            top_k=5,
        )

        assert result == []

    def test_extract_features_skips_hybrid_index_0(self):
        """Test that index 0 (hybrid_probability) is skipped."""
        from pff.validators.ensembles.advanced_trainer import AdvancedEnsembleTrainer

        trainer = AdvancedEnsembleTrainer.__new__(AdvancedEnsembleTrainer)
        trainer.logger = MagicMock()

        feature_names = ["hybrid_probability", "rule_a", "rule_b"]
        importances = np.array([0.90, 0.05, 0.05])  # hybrid has highest

        result = trainer._extract_top_symbolic_features(
            importances=importances,
            feature_names=feature_names,
            top_k=10,
        )

        # Should NOT include hybrid_probability despite it having highest importance
        names = [f["name"] for f in result]
        assert "hybrid_probability" not in names
        assert "rule_a" in names
        assert "rule_b" in names


class TestSymbolicFeaturesInReport:
    """Test that symbolic features appear in final metrics report."""

    def test_extract_method_returns_correct_format(self):
        """Test that _extract_top_symbolic_features returns correct dict format."""
        from pff.validators.ensembles.advanced_trainer import AdvancedEnsembleTrainer

        trainer = AdvancedEnsembleTrainer.__new__(AdvancedEnsembleTrainer)
        trainer.logger = MagicMock()

        feature_names = ["hybrid", "rule_conf", "rule_cov", "emb_sim"]
        importances = np.array([0.10, 0.45, 0.35, 0.10])

        result = trainer._extract_top_symbolic_features(
            importances=importances,
            feature_names=feature_names,
            top_k=3,
        )

        # Should be list of dicts with 'name' and 'importance' keys
        assert len(result) >= 1
        assert "name" in result[0]
        assert "importance" in result[0]
        assert isinstance(result[0]["name"], str)
        assert isinstance(result[0]["importance"], float)

    def test_extract_method_values_are_correct(self):
        """Test that extracted values match input."""
        from pff.validators.ensembles.advanced_trainer import AdvancedEnsembleTrainer

        trainer = AdvancedEnsembleTrainer.__new__(AdvancedEnsembleTrainer)
        trainer.logger = MagicMock()

        feature_names = ["hybrid", "rule_a", "rule_b"]
        importances = np.array([0.10, 0.55, 0.35])

        result = trainer._extract_top_symbolic_features(
            importances=importances,
            feature_names=feature_names,
            top_k=5,
        )

        # First result should be rule_a with highest importance
        assert result[0]["name"] == "rule_a"
        assert result[0]["importance"] == 0.55
        assert result[1]["name"] == "rule_b"
        assert result[1]["importance"] == 0.35

    def test_extract_method_json_serializable(self):
        """Test that output can be serialized to JSON."""
        from pff.validators.ensembles.advanced_trainer import AdvancedEnsembleTrainer

        trainer = AdvancedEnsembleTrainer.__new__(AdvancedEnsembleTrainer)
        trainer.logger = MagicMock()

        feature_names = ["hybrid", "rule_conf", "rule_cov"]
        importances = np.array([0.10, 0.60, 0.30])

        result = trainer._extract_top_symbolic_features(
            importances=importances,
            feature_names=feature_names,
            top_k=2,
        )

        # Should be JSON serializable for report writing
        json_str = json.dumps(result)
        assert json_str is not None
        
        # Parse back and verify
        parsed = json.loads(json_str)
        assert len(parsed) == 2
        assert parsed[0]["name"] == "rule_conf"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-q"])
