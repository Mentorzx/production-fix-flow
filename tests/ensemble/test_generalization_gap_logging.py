"""Tests for P2.1 - Generalization gap logging.

Verifies that AdvancedEnsembleTrainer computes and logs the generalization gap
between OOF (CV) and holdout metrics, and persists it in metrics_all.json.

Author: PFF Team
Date: 2025-11-27
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest


class TestGeneralizationGapComputation:
    """Test the generalization gap computation logic."""

    @pytest.fixture
    def mock_file_manager(self):
        """Create a mock FileManager."""
        mock_fm = MagicMock()
        mock_fm.read.return_value = {
            "balancing": {"symbolic_dominance_threshold": 0.85},
            "ensemble_weights": {"neural": 0.2, "rules": 0.2, "lightgbm": 0.6},
            "adaptive_weighting": {"enabled": False},
        }
        return mock_fm

    def test_compute_generalization_gap_normal_case(self, mock_file_manager):
        """Test generalization gap with normal CV and holdout metrics."""
        with patch(
            "pff.validators.ensembles.advanced_trainer.FileManager",
            return_value=mock_file_manager,
        ):
            from pff.validators.ensembles.advanced_trainer import AdvancedEnsembleTrainer

            with patch.object(AdvancedEnsembleTrainer, "_resolve_lightgbm_path"):
                trainer = AdvancedEnsembleTrainer(
                    neural_model_path="/fake/path",
                    rules_path="/fake/rules.tsv",
                    lightgbm_model_path="/fake/lgb.bin",
                    file_manager=mock_file_manager,
                )

                cv_results = {
                    "roc_auc_test_mean": 0.85,
                    "roc_auc_test_std": 0.02,
                    "f1_test_mean": 0.75,
                }
                holdout_metrics = {
                    "test_auc_roc": 0.82,
                    "test_f1_score": 0.72,
                }

                gap = trainer._compute_generalization_gap(
                    cv_results, holdout_metrics, "roc_auc"
                )

                assert gap["oof_metric"] == 0.85
                assert gap["holdout_metric"] == 0.82
                assert abs(gap["gap"] - 0.03) < 0.001
                assert gap["gap_percentage"] > 0  # Positive gap = overfitting

    def test_compute_generalization_gap_no_cv_results(self, mock_file_manager):
        """Test generalization gap when CV results are missing."""
        with patch(
            "pff.validators.ensembles.advanced_trainer.FileManager",
            return_value=mock_file_manager,
        ):
            from pff.validators.ensembles.advanced_trainer import AdvancedEnsembleTrainer

            with patch.object(AdvancedEnsembleTrainer, "_resolve_lightgbm_path"):
                trainer = AdvancedEnsembleTrainer(
                    neural_model_path="/fake/path",
                    rules_path="/fake/rules.tsv",
                    lightgbm_model_path="/fake/lgb.bin",
                    file_manager=mock_file_manager,
                )

                holdout_metrics = {"test_auc_roc": 0.82}

                gap = trainer._compute_generalization_gap(
                    None, holdout_metrics, "roc_auc"
                )

                assert gap["oof_metric"] == 0.0
                assert gap["holdout_metric"] == 0.82
                assert gap["gap"] == -0.82  # OOF=0, holdout=0.82

    def test_compute_generalization_gap_negative_gap(self, mock_file_manager):
        """Test generalization gap when holdout is better than OOF (negative gap)."""
        with patch(
            "pff.validators.ensembles.advanced_trainer.FileManager",
            return_value=mock_file_manager,
        ):
            from pff.validators.ensembles.advanced_trainer import AdvancedEnsembleTrainer

            with patch.object(AdvancedEnsembleTrainer, "_resolve_lightgbm_path"):
                trainer = AdvancedEnsembleTrainer(
                    neural_model_path="/fake/path",
                    rules_path="/fake/rules.tsv",
                    lightgbm_model_path="/fake/lgb.bin",
                    file_manager=mock_file_manager,
                )

                cv_results = {"roc_auc_test_mean": 0.75}
                holdout_metrics = {"test_auc_roc": 0.80}

                gap = trainer._compute_generalization_gap(
                    cv_results, holdout_metrics, "roc_auc"
                )

                assert gap["gap"] < 0  # Negative gap = holdout better
                assert gap["gap_percentage"] < 0

    def test_compute_generalization_gap_f1_metric(self, mock_file_manager):
        """Test generalization gap with F1 metric instead of AUC."""
        with patch(
            "pff.validators.ensembles.advanced_trainer.FileManager",
            return_value=mock_file_manager,
        ):
            from pff.validators.ensembles.advanced_trainer import AdvancedEnsembleTrainer

            with patch.object(AdvancedEnsembleTrainer, "_resolve_lightgbm_path"):
                trainer = AdvancedEnsembleTrainer(
                    neural_model_path="/fake/path",
                    rules_path="/fake/rules.tsv",
                    lightgbm_model_path="/fake/lgb.bin",
                    file_manager=mock_file_manager,
                )

                cv_results = {"f1_test_mean": 0.78}
                holdout_metrics = {"test_f1_score": 0.74}

                gap = trainer._compute_generalization_gap(
                    cv_results, holdout_metrics, "f1"
                )

                assert gap["oof_metric"] == 0.78
                assert gap["holdout_metric"] == 0.74
                assert abs(gap["gap"] - 0.04) < 0.001


class TestGeneralizationGapInReport:
    """Test that generalization gap is included in metrics_all.json."""

    @pytest.fixture
    def mock_trainer_with_model(self):
        """Create a trainer with a mock ensemble model."""
        from pff.validators.ensembles.advanced_trainer import AdvancedEnsembleTrainer

        with patch(
            "pff.validators.ensembles.advanced_trainer.FileManager"
        ) as mock_fm_class:
            mock_fm = MagicMock()
            mock_fm.read.return_value = {
                "balancing": {"symbolic_dominance_threshold": 0.85},
                "ensemble_weights": {"neural": 0.2, "rules": 0.2, "lightgbm": 0.6},
                "adaptive_weighting": {"enabled": False},
            }
            mock_fm_class.return_value = mock_fm

            with patch.object(AdvancedEnsembleTrainer, "_resolve_lightgbm_path"):
                trainer = AdvancedEnsembleTrainer(
                    neural_model_path="/fake/path",
                    rules_path="/fake/rules.tsv",
                    lightgbm_model_path="/fake/lgb.bin",
                    file_manager=mock_fm,
                )

                # Setup mock ensemble model
                mock_model = Mock()
                mock_model.predict.return_value = np.array([0, 1, 0, 1])
                mock_model.predict_proba.return_value = np.array([
                    [0.8, 0.2], [0.3, 0.7], [0.9, 0.1], [0.2, 0.8]
                ])
                
                # For hierarchical mode: provide feature_importances_ on the model itself
                # (used when "xgboost" is not in named_steps)
                mock_model.feature_importances_ = np.array([0.4, 0.3, 0.2, 0.1])
                
                # XGBoost config attributes (used in report generation)
                mock_model.max_depth = 6
                mock_model.colsample_bytree = 0.8
                mock_model.reg_alpha = 0.1
                mock_model.subsample = 0.9

                # Mock feature union
                mock_features = Mock()
                mock_base_union = Mock()
                mock_symbolic = Mock()
                mock_symbolic.enable_grouping = False
                mock_symbolic.rules_ = [f"rule_{i}" for i in range(5)]
                mock_base_union.transformer_list = [
                    ("hybrid", Mock()),
                    ("symbolic", mock_symbolic),
                ]
                mock_features.base_union = mock_base_union

                # Mock meta-learner
                mock_meta = Mock()
                mock_meta.feature_importances_ = np.array([0.3, 0.2, 0.15, 0.15, 0.1, 0.1])
                mock_meta.max_depth = 6
                mock_meta.colsample_bytree = 0.8
                mock_meta.reg_alpha = 0.1
                mock_meta.subsample = 0.9

                mock_model.named_steps = {
                    "features": mock_features,
                    "meta_learner": mock_meta,
                }

                trainer.ensemble_model = mock_model
                trainer.output_dir = Path("/tmp/test_output")

                return trainer, mock_fm

    def test_generalization_gap_in_metrics_dict(self, mock_trainer_with_model):
        """Test that generalization gap is added to the metrics report."""
        trainer, mock_fm = mock_trainer_with_model

        X_test = np.random.randn(4, 10)
        y_test = np.array([0, 1, 0, 1])
        cv_results = {
            "roc_auc_test_mean": 0.88,
            "f1_test_mean": 0.76,
        }

        # Capture what gets saved
        saved_data = {}

        def capture_save(data, path):
            saved_data["report"] = data
            saved_data["path"] = path

        mock_fm.save = capture_save

        with patch("pff.validators.ensembles.advanced_trainer.FileManager", return_value=mock_fm):
            trainer._save_final_metrics_report(X_test, y_test, cv_results=cv_results)

        assert "report" in saved_data
        report = saved_data["report"]

        # Verify generalization_gap section exists
        assert "generalization_gap" in report
        gap_section = report["generalization_gap"]
        
        assert "metric" in gap_section
        assert "oof_value" in gap_section
        assert "holdout_value" in gap_section
        assert "gap" in gap_section
        assert "gap_percentage" in gap_section
        
        assert gap_section["metric"] == "roc_auc"
        assert gap_section["oof_value"] == 0.88


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-q"])
