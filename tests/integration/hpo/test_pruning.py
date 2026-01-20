"""Tests for trial pruning and early stopping in DSLFM training."""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from pff.domain.learning.dslfm.dslfm_kgc import DSLFMKGCConfig
from pff.domain.learning.dslfm.kgc_manager import DSLFMKGCManager, KGCTrainingConfig


class TestPruning(unittest.TestCase):
    def setUp(self):
        # Minimal configurations for fast testing (CPU)
        self.model_config = DSLFMKGCConfig(
            num_entities=10,
            num_relations=2,
            entity_dim=8,
            feature_dim=8,
            hidden_dim=8,
        )
        self.train_config = KGCTrainingConfig(
            epochs=10,
            batch_size=4,
            validate_every=1,
            early_stopping_patience=3,
        )

        # Mock triples
        self.train_triples = np.array([[0, 0, 1], [1, 0, 0], [2, 1, 3], [3, 1, 2]])
        self.valid_triples = np.array([[0, 0, 2], [1, 0, 3]])

    @patch("pff.domain.learning.dslfm.kgc_manager.DSLFMKGCModel")
    @patch("pff.domain.learning.dslfm.kgc_manager._bind_evaluate")
    @patch("pff.domain.learning.dslfm.kgc_manager.TripleDataset")
    @patch("pff.domain.learning.dslfm.kgc_manager.DataLoader")
    def test_trial_pruning(self, mock_loader, mock_dataset, mock_bind, mock_model_class):
        """Verify that training stops when trial.should_prune() is True."""
        # Setup mock model
        mock_model = MagicMock()
        mock_model.device = torch.device("cpu")
        # Provide a dummy parameter
        param = torch.nn.Parameter(torch.randn(1))
        mock_model.parameters.return_value = [param]
        mock_model_class.return_value = mock_model
        mock_bind.return_value = mock_model

        # Mock evaluation to return static MRR
        manager = DSLFMKGCManager(self.model_config, self.train_config)
        manager._validate = MagicMock(return_value={"mrr": 0.5})
        manager._compute_binary_metrics_internal = MagicMock(return_value={"mcc": 0.5})
        manager._train_epoch = MagicMock(return_value={"loss": 0.1})
        manager._save_checkpoint = MagicMock()

        # Mock trial
        mock_trial = MagicMock()
        # Return True on the 3rd epoch (epoch 2)
        mock_trial.should_prune.side_effect = [False, False, True, True, True]

        # Run training
        import optuna

        with self.assertRaises(optuna.TrialPruned):
            manager.train(self.train_triples, self.valid_triples, trial=mock_trial)

        # Assertions
        # Epochs 0, 1 were processed, epoch 2 started and then pruned
        self.assertEqual(manager.current_epoch, 2)
        self.assertEqual(mock_trial.report.call_count, 3)
        self.assertEqual(mock_trial.should_prune.call_count, 3)

    @patch("pff.domain.learning.dslfm.kgc_manager.DSLFMKGCModel")
    @patch("pff.domain.learning.dslfm.kgc_manager._bind_evaluate")
    @patch("pff.domain.learning.dslfm.kgc_manager.TripleDataset")
    @patch("pff.domain.learning.dslfm.kgc_manager.DataLoader")
    def test_early_stopping(self, mock_loader, mock_dataset, mock_bind, mock_model_class):
        """Verify that training stops when patience is exceeded."""
        # Setup mock model
        mock_model = MagicMock()
        mock_model.device = torch.device("cpu")
        # Provide a dummy parameter
        param = torch.nn.Parameter(torch.randn(1))
        mock_model.parameters.return_value = [param]
        mock_model_class.return_value = mock_model
        mock_bind.return_value = mock_model

        # Mock evaluation to return stagnant MRR
        manager = DSLFMKGCManager(self.model_config, self.train_config)
        manager._validate = MagicMock(return_value={"mrr": 0.1})
        manager._compute_binary_metrics_internal = MagicMock(return_value={"mcc": 0.1})
        manager._train_epoch = MagicMock(return_value={"loss": 0.1})
        manager._save_checkpoint = MagicMock()

        # Run training (patience=3)
        # Epoch 0: MRR 0.1 (new best) -> patience 0
        # Epoch 1: MRR 0.1 (not better) -> patience 1
        # Epoch 2: MRR 0.1 (not better) -> patience 2
        # Epoch 3: MRR 0.1 (not better) -> patience 3 -> STOP
        stats = manager.train(self.train_triples, self.valid_triples)

        # Assertions
        self.assertEqual(stats["epochs_trained"], 4)
        self.assertEqual(manager.patience_counter, 3)

    @patch("pff.domain.learning.dslfm.kgc_manager.DSLFMKGCModel")
    @patch("pff.domain.learning.dslfm.kgc_manager._bind_evaluate")
    @patch("pff.domain.learning.dslfm.kgc_manager.TripleDataset")
    @patch("pff.domain.learning.dslfm.kgc_manager.DataLoader")
    def test_time_budget_pruning(self, mock_loader, mock_dataset, mock_bind, mock_model_class):
        """Verify that training stops when time budget is exceeded."""
        # Setup mock model
        mock_model = MagicMock()
        mock_model.device = torch.device("cpu")
        param = torch.nn.Parameter(torch.randn(1))
        mock_model.parameters.return_value = [param]
        mock_model_class.return_value = mock_model
        mock_bind.return_value = mock_model

        # Enable time budget
        self.train_config.time_budget = {
            "enabled": True,
            "max_total_time_s": 5.0,
        }

        manager = DSLFMKGCManager(self.model_config, self.train_config)
        manager._validate = MagicMock(return_value={"mrr": 0.5})
        manager._compute_binary_metrics_internal = MagicMock(return_value={"mcc": 0.5})
        manager._train_epoch = MagicMock(return_value={"loss": 0.1})
        manager._save_checkpoint = MagicMock()

        # Mock time estimator to return True (budget exceeded) on 2nd check
        manager.time_estimator.check_budget = MagicMock(side_effect=[False, True])

        # Run training
        stats = manager.train(self.train_triples, self.valid_triples)

        # Assertions
        # Epoch 0 (first check), then Epoch 1 (returns True) -> STOP
        self.assertEqual(stats["epochs_trained"], 2)
        self.assertEqual(manager.time_estimator.check_budget.call_count, 2)
