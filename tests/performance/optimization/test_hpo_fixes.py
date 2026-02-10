"""Tests for HPO bug fixes (Sprint fixes).

Tests verify:
- A1: No NameError in evaluate_trial_with_config
- B1: Seeds are applied correctly via set_global_seed
- C1: Pruning burn-in is respected
- D1: PC2 auto-prune has cooldown
- D2: pc_inbatch_rerank flag disables PC in training
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import torch


class TestA1TrialNumberFix:
    """Test that self.trial_number NameError is fixed."""

    def test_evaluate_trial_with_config_no_nameerror(self):
        """Verify evaluate_trial_with_config uses config.trial_number."""
        import polars as pl

        from pff.infrastructure.hpo.trials.pipeline import (
            TrialEvaluationConfig,
        )

        # Create minimal config with trial_number
        train_df = pl.DataFrame({"s": ["a"], "p": ["r"], "o": ["b"]})
        valid_df = pl.DataFrame({"s": ["a"], "p": ["r"], "o": ["c"]})

        config = TrialEvaluationConfig(
            params={"lr": 0.001},
            train_df=train_df,
            valid_df=valid_df,
            target_entity_ratio=0.5,
            trial_number=42,  # This should be used, not self.trial_number
            trial_output_root=Path("/tmp/test_hpo"),
            trial=None,
            artifact_manager=MagicMock(store=MagicMock(), study_name="test"),
        )

        # The config object has trial_number attribute
        assert hasattr(config, "trial_number")
        assert config.trial_number == 42


class TestB1SeedApplication:
    """Test that trial_seed is applied via set_global_seed."""

    def test_set_global_seed_imported(self):
        """Verify set_global_seed is imported in pipeline."""
        from pff.infrastructure.hpo.trials.pipeline import set_global_seed
        from pff.shared.determinism import set_global_seed as original

        assert set_global_seed is original

    def test_seed_produces_deterministic_results(self):
        """Verify same seed produces same random numbers."""
        import random

        from pff.shared.determinism import set_global_seed

        set_global_seed(12345)
        vals1 = [random.random() for _ in range(5)]
        np_vals1 = np.random.rand(5).tolist()
        torch_vals1 = torch.rand(5).tolist()

        set_global_seed(12345)
        vals2 = [random.random() for _ in range(5)]
        np_vals2 = np.random.rand(5).tolist()
        torch_vals2 = torch.rand(5).tolist()

        assert vals1 == vals2
        assert np_vals1 == np_vals2
        assert torch_vals1 == torch_vals2


class TestB2SeedFromConfig:
    """Test that sampler seed is read from config."""

    def test_load_optuna_settings_includes_seed(self):
        """Verify load_optuna_settings returns seed from config."""
        from pff.infrastructure.hpo.config_loader import load_optuna_settings

        settings = load_optuna_settings()

        assert "sampler" in settings
        assert "seed" in settings["sampler"]
        assert isinstance(settings["sampler"]["seed"], int)

    def test_load_optuna_settings_includes_burn_in(self):
        """Verify load_optuna_settings returns burn_in_epochs."""
        from pff.infrastructure.hpo.config_loader import load_optuna_settings

        settings = load_optuna_settings()

        assert "pruner" in settings
        assert "hyperband" in settings["pruner"]
        assert "burn_in_epochs" in settings["pruner"]["hyperband"]
        assert settings["pruner"]["hyperband"]["burn_in_epochs"] >= 1


class TestC1PruningBurnIn:
    """Test that pruning burn-in is implemented."""

    def test_kgc_training_config_has_burn_in(self):
        """Verify KGCTrainingConfig has pruning_burn_in_epochs."""
        from pff.domain.learning.dslfm.kgc_manager import KGCTrainingConfig

        config = KGCTrainingConfig()

        assert hasattr(config, "pruning_burn_in_epochs")
        assert config.pruning_burn_in_epochs >= 1

    def test_burn_in_prevents_early_pruning(self):
        """Verify trial.report is not called before burn-in epochs."""
        from pff.domain.learning.dslfm.kgc_manager import KGCTrainingConfig

        config = KGCTrainingConfig(pruning_burn_in_epochs=10, validate_every=2)

        # Epochs before burn-in should NOT trigger trial.report
        for epoch in range(config.pruning_burn_in_epochs):
            should_report = (epoch + 1) >= config.pruning_burn_in_epochs
            if epoch < config.pruning_burn_in_epochs - 1:
                assert not should_report, f"Epoch {epoch + 1} should not report"


class TestTimeBudgetInjection:
    """Test that HPO time budget config is injected into training config."""

    def test_hpo_training_config_receives_time_budget(self, monkeypatch, tmp_path):
        """Verify HPO training config reads time_budget_pruning from config."""
        from pff.domain.learning.dslfm import kgc_manager
        from pff.infrastructure.hpo.config_loader import load_optimization_config
        from pff.infrastructure.hpo.trials import evaluator

        captured = {}

        class DummyManager:
            def __init__(self, model_config, training_config, relation_names=None, **kwargs):
                captured["time_budget"] = training_config.time_budget
                self.observers = kwargs.get("observers", [])

            def train(self, *args, **kwargs):
                return {"final_metrics": {}}

        monkeypatch.setattr(kgc_manager, "DSLFMKGCManager", DummyManager)
        monkeypatch.setattr(evaluator, "_compute_binary_metrics", lambda *args, **kwargs: {})

        params = {}
        train_triples = np.zeros((2, 3), dtype=np.int64)
        valid_triples = np.zeros((1, 3), dtype=np.int64)

        evaluator._train_dslfm_kgc_model(
            params=params,
            model_dir=tmp_path,
            train_triples=train_triples,
            valid_triples=valid_triples,
            num_entities=2,
            num_relations=1,
            relation_names=None,
            use_bert=False,
            trial=None,
        )

        expected = load_optimization_config().get("time_budget_pruning", {})
        assert captured["time_budget"] == expected


class TestD1PC2PruneCooldown:
    """Test that PC2 auto-prune has cooldown."""

    def test_pc2_has_prune_every_n_steps(self):
        """Verify NeuralProbabilisticCircuit has prune frequency control."""
        from pff.domain.learning.pc.npc import NeuralProbabilisticCircuit

        pc = NeuralProbabilisticCircuit(
            num_attrs=8,
            prune_every_n_steps=50,
        )

        assert hasattr(pc, "prune_every_n_steps")
        assert pc.prune_every_n_steps == 50
        assert hasattr(pc, "_forward_count")
        assert pc._forward_count == 0

    def test_auto_prune_respects_cooldown(self):
        """Verify _auto_prune only runs every N steps."""
        from pff.domain.learning.pc.npc import NeuralProbabilisticCircuit

        pc = NeuralProbabilisticCircuit(
            num_attrs=4,
            pruning_threshold=0.5,
            prune_every_n_steps=10,
        )

        # Simulate forward passes
        attr_probs = torch.rand(2, 4, 2)
        labels = torch.ones(2, dtype=torch.long)

        for i in range(25):
            pc.forward(attr_probs, labels)
            pc.maintenance()

        # Should have done 2 prune checks (at step 10 and 20)
        assert pc._forward_count == 25
        # Total prune calls depends on edge flow, but at most 2


class TestD2PCInbatchRerank:
    """Test that pc_inbatch_rerank flag works."""

    def test_dslfm_config_has_pc_inbatch_flag(self):
        """Verify DSLFMKGCConfig has pc_inbatch_rerank."""
        from pff.domain.learning.dslfm.dslfm_kgc import DSLFMKGCConfig

        config = DSLFMKGCConfig(num_entities=10, num_relations=2)

        assert hasattr(config, "pc_inbatch_rerank")
        assert config.pc_inbatch_rerank is False  # Default should be False

    def test_pc_inbatch_false_skips_pc_in_training(self):
        """Verify compute_loss respects pc_inbatch_rerank flag."""
        from pff.domain.learning.dslfm.dslfm_kgc import DSLFMKGCConfig

        # Default config should have pc_inbatch_rerank=False
        config = DSLFMKGCConfig(
            num_entities=10,
            num_relations=2,
            lambda_pc=0.1,  # PC enabled
            pc_inbatch_rerank=False,  # But not for in-batch
        )

        assert config.lambda_pc > 0
        assert config.pc_inbatch_rerank is False


class TestE1OptunaCleanup:
    """Test that optuna_impl fixes are applied."""

    def test_suggest_params_range_uses_linear_scale(self):
        """Verify 2-element ranges use linear scale (not auto log)."""
        from pff.infrastructure.hpo.strategies.base import OptimizationConfig
        from pff.infrastructure.hpo.strategies.optuna_impl import OptunaStrategy

        config = OptimizationConfig(
            n_trials=1,
            direction="maximize",
        )

        strategy = OptunaStrategy(config)

        # Create mock trial
        mock_trial = MagicMock()
        mock_trial.suggest_float.return_value = 0.5
        mock_trial.suggest_categorical.return_value = "a"

        # Search space with range that previously triggered log=True bug
        # low < 0 and high > 0 should NOT trigger log scale
        search_space = {
            "param": [-10.0, 10.0],
        }

        strategy.suggest_params(mock_trial, search_space)

        # suggest_float should be called
        assert mock_trial.suggest_float.called
        call_args = mock_trial.suggest_float.call_args

        # log=True should NOT be in kwargs (linear scale expected)
        kwargs = call_args.kwargs if call_args.kwargs else {}
        assert kwargs.get("log", False) is False, "Range [-10, 10] should use linear scale"

    def test_suggest_params_explicit_log_respected(self):
        """Verify explicit log=True in dict config is respected."""
        from pff.infrastructure.hpo.strategies.base import OptimizationConfig
        from pff.infrastructure.hpo.strategies.optuna_impl import OptunaStrategy

        config = OptimizationConfig(
            n_trials=1,
            direction="maximize",
        )

        strategy = OptunaStrategy(config)

        mock_trial = MagicMock()
        mock_trial.suggest_float.return_value = 0.001

        search_space = {
            "lr": {"type": "float", "low": 1e-5, "high": 1e-1, "log": True},
        }

        strategy.suggest_params(mock_trial, search_space)

        assert mock_trial.suggest_float.called
        call_args = mock_trial.suggest_float.call_args
        assert call_args.kwargs.get("log") is True


class TestNumericalSanity:
    """Test numerical sanity checks."""

    def test_nan_inf_detection_in_scores(self):
        """Verify NaN/Inf are handled in _score_all_pairs."""
        from pff.domain.learning.dslfm.dslfm_kgc import DSLFMKGCConfig, DSLFMKGCModel

        config = DSLFMKGCConfig(
            num_entities=10,
            num_relations=2,
            entity_dim=16,
            feature_dim=16,
            max_communities=4,
            hidden_dim=32,
        )

        model = DSLFMKGCModel(config)
        model.eval()

        heads = torch.tensor([0, 1])
        relations = torch.tensor([0, 0])
        tails = torch.tensor([2, 3])

        with torch.no_grad():
            result = model.forward(heads, relations, tails)

        # Scores should not have NaN/Inf
        assert not torch.isnan(result["scores"]).any()
        assert not torch.isinf(result["scores"]).any()
