"""Regression tests for objective sampling bounds."""

from __future__ import annotations

from optuna.distributions import IntDistribution
from optuna.trial import FixedTrial

from pff.infrastructure.hpo.trials.objective import (
    _suggest_dslfm_params,
    collect_dslfm_distributions,
)


def test_objective_respects_adaptive_epoch_and_patience_bounds() -> None:
    """Epoch/patience distributions must not be force-expanded by hardcoded floors."""
    hpo_ranges = {
        "kge": {
            "embedding_dim": {"choices": [128, 256]},
            "max_communities": {"choices": [64, 128]},
            "ibp_alpha": {"low": 1.0, "high": 5.0},
            "batch_size": {"low": 256, "high": 512},
            "negative_sample_size": {"low": 256, "high": 512},
            "adversarial_temperature": {"low": 0.5, "high": 3.0},
            "learning_rate": {"low": 1e-4, "high": 1e-3},
            "self_adversarial": {"choices": [False, True]},
            "use_bert_default": False,
        },
        "training": {
            "epochs": {"low": 90, "high": 130},
            "use_compile": False,
        },
        "logic": {
            "lambda_logic": {"low": 0.0, "high": 0.05},
            "t_norm": {"choices": ["product", "godel"]},
            "attr_hidden_dim": {"choices": [256, 512]},
        },
        "pc": {
            "lambda_pc": {"low": 0.0, "high": 0.05},
            "pruning_threshold": {"low": 1e-3, "high": 0.1},
            "rebuild_every": {"low": 0, "high": 30},
            "max_circuit_depth": {"choices": [2, 3, 4]},
        },
        "regularization": {"lambda_sum_cap": 0.1},
        "contrastive": {
            "temperature_low": 0.02,
            "temperature_high": 0.08,
            "num_global_negatives_low": 32,
            "num_global_negatives_high": 96,
        },
        "architecture": {"kl_weight_low": 1e-4, "kl_weight_high": 1e-2},
    }
    adaptive_bounds = {
        "epochs": (100, 120),
        "early_stopping_patience": (6, 10),
        "batch_size": (256, 512),
        "validate_every": (3, 5),
        "min_delta": (1e-4, 2e-3),
    }

    distributions = collect_dslfm_distributions(
        hpo_ranges,
        num_train=13_796,
        num_valid=1_148,
        num_entities=3_317,
        num_relations=46,
        adaptive_bounds=adaptive_bounds,
    )

    epochs_dist = distributions.get("dslfm_epochs")
    patience_dist = distributions.get("early_stopping_patience")
    assert isinstance(epochs_dist, IntDistribution)
    assert isinstance(patience_dist, IntDistribution)
    assert epochs_dist.low >= 90
    assert epochs_dist.high <= 130
    assert patience_dist.low >= 5
    assert patience_dist.high <= 10


def test_negative_sample_size_aligns_step_range() -> None:
    """Negative sample bounds should align to the step size to avoid warnings."""
    hpo_ranges = {
        "kge": {
            "embedding_dim": {"choices": [128, 256]},
            "max_communities": {"choices": [64, 128]},
            "ibp_alpha": {"low": 1.0, "high": 5.0},
            "batch_size": {"low": 256, "high": 512},
            "negative_sample_size": {"low": 384, "high": 672},
            "adversarial_temperature": {"low": 0.5, "high": 3.0},
            "learning_rate": {"low": 1e-4, "high": 1e-3},
            "self_adversarial": {"choices": [False, True]},
            "use_bert_default": False,
        },
        "training": {
            "epochs": {"low": 90, "high": 130},
            "use_compile": False,
        },
        "logic": {
            "lambda_logic": {"low": 0.0, "high": 0.05},
            "t_norm": {"choices": ["product", "godel"]},
            "attr_hidden_dim": {"choices": [256, 512]},
        },
        "pc": {
            "lambda_pc": {"low": 0.0, "high": 0.05},
            "pruning_threshold": {"low": 1e-3, "high": 0.1},
            "rebuild_every": {"low": 0, "high": 30},
            "max_circuit_depth": {"choices": [2, 3, 4]},
        },
        "regularization": {"lambda_sum_cap": 0.1},
        "contrastive": {
            "temperature_low": 0.02,
            "temperature_high": 0.08,
            "num_global_negatives_low": 32,
            "num_global_negatives_high": 96,
        },
        "architecture": {"kl_weight_low": 1e-4, "kl_weight_high": 1e-2},
    }
    adaptive_bounds = {
        "epochs": (100, 120),
        "early_stopping_patience": (6, 10),
        "batch_size": (256, 512),
        "validate_every": (3, 5),
        "min_delta": (1e-4, 2e-3),
    }

    distributions = collect_dslfm_distributions(
        hpo_ranges,
        num_train=13_796,
        num_valid=1_148,
        num_entities=3_317,
        num_relations=46,
        adaptive_bounds=adaptive_bounds,
    )

    neg_dist = distributions.get("negative_sample_size")
    assert isinstance(neg_dist, IntDistribution)
    assert neg_dist.low == 384
    assert neg_dist.high == 640


def test_objective_keeps_validation_cache_refresh_enabled_by_default() -> None:
    """Validation should refresh cached entity latents to avoid stale ranking metrics."""
    hpo_ranges = {
        "kge": {
            "embedding_dim": {"choices": [128, 256]},
            "max_communities": {"choices": [64, 128]},
            "ibp_alpha": {"low": 1.0, "high": 5.0},
            "batch_size": {"low": 256, "high": 512},
            "negative_sample_size": {"low": 256, "high": 512},
            "adversarial_temperature": {"low": 0.5, "high": 3.0},
            "learning_rate": {"low": 1e-4, "high": 1e-3},
            "self_adversarial": {"choices": [False, True]},
            "use_bert_default": False,
        },
        "training": {"epochs": {"low": 90, "high": 130}, "use_compile": False},
        "logic": {
            "lambda_logic": {"low": 0.0, "high": 0.05},
            "t_norm": {"choices": ["product", "godel"]},
            "attr_hidden_dim": {"choices": [256, 512]},
        },
        "pc": {
            "lambda_pc": {"low": 0.0, "high": 0.05},
            "pruning_threshold": {"low": 1e-3, "high": 0.1},
            "rebuild_every": {"low": 0, "high": 30},
            "max_circuit_depth": {"choices": [2, 3, 4]},
        },
        "regularization": {"lambda_sum_cap": 0.1},
        "contrastive": {
            "temperature_low": 0.02,
            "temperature_high": 0.08,
            "num_global_negatives_low": 32,
            "num_global_negatives_high": 96,
        },
        "architecture": {"kl_weight_low": 1e-4, "kl_weight_high": 1e-2},
    }
    adaptive_bounds = {
        "epochs": (100, 120),
        "early_stopping_patience": (6, 10),
        "batch_size": (256, 512),
        "validate_every": (3, 5),
        "min_delta": (1e-4, 2e-3),
    }
    trial = FixedTrial(
        {
            "embedding_dim": 128,
            "max_communities": 64,
            "ibp_alpha": 2.0,
            "dslfm_epochs": 100,
            "early_stopping_patience": 6,
            "batch_size": 256,
            "negative_sample_size": 256,
            "adversarial_temperature": 1.0,
            "self_adversarial": True,
            "learning_rate": 5e-4,
            "lambda_logic": 0.01,
            "t_norm": "product",
            "attr_hidden_dim": 256,
            "lambda_pc": 0.01,
            "pruning_threshold": 0.01,
            "rebuild_every": 10,
            "max_circuit_depth": 2,
            "min_delta": 1e-4,
            "validate_every": 4,
            "contrastive_temperature": 0.05,
            "num_global_negatives": 64,
            "kl_weight": 1e-3,
        }
    )

    params = _suggest_dslfm_params(
        trial,
        hpo_ranges,
        num_train=5000,
        num_valid=5000,
        num_entities=5312,
        num_relations=44,
        adaptive_bounds=adaptive_bounds,
    )

    assert params["refresh_cache_on_val"] is True
