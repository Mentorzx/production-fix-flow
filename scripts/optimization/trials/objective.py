from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
from sklearn.model_selection import train_test_split

from pff import settings
from pff.utils import logger
from pff.utils.core.file_manager import FileManager
from pff.utils.hash import stable_hash
from pff.validators.ensembles.advanced_trainer import AdvancedEnsembleTrainer, SymbolicBalanceError
from pff.validators.ensembles.data_loader import EnsembleDataLoader
from pff.validators.ensembles.ensemble_wrappers.transformers import SymbolicCoverageError
from pff.validators.ensembles.hierarchical import load_hierarchical_config
from pff.validators.kg.anyburl import AnyBURLLearner
from pff.validators.kg.config import KGConfig
from pff.validators.kg.rule_filter import AnyBURLRuleFilter, RuleFilterConfig
from pff.validators.rotate.lightgbm_trainer import RotatELightGBMTrainer

from .artifacts import TrialArtifactManager
from .bounds import get_range, load_metric_bounds, get_rule_component_weights, normalize_metric, blend_scores
from .config_loader import load_ensemble_hpo_bounds
from .pipeline import evaluate_trial
from .constants import KGE_MODEL_ROTATE


# ============================================================================
# Builder Pattern: SymbolicRetryParamsBuilder
# ============================================================================

@dataclass
class SymbolicRetryParamsBuilder:
    """
    Builder Pattern for constructing symbolic retry parameters.
    
    Provides a fluent API for building fallback parameters when
    symbolic coverage is insufficient.
    
    Design Pattern: Builder
    - Separates construction of complex params from representation
    - Allows step-by-step configuration
    - Provides fluent interface for readability
    
    Example:
        params = (
            SymbolicRetryParamsBuilder(current_params)
            .with_lower_feature_threshold(0.5)
            .with_higher_symbolic_ratio()
            .with_normalized_weights()
            .build()
        )
    """
    
    base_params: dict[str, Any]
    _params: dict[str, Any] = field(default_factory=dict, init=False)
    
    def __post_init__(self) -> None:
        """Initialize working params from base."""
        self._params = dict(self.base_params) if self.base_params else {}
    
    def with_lower_feature_threshold(self, factor: float = 0.5) -> "SymbolicRetryParamsBuilder":
        """
        Reduce feature selection threshold to allow more symbolic features.
        
        Args:
            factor: Multiplier for threshold reduction (default 0.5)
            
        Returns:
            Self for chaining
        """
        current = self._params.get("feature_selection_threshold", 0.15)
        self._params["feature_selection_threshold"] = float(
            np.clip(current * factor, 0.03, 0.2)
        )
        return self
    
    def with_higher_symbolic_ratio(self, min_ratio: float = 0.3, max_ratio: float = 0.42) -> "SymbolicRetryParamsBuilder":
        """
        Adjust target symbolic ratio to higher range.
        
        Args:
            min_ratio: Minimum symbolic ratio
            max_ratio: Maximum symbolic ratio
            
        Returns:
            Self for chaining
        """
        current = self._params.get("target_symbolic_ratio", 0.38)
        self._params["target_symbolic_ratio"] = float(
            np.clip(current, min_ratio, max_ratio)
        )
        return self
    
    def with_lower_rules_threshold(self, factor: float = 0.8) -> "SymbolicRetryParamsBuilder":
        """
        Reduce rules threshold to accept more rules.
        
        Args:
            factor: Multiplier for threshold reduction
            
        Returns:
            Self for chaining
        """
        current = self._params.get("rules_threshold", 0.3)
        self._params["rules_threshold"] = float(
            np.clip(current * factor, 0.15, 0.45)
        )
        return self
    
    def with_adjusted_weights(
        self,
        rules_range: tuple[float, float] = (0.12, 0.25),
        lightgbm_range: tuple[float, float] = (0.5, 0.65),
        neural_range: tuple[float, float] = (0.15, 0.45),
    ) -> "SymbolicRetryParamsBuilder":
        """
        Adjust component weights with clipping.
        
        Args:
            rules_range: Min/max for rules weight
            lightgbm_range: Min/max for lightgbm weight
            neural_range: Min/max for neural weight
            
        Returns:
            Self for chaining
        """
        self._params["rules_weight"] = float(
            np.clip(self._params.get("rules_weight", 0.18), *rules_range)
        )
        self._params["lightgbm_weight"] = float(
            np.clip(self._params.get("lightgbm_weight", 0.55), *lightgbm_range)
        )
        # Neural weight is derived
        neural = 1.0 - self._params["rules_weight"] - self._params["lightgbm_weight"]
        self._params["neural_weight"] = float(np.clip(neural, *neural_range))
        return self
    
    def with_normalized_weights(self) -> "SymbolicRetryParamsBuilder":
        """
        Normalize weights to sum to 1.0.
        
        Returns:
            Self for chaining
        """
        neural = self._params.get("neural_weight", 0.3)
        rules = self._params.get("rules_weight", 0.18)
        lightgbm = self._params.get("lightgbm_weight", 0.55)
        
        total = neural + rules + lightgbm
        if total > 0:
            scale = 1.0 / total
            self._params["neural_weight"] = neural * scale
            self._params["rules_weight"] = rules * scale
            self._params["lightgbm_weight"] = lightgbm * scale
        
        return self
    
    def build(self) -> dict[str, Any] | None:
        """
        Build and return the final parameters.
        
        Returns:
            Constructed parameters or None if invalid
        """
        total = (
            self._params.get("neural_weight", 0)
            + self._params.get("rules_weight", 0)
            + self._params.get("lightgbm_weight", 0)
        )
        if total <= 0:
            return None
        return self._params


def _derive_symbolic_retry_params(current_params: dict[str, Any]) -> dict[str, Any] | None:
    """
    Generate fallback params biased toward higher symbolic coverage.
    
    Uses SymbolicRetryParamsBuilder for clean construction.
    
    Args:
        current_params: Current trial parameters
        
    Returns:
        Adjusted parameters or None if invalid
    """
    if not current_params:
        return None

    return (
        SymbolicRetryParamsBuilder(current_params)
        .with_lower_feature_threshold(0.5)
        .with_higher_symbolic_ratio()
        .with_lower_rules_threshold(0.8)
        .with_adjusted_weights()
        .with_normalized_weights()
        .build()
    )


def maybe_enqueue_symbolic_retry(
    source_trial,
    failed_params: dict[str, Any],
    *,
    reason: str,
    symbolic_retry_state: dict[str, int],
    max_symbolic_retry_enqueues: int,
) -> None:
    """Enqueue a symbolic retry with adjusted parameters."""
    fallback_params = _derive_symbolic_retry_params(failed_params)
    if not fallback_params:
        return
    if symbolic_retry_state["enqueues"] >= max_symbolic_retry_enqueues:
        logger.warning("Symbolic re-enqueue limit reached; proceeding without adjustments")
        return
    study = getattr(source_trial, "study", None)
    if study is None:
        logger.warning("Cannot enqueue symbolic retry because trial study is missing")
        return
    try:
        study.enqueue_trial(fallback_params, skip_if_exists=True)
        symbolic_retry_state["enqueues"] += 1
        logger.info(
            " Reenfileirando tentativa simbólica com parâmetros ajustados "
            f"(motivo: {reason}) (retry {symbolic_retry_state['enqueues']}/{max_symbolic_retry_enqueues})"
        )
    except Exception as enqueue_exc:  # noqa: BLE001
        logger.warning(f"Failed to enqueue symbolic retry: {enqueue_exc}")


def kg_objective(
    trial,
    *,
    train_df: pl.DataFrame,
    valid_df: pl.DataFrame,
    target_entity_ratio: float,
    trial_runs_dir: Path,
    rule_filter: AnyBURLRuleFilter | None,
    hpo_ranges: dict[str, dict[str, int | float]],
    file_manager: FileManager,
    artifact_manager: TrialArtifactManager,
    symbolic_retry_state: dict[str, int],
    max_symbolic_retry_enqueues: int,
) -> float:
    """Optuna objective optimized for the PFF KG ensemble (delegates to trial pipeline)."""
    if trial_runs_dir is None:
        raise RuntimeError("Trial output directory not initialized")

    ensemble_hpo_bounds = load_ensemble_hpo_bounds(file_manager)
    cyclic_range = hpo_ranges.get("max_length_cyclic", {"low": 3, "high": 4})
    acyclic_range = hpo_ranges.get("max_length_acyclic", {"low": 3, "high": 5})
    conf_quantile_range = hpo_ranges.get("confidence_quantile", {"low": 0.5, "high": 0.9})
    support_quantile_range = hpo_ranges.get("support_quantile", {"low": 0.3, "high": 0.8})
    target_ratio_range = hpo_ranges.get("target_ratio", {"low": 0.2, "high": 0.5})
    nw_low, nw_high = get_range(ensemble_hpo_bounds, ["weights", "neural_weight"], 0.35, 0.55)
    rw_low, rw_high = get_range(ensemble_hpo_bounds, ["weights", "rules_weight"], 0.1, 0.22)
    lw_low, lw_high = get_range(ensemble_hpo_bounds, ["weights", "lightgbm_weight"], 0.4, 0.65)
    nt_low, nt_high = get_range(ensemble_hpo_bounds, ["thresholds", "neural_threshold"], 0.3, 0.7)
    rt_low, rt_high = get_range(ensemble_hpo_bounds, ["thresholds", "rules_threshold"], 0.3, 0.7)
    lt_low, lt_high = get_range(ensemble_hpo_bounds, ["thresholds", "lightgbm_threshold"], 0.3, 0.7)
    tsr_low, tsr_high = get_range(ensemble_hpo_bounds, ["target_symbolic_ratio"], 0.3, 0.42)
    fst_low, fst_high = get_range(ensemble_hpo_bounds, ["feature_selection_threshold"], 0.35, 0.6)
    kge_bounds = ensemble_hpo_bounds.get("kge", {})
    neg_low, neg_high = get_range(kge_bounds, ["negative_ratio"], 0.4, 0.8)
    batch_low, batch_high = get_range(kge_bounds, ["batch_size"], 256, 640)

    raw_embedding_choices = kge_bounds.get("embedding_dim", {}).get("choices", [128])
    embedding_choices = [int(choice) for choice in raw_embedding_choices] if raw_embedding_choices else [128]

    raw_self_adv_choices = kge_bounds.get("self_adversarial", {}).get("choices", [False])
    self_adv_choices = [bool(choice) for choice in raw_self_adv_choices] if raw_self_adv_choices else [False]

    # P6: Load hierarchical config to conditionally suggest hierarchical routing params
    hierarchical_config = load_hierarchical_config()
    hierarchical_bounds = ensemble_hpo_bounds.get("hierarchical", {})

    params = {
        "neural_weight": trial.suggest_float("neural_weight", float(nw_low), float(nw_high)),
        "rules_weight": trial.suggest_float("rules_weight", float(rw_low), float(rw_high)),
        "lightgbm_weight": trial.suggest_float("lightgbm_weight", float(lw_low), float(lw_high)),
        "rule_confidence": trial.suggest_float("rule_confidence", 0.5, 0.95),
        "rule_support": trial.suggest_int("rule_support", 5, 50),
        "max_rule_length": trial.suggest_int("max_rule_length", 2, 5),
        "confidence_quantile": trial.suggest_float(
            "confidence_quantile", float(conf_quantile_range.get("low", 0.5)), float(conf_quantile_range.get("high", 0.9))
        ),
        "support_quantile": trial.suggest_float(
            "support_quantile", float(support_quantile_range.get("low", 0.3)), float(support_quantile_range.get("high", 0.8))
        ),
        "target_ratio": trial.suggest_float(
            "target_ratio", float(target_ratio_range.get("low", 0.2)), float(target_ratio_range.get("high", 0.5))
        ),
        "max_length_cyclic": trial.suggest_int("max_length_cyclic", int(cyclic_range.get("low", 3)), int(cyclic_range.get("high", 4))),
        "max_length_acyclic": trial.suggest_int("max_length_acyclic", int(acyclic_range.get("low", 3)), int(acyclic_range.get("high", 5))),
        "meta_learning_rate": trial.suggest_float("meta_learning_rate", 1e-4, 1e-1, log=True),
        "meta_n_estimators": trial.suggest_int("meta_n_estimators", 50, 300),
        "negative_ratio": trial.suggest_float("negative_ratio", float(neg_low), float(neg_high)),
        "target_symbolic_ratio": trial.suggest_float("target_symbolic_ratio", float(tsr_low), float(tsr_high)),
        "neural_threshold": trial.suggest_float("neural_threshold", float(nt_low), float(nt_high)),
        "rules_threshold": trial.suggest_float("rules_threshold", float(rt_low), float(rt_high)),
        "lightgbm_threshold": trial.suggest_float("lightgbm_threshold", float(lt_low), float(lt_high)),
        "ensemble_voting": trial.suggest_categorical("ensemble_voting", ["soft", "hard"]),
        "feature_selection_threshold": trial.suggest_float("feature_selection_threshold", float(fst_low), float(fst_high)),
        "kge_model": KGE_MODEL_ROTATE,
        "embedding_dim": trial.suggest_categorical("embedding_dim", embedding_choices),
        "gamma": trial.suggest_float("gamma", 9.0, 18.0),
        "epsilon": trial.suggest_float("epsilon", 1.5, 2.5),
        # Training epochs bounded; early stopping will cap sooner if convergence happens
        "rotate_epochs": trial.suggest_int("rotate_epochs", 60, 80),
        "batch_size": trial.suggest_int("batch_size", int(batch_low), int(batch_high)),
        "negative_sample_size": trial.suggest_int("negative_sample_size", 64, 200),
        "adversarial_temperature": trial.suggest_float("adversarial_temperature", 0.5, 2.0),
        "self_adversarial": trial.suggest_categorical("self_adversarial", self_adv_choices),
        "regularization_weight": trial.suggest_float("regularization_weight", 1e-5, 1e-3, log=True),
    }

    # P6: Add hierarchical routing params when hierarchical mode is active
    if hierarchical_config.is_hierarchical:
        sht_low, sht_high = get_range(hierarchical_bounds, ["symbolic_high_threshold"], 0.60, 0.85)
        slt_low, slt_high = get_range(hierarchical_bounds, ["symbolic_low_threshold"], 0.20, 0.40)
        nft_low, nft_high = get_range(hierarchical_bounds, ["neural_fallback_threshold"], 0.40, 0.70)
        bws_low, bws_high = get_range(hierarchical_bounds, ["blend_weight_symbolic"], 0.40, 0.70)
        
        params["hierarchical_symbolic_high_threshold"] = trial.suggest_float(
            "hierarchical_symbolic_high_threshold", float(sht_low), float(sht_high)
        )
        params["hierarchical_symbolic_low_threshold"] = trial.suggest_float(
            "hierarchical_symbolic_low_threshold", float(slt_low), float(slt_high)
        )
        params["hierarchical_neural_fallback_threshold"] = trial.suggest_float(
            "hierarchical_neural_fallback_threshold", float(nft_low), float(nft_high)
        )
        params["hierarchical_blend_weight_symbolic"] = trial.suggest_float(
            "hierarchical_blend_weight_symbolic", float(bws_low), float(bws_high)
        )

    score = evaluate_trial(
        params=params,
        train_df=train_df,
        valid_df=valid_df,
        target_entity_ratio=target_entity_ratio,
        trial_number=trial.number,
        trial_output_root=trial_runs_dir,
        rule_filter=rule_filter,
        trial=trial,
        artifact_manager=artifact_manager,
    )
    return score
