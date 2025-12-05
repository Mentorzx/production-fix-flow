"""
Trial Evaluation Pipeline Module

Orchestrates the complete evaluation of HPO trials including:
- KGE model training (RotatE)
- Rule learning (AnyBURL)
- Ensemble scoring (LightGBM + XGBoost)

Design Patterns:
- Template Method: Defines the skeleton of trial evaluation algorithm
- Strategy Pattern: Different KGE models (RotatE, TransE) are strategies
- Factory Pattern: Model and evaluator creation
- Observer Pattern: Trial progress monitoring via callbacks
- Parameter Object: TrialEvaluationConfig encapsulates evaluation parameters
"""

from __future__ import annotations

import asyncio
import json
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from pff import settings
from pff.config import (
    ENSEMBLE_CONFIG_PATH,
    ENSEMBLE_HPO_CONFIG_PATH,
    KG_PIPELINE_CONFIG_PATH,
    RULE_FILTER_CONFIG_PATH,
)
from pff.utils import logger, KGEDASEvaluator
from pff.utils.hash import stable_hash
from pff.utils.metrics.calibration import compute_ece, prediction_entropy
from pff.utils.ops.global_interrupt_manager import check_interruption
from pff.validators.ensembles.advanced_trainer import AdvancedEnsembleTrainer, SymbolicBalanceError
from pff.validators.ensembles.data_loader import EnsembleDataLoader
from pff.validators.ensembles.ensemble_wrappers.transformers import SymbolicCoverageError
from pff.validators.ensembles.hierarchical import load_hierarchical_config
from pff.validators.kg.anyburl import AnyBURLLearner
from pff.validators.kg.config import KGConfig
from pff.validators.kg.rule_filter import AnyBURLRuleFilter, RuleFilterConfig
from pff.validators.rotate.lightgbm_trainer import RotatELightGBMTrainer

from .artifacts import TrialArtifactManager
from .bounds import (
    blend_scores,
    get_range,
    get_rule_component_weights,
    load_metric_bounds,
    normalize_metric,
)
from .config_loader import get_cached_config, load_ensemble_hpo_bounds, load_trial_constraints
from .embedding_cache import compute_data_hash
from .helpers import default_anyburl_metrics
from .constants import KGE_MODEL_ROTATE
from .evaluator import _train_rotate_model
from ..shared import get_file_manager


# ============================================================================
# Parameter Object Pattern: TrialEvaluationConfig
# ============================================================================

@dataclass
class TrialEvaluationConfig:
    """
    Parameter Object for trial evaluation configuration.
    
    Encapsulates all parameters needed for trial evaluation, reducing
    function signature complexity and improving maintainability.
    
    Design Pattern: Parameter Object
    - Groups related parameters into a single object
    - Reduces function parameter count from 10+ to 1
    - Provides default values and type safety via dataclass
    
    Example:
        config = TrialEvaluationConfig(
            params=trial_params,
            train_df=train_df,
            valid_df=valid_df,
            target_entity_ratio=0.3,
            trial_number=1,
            trial_output_root=Path("outputs/trials"),
        )
        score = evaluate_trial_with_config(config)
    """
    
    params: dict[str, Any]
    train_df: pl.DataFrame
    valid_df: pl.DataFrame
    target_entity_ratio: float
    trial_number: int
    trial_output_root: Path
    rule_filter: AnyBURLRuleFilter | None = None
    trial: Any | None = None
    artifact_manager: TrialArtifactManager | None = field(default=None)
    
    def __post_init__(self) -> None:
        """Initialize artifact_manager if not provided."""
        if self.artifact_manager is None:
            self.artifact_manager = TrialArtifactManager()


def set_reproducible_seed(seed: int) -> None:
    """
    Set reproducible seed across all random number generators.

    Centralizes seed setup to ensure consistency and avoid duplication.
    Sets seeds for: random, numpy, torch (CPU and CUDA if available).

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def mirror_directory(source: Path, destination: Path) -> None:
    """Mirror source directory into destination using symlink when possible."""
    fm = get_file_manager()
    fm.delete_directory(destination, ignore_errors=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        destination.symlink_to(source, target_is_directory=True)
    except Exception:
        fm.copy_directory(source, destination)


def _extract_head_relation(rule: str) -> str | None:
    """Extract head relation from an AnyBURL rule string."""
    match = re.match(r"\s*([^\s(]+)\s*\(", rule)
    if match:
        return match.group(1)
    return None


def _compute_relation_metrics(
    filtered_metadata: list[dict[str, Any]],
    relation_map_path: Path,
) -> dict[str, float | None]:
    """Compute relation-level coverage metrics from filtered rules."""
    fm = get_file_manager()
    candidates = [
        relation_map_path,
        settings.OUTPUTS_DIR / "kg" / "pyclause" / "relation_map.parquet",
        settings.ROOT_DIR / "outputs" / "kg" / "pyclause" / "relation_map.parquet",
    ]

    rel_map = None
    for candidate in candidates:
        try:
            if candidate.exists():
                rel_map = fm.read(candidate)
                break
        except Exception:
            continue

    if rel_map is None:
        logger.warning(
            f"Failed to load relation map for coverage metrics (candidates tried: {candidates})"
        )
        return {"relation_coverage": None, "rules_per_relation": None}

    if hasattr(rel_map, "columns") and "label" in rel_map.columns:
        total_relations = len(rel_map["label"])
    elif hasattr(rel_map, "columns") and "relation" in rel_map.columns:
        total_relations = len(rel_map["relation"])
    else:
        total_relations = len(rel_map)

    if total_relations <= 0:
        return {"relation_coverage": None, "rules_per_relation": None}

    head_relations: set[str] = set()
    for meta in filtered_metadata:
        relation = _extract_head_relation(str(meta.get("rule", "")))
        if relation:
            head_relations.add(relation)

    coverage = len(head_relations) / float(total_relations)
    rules_per_relation = len(filtered_metadata) / float(total_relations)
    return {
        "relation_coverage": float(coverage),
        "rules_per_relation": float(rules_per_relation),
    }


class TrialEvaluationPipeline:
    """Template Method for KG ensemble trials (setup → train → evaluate → score)."""

    def __init__(
        self,
        params: dict[str, Any],
        train_df: pl.DataFrame,
        valid_df: pl.DataFrame,
        *,
        target_entity_ratio: float,
        trial_number: int,
        trial_output_root: Path,
        rule_filter: AnyBURLRuleFilter | None,
        trial: Any | None,
        artifact_manager: TrialArtifactManager,
    ) -> None:
        self.params = params
        self.train_df = train_df
        self.valid_df = valid_df
        self.target_entity_ratio = target_entity_ratio
        self.trial_number = trial_number
        self.trial_output_root = trial_output_root
        self.rule_filter = rule_filter
        self.trial = trial
        self.artifact_manager = artifact_manager

        self.file_manager = get_file_manager()  # Singleton via shared module
        constraints = load_trial_constraints(self.file_manager)
        self.symbolic_params: dict[str, Any] = {}
        self.rule_metadata_lookup: dict[str, dict[str, Any]] = {}
        self.symbolic_contribution_ratio: float | None = None
        self.hybrid_contribution_ratio: float | None = None
        self.dominance_violation_message: str | None = None

        # Compute data hash for embedding cache (SOTA: avoid redundant training)
        self.data_hash = compute_data_hash(train_df, valid_df)

        self.coverage_gate = constraints["coverage_gate"]
        self.dominance_gate = constraints["dominance_gate"]
        self.symbolic_max_rules = constraints["symbolic_max_rules"]
        self.min_symbolic_activation = constraints["min_symbolic_activation"]

        outputs_root = settings.OUTPUTS_DIR / "optimization" / "kg_ensemble"
        self.trial_output_root = outputs_root / "trials"

        self.trial_dir: Path | None = None
        self.config_dir: Path | None = None
        self.models_dir: Path | None = None
        self.kge_model_dir: Path | None = None
        self.lightgbm_model_dir: Path | None = None
        self.kge_checkpoint_path: Path | None = None
        self.rules_path: Path | None = None
        self.lightgbm_model_path: Path | None = None
        self.xgboost_model_path: Path | None = None

        self.kge_metrics: dict[str, float] = {}
        self.lightgbm_metrics: dict[str, float] = {}
        self.anyburl_metrics: dict[str, float] = {}
        self.anyburl_classifier_metrics: dict[str, Any] = {}
        self.hybrid_eval_metrics: dict[str, Any] = {}
        self.xgboost_metrics: dict[str, Any] = {}
        self.ensemble_summary_metrics: dict[str, Any] = {}
        self.base_learner_agreement: float | None = None
        self.ensemble_ece: float | None = None
        self.ensemble_entropy: float | None = None

        self.composite_score: float = 0.0
        self.base_score: float = 0.0
        self.elapsed_time: float = 0.0
        self.trial_seed: int = 0
        
        # Hierarchical ensemble routing statistics (populated when hierarchical mode is active)
        self.hierarchical_routing_stats: dict[str, float] = {}

    @staticmethod
    def _compute_neural_symbolic_synergy(
        ensemble_f1: float | None, baseline_f1: float | None
    ) -> float | None:
        """Compute synergy as the uplift of ensemble F1 over the best baseline."""
        if ensemble_f1 is None or baseline_f1 is None:
            return None
        return ensemble_f1 - baseline_f1

    def run(self) -> float:
        """Execute the full trial lifecycle."""
        check_interruption()
        start_time = time.time()
        self._setup_trial()
        check_interruption()
        self._prepare_symbolic_limits()
        check_interruption()
        trial_kg_config = self._prepare_kg_config()
        check_interruption()
        self._train_kge(trial_kg_config)
        check_interruption()
        self._train_lightgbm()
        check_interruption()
        self._train_rules(trial_kg_config)
        check_interruption()
        self._evaluate_models()
        check_interruption()
        self._compute_score()
        self.elapsed_time = time.time() - start_time
        self._record_result()
        return self.composite_score

    def _setup_trial(self) -> None:
        """Setup deterministic seeds and trial directories."""
        check_interruption()
        logger.info(
            f"Visao do dataset do trial: treino={len(self.train_df):,} | validacao={len(self.valid_df):,}"
        )

        self.trial_seed = stable_hash(tuple(sorted(self.params.items())), truncate=16) & (2**32 - 1)
        # Use centralized seed helper for reproducibility
        set_reproducible_seed(self.trial_seed)

        self.trial_dir = self.trial_output_root / f"trial_{self.trial_number:04d}"
        self.file_manager.delete_directory(self.trial_dir, ignore_errors=True)
        self.trial_dir.mkdir(parents=True, exist_ok=True)

        self.config_dir = self.trial_dir / "config"
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir = self.trial_dir / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.kge_model_dir = self.models_dir / KGE_MODEL_ROTATE
        self.kge_model_dir.mkdir(parents=True, exist_ok=True)
        self.lightgbm_model_dir = self.models_dir / "lightgbm"
        self.lightgbm_model_dir.mkdir(parents=True, exist_ok=True)

    def _prepare_symbolic_limits(self) -> None:
        """Load symbolic thresholds and activation limits from config."""
        check_interruption()
        ensemble_config_path = ENSEMBLE_CONFIG_PATH
        try:
            ensemble_cfg = self.file_manager.read(ensemble_config_path) or {}
            for base_model in ensemble_cfg.get("base_models", []):
                if base_model.get("type") == "symbolic":
                    self.symbolic_params = base_model.get("params", {})
                    break
        except Exception as cfg_exc:
            logger.warning(f"Failed to load ensemble.yaml for symbolic limits: {cfg_exc}")
            self.symbolic_params = {}

        constraints = load_trial_constraints(self.file_manager)
        self.coverage_gate = float(self.symbolic_params.get("min_coverage_threshold", constraints["coverage_gate"]))
        dominance_raw = float(self.symbolic_params.get("dominance_max_ratio", constraints["dominance_gate"]))
        self.dominance_gate = float(np.clip(dominance_raw, 0.55, 1.0))

        max_symbolic_rules_cfg = self.symbolic_params.get("max_rules", constraints["symbolic_max_rules"])
        self.symbolic_max_rules = (
            int(max_symbolic_rules_cfg)
            if isinstance(max_symbolic_rules_cfg, (int, float)) and max_symbolic_rules_cfg > 0
            else None
        )
        default_activation_ratio = float(
            self.symbolic_params.get("min_activation_ratio", constraints["min_symbolic_activation"])
        )
        raw_feature_threshold = self.params.get("feature_selection_threshold")
        self.min_symbolic_activation = default_activation_ratio
        try:
            if raw_feature_threshold is not None:
                self.min_symbolic_activation = float(
                    np.clip(float(raw_feature_threshold) * 0.1, 0.005, 0.05)
                )
        except (TypeError, ValueError):
            logger.debug(f"Invalid feature_selection_threshold received: {raw_feature_threshold}")

    def _prepare_kg_config(self) -> KGConfig:
        """Build per-trial KG config rooted in outputs.

        This method mirrors the global KG data directory into the trial's output
        directory so that AnyBURL and other components can find the required
        parquet files (train.parquet, valid.parquet, test.parquet).
        """
        if self.trial_dir is None or self.config_dir is None:
            raise RuntimeError("Trial directories not initialized")

        kg_config_path = KG_PIPELINE_CONFIG_PATH
        kg_config_data = self.file_manager.read(kg_config_path)
        kg_config_data.setdefault("paths", {})
        kg_config_data["paths"]["data_dir"] = str(settings.DATA_DIR)
        kg_config_data["paths"]["output_dir"] = str(self.trial_dir / "outputs")
        graph_subdir = kg_config_data["paths"].get("graph_subdir", "models/kg")
        kg_config_data["paths"]["graph_subdir"] = graph_subdir

        # Mirror global KG data to trial output directory
        # AnyBURL expects train/valid/test.parquet in trial_dir/outputs/kg/
        global_kg_dir = settings.OUTPUTS_DIR / "kg"
        trial_kg_dir = self.trial_dir / "outputs" / graph_subdir
        if global_kg_dir.exists():
            mirror_directory(global_kg_dir, trial_kg_dir)
            logger.debug(f"KG data mirrored: {global_kg_dir} -> {trial_kg_dir}")
        else:
            # Fallback: create directory and log warning
            trial_kg_dir.mkdir(parents=True, exist_ok=True)
            logger.warning(f"Global KG directory not found at {global_kg_dir}; trial may fail")

        # Mirror pyclause directory for homogenized data files
        # AnyBURL uses train.homogenized.parquet for optimized rule learning
        global_pyclause_dir = settings.OUTPUTS_DIR / "pyclause"
        trial_pyclause_dir = self.trial_dir / "outputs" / "pyclause"
        if global_pyclause_dir.exists():
            mirror_directory(global_pyclause_dir, trial_pyclause_dir)
            logger.debug(f"PyClause data mirrored: {global_pyclause_dir} -> {trial_pyclause_dir}")

        kg_config_data.setdefault("anyburl", {})
        kg_config_data["anyburl"]["MAX_LENGTH_CYCLIC"] = self.params.get("max_length_cyclic", 3)
        kg_config_data["anyburl"]["MAX_LENGTH_ACYCLIC"] = self.params.get("max_length_acyclic", 3)

        trial_kg_config_path = self.config_dir / "kg.yaml"
        self.file_manager.save(kg_config_data, trial_kg_config_path)
        return KGConfig(trial_kg_config_path)

    def _train_kge(self, trial_kg_config: KGConfig) -> None:
        """Train RotatE model with trial hyperparameters."""
        check_interruption()
        if self.kge_model_dir is None:
            raise RuntimeError("KGE model directory not initialized")
        kge_checkpoint_dir = self.kge_model_dir / "checkpoints"
        kge_checkpoint_dir.mkdir(parents=True, exist_ok=True)

        kge_manager, kge_metrics, kge_checkpoint_path = _train_rotate_model(
            self.params,
            self.kge_model_dir,
            kge_checkpoint_dir,
            self.config_dir or self.kge_model_dir,
            self.file_manager,
            trial=self.trial,
            enable_embedding_cache=True,
            data_hash=self.data_hash,
        )
        self.kge_metrics = kge_metrics
        self.kge_checkpoint_path = kge_checkpoint_path
        self.kge_manager = kge_manager

    def _train_lightgbm(self) -> None:
        """Train LightGBM hybrid model using RotatE embeddings."""
        check_interruption()
        if self.lightgbm_model_dir is None or self.kge_model_dir is None:
            raise RuntimeError("Model directories not initialized")

        logger.info("Treinando modelo híbrido LightGBM...")
        trainer = RotatELightGBMTrainer(self.kge_manager)
        lightgbm_metrics_raw = trainer.train_hybrid_model(force_retrain=True)
        self.lightgbm_metrics = {k: float(v) for k, v in lightgbm_metrics_raw.items()}
        
        # FIX: Normalize metric keys - trainer returns val_* but some code expects without prefix
        # Create alias for backward compatibility
        if "val_auc" in self.lightgbm_metrics and "auc" not in self.lightgbm_metrics:
            self.lightgbm_metrics["auc"] = self.lightgbm_metrics["val_auc"]
        if "val_f1" in self.lightgbm_metrics and "f1" not in self.lightgbm_metrics:
            self.lightgbm_metrics["f1"] = self.lightgbm_metrics["val_f1"]
        
        global_lightgbm_path = settings.OUTPUTS_DIR / "rotate" / "lightgbm_model.bin"
        self.lightgbm_model_path = self.lightgbm_model_dir / "lightgbm_model.bin"
        
        if global_lightgbm_path.exists() and not self.lightgbm_model_path.exists():
            self.file_manager.copy_file(global_lightgbm_path, self.lightgbm_model_path)
            logger.debug(f"LightGBM model copied to trial directory: {self.lightgbm_model_path}")

    def _train_rules(self, trial_kg_config: KGConfig) -> None:
        """Train and filter AnyBURL rules."""
        check_interruption()
        logger.info("Aprendendo regras com AnyBURL...")
        anyburl_learner = AnyBURLLearner()
        asyncio.run(anyburl_learner.learn_rules(trial_kg_config))

        self.rules_path = trial_kg_config.get_rules_path()
        from .helpers import default_anyburl_metrics

        self.anyburl_metrics = default_anyburl_metrics(
            conf_threshold=float(self.params.get("rule_confidence", max(self.target_entity_ratio, 0.5))),
            support_threshold=float(self.params.get("rule_support", 5)),
        )

        if self.rules_path.exists():
            filter_instance = self.rule_filter or AnyBURLRuleFilter(RuleFilterConfig())
            try:
                filter_result = filter_instance.filter_rules(
                    rules_path=self.rules_path,
                    output_dir=self.trial_dir / "anyburl" if self.trial_dir else Path("anyburl"),
                    rule_confidence=float(self.params.get("rule_confidence", 0.5)),
                    rule_support=float(self.params.get("rule_support", 5)),
                    target_entity_ratio=self.target_entity_ratio,
                    max_rules=self.symbolic_max_rules,
                )
                self.rules_path = filter_result.filtered_rules_path
                self.anyburl_metrics = filter_result.metrics
                self.rule_metadata_lookup = filter_result.metadata_lookup
                relation_metrics = _compute_relation_metrics(
                    filter_result.filtered_metadata,
                    trial_kg_config.get_relation_map_path(),
                )
                self.anyburl_metrics.update(relation_metrics)
            except Exception as rule_exc:
                logger.warning(f"Failed to filter AnyBURL rules: {rule_exc}")
                logger.warning("Continuing with unfiltered rules; symbolic metrics may be degraded")
        else:
            logger.warning("Rules file not found after AnyBURL execution")

    def _evaluate_models(self) -> None:
        """Evaluate hybrid/ensemble models and gather metrics."""
        check_interruption()
        if self.lightgbm_model_path is None or self.kge_model_dir is None:
            logger.warning("Skipping ensemble evaluation because LightGBM model artifact is missing")
            return

        try:
            logger.info("Preparando datasets para avaliação híbrida/XGBoost...")
            loader = EnsembleDataLoader()
            X_train_samples, y_train_samples, X_test_samples, y_test_samples = loader.load_ensemble_data()

            X_train_samples = list(X_train_samples)
            X_test_samples = list(X_test_samples)
            y_train_samples = np.asarray(y_train_samples, dtype=int)
            y_test_samples = np.asarray(y_test_samples, dtype=int)

            if len(X_train_samples) == 0 or len(np.unique(y_train_samples)) < 2:
                raise ValueError("EnsembleDataLoader returned insufficient training data for evaluation")

            X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
                X_train_samples,
                y_train_samples,
                test_size=0.2,
                random_state=self.trial_seed,
                stratify=y_train_samples,
            )

            X_train_np = np.array(X_train_split, dtype=object)
            X_val_np = np.array(X_val_split, dtype=object)
            y_train_np = np.array(y_train_split)
            y_val_np = np.array(y_val_split)
            X_test_np = np.array(X_test_samples, dtype=object)
            y_test_np = np.array(y_test_samples)

            temp_outputs_dir = self.trial_dir / "runtime_outputs" if self.trial_dir else Path("runtime_outputs")
            self.file_manager.delete_directory(temp_outputs_dir, ignore_errors=True)
            temp_rotate_dir = temp_outputs_dir / "rotate"

            original_outputs_dir = settings.OUTPUTS_DIR
            orig_rotate_dir = original_outputs_dir / "rotate"
            mirror_directory(orig_rotate_dir, temp_rotate_dir)

            orig_pyclause_dir = original_outputs_dir / "pyclause"
            temp_pyclause_dir = temp_outputs_dir / "pyclause"
            mirror_directory(orig_pyclause_dir, temp_pyclause_dir)
            if orig_pyclause_dir.exists():
                logger.info(f"Mapeamentos vinculados de {orig_pyclause_dir} para {temp_pyclause_dir}")

            trial_embeddings = self.kge_model_dir / "node_embeddings.pkl"
            if trial_embeddings.exists():
                self.file_manager.copy_file(trial_embeddings, temp_rotate_dir / "node_embeddings.pkl")

            for metadata_name in ["lightgbm_metadata.pkl", "hybrid_metrics.json"]:
                src_meta = self.lightgbm_model_dir / metadata_name
                if src_meta.exists():
                    dest_meta_dir = temp_outputs_dir / "rotate"
                    dest_meta_dir.mkdir(parents=True, exist_ok=True)
                    self.file_manager.copy_file(src_meta, dest_meta_dir / metadata_name)

            settings.OUTPUTS_DIR = temp_outputs_dir
            try:
                ensemble_output_dir = self.models_dir / "ensemble" if self.models_dir else Path("ensemble")
                ensemble_trainer = AdvancedEnsembleTrainer(
                    neural_model_path=str(self.kge_model_dir),
                    rules_path=str(self.rules_path),
                    lightgbm_model_path=str(self.lightgbm_model_path),
                    output_dir=ensemble_output_dir,
                    force_symbolic_contribution=False,
                    min_symbolic_activation=self.min_symbolic_activation,
                )

                try:
                    ensemble_trainer.train(X_train_np, y_train_np, X_val_np, y_val_np)
                    ensemble_trainer.save_model()
                except SymbolicBalanceError as dominance_exc:
                    logger.warning(f"Symbolic dominance detected during training (ignored): {dominance_exc}")
                except SymbolicCoverageError:
                    raise

                trainer_balance = getattr(ensemble_trainer, "feature_balance", None)
                if trainer_balance and isinstance(trainer_balance, dict):
                    # feature_balance is a dict with 'symbolic' and 'hybrid' keys
                    self.symbolic_contribution_ratio = trainer_balance.get("symbolic", 0.0)
                    self.hybrid_contribution_ratio = trainer_balance.get("hybrid", 0.0)
                    if self.symbolic_contribution_ratio > self.dominance_gate:
                        self.dominance_violation_message = (
                            f"Symbolic contribution {self.symbolic_contribution_ratio:.3f} exceeds dominance limit {self.dominance_gate:.3f}"
                        )

                self.xgboost_model_path = ensemble_trainer.output_dir / "stacking_model_advanced.joblib"
                evaluation_result = ensemble_trainer.evaluate(X_test_np, y_test_np)

                self.anyburl_classifier_metrics = evaluation_result.get("anyburl_classifier_metrics", {})
                self.hybrid_eval_metrics = evaluation_result.get("hybrid_metrics", {})
                self.lightgbm_metrics["auc"] = evaluation_result.get("lightgbm_auc", self.lightgbm_metrics.get("auc"))
                self.xgboost_metrics = evaluation_result.get("xgboost_metrics", {})
                ensemble_metrics = evaluation_result.get("ensemble_metrics") or {}
                if not ensemble_metrics:
                    ensemble_metrics = {
                        "f1": evaluation_result.get("test_f1_score"),
                        "auc": evaluation_result.get("test_auc_roc"),
                        "accuracy": evaluation_result.get("test_accuracy"),
                        "precision": evaluation_result.get("test_precision"),
                        "recall": evaluation_result.get("test_recall"),
                    }
                self.ensemble_summary_metrics = {
                    key: value for key, value in ensemble_metrics.items() if value is not None
                }
                base_learner_agreement = None
                baseline_test_auc = None
                ensemble_preds = None
                baseline_probs = None
                ensemble_proba = None
                try:
                    # Avoid object/string features (hierarchical uses meta-learner on precomputed features)
                    if (
                        ensemble_trainer.ensemble_model is not None
                        and np.issubdtype(X_test_np.dtype, np.number)
                    ):
                        ensemble_preds = ensemble_trainer.ensemble_model.predict(X_test_np)
                        ensemble_proba = ensemble_trainer.ensemble_model.predict_proba(X_test_np)[:, 1]
                    elif ensemble_trainer.ensemble_model is not None:
                        logger.warning(
                            "Ensemble predictions skipped for agreement (non-numeric features in X_test)"
                        )
                except Exception as pred_exc:  # noqa: BLE001
                    logger.warning(f"Failed to compute ensemble predictions for agreement: {pred_exc}")

                try:
                    baseline_trainer = RotatELightGBMTrainer(self.kge_manager)
                    baseline_trainer.lightgbm_model = lgb.Booster(model_file=str(self.lightgbm_model_path))
                    baseline_probs = baseline_trainer.predict_samples(list(X_test_np))
                    if baseline_probs is not None:
                        if ensemble_preds is not None and len(baseline_probs) == len(ensemble_preds):
                            baseline_labels = (baseline_probs > 0.5).astype(int)
                            base_learner_agreement = float(np.mean(baseline_labels == ensemble_preds))
                        if len(np.unique(y_test_np)) > 1:
                            baseline_test_auc = float(roc_auc_score(y_test_np, baseline_probs))
                except Exception as base_exc:  # noqa: BLE001
                    logger.warning(f"Failed to compute base learner agreement: {base_exc}")

                try:
                    if baseline_probs is not None and len(np.unique(y_test_np)) > 1:
                        self.lightgbm_metrics["ece"] = compute_ece(baseline_probs, y_test_np)
                        self.lightgbm_metrics["entropy"] = prediction_entropy(baseline_probs, average=True)
                except Exception as calib_exc:  # noqa: BLE001
                    logger.warning(f"Failed to compute LightGBM calibration metrics: {calib_exc}")

                try:
                    if ensemble_proba is not None and len(np.unique(y_test_np)) > 1:
                        self.ensemble_ece = compute_ece(ensemble_proba, y_test_np)
                        self.ensemble_entropy = prediction_entropy(ensemble_proba, average=True)
                except Exception as ensemble_calib_exc:  # noqa: BLE001
                    logger.warning(f"Failed to compute ensemble calibration metrics: {ensemble_calib_exc}")

                if self.xgboost_metrics is not None:
                    self.xgboost_metrics["test_auc"] = self.xgboost_metrics.get("test_auc_roc", self.xgboost_metrics.get("auc_roc"))
                    if base_learner_agreement is not None:
                        self.xgboost_metrics["base_learner_agreement"] = base_learner_agreement
                    if baseline_test_auc is not None:
                        self.xgboost_metrics["lightgbm_test_auc"] = baseline_test_auc
                self.base_learner_agreement = base_learner_agreement
                self.rule_metadata_lookup.update(evaluation_result.get("rule_metadata", {}))
                contribution_metrics = evaluation_result.get("contribution_metrics", {})
                if contribution_metrics:
                    self.symbolic_contribution_ratio = contribution_metrics.get("symbolic_ratio", self.symbolic_contribution_ratio)
                    self.hybrid_contribution_ratio = contribution_metrics.get("hybrid_ratio", self.hybrid_contribution_ratio)
                    if self.symbolic_contribution_ratio is not None and self.symbolic_contribution_ratio > self.dominance_gate:
                        self.dominance_violation_message = (
                            f"Symbolic contribution {self.symbolic_contribution_ratio:.3f} exceeds dominance limit {self.dominance_gate:.3f}"
                        )
                    if self.hybrid_contribution_ratio is not None and self.hybrid_contribution_ratio < 0.10:
                        logger.warning(
                            f"Hybrid contribution is low ({self.hybrid_contribution_ratio:.2%}); neural component may be underutilized"
                        )
            finally:
                settings.OUTPUTS_DIR = original_outputs_dir
        except Exception as ensemble_exc:
            logger.warning(f"Failed to run XGBoost ensemble evaluation: {ensemble_exc}")

    def _compute_score(self) -> None:
        """Compute composite score and guardrails."""
        check_interruption()
        self.anyburl_metrics.setdefault("coverage", 0.0)
        self.anyburl_metrics.setdefault("positive_rule_coverage", 0.0)

        # Use symbolic contribution ratio as coverage proxy when available
        # This reflects actual rule activation during ensemble training
        coverage_val = float(self.anyburl_metrics.get("coverage", 0.0))
        relation_coverage = self.anyburl_metrics.get("relation_coverage")
        if relation_coverage is not None:
            coverage_val = max(coverage_val, float(relation_coverage))
        if self.symbolic_contribution_ratio is not None and self.symbolic_contribution_ratio > 0:
            # Symbolic contribution is a better proxy for coverage than rule filter metrics
            # Scale to [0, 1] range (contribution can exceed 1.0 in dominance cases)
            coverage_val = min(self.symbolic_contribution_ratio, 1.0)
            self.anyburl_metrics["coverage"] = coverage_val
            logger.debug(f"Using symbolic contribution {self.symbolic_contribution_ratio:.3f} as coverage proxy")

        if coverage_val < self.coverage_gate:
            warning_msg = f"Symbolic coverage {coverage_val:.3f} below required target {self.coverage_gate:.3f}"
            logger.warning(warning_msg)
            raise SymbolicCoverageError(warning_msg)
        if self.dominance_violation_message:
            logger.warning(self.dominance_violation_message)

        neural_w = float(self.params.get("neural_weight", 0.0))
        rules_w = float(self.params.get("rules_weight", 0.0))
        lgbm_w = float(self.params.get("lightgbm_weight", 0.0))

        safe_neural_w = max(neural_w, 0.05)
        safe_rules_w = max(rules_w, 0.05)
        safe_lgbm_w = min(max(lgbm_w, 0.05), 0.70)

        scoring_config = get_cached_config(ENSEMBLE_HPO_CONFIG_PATH, self.file_manager).get("scoring", {})
        metric_bounds = load_metric_bounds(self.file_manager)
        kge_low, kge_high = get_range(metric_bounds, ["kge", "mrr"], 0.15, 0.75)
        rules_conf_low, rules_conf_high = get_range(metric_bounds, ["rules", "confidence"], 0.4, 0.95)
        rules_rec_low, rules_rec_high = get_range(metric_bounds, ["rules", "recall"], 0.05, 0.5)
        rules_cov_low, rules_cov_high = get_range(metric_bounds, ["rules", "coverage"], 0.05, 0.5)
        lgb_auc_low, lgb_auc_high = get_range(metric_bounds, ["learner", "lgbm_auc"], 0.6, 0.99)
        pr_auc_low, pr_auc_high = get_range(metric_bounds, ["learner", "lgbm_pr_auc"], 0.5, 0.99)
        mcc_low, mcc_high = get_range(metric_bounds, ["learner", "lgbm_mcc"], 0.0, 0.9)
        gen_gap_low, gen_gap_high = get_range(metric_bounds, ["learner", "generalization_gap"], 0.0, 0.20)
        hybrid_f1_low, hybrid_f1_high = get_range(metric_bounds, ["learner", "hybrid_f1"], 0.45, 0.9)
        xgb_f1_low, xgb_f1_high = get_range(metric_bounds, ["learner", "xgb_f1"], 0.45, 0.9)

        kge_component = normalize_metric(self.kge_metrics["mrr"], low=kge_low, high=kge_high)
        rules_conf_component = normalize_metric(
            self.anyburl_metrics["avg_confidence"], low=rules_conf_low, high=rules_conf_high
        )
        rules_recall_component = normalize_metric(
            self.anyburl_classifier_metrics.get("recall", 0.0), low=rules_rec_low, high=rules_rec_high
        )
        rules_cov_component = normalize_metric(
            self.anyburl_metrics.get("coverage", 0.0), low=rules_cov_low, high=rules_cov_high
        )
        relation_cov_low, relation_cov_high = get_range(metric_bounds, ["rules", "relation_coverage"], 0.05, 0.60)
        rules_per_rel_low, rules_per_rel_high = get_range(metric_bounds, ["rules", "rules_per_relation"], 1.0, 80.0)
        relation_cov_component = normalize_metric(
            self.anyburl_metrics.get("relation_coverage", 0.0),
            low=relation_cov_low,
            high=relation_cov_high,
        )
        rules_per_rel_component = normalize_metric(
            self.anyburl_metrics.get("rules_per_relation", 0.0),
            low=rules_per_rel_low,
            high=rules_per_rel_high,
        )
        conf_weight, recall_weight, coverage_weight = get_rule_component_weights(self.file_manager)
        rules_per_rel_weight = max(0.0, min(1.0, float(scoring_config.get("rules_per_relation_weight", 0.10))))
        # Rules per relation complements coverage; rescale other weights to keep the sum at 1.0 without double penalizing activation
        scaled_conf = conf_weight * (1.0 - rules_per_rel_weight)
        scaled_recall = recall_weight * (1.0 - rules_per_rel_weight)
        scaled_coverage = coverage_weight * (1.0 - rules_per_rel_weight)
        rules_component = blend_scores(
            [
                (rules_conf_component, scaled_conf),
                (rules_recall_component, scaled_recall),
                (rules_cov_component, scaled_coverage),
                (rules_per_rel_component, rules_per_rel_weight),
            ]
        )

        # FIX: LightGBM metrics use "val_auc" key from RotatELightGBMTrainer
        lgbm_auc_raw = self.lightgbm_metrics.get("val_auc") or self.lightgbm_metrics.get("auc") or 0.0
        lgbm_auc_component = normalize_metric(lgbm_auc_raw, low=lgb_auc_low, high=lgb_auc_high)
        pr_auc_raw = float(self.lightgbm_metrics.get("pr_auc") or 0.0)
        pr_auc_component = normalize_metric(pr_auc_raw, low=pr_auc_low, high=pr_auc_high)
        mcc_raw = float(self.lightgbm_metrics.get("mcc") or 0.0)
        mcc_component = normalize_metric(mcc_raw, low=mcc_low, high=mcc_high)
        generalization_gap_raw = float(self.lightgbm_metrics.get("generalization_gap") or 0.0)
        generalization_gap_component = normalize_metric(
            generalization_gap_raw, low=gen_gap_low, high=gen_gap_high
        )
        
        # FIX: hybrid_eval_metrics from ensemble evaluation uses "f1" key
        hybrid_f1_raw = self.hybrid_eval_metrics.get("f1") or 0.0
        hybrid_f1_component = normalize_metric(hybrid_f1_raw, low=hybrid_f1_low, high=hybrid_f1_high)
        
        # XGBoost metrics from AdvancedEnsembleTrainer.evaluate() use "test_f1_score" key
        xgb_f1_raw = (self.xgboost_metrics.get("test_f1_score") if self.xgboost_metrics else None) or 0.0
        xgb_f1_component = normalize_metric(xgb_f1_raw, low=xgb_f1_low, high=xgb_f1_high)
        xgb_auc_low, xgb_auc_high = get_range(metric_bounds, ["learner", "xgb_test_auc"], 0.6, 0.99)
        xgb_auc_raw = 0.0
        if self.xgboost_metrics:
            xgb_auc_raw = (
                self.xgboost_metrics.get("test_auc")
                or self.xgboost_metrics.get("test_auc_roc")
                or self.xgboost_metrics.get("auc_roc")
                or 0.0
            )
        xgb_auc_component = normalize_metric(xgb_auc_raw, low=xgb_auc_low, high=xgb_auc_high)
        agreement_low, agreement_high = get_range(metric_bounds, ["learner", "base_learner_agreement"], 0.4, 0.95)
        base_learner_agreement_component = normalize_metric(
            self.base_learner_agreement or 0.0, low=agreement_low, high=agreement_high
        )
        lgb_ece_low, lgb_ece_high = get_range(metric_bounds, ["learner", "lightgbm_ece"], 0.0, 0.10)
        lgb_entropy_low, lgb_entropy_high = get_range(metric_bounds, ["learner", "lightgbm_entropy"], 0.0, 0.70)
        lgb_ece_component = normalize_metric(self.lightgbm_metrics.get("ece", 0.0), low=lgb_ece_low, high=lgb_ece_high)
        lgb_entropy_component = normalize_metric(
            self.lightgbm_metrics.get("entropy", 0.0), low=lgb_entropy_low, high=lgb_entropy_high
        )
        ensemble_ece_low, ensemble_ece_high = get_range(metric_bounds, ["ensemble", "ensemble_ece"], 0.0, 0.10)
        ensemble_entropy_low, ensemble_entropy_high = get_range(metric_bounds, ["ensemble", "ensemble_entropy"], 0.0, 0.70)
        ensemble_ece_component = normalize_metric(self.ensemble_ece or 0.0, low=ensemble_ece_low, high=ensemble_ece_high)
        ensemble_entropy_component = normalize_metric(
            self.ensemble_entropy or 0.0, low=ensemble_entropy_low, high=ensemble_entropy_high
        )
        
        learner_weights = scoring_config.get("learner_weights", {})
        scoring_method = scoring_config.get("scoring_method", "weighted_avg")
        # AUC mantém peso principal; PR-AUC e MCC complementam cenários desbalanceados
        learner_component = blend_scores(
            [
                (lgbm_auc_component, learner_weights.get("auc", 0.30)),
                (pr_auc_component, learner_weights.get("pr_auc", 0.25)),
                (mcc_component, learner_weights.get("mcc", 0.15)),
                (hybrid_f1_component, learner_weights.get("hybrid_f1", 0.15)),
                (xgb_f1_component, learner_weights.get("xgb_f1", 0.10)),
                (base_learner_agreement_component, learner_weights.get("agreement", 0.05)),
            ]
        )
        
        # Debug: Log normalized components to help diagnose scoring issues
        logger.debug(
            f"Learner components: lgbm_auc={lgbm_auc_raw:.4f}→{lgbm_auc_component:.4f}, "
            f"pr_auc={pr_auc_raw:.4f}→{pr_auc_component:.4f}, "
            f"mcc={mcc_raw:.4f}→{mcc_component:.4f}, "
            f"hybrid_f1={hybrid_f1_raw:.4f}→{hybrid_f1_component:.4f}, "
            f"xgb_f1={xgb_f1_raw:.4f}→{xgb_f1_component:.4f}, "
            f"xgb_auc={xgb_auc_raw:.4f}→{xgb_auc_component:.4f}, "
            f"agreement={self.base_learner_agreement or 0.0:.4f}→{base_learner_agreement_component:.4f}, "
            f"gen_gap={generalization_gap_raw:.4f}→{generalization_gap_component:.4f}, "
            f"lgb_ece={self.lightgbm_metrics.get('ece', 0.0):.4f}→{lgb_ece_component:.4f}, "
            f"lgb_entropy={self.lightgbm_metrics.get('entropy', 0.0):.4f}→{lgb_entropy_component:.4f}, "
            f"ensemble_ece={(self.ensemble_ece or 0.0):.4f}→{ensemble_ece_component:.4f}, "
            f"ensemble_entropy={(self.ensemble_entropy or 0.0):.4f}→{ensemble_entropy_component:.4f} "
            f"→ blend={learner_component:.4f}"
        )

        ensemble_f1 = None
        if self.ensemble_summary_metrics:
            ensemble_f1 = (
                self.ensemble_summary_metrics.get("f1")
                or self.ensemble_summary_metrics.get("f1_score")
            )
        if ensemble_f1 is None and self.xgboost_metrics:
            ensemble_f1 = self.xgboost_metrics.get("test_f1_score")

        baseline_f1_candidates: list[float] = []
        if "val_f1" in self.lightgbm_metrics:
            baseline_f1_candidates.append(float(self.lightgbm_metrics["val_f1"]))
        if self.hybrid_eval_metrics:
            hybrid_f1_metric = self.hybrid_eval_metrics.get("f1") or self.hybrid_eval_metrics.get("f1_score")
            if hybrid_f1_metric is not None:
                baseline_f1_candidates.append(float(hybrid_f1_metric))
        baseline_f1 = max(baseline_f1_candidates) if baseline_f1_candidates else None
        neural_symbolic_synergy = self._compute_neural_symbolic_synergy(ensemble_f1, baseline_f1)

        edas_score = None
        if scoring_method == "edas":
            edas_metrics = {
                "neural": kge_component,
                "rules": rules_component,
                "learner": learner_component,
            }
            if any(v is None for v in edas_metrics.values()):
                logger.warning("Metric missing for EDAS; falling back to weighted_avg")
            else:
                edas_weights = {
                    "neural": safe_neural_w,
                    "rules": safe_rules_w,
                    "learner": safe_lgbm_w,
                }
                edas_result = KGEDASEvaluator().compute_score(edas_metrics, edas_weights)
                edas_score = edas_result.score
                self.ensemble_summary_metrics.setdefault("edas", {})
                self.ensemble_summary_metrics["edas"] = {
                    "score": edas_result.score,
                    "positive_distance": edas_result.positive_distance,
                    "negative_distance": edas_result.negative_distance,
                    "reference": edas_result.reference,
                }

        if edas_score is None:
            self.base_score = blend_scores(
                [
                    (kge_component, safe_neural_w),
                    (rules_component, safe_rules_w),
                    (learner_component, safe_lgbm_w),
                ]
            )
        else:
            self.base_score = edas_score
        base_score_without_synergy = self.base_score

        synergy_bonus_coeff = float(scoring_config.get("synergy_bonus_coeff", 0.10))
        synergy_penalty_coeff = float(scoring_config.get("synergy_penalty_coeff", 0.05))
        synergy_max_bonus = float(scoring_config.get("synergy_max_bonus", 0.08))
        synergy_max_penalty = float(scoring_config.get("synergy_max_penalty", 0.04))
        synergy_adjustment = 0.0
        if neural_symbolic_synergy is not None:
            synergy_keys = self.ensemble_summary_metrics or {}
            if "synergy" not in synergy_keys and "neural_symbolic_synergy" not in synergy_keys:
                if neural_symbolic_synergy > 0:
                    synergy_adjustment = min(neural_symbolic_synergy * synergy_bonus_coeff, synergy_max_bonus)
                else:
                    synergy_adjustment = max(
                        neural_symbolic_synergy * synergy_penalty_coeff, -synergy_max_penalty
                    )
        self.base_score = max(0.0, self.base_score + synergy_adjustment)

        min_weight = min(neural_w, rules_w, lgbm_w)
        weight_penalty = max(0.0, 0.05 - min_weight)
        coverage_target = max(self.coverage_gate, 0.05)
        coverage_penalty = max(0.0, coverage_target - self.anyburl_metrics.get("coverage", 0.0))
        rules_weight_target = 0.25
        rules_weight_penalty = max(0.0, rules_weight_target - rules_w)
        overweight = max(0.0, lgbm_w - 0.70)

        fallback_dominance_target = float(scoring_config.get("fallback_dominance_target", 0.70))
        symbolic_dominance_penalty_coeff = float(scoring_config.get("symbolic_dominance_penalty_coeff", 0.60))
        min_neural_target = float(scoring_config.get("min_neural_contribution", 0.20))
        # Progressive penalty thresholds
        soft_threshold = float(scoring_config.get("symbolic_soft_threshold", 0.65))
        hard_threshold = float(scoring_config.get("symbolic_hard_threshold", 0.85))

        dominance_target = float(self.params.get("target_symbolic_ratio", fallback_dominance_target))
        symbolic_dominance_penalty = 0.0
        
        # P5: Check if hierarchical mode is enabled - if so, skip symbolic dominance penalty
        # In hierarchical mode, modality separation is handled by the architecture itself
        hierarchical_config = load_hierarchical_config()
        skip_symbolic_penalty = hierarchical_config.is_hierarchical and not hierarchical_config.should_apply_symbolic_dominance_penalty
        
        # P6: Collect hierarchical routing statistics for MLflow metrics
        if hierarchical_config.is_hierarchical:
            self.hierarchical_routing_stats = {
                "architecture_type": 1.0,  # 1.0 = hierarchical, 0.0 = flat
                "symbolic_high_threshold": hierarchical_config.decision_router.symbolic_confidence_threshold,
                "symbolic_low_threshold": hierarchical_config.decision_router.symbolic_low_threshold,
                "neural_fallback_threshold": hierarchical_config.decision_router.neural_confidence_threshold,
            }
        else:
            self.hierarchical_routing_stats = {"architecture_type": 0.0}
        
        if skip_symbolic_penalty:
            logger.debug("Symbolic dominance penalty disabled (hierarchical mode active)")
        elif self.symbolic_contribution_ratio is not None:
            sym_ratio = self.symbolic_contribution_ratio
            # Progressive penalty by bands:
            # - Below soft_threshold: no penalty
            # - soft_threshold to hard_threshold: ramp with partial coeff
            # - Above hard_threshold: full penalty
            if sym_ratio <= soft_threshold:
                symbolic_dominance_penalty = 0.0
            elif sym_ratio <= hard_threshold:
                # Linear ramp from 0 to 0.5 in the soft-hard band
                band_progress = (sym_ratio - soft_threshold) / (hard_threshold - soft_threshold)
                symbolic_dominance_penalty = band_progress * 0.5
            else:
                # Hard cut: full penalty above hard_threshold
                excess_above_hard = sym_ratio - hard_threshold
                symbolic_dominance_penalty = 0.5 + excess_above_hard / (1.0 - hard_threshold) * 0.5
                symbolic_dominance_penalty = min(1.0, symbolic_dominance_penalty)

        neural_contribution_penalty = 0.0
        if self.hybrid_contribution_ratio is not None:
            if self.hybrid_contribution_ratio < min_neural_target:
                neural_contribution_penalty = (min_neural_target - self.hybrid_contribution_ratio) / min_neural_target
                logger.warning(f"Low neural contribution: {self.hybrid_contribution_ratio:.2%} < {min_neural_target:.0%}")

        # Threshold derived from config bound (half of allowed gap range)
        gen_gap_penalty_coeff = float(scoring_config.get("generalization_gap_penalty_coeff", 0.15))
        gen_gap_threshold = gen_gap_high / 2.0
        gen_gap_penalty = 0.0
        if generalization_gap_raw > gen_gap_threshold:
            gap_range = max(gen_gap_high - gen_gap_threshold, 1e-6)
            gen_gap_penalty = (generalization_gap_raw - gen_gap_threshold) / gap_range
        gen_gap_penalty = max(0.0, min(1.0, gen_gap_penalty))

        self.composite_score = self.base_score
        penalty_factors = [
            ("weight_penalty", 0.40, weight_penalty),
            ("coverage_penalty", 0.45, coverage_penalty),
            ("rules_weight_penalty", 0.35, rules_weight_penalty),
            ("overweight", 0.20, overweight),
            ("symbolic_dominance", symbolic_dominance_penalty_coeff, symbolic_dominance_penalty),
            ("neural_contribution", 0.60, neural_contribution_penalty),
        ]
        if gen_gap_penalty_coeff > 0:
            penalty_factors.append(("generalization_gap", gen_gap_penalty_coeff, gen_gap_penalty))
        calibration_cfg = scoring_config.get("calibration_penalty", {})
        lgb_ece_coeff = float(calibration_cfg.get("lightgbm_ece_coeff", 0.0))
        ensemble_ece_coeff = float(calibration_cfg.get("ensemble_ece_coeff", 0.0))
        lgb_entropy_coeff = float(calibration_cfg.get("lightgbm_entropy_coeff", 0.0))
        ensemble_entropy_coeff = float(calibration_cfg.get("ensemble_entropy_coeff", 0.0))
        if lgb_ece_coeff > 0:
            penalty_factors.append(("lightgbm_ece", lgb_ece_coeff, lgb_ece_component))
        if ensemble_ece_coeff > 0:
            penalty_factors.append(("ensemble_ece", ensemble_ece_coeff, ensemble_ece_component))
        if lgb_entropy_coeff > 0:
            penalty_factors.append(("lightgbm_entropy", lgb_entropy_coeff, lgb_entropy_component))
        if ensemble_entropy_coeff > 0:
            penalty_factors.append(("ensemble_entropy", ensemble_entropy_coeff, ensemble_entropy_component))
        for name, coeff, penalty in penalty_factors:
            multiplier = 1.0 - coeff * min(1.0, penalty)
            if penalty > 0:
                logger.debug(f"Penalty '{name}': coeff={coeff:.2f}, raw={penalty:.4f}, multiplier={multiplier:.4f}")
            self.composite_score *= multiplier
        self.composite_score = max(0.0, self.composite_score)
        
        # Log score breakdown for diagnosis
        logger.debug(
            f"Score breakdown: base={base_score_without_synergy:.4f} synergy={synergy_adjustment:+.4f} "
            f"→ base_synergy={self.base_score:.4f} → composite={self.composite_score:.4f} "
            f"(symbolic_contrib={self.symbolic_contribution_ratio or 0:.2%}, "
            f"hybrid_contrib={self.hybrid_contribution_ratio or 0:.2%})"
        )

        self.ensemble_metrics = {
            "weighted_score": self.composite_score,
            "base_weighted_score": self.base_score,
            "base_weighted_score_no_synergy": base_score_without_synergy,
            "synergy_adjustment": synergy_adjustment,
            "kge_mrr": self.kge_metrics.get("mrr", 0.0),
            "kge_hits@3": self.kge_metrics.get("hits@3"),
            "kge_mean_rank": self.kge_metrics.get("mean_rank"),
            "rules_avg_confidence": self.anyburl_metrics["avg_confidence"],
            "rules_coverage": self.anyburl_metrics.get("coverage", 0.0),
            "relation_coverage": self.anyburl_metrics.get("relation_coverage"),
            "rules_per_relation": self.anyburl_metrics.get("rules_per_relation"),
            "lightgbm_auc": self.lightgbm_metrics.get("val_auc", self.lightgbm_metrics.get("auc", 0.0)),
            "lightgbm_pr_auc": self.lightgbm_metrics.get("pr_auc"),
            "lightgbm_mcc": self.lightgbm_metrics.get("mcc"),
            "lightgbm_generalization_gap": self.lightgbm_metrics.get("generalization_gap"),
            "lightgbm_train_auc": self.lightgbm_metrics.get("train_auc"),
            "lightgbm_ece": self.lightgbm_metrics.get("ece"),
            "lightgbm_entropy": self.lightgbm_metrics.get("entropy"),
            "normalized_neural": kge_component,
            "normalized_rules": rules_component,
            "normalized_learner": learner_component,
            "normalized_relation_coverage": relation_cov_component,
            "normalized_rules_per_relation": rules_per_rel_component,
            "normalized_xgb_auc": xgb_auc_component,
            "normalized_base_learner_agreement": base_learner_agreement_component,
            "normalized_lightgbm_pr_auc": pr_auc_component,
            "normalized_lightgbm_mcc": mcc_component,
            "normalized_generalization_gap": generalization_gap_component,
            "normalized_lightgbm_ece": lgb_ece_component,
            "normalized_lightgbm_entropy": lgb_entropy_component,
            "ensemble_ece": self.ensemble_ece,
            "ensemble_entropy": self.ensemble_entropy,
            "normalized_ensemble_ece": ensemble_ece_component,
            "normalized_ensemble_entropy": ensemble_entropy_component,
            "weight_penalty": weight_penalty,
            "coverage_penalty": coverage_penalty,
            "rules_weight_penalty": rules_weight_penalty,
            "symbolic_dominance_penalty": symbolic_dominance_penalty,
            "generalization_gap_penalty_ratio": gen_gap_penalty,
            "generalization_gap_threshold": gen_gap_threshold,
            "generalization_gap_penalty": gen_gap_penalty * gen_gap_penalty_coeff if gen_gap_penalty_coeff > 0 else 0.0,
            "normalized_weights": {
                "neural": neural_w,
                "rules": rules_w,
                "lightgbm": lgbm_w,
            },
            "symbolic_contribution": self.symbolic_contribution_ratio,
            "hybrid_contribution": self.hybrid_contribution_ratio,
            "neural_symbolic_synergy": neural_symbolic_synergy,
        }

        if self.ensemble_summary_metrics:
            self.ensemble_summary_metrics["weighted_score"] = self.composite_score
            self.ensemble_summary_metrics.update(
                {
                    "normalized_weighted_score": self.base_score,
                    "base_weighted_score_no_synergy": base_score_without_synergy,
                    "synergy_adjustment": synergy_adjustment,
                    "normalized_neural": kge_component,
                    "normalized_rules": rules_component,
                    "normalized_learner": learner_component,
                    "normalized_lightgbm_pr_auc": pr_auc_component,
                    "normalized_lightgbm_mcc": mcc_component,
                    "normalized_generalization_gap": generalization_gap_component,
                }
            )
        if neural_symbolic_synergy is not None:
            self.ensemble_summary_metrics.setdefault("f1", ensemble_f1)
            self.ensemble_summary_metrics["neural_symbolic_synergy"] = neural_symbolic_synergy
        if gen_gap_penalty_coeff > 0 and gen_gap_penalty > 0:
            self.ensemble_summary_metrics["generalization_gap_penalty"] = gen_gap_penalty * gen_gap_penalty_coeff
        if self.base_learner_agreement is not None:
            self.ensemble_summary_metrics["base_learner_agreement"] = self.base_learner_agreement
            self.ensemble_summary_metrics["normalized_base_learner_agreement"] = base_learner_agreement_component
        if xgb_auc_raw:
            self.ensemble_summary_metrics["xgb_test_auc"] = xgb_auc_raw
            self.ensemble_summary_metrics["normalized_xgb_auc"] = xgb_auc_component
        if self.ensemble_ece is not None:
            self.ensemble_summary_metrics["ensemble_ece"] = self.ensemble_ece
            self.ensemble_summary_metrics["normalized_ensemble_ece"] = ensemble_ece_component
        if self.ensemble_entropy is not None:
            self.ensemble_summary_metrics["ensemble_entropy"] = self.ensemble_entropy
            self.ensemble_summary_metrics["normalized_ensemble_entropy"] = ensemble_entropy_component

        logger.info("=" * 70)
        logger.info("Métricas individuais")
        logger.info("=" * 70)
        logger.info(
            f"KGE → MRR: {self.kge_metrics.get('mrr', 0.0):.4f} | "
            f"Hits@1: {self.kge_metrics.get('hits@1', 0.0):.4f} | "
            f"Hits@3: {self.kge_metrics.get('hits@3', 0.0):.4f} | "
            f"Hits@10: {self.kge_metrics.get('hits@10', 0.0):.4f} | "
            f"Mean rank: {self.kge_metrics.get('mean_rank', 0.0):.2f} | "
            f"Best val MRR: {self.kge_metrics.get('best_val_mrr', 0.0):.4f}"
        )
        anyburl_rule_count = int(round(self.anyburl_metrics.get("rule_count", 0.0)))
        relation_cov = self.anyburl_metrics.get("relation_coverage") or 0.0
        rules_per_relation = self.anyburl_metrics.get("rules_per_relation") or 0.0
        logger.info(
            f"AnyBURL → rules={anyburl_rule_count} | "
            f"avg_conf={self.anyburl_metrics.get('avg_confidence', 0.0):.4f} | "
            f"avg_support={self.anyburl_metrics.get('avg_support', 0.0):.2f} | "
            f"high_conf_ratio={self.anyburl_metrics.get('high_confidence_ratio', 0.0):.2f} | "
            f"relation_cov={relation_cov:.3f} | "
            f"rules_per_relation={rules_per_relation:.2f}"
        )

        def _format_metric_value(val: Any) -> str:
            if isinstance(val, (int, float)):
                return f"{float(val):.4f}"
            if isinstance(val, (list, tuple, set)):
                return json.dumps(list(val))
            if isinstance(val, dict):
                return json.dumps(val, ensure_ascii=False)
            return str(val)

        logger.info("Métricas do LightGBM:")
        for metric_name in ["auc", "f1", "accuracy", "precision", "recall", "pr_auc", "mcc", "generalization_gap", "train_auc"]:
            if metric_name in self.lightgbm_metrics:
                logger.info(f"  {metric_name.upper()}: {_format_metric_value(self.lightgbm_metrics[metric_name])}")
        if "ece" in self.lightgbm_metrics or "entropy" in self.lightgbm_metrics:
            logger.info("  CALIBRAÇÃO:")
            if "ece" in self.lightgbm_metrics:
                logger.info(f"    ECE: {_format_metric_value(self.lightgbm_metrics['ece'])}")
            if "entropy" in self.lightgbm_metrics:
                logger.info(f"    ENTROPIA: {_format_metric_value(self.lightgbm_metrics['entropy'])}")
        if self.hybrid_eval_metrics:
            logger.info("Métricas do híbrido (RotatE + LightGBM):")
            for metric_name, metric_value in self.hybrid_eval_metrics.items():
                logger.info(f"  {metric_name.upper()}: {_format_metric_value(metric_value)}")
        if self.anyburl_classifier_metrics:
            logger.info("Métricas do classificador AnyBURL:")
            for metric_name, metric_value in self.anyburl_classifier_metrics.items():
                logger.info(f"  {metric_name.upper()}: {_format_metric_value(metric_value)}")
        if self.xgboost_metrics:
            logger.info("Métricas do ensemble XGBoost:")
            for metric_name, metric_value in self.xgboost_metrics.items():
                logger.info(f"  {metric_name.upper()}: {_format_metric_value(metric_value)}")
        if self.ensemble_ece is not None or self.ensemble_entropy is not None:
            logger.info("Calibração do ensemble (stacking):")
            if self.ensemble_ece is not None:
                logger.info(f"  ECE: {_format_metric_value(self.ensemble_ece)}")
            if self.ensemble_entropy is not None:
                logger.info(f"  ENTROPIA: {_format_metric_value(self.ensemble_entropy)}")
        if self.ensemble_summary_metrics:
            logger.info("Resumo final do ensemble:")
            for metric_name, metric_value in self.ensemble_summary_metrics.items():
                if metric_value is None:
                    continue
                logger.info(f"  {metric_name.upper()}: {_format_metric_value(metric_value)}")
        if weight_penalty > 0:
            logger.warning(f"Ensemble weight imbalance detected (min weight {min_weight:.3f})")
        if coverage_penalty > 0:
            logger.warning(
                f"Rule coverage below target ({self.anyburl_metrics.get('coverage', 0.0):.3f} < {coverage_target:.3f})"
            )
        if symbolic_dominance_penalty > 0 and self.symbolic_contribution_ratio is not None:
            logger.warning(
                f"Symbolic dominance detected ({self.symbolic_contribution_ratio:.2%} > {dominance_target:.0%})"
            )

        logger.debug(
            f"Weights: neural={neural_w:.3f}, rules={rules_w:.3f}, "
            f"lgbm={lgbm_w:.3f}, base_norm={self.base_score:.4f}"
        )
        logger.success(
            f"Avaliacao do trial concluida: score={self.composite_score:.4f}, tempo={self.elapsed_time / 60.0:.1f}min"
        )

    def _record_result(self) -> None:
        """Persist trial result into the artifact manager."""
        check_interruption()
        model_paths: dict[str, Path] = {}
        if self.kge_checkpoint_path and self.kge_checkpoint_path.exists():
            model_paths["rotate"] = self.kge_checkpoint_path
        if self.rules_path and self.rules_path.exists():
            model_paths["anyburl"] = self.rules_path
        if self.lightgbm_model_path and self.lightgbm_model_path.exists():
            model_paths["lightgbm"] = self.lightgbm_model_path
        if self.xgboost_model_path and self.xgboost_model_path.exists():
            model_paths["xgboost"] = self.xgboost_model_path

        if self.kge_checkpoint_path is None:
            self.kge_checkpoint_path = Path("")
        if self.rules_path is None:
            self.rules_path = Path("")
        if self.lightgbm_model_path is None:
            self.lightgbm_model_path = Path("")

        model_metrics: dict[str, Any] = {
            "rotate": self.kge_metrics,
            "anyburl": self.anyburl_metrics,
            "lightgbm": self.lightgbm_metrics,
        }
        if self.anyburl_classifier_metrics:
            model_metrics["xgboost"] = self.xgboost_metrics
        if self.ensemble_summary_metrics:
            model_metrics["ensemble"] = self.ensemble_summary_metrics

        trial_result = {
            "composite_score": self.composite_score,
            "ensemble_metrics": self.ensemble_metrics,
            "model_metrics": model_metrics,
            "params": dict(self.params),
            "trial_number": self.trial_number,
            "trial_dir": self.trial_dir,
            "model_paths": model_paths,
            "models_trained": {
                "rotate": self.kge_checkpoint_path.exists(),
                "anyburl": self.rules_path.exists(),
                "lightgbm": self.lightgbm_model_path.exists(),
                "xgboost": bool(self.xgboost_model_path and self.xgboost_model_path.exists()),
                "ensemble": bool(self.xgboost_model_path and self.xgboost_model_path.exists()),
            },
            "elapsed_time": self.elapsed_time,
        }
        self.artifact_manager.record_result(self.trial_number, trial_result)


def evaluate_trial_with_config(config: TrialEvaluationConfig) -> float:
    """
    Evaluate a trial using the Parameter Object pattern.
    
    SOTA: Preferred method for trial evaluation with clean API.
    
    Args:
        config: TrialEvaluationConfig with all trial parameters
        
    Returns:
        Trial score (float, higher is better)
        
    Example:
        config = TrialEvaluationConfig(
            params=suggested_params,
            train_df=train_df,
            valid_df=valid_df,
            target_entity_ratio=0.3,
            trial_number=trial.number,
            trial_output_root=output_dir,
        )
        score = evaluate_trial_with_config(config)
    """
    pipeline = TrialEvaluationPipeline(
        params=config.params,
        train_df=config.train_df,
        valid_df=config.valid_df,
        target_entity_ratio=config.target_entity_ratio,
        trial_number=config.trial_number,
        trial_output_root=config.trial_output_root,
        rule_filter=config.rule_filter,
        trial=config.trial,
        artifact_manager=config.artifact_manager or TrialArtifactManager(),
    )
    score = pipeline.run()

    if config.trial is not None:
        metrics_payload = {
            "composite_score": score,
            "kge_mrr": float(pipeline.kge_metrics.get("mrr", 0.0)),
            "kge_hits@3": float(pipeline.kge_metrics.get("hits@3", 0.0)),
            "lightgbm_auc": float(pipeline.lightgbm_metrics.get("val_auc", pipeline.lightgbm_metrics.get("auc", 0.0))),
            "lightgbm_pr_auc": float(pipeline.lightgbm_metrics.get("pr_auc") or 0.0),
            "lightgbm_mcc": float(pipeline.lightgbm_metrics.get("mcc") or 0.0),
            "lightgbm_ece": float(pipeline.lightgbm_metrics.get("ece") or 0.0),
            "lightgbm_entropy": float(pipeline.lightgbm_metrics.get("entropy") or 0.0),
            "ensemble_f1": float(
                (pipeline.ensemble_summary_metrics or {}).get("f1")
                or (pipeline.xgboost_metrics or {}).get("test_f1_score")
                or 0.0
            ),
            "ensemble_ece": float(pipeline.ensemble_ece or 0.0),
            "ensemble_entropy": float(pipeline.ensemble_entropy or 0.0),
            "base_learner_agreement": float(pipeline.base_learner_agreement or 0.0),
            "relation_coverage": float(pipeline.anyburl_metrics.get("relation_coverage") or 0.0),
            "rules_per_relation": float(pipeline.anyburl_metrics.get("rules_per_relation") or 0.0),
        }
        # P6: Add hierarchical routing statistics to metrics payload for MLflow tracking
        if pipeline.hierarchical_routing_stats:
            for key, value in pipeline.hierarchical_routing_stats.items():
                metrics_payload[f"hierarchical_{key}"] = float(value)
        try:
            for key, value in metrics_payload.items():
                config.trial.set_user_attr(key, value)
        except Exception as attr_exc:  # noqa: BLE001
            logger.warning(f"Failed to attach metrics to trial attrs: {attr_exc}")

    return score


def evaluate_trial(
    params: dict[str, Any],
    train_df: pl.DataFrame,
    valid_df: pl.DataFrame,
    *,
    target_entity_ratio: float,
    trial_number: int,
    trial_output_root: Path,
    rule_filter: AnyBURLRuleFilter | None = None,
    trial: Any | None = None,
    artifact_manager: TrialArtifactManager | None = None,
) -> float:
    """
    Evaluate a trial (legacy API, prefer evaluate_trial_with_config).
    
    Args:
        params: Trial hyperparameters dictionary
        train_df: Training DataFrame (Polars)
        valid_df: Validation DataFrame (Polars)
        target_entity_ratio: Target entity coverage ratio
        trial_number: Trial number for identification
        trial_output_root: Root directory for trial outputs
        rule_filter: Optional rule filter instance
        trial: Optional Optuna trial for pruning
        artifact_manager: Optional artifact manager
        
    Returns:
        Trial score (float, higher is better)
    """
    config = TrialEvaluationConfig(
        params=params,
        train_df=train_df,
        valid_df=valid_df,
        target_entity_ratio=target_entity_ratio,
        trial_number=trial_number,
        trial_output_root=trial_output_root,
        rule_filter=rule_filter,
        trial=trial,
        artifact_manager=artifact_manager,
    )
    return evaluate_trial_with_config(config)
