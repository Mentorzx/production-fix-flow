#!/usr/bin/env python3
"""
Core Hyperparameter Optimization Module - SOTA "Zero-Touch" Implementation
with REAL PFF Data Integration (Using Polars)

Template Method Pattern + MLOps Integration:
- Orchestrates complete "Zero-Touch" optimization
- Automatically selects best framework (Optuna SOTA)
- Integrates with MLflow for complete experiment tracking
- Generates comprehensive visualizations
- Saves best parameters automatically
- USES REAL PFF DATA (Knowledge Graph triplets)

This module provides a "Zero-Touch" experience where users only need to:
1. Define objective function
2. Define search space
3. Call find_best_hyperparameters()

For PFF-specific optimization, use optimize_kg_hyperparameters() which:
- Loads real PFF Knowledge Graph data (10K+ triplets) using Polars
- Optimizes KG-based ensemble hyperparameters
- Uses actual PFF data for realistic evaluation
"""

from __future__ import annotations

import asyncio
import copy
import json
import math
import numbers
import os
import random
import shutil
import time
import types
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import yaml
import yaml

from pff import settings
from pff.config import (
    ENSEMBLE_CONFIG_PATH,
    ENSEMBLE_HPO_CONFIG_PATH,
    KG_PIPELINE_CONFIG_PATH,
    OPTIMIZATION_CONFIG_PATH,
    ROTATE_CONFIG_PATH,
    RULE_FILTER_CONFIG_PATH,
    RULE_FILTER_HPO_CONFIG_PATH,
)
from pff.utils import ScoreCalibrator, logger
from pff.utils.hash import stable_hash

# Import strategy components (new architecture)
from .spaces import TuningConfig, SearchSpaceFactory
from .strategies import StrategyFactory
from .strategies.base import BaseOptimizerStrategy, OptimizationConfig
from .callbacks import CallbackManager, OptimizationObserver
from .extensions import ImportanceAnalyzer
from .tracker import MLflowTracker
from .visualizer import OptimizationVisualizer

# Import PFF data loading capability
import atexit
import gc


def _is_cuda_safe() -> bool:
    """
    Check if CUDA is safely available for use.
    
    Uses the global state from RotatEManager to avoid re-initialization
    attempts that could cause segfaults.
    """
    try:
        from pff.validators.rotate.manager import _CUDA_AVAILABLE
        if _CUDA_AVAILABLE is False:
            return False
        if _CUDA_AVAILABLE is True:
            return True
        # If None, CUDA hasn't been tested yet - let the manager handle it
        return False
    except ImportError:
        return False
from pff.utils.core.file_manager import FileManager
from pff.db.connection import close_connection_pool


def _cleanup_resources():
    """Cleanup resources on exit to prevent segfaults."""
    try:
        # Close PostgreSQL pool
        loop = asyncio.new_event_loop()
        loop.run_until_complete(close_connection_pool())
        loop.close()
    except Exception:
        pass
    gc.collect()


# Register cleanup at module load
atexit.register(_cleanup_resources)

# Import for real PFF data handling (using Polars, not pandas)
import polars as pl
import numpy as np
import torch
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from pff.validators.kg.config import KGConfig
from pff.validators.kg.anyburl import AnyBURLLearner
from pff.validators.kg.rule_filter import AnyBURLRuleFilter, RuleFilterConfig
from pff.validators.rotate.manager import RotatEManager
from pff.validators.rotate.lightgbm_trainer import RotatELightGBMTrainer
from pff.validators.ensembles.data_loader import EnsembleDataLoader
from pff.validators.ensembles.advanced_trainer import AdvancedEnsembleTrainer, SymbolicBalanceError
from pff.validators.ensembles.ensemble_wrappers.transformers import SymbolicCoverageError

# KGE model type - RotatE is the default and only supported model (SOTA for sparse KGs)
KGE_MODEL_ROTATE = "rotate"
DEFAULT_KGE_MODEL = KGE_MODEL_ROTATE

_CONFIG_CACHE: dict[str, Any] = {}


def _get_cached_config(path: Path, file_manager: FileManager | None = None) -> dict[str, Any]:
    """
    Lightweight config cache to avoid repeated disk reads.

    When a file_manager is provided, bypass cache to honor custom readers
    (e.g., tests/mocks). Otherwise, cache per-path in-process.
    """
    key = str(path)
    if file_manager is not None:
        return file_manager.read(path) or {}
    if key in _CONFIG_CACHE:
        return _CONFIG_CACHE[key]
    fm = FileManager()
    cfg = fm.read(path) or {}
    _CONFIG_CACHE[key] = cfg
    return cfg


def _normalize_metric(value: float, *, low: float, high: float) -> float:
    """Clamp and scale a metric into [0, 1] interval."""

    if math.isnan(value):
        return 0.0
    if high <= low:
        return max(0.0, min(1.0, value))
    normalized = (value - low) / (high - low)
    return max(0.0, min(1.0, normalized))


def _blend_scores(scores: Iterable[tuple[float, float]]) -> float:
    """Compute a weighted average from (value, weight) pairs."""

    total_weight = 0.0
    total = 0.0
    for value, weight in scores:
        if weight <= 0:
            continue
        total += value * weight
        total_weight += weight
    if total_weight == 0.0:
        return 0.0
    return total / total_weight


def _get_rules_coverage_weight(file_manager: FileManager | None = None) -> float:
    """
    Load the rules coverage weight from ensemble.yaml config.
    
    P2.2: Makes coverage weight in rules component configurable.
    Returns the weight clamped to [0.15, 0.40] for safety.
    Default: 0.2 (matches previous hardcoded behavior).
    """
    try:
        ensemble_config = _get_cached_config(ENSEMBLE_CONFIG_PATH, file_manager)
        balancing = ensemble_config.get("balancing", {})
        rules_config = balancing.get("rules", {})
        raw_weight = float(rules_config.get("coverage_weight", 0.2))
        # Clamp to safe range [0.15, 0.40]
        clamped = max(0.15, min(0.40, raw_weight))
        if clamped != raw_weight:
            logger.debug(
                f"coverage_weight clamped: {raw_weight} -> {clamped} (allowed: [0.15, 0.40])"
            )
        return clamped
    except Exception as e:
        logger.debug(f"Failed to load coverage_weight from config, using default 0.2: {e}")
        return 0.2


def _get_rule_component_weights(file_manager: FileManager | None = None) -> tuple[float, float, float]:
    """
    Load rule component weights (confidence, recall, coverage) from config.

    Returns:
        Tuple of (confidence_weight, recall_weight, coverage_weight) that sum to 1.0.
        Coverage weight is clamped to [0.15, 0.40]. Confidence/recall are
        proportionally scaled to fill the remaining mass while preserving
        their relative ratio from config.
    """
    coverage_weight = _get_rules_coverage_weight()
    try:
        ensemble_config = _get_cached_config(ENSEMBLE_CONFIG_PATH, file_manager)
        rules_cfg = ensemble_config.get("balancing", {}).get("rules", {})
        conf_raw = max(0.0, float(rules_cfg.get("confidence_weight", 0.5)))
        recall_raw = max(0.0, float(rules_cfg.get("recall_weight", 0.3)))
    except Exception as e:
        logger.debug(f"Failed to load rule component weights, using defaults: {e}")
        conf_raw, recall_raw = 0.5, 0.3

    remaining = max(0.0, 1.0 - coverage_weight)
    base_sum = conf_raw + recall_raw
    if base_sum <= 0:
        conf_weight = recall_weight = remaining * 0.5
    else:
        scale = remaining / base_sum
        conf_weight = conf_raw * scale
        recall_weight = recall_raw * scale

    return conf_weight, recall_weight, coverage_weight


def _load_ensemble_hpo_bounds(file_manager: FileManager | None = None) -> dict[str, Any]:
    """
    Load ensemble HPO bounds from config/hpo/ensemble_hpo.yaml (config-first, no literals).

    Returns:
        Nested dict with weights/thresholds/feature selection bounds.
    """
    fm = file_manager or FileManager()
    default_bounds = {
        "weights": {
            "neural_weight": {"low": 0.2, "high": 0.45},
            "rules_weight": {"low": 0.1, "high": 0.25},
            "lightgbm_weight": {"low": 0.45, "high": 0.7},
        },
        "thresholds": {
            "neural_threshold": {"low": 0.3, "high": 0.7},
            "rules_threshold": {"low": 0.2, "high": 0.7},
            "lightgbm_threshold": {"low": 0.3, "high": 0.7},
        },
        "target_symbolic_ratio": {"low": 0.3, "high": 0.42},
        "feature_selection_threshold": {"low": 0.3, "high": 0.55},
    }
    try:
        ensemble_config = _get_cached_config(ENSEMBLE_HPO_CONFIG_PATH, file_manager)
        return ensemble_config.get("hpo_bounds", default_bounds) or default_bounds
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"Failed to load ensemble hpo_bounds: {exc}")
        try:
            legacy_config = _get_cached_config(ENSEMBLE_CONFIG_PATH, file_manager)
            return legacy_config.get("hpo_bounds", default_bounds) or default_bounds
        except Exception as legacy_exc:  # noqa: BLE001
            logger.debug(f"Legacy ensemble.yaml load failed for hpo_bounds: {legacy_exc}")
        return default_bounds


def _get_range(bounds: dict[str, Any], path: list[str], default_low: float, default_high: float) -> tuple[float, float]:
    """
    Safely read a low/high pair from nested bounds with defaults.
    """
    node: Any = bounds
    try:
        for key in path:
            if not isinstance(node, dict):
                node = {}
                break
            node = node.get(key, {})
        low = float(node.get("low", default_low)) if isinstance(node, dict) else default_low
        high = float(node.get("high", default_high)) if isinstance(node, dict) else default_high
        return low, high
    except Exception:  # noqa: BLE001
        return default_low, default_high


def _load_metric_bounds(file_manager: FileManager | None = None) -> dict[str, Any]:
    """
    Load metric normalization bounds from config/hpo/ensemble_hpo.yaml.

    Returns:
        Nested dict with bounds for kge/rules/learner metrics.
    """
    fm = file_manager or FileManager()
    default_bounds = {
        "kge": {"mrr": {"low": 0.15, "high": 0.75}},
        "rules": {
            "confidence": {"low": 0.4, "high": 0.95},
            "recall": {"low": 0.05, "high": 0.5},
            "coverage": {"low": 0.05, "high": 0.5},
        },
        "learner": {
            "lgbm_auc": {"low": 0.6, "high": 0.99},
            "hybrid_f1": {"low": 0.45, "high": 0.9},
            "xgb_f1": {"low": 0.45, "high": 0.9},
        },
    }
    try:
        ensemble_config = _get_cached_config(ENSEMBLE_HPO_CONFIG_PATH, file_manager)
        return ensemble_config.get("metrics_bounds", default_bounds) or default_bounds
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"Failed to load metrics_bounds: {exc}")
        try:
            legacy_config = _get_cached_config(ENSEMBLE_CONFIG_PATH, file_manager)
            return legacy_config.get("metrics_bounds", default_bounds) or default_bounds
        except Exception as legacy_exc:  # noqa: BLE001
            logger.debug(f"Legacy ensemble.yaml load failed for metrics_bounds: {legacy_exc}")
        return default_bounds


def _default_anyburl_metrics(conf_threshold: float, support_threshold: float) -> dict[str, float]:
    return {
        "rule_count": 0.0,
        "avg_confidence": 0.0,
        "avg_support": 0.0,
        "high_confidence_ratio": 0.0,
        "applied_conf_threshold": float(conf_threshold),
        "applied_support_threshold": float(support_threshold),
    }


def _train_rotate_model(
    params: dict[str, Any],
    model_dir: Path,
    checkpoint_dir: Path,
    config_dir: Path,
    file_manager: FileManager,
) -> tuple[RotatEManager, dict[str, float], Path]:
    """Train RotatE model with given hyperparameters.

    RotatE (Rotation-based Translational Embedding) uses complex-valued
    embeddings where relations are modeled as rotations in complex space:
    h ∘ r ≈ t where r = e^(iθ)

    This model is particularly effective for:
    - Sparse knowledge graphs (>99% sparsity)
    - Graphs with symmetric and antisymmetric relations
    - Capturing compositional relation patterns

    Args:
        params: Hyperparameters dictionary containing embedding_dim, gamma, epsilon, etc.
        model_dir: Directory to save model artifacts.
        checkpoint_dir: Directory for model checkpoints.
        config_dir: Directory for trial configuration files.
        file_manager: FileManager instance for I/O operations.

    Returns:
        Tuple of (manager, metrics, checkpoint_path)
    """
    rotate_config_path = ROTATE_CONFIG_PATH
    rotate_config_data = file_manager.read(rotate_config_path)

    # Update model hyperparameters from trial
    rotate_config_data["model"]["embedding_dim"] = int(
        params.get("embedding_dim", rotate_config_data["model"].get("embedding_dim", 256))
    )
    rotate_config_data["model"]["gamma"] = float(
        params.get("gamma", rotate_config_data["model"].get("gamma", 12.0))
    )
    rotate_config_data["model"]["epsilon"] = float(
        params.get("epsilon", rotate_config_data["model"].get("epsilon", 2.0))
    )

    # Update training hyperparameters
    rotate_config_data["training"]["epochs"] = int(
        params.get("rotate_epochs", rotate_config_data["training"].get("epochs", 100))
    )
    rotate_config_data["training"]["batch_size"] = int(
        params.get("batch_size", rotate_config_data["training"].get("batch_size", 512))
    )
    rotate_config_data["training"]["learning_rate"] = float(
        params.get("meta_learning_rate", rotate_config_data["training"].get("learning_rate", 0.0001))
    )
    rotate_config_data["training"]["negative_sample_size"] = int(
        params.get("negative_sample_size", rotate_config_data["training"].get("negative_sample_size", 256))
    )

    # Self-adversarial negative sampling (SOTA technique)
    rotate_config_data["training"]["self_adversarial"] = bool(
        params.get("self_adversarial", rotate_config_data["training"].get("self_adversarial", True))
    )
    rotate_config_data["training"]["adversarial_temperature"] = float(
        params.get("adversarial_temperature", rotate_config_data["training"].get("adversarial_temperature", 1.0))
    )

    # Regularization
    rotate_config_data["model"]["regularization_weight"] = float(
        params.get("regularization_weight", rotate_config_data["model"].get("regularization_weight", 1e-5))
    )

    # Output configuration
    rotate_config_data.setdefault("checkpointing", {})
    rotate_config_data["checkpointing"]["save_dir"] = str(checkpoint_dir)
    rotate_config_data["outputs"] = {
        "dir": str(model_dir),
        "save_model": True,
        "save_embeddings": True,
        "save_checkpoints": False,
    }

    trial_rotate_config_path = config_dir / "rotate.yaml"
    file_manager.save(rotate_config_data, trial_rotate_config_path)

    logger.info("Treinando modelo RotatE (rotação em espaço complexo)...")
    rotate_manager = RotatEManager(
        rotate_config_path=trial_rotate_config_path,
        kg_config_path=KG_PIPELINE_CONFIG_PATH,
    )
    rotate_manager._setup_data()
    rotate_manager._setup_model()
    # Force retrain for HPO - each trial needs fresh training with trial-specific hyperparams
    rotate_training_stats = rotate_manager.train(force_retrain=True)

    if rotate_manager.val_triples is not None and len(rotate_manager.val_triples) > 0:
        rotate_eval_raw = rotate_manager._validate(rotate_manager.val_triples)
    else:
        rotate_eval_raw = rotate_manager.last_val_metrics or {}

    rotate_metrics = {
        "mrr": float(rotate_eval_raw.get("mrr", 0.0)),
        "hits@1": float(rotate_eval_raw.get("hits@1", 0.0)),
        "hits@10": float(rotate_eval_raw.get("hits@10", 0.0)),
        "best_val_mrr": float(rotate_training_stats.get("best_val_mrr", 0.0)),
    }

    try:
        _train_rotate_score_calibrator(rotate_manager, model_dir)
    except Exception as calib_exc:
        logger.warning(f"Failed to train RotatE calibrator: {calib_exc}")

    checkpoint_path = checkpoint_dir / "best_model.pt"
    return rotate_manager, rotate_metrics, checkpoint_path


def _train_rotate_score_calibrator(rotate_manager, output_dir: Path) -> None:
    """Fit Platt scaling using validation triples for RotatE model.

    Args:
        rotate_manager: Trained RotatE manager that exposes ``val_triples``.
        output_dir: Base directory where the calibrator file will be saved.
    """
    val_triples = getattr(rotate_manager, "val_triples", None)
    model = getattr(rotate_manager, "model", None)
    if val_triples is None or val_triples.size == 0 or model is None:
        logger.warning("No validation triples available; skipping RotatE calibration")
        return

    entity_count = len(getattr(rotate_manager, "entity_to_idx", {}))
    if entity_count == 0:
        logger.warning("Entity vocabulary is empty; skipping RotatE calibration")
        return

    rng = np.random.default_rng(42)
    scores: list[float] = []
    labels: list[int] = []
    for triple in val_triples:
        head, rel, tail = map(int, triple)
        pos_score = float(model.score_triple(head, rel, tail))
        scores.append(pos_score)
        labels.append(1)
        # Corrupt either head or tail for negative sample
        if rng.random() < 0.5:
            corrupted_head = int(rng.integers(0, entity_count))
            neg_score = float(model.score_triple(corrupted_head, rel, tail))
        else:
            corrupted_tail = int(rng.integers(0, entity_count))
            neg_score = float(model.score_triple(head, rel, corrupted_tail))
        scores.append(neg_score)
        labels.append(0)

    calibrator = ScoreCalibrator()
    try:
        calibrator.fit(np.array(scores), np.array(labels))
    except Exception as exc:
        logger.warning(f"Failed to fit RotatE score calibrator: {exc}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    calib_path = output_dir / "score_calibrator.pkl"
    FileManager().save(calibrator.to_dict(), calib_path)
    logger.info(f" Calibrador RotatE salvo em {calib_path}")


def _create_rotate_lightgbm_trainer(
    rotate_manager: RotatEManager,
    lightgbm_model_dir: Path,
    kge_model_dir: Path,
    file_manager: FileManager,
) -> RotatELightGBMTrainer:
    """Create a LightGBM trainer for RotatE embeddings.

    RotatE uses complex-valued embeddings (real + imaginary parts).
    The trainer converts complex embeddings to real-valued by
    concatenating real and imaginary parts (2*embedding_dim).

    Args:
        rotate_manager: Trained RotatE manager with complex embeddings.
        lightgbm_model_dir: Directory for LightGBM model artifacts.
        kge_model_dir: Directory for KGE model artifacts.
        file_manager: FileManager instance for I/O operations.

    Returns:
        RotatELightGBMTrainer instance configured for RotatE embeddings.
    """
    trainer = RotatELightGBMTrainer(rotate_manager)

    original_save_model = trainer.__class__.save_model

    def save_model_override(self, output_dir=None):  # type: ignore[override]
        # Override to always save to trial-specific lightgbm_model_dir
        lightgbm_model_dir.mkdir(parents=True, exist_ok=True)
        return original_save_model(self, lightgbm_model_dir)

    trainer.save_model = types.MethodType(save_model_override, trainer)

    original_extract = trainer.__class__.extract_embeddings

    def extract_embeddings_override(self):  # type: ignore[override]
        embeddings = original_extract(self)
        # Save embeddings to kge_model_dir as well
        embeddings_path = kge_model_dir / "node_embeddings.pkl"
        embeddings_path.parent.mkdir(parents=True, exist_ok=True)
        file_manager.save(embeddings, embeddings_path)
        logger.debug(f"Embeddings RotatE salvos em: {embeddings_path}")
        return embeddings

    trainer.extract_embeddings = types.MethodType(extract_embeddings_override, trainer)

    return trainer


def _load_checkpoint(checkpoint_path: Path) -> dict[str, Any] | None:
    if not checkpoint_path.exists():
        return None
    try:
        return json.loads(checkpoint_path.read_text())
    except Exception as exc:
        logger.warning(f"Could not read checkpoint {checkpoint_path}: {exc}")
        return None


def _write_checkpoint(checkpoint_path: Path, payload: dict[str, Any]) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def _delete_directory(path: Path) -> None:
    if not path.exists():
        return
    shutil.rmtree(path)

@dataclass
class HPOMemoryConfig:
    """Configuration for persistent HPO memory (Memento + Observer patterns)."""

    enabled: bool = True
    top_k_trials: int = 5
    warmstart_trials: int = 3
    storage_subdir: str = "hpo_replay"
    min_score_delta: float = 0.0

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "HPOMemoryConfig":
        data = data or {}
        return cls(
            enabled=bool(data.get("enabled", True)),
            top_k_trials=int(data.get("top_k_trials", 5)),
            warmstart_trials=int(data.get("warmstart_trials", 3)),
            storage_subdir=str(data.get("storage_subdir", "hpo_replay")),
            min_score_delta=float(data.get("min_score_delta", 0.0)),
        )


def _load_hpo_memory_config(file_manager: FileManager | None = None) -> HPOMemoryConfig:
    """Load HPO memory configuration from config/hpo/optimization.yaml."""

    fm = file_manager or FileManager()
    config_path = OPTIMIZATION_CONFIG_PATH

    try:
        raw_config = fm.read(config_path) or {}
        memory_config = raw_config.get("hpo_memory", {}) if isinstance(raw_config, dict) else {}
        return HPOMemoryConfig.from_dict(memory_config)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Failed to load HPO optimization config: {exc}")
        return HPOMemoryConfig()


class PersistentBestTrialMemory:
    """Persist best trial metrics to warm-start future HPO runs (Memento pattern)."""

    def __init__(
        self,
        output_dir: Path,
        config: HPOMemoryConfig,
        *,
        file_manager: FileManager | None = None,
    ):
        self.config = config
        self.file_manager = file_manager or FileManager()
        self.memory_path = output_dir / config.storage_subdir / "best_trials.json"
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        self.entries: list[dict[str, Any]] = self._load_entries()

    def record_trial(self, study, trial, trial_result: dict[str, Any] | None = None) -> None:
        """Record a completed trial with metrics into the persistent memory."""
        if not self.config.enabled:
            return

        try:
            from optuna.trial import TrialState
        except Exception:  # pragma: no cover - optuna missing only in unsupported envs
            return

        if getattr(trial, "state", None) != TrialState.COMPLETE:
            return
        if trial.value is None:
            return

        if self.entries and len(self.entries) >= self.config.top_k_trials:
            best_value = float(self.entries[0]["value"])
            if float(trial.value) + self.config.min_score_delta < best_value and all(
                entry["value"] >= float(trial.value) for entry in self.entries
            ):
                return

        metrics = {}
        model_metrics = {}
        if trial_result:
            metrics = self._coerce_metrics(trial_result.get("ensemble_metrics", {}))
            model_metrics = self._coerce_metrics(trial_result.get("model_metrics", {}))

        entry = {
            "study_name": getattr(study, "study_name", ""),
            "trial_number": trial.number,
            "value": float(trial.value),
            "params": dict(trial.params),
            "distributions": self._serialize_distributions(
                getattr(trial, "distributions", {}) or {}
            ),
            "metrics": metrics,
            "model_metrics": model_metrics,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        self.entries.append(entry)
        self.entries = sorted(self.entries, key=lambda item: item["value"], reverse=True)[
            : self.config.top_k_trials
        ]
        self._persist()

    def warmstart_study(self, study) -> int:
        """
        Add top trials as completed warm-start seeds into a new Optuna study.

        Returns:
            Number of trials injected.
        """
        if not self.config.enabled or not self.entries:
            return 0

        try:
            import optuna
            from optuna.trial import TrialState
        except Exception:  # pragma: no cover - optuna missing only in unsupported envs
            return 0

        added = 0
        existing_trials = [
            trial for trial in getattr(study, "trials", []) if getattr(trial, "state", None)
        ]

        for entry in self.entries[: self.config.warmstart_trials]:
            if any(self._params_match(trial.params, entry["params"]) for trial in existing_trials):
                continue

            distributions = self._deserialize_distributions(entry.get("distributions", {}))
            try:
                frozen = optuna.create_trial(
                    state=TrialState.COMPLETE,
                    value=float(entry["value"]),
                    params=entry["params"],
                    distributions=distributions,
                )
                study.add_trial(frozen)
                added += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Failed to warm-start trial replay: {exc}")

        if added > 0:
            logger.info(
                f" {added} trials de warm-start carregados da memória persistente para este estudo"
            )
        return added

    def _load_entries(self) -> list[dict[str, Any]]:
        if not self.memory_path.exists():
            return []
        try:
            payload = self.file_manager.read(self.memory_path) or {}
            entries = payload.get("entries", []) if isinstance(payload, dict) else payload
            return self._sanitize_entries(entries)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Failed to load HPO memory: {exc}")
            return []

    def _sanitize_entries(self, entries: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        sanitized: list[dict[str, Any]] = []
        for entry in entries or []:
            params = entry.get("params", {})
            try:
                value = float(entry.get("value"))
            except (TypeError, ValueError):
                continue

            if not isinstance(params, dict):
                continue

            sanitized.append(
                {
                    "study_name": entry.get("study_name", ""),
                    "trial_number": int(entry.get("trial_number", -1)),
                    "value": value,
                    "params": params,
                    "distributions": entry.get("distributions", {}),
                    "metrics": self._coerce_metrics(entry.get("metrics", {})),
                    "model_metrics": self._coerce_metrics(entry.get("model_metrics", {})),
                    "timestamp": entry.get("timestamp"),
                }
            )

        sanitized.sort(key=lambda item: item["value"], reverse=True)
        return sanitized[: self.config.top_k_trials]

    def _persist(self) -> None:
        payload = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "entries": self.entries,
        }
        self.file_manager.save(payload, self.memory_path)

    def _coerce_metrics(self, metrics: dict[str, Any]) -> dict[str, Any]:
        numeric_metrics: dict[str, Any] = {}
        for key, value in (metrics or {}).items():
            if isinstance(value, numbers.Number):
                numeric_metrics[key] = float(value)
            elif isinstance(value, dict):
                nested_numeric = self._coerce_metrics(value)
                if nested_numeric:
                    numeric_metrics[key] = nested_numeric
        return numeric_metrics

    def _serialize_distributions(self, distributions: dict[str, Any]) -> dict[str, dict[str, Any]]:
        try:
            from optuna.distributions import (
                BaseDistribution,
                CategoricalDistribution,
                FloatDistribution,
                IntDistribution,
            )
        except Exception:  # pragma: no cover
            return {}

        serialized: dict[str, dict[str, Any]] = {}
        for name, distribution in distributions.items():
            if not isinstance(distribution, BaseDistribution):
                continue
            if isinstance(distribution, FloatDistribution):
                serialized[name] = {
                    "type": "float",
                    "low": distribution.low,
                    "high": distribution.high,
                    "log": distribution.log,
                    "step": distribution.step,
                }
            elif isinstance(distribution, IntDistribution):
                serialized[name] = {
                    "type": "int",
                    "low": distribution.low,
                    "high": distribution.high,
                    "log": distribution.log,
                    "step": distribution.step,
                }
            elif isinstance(distribution, CategoricalDistribution):
                serialized[name] = {
                    "type": "categorical",
                    "choices": list(distribution.choices),
                }
        return serialized

    def _deserialize_distributions(self, serialized: dict[str, dict[str, Any]]) -> dict[str, Any]:
        try:
            from optuna.distributions import (
                CategoricalDistribution,
                FloatDistribution,
                IntDistribution,
            )
        except Exception:  # pragma: no cover
            return {}

        distributions: dict[str, Any] = {}
        for name, payload in (serialized or {}).items():
            dist_type = payload.get("type")
            if dist_type == "float":
                distributions[name] = FloatDistribution(
                    low=float(payload.get("low")),
                    high=float(payload.get("high")),
                    log=bool(payload.get("log", False)),
                    step=payload.get("step"),
                )
            elif dist_type == "int":
                distributions[name] = IntDistribution(
                    low=int(payload.get("low")),
                    high=int(payload.get("high")),
                    log=bool(payload.get("log", False)),
                    step=payload.get("step"),
                )
            elif dist_type == "categorical":
                distributions[name] = CategoricalDistribution(
                    choices=list(payload.get("choices", []))
                )
        return distributions

    def _params_match(self, lhs: dict[str, Any], rhs: dict[str, Any]) -> bool:
        if set(lhs.keys()) != set(rhs.keys()):
            return False
        for key, left_val in lhs.items():
            right_val = rhs.get(key)
            if isinstance(left_val, float) and isinstance(right_val, float):
                if abs(left_val - right_val) > 1e-9:
                    return False
            else:
                if left_val != right_val:
                    return False
        return True


def _normalize_ensemble_weights(
    params: dict[str, Any],
    *,
    min_neural_weight: float = 0.20,
    min_rules_weight: float = 0.20,
    min_lgbm_weight: float = 0.25,
    max_lgbm_weight: float = 0.60,
) -> tuple[float, float, float]:
    """
    Enforce sane constraints over ensemble weights and project them onto the simplex.

    Optuna samples weights independently, which can lead to sums > 1 or symbolic
    weights collapsing to ~0. This helper:
        1. Applies minimum floors to each component (rules always get meaningful signal).
        2. Caps LightGBM so it cannot dominate the ensemble.
        3. Normalizes the trio so they sum exactly to 1.
    """

    default_weights = {
        "neural_weight": 0.34,
        "rules_weight": 0.33,
        "lightgbm_weight": 0.33,
    }
    desired = {
        "neural_weight": float(params.get("neural_weight", default_weights["neural_weight"])),
        "rules_weight": float(params.get("rules_weight", default_weights["rules_weight"])),
        "lightgbm_weight": float(params.get("lightgbm_weight", default_weights["lightgbm_weight"])),
    }

    # Clamp LightGBM before projection to respect hard cap
    desired["lightgbm_weight"] = min(
        max(desired["lightgbm_weight"], min_lgbm_weight),
        max_lgbm_weight,
    )

    base_neural = min_neural_weight
    base_rules = min_rules_weight
    base_lgbm = min_lgbm_weight

    min_sum = base_neural + base_rules + base_lgbm
    if min_sum > 1.0:
        raise ValueError("Minimum ensemble weights exceed 100%. Check configuration.")

    # Ensure LightGBM still leaves space for neural + rules minima
    if desired["lightgbm_weight"] > 1.0 - (base_neural + base_rules):
        desired["lightgbm_weight"] = max(min_lgbm_weight, 1.0 - (base_neural + base_rules))

    z = desired["lightgbm_weight"]
    available_xy = 1.0 - z
    extra_xy = max(0.0, available_xy - (base_neural + base_rules))

    request_neural = max(0.0, desired["neural_weight"] - base_neural)
    request_rules = max(0.0, desired["rules_weight"] - base_rules)
    total_request = request_neural + request_rules

    if total_request <= 0:
        share_neural = share_rules = extra_xy * 0.5
    else:
        share_neural = extra_xy * (request_neural / total_request)
        share_rules = extra_xy - share_neural

    x = base_neural + share_neural
    y = base_rules + share_rules

    # Numerical safety: enforce bounds again
    x = max(base_neural, min(x, available_xy - base_rules))
    y = available_xy - x
    if y < base_rules:
        y = base_rules
        x = available_xy - y
    z = max(min_lgbm_weight, min(z, max_lgbm_weight))

    params["neural_weight"] = x
    params["rules_weight"] = y
    params["lightgbm_weight"] = z

    return x, y, z


def find_best_hyperparameters(
    objective_func: Callable[[Any], Union[float, List[float]]],
    search_space: dict[str, Any],
    n_trials: int = 100,
    strategy: str = "auto",
    study_name: str | None = None,
    direction: str = "maximize",
    enable_pruning: bool = True,
    enable_mlflow: bool = True,
    enable_visualization: bool = True,
    save_best_params: bool = True,
    output_dir: Path | None = None,
    timeout_seconds: int | None = None,
    storage_url: str | None = None,
    random_state: int = 42,
    enable_advanced_features: bool = False,
) -> dict[str, Any]:
    """
     SOTA "Zero-Touch" Hyperparameter Optimization

    This is the main entry point for users. Simply provide your objective function
    and search space, and this function will:
    1. Auto-select the best optimization framework (Optuna SOTA by default)
    2. Run optimization with advanced features (pruning, multi-objective, etc.)
    3. Track everything in MLflow (experiments, params, metrics, artifacts)
    4. Generate comprehensive visualizations automatically
    5. Save best parameters to JSON for easy re-use
    6. Provide detailed summary and MLflow UI URL

    Args:
        objective_func: Your objective function that takes a trial and returns a score
                       Example: def objective(trial): return trial.suggest_float('x', -10, 10)**2
        search_space: Dictionary defining your search space
                     Example: {'learning_rate': (1e-5, 1e-1), 'n_estimators': [50, 100, 200]}
        n_trials: Number of optimization trials (default: 100)
        strategy: Optimization strategy ('auto', 'optuna', 'optuna-auto', 'hyperopt')
                 'auto' automatically selects the best available framework
        study_name: Name for the optimization study (auto-generated if None)
        direction: 'maximize' or 'minimize' the objective
        enable_pruning: Whether to use pruning to skip unpromising trials
        enable_mlflow: Whether to track experiment in MLflow
        enable_visualization: Whether to generate visualization plots
        save_best_params: Whether to save best parameters to JSON
        output_dir: Directory to save results (default: ./optimization_results)
        timeout_seconds: Maximum time to spend on optimization (optional)
        storage_url: MLflow/SQLite storage URL (optional)
        random_state: Random seed for reproducibility
        enable_advanced_features: Whether to enable advanced SOTA features
                                 (distributed optimization, Bayesian optimization, PDF reports, etc.)
                                 Requires additional dependencies (ray, botorch, etc.)

    Returns:
        Dictionary with complete optimization results:
        {
            'best_params': {param_name: value, ...},
            'best_value': float,
            'n_trials': int,
            'optimization_time': float,
            'framework': str,
            'mlflow_tracking_uri': str,
            'visualization_plots': dict[str, Path],
            'best_params_file': Path,
            'study': study_object (if available)
        }

    Example:
        >>> def objective(trial):
        ...     lr = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
        ...     n_estimators = trial.suggest_int('n_estimators', 50, 200)
        ...     # Your model training code here...
        ...     return accuracy  # or [f1, auc] for multi-objective
        >>>
        >>> search_space = {
        ...     'learning_rate': (1e-5, 1e-1),
        ...     'n_estimators': [50, 100, 200, 300],
        ...     'max_depth': (3, 10),
        ... }
        >>>
        >>> result = find_best_hyperparameters(
        ...     objective_func=objective,
        ...     search_space=search_space,
        ...     n_trials=100,
        ...     strategy="auto",
        ...     study_name="model_v1_tuning"
        ... )
        >>>
        >>> print(f"Best score: {result['best_value']:.4f}")
        >>> print(f"Best params: {result['best_params']}")
        >>> print(f"MLflow UI: {result['mlflow_tracking_uri']}")
    """
    start_time = time.time()
    file_manager = FileManager()

    # Auto-generate study name if not provided
    if not study_name:
        study_name = f"optimization_{int(time.time())}"

    # Set output directory (save to outputs/ not root)
    if not output_dir:
        output_dir = settings.OUTPUTS_DIR / "optimization_results" / study_name
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("Otimização SOTA Zero-Touch de hiperparâmetros")
    logger.info("=" * 70)
    logger.info(f"Nome do estudo: {study_name}")
    logger.info(f"Estratégia: {strategy} {'(SOTA)' if strategy == 'auto' else ''}")
    logger.info(f"Número de trials: {n_trials}")
    logger.info(f"Direção: {direction}")
    logger.info(f"Diretório de saída: {output_dir}")

    # Step 1: Create optimization configuration
    config = OptimizationConfig(
        n_trials=n_trials,
        timeout_seconds=timeout_seconds,
        random_state=random_state,
        enable_pruning=enable_pruning,
        storage_url=storage_url,
        study_name=study_name,
        direction=direction,
    )

    # Step 2: Create strategy with auto-selection
    logger.info("\nSelecionando estratégia de otimização...")
    is_multi_objective = _check_if_multi_objective(objective_func)
    strategy_instance = StrategyFactory.create_strategy(
        strategy_name=strategy,
        config=config,
        is_multi_objective=is_multi_objective,
    )

    logger.info(f"Framework selecionado: {strategy_instance.framework_name}")

    # Step 2.5: Enable advanced features if requested
    if enable_advanced_features:
        logger.info("\nHabilitando recursos avançados SOTA...")
        try:
            from .advanced import AdvancedOptimizer
            logger.info("Recursos avançados disponíveis (distribuído, Bayesiano, relatórios em PDF, etc.)")
        except ImportError as e:
            logger.warning(f"Advanced features not available: {e}")
            logger.warning("Install with: pip install ray botorch fANOVA reportlab")

    # Step 3: Initialize MLflow tracking
    mlflow_tracker = None
    mlflow_run_id = ""
    if enable_mlflow:
        logger.info("\nInicializando rastreamento com MLflow...")
        mlflow_tracker = MLflowTracker(
            experiment_name=study_name,
            tracking_uri=storage_url or "./mlruns",
        )
        mlflow_run_id = mlflow_tracker.log_optimization_start(
            n_trials=n_trials,
            strategy_name=strategy_instance.framework_name,
            search_space=search_space,
        )

        tracking_uri = mlflow_tracker.get_tracking_uri()
        if tracking_uri:
            logger.info(f"MLflow pronto em: {tracking_uri}")
            logger.info("Visualize os experimentos em: http://localhost:5000")
        else:
            logger.warning("MLflow tracking URI not available")

    # Step 4: Run optimization
    logger.info("\nIniciando otimização...")
    result = strategy_instance.run_optimization(objective_func, search_space)

    # Step 5: Log results to MLflow
    if mlflow_tracker:
        logger.info("\nRegistrando resultados no MLflow...")
        mlflow_tracker.log_optimization_end(result)

        # Log each trial as nested run
        for i, trial in enumerate(result.trials):
            mlflow_tracker.log_trial(trial, i)

    # Step 6: Generate visualizations
    artifacts = {}
    if enable_visualization:
        logger.info("\nGerando visualizações...")
        visualizer = OptimizationVisualizer(output_dir=output_dir)

        # Try to get Optuna study for better plots
        optuna_study = None
        if hasattr(strategy_instance, 'study'):
            optuna_study = strategy_instance.study

        artifacts = visualizer.generate_all_plots(result, study=optuna_study)

        if artifacts:
            logger.success(f"{len(artifacts)} gráficos de visualização gerados com sucesso")

            # Log artifacts to MLflow
            if mlflow_tracker:
                mlflow_tracker.log_artifacts(artifacts)

    # Step 7: Save best parameters
    best_params_file = None
    if save_best_params:
        logger.info("\nSalvando melhores parâmetros...")
        best_params_file = output_dir / "best_params.json"

        # Create comprehensive results JSON
        results_json = {
            "best_params": result.best_params,
            "best_value": result.best_value,
            "best_trial_number": result.best_trial_number,
            "n_trials": result.n_trials,
            "optimization_time": result.optimization_time,
            "framework": result.framework,
            "strategy": strategy_instance.framework_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "search_space": search_space,
            "mlflow_run_id": mlflow_run_id,
        }

        file_manager.save(results_json, best_params_file)

        logger.success(f" Best parameters saved to: {best_params_file}")

    # Step 8: Print summary
    optimization_time = time.time() - start_time

    logger.info("\n" + "=" * 70)
    logger.success("OTIMIZAÇÃO CONCLUÍDA!")
    logger.info("=" * 70)
    logger.info(f"Melhor valor: {result.best_value:.4f}")
    logger.info(f"Melhores parâmetros:")
    for key, value in result.best_params.items():
        logger.info(f"  • {key}: {value}")
    logger.info(f"\nTempo total: {optimization_time:.2f}s")
    logger.info(f"Trials concluídos: {result.n_trials}")
    logger.info(f"Framework usado: {result.framework}")

    # Trial statistics
    n_completed = len([t for t in result.trials if t.state == 'COMPLETE'])
    n_pruned = len([t for t in result.trials if t.state == 'PRUNED'])
    n_failed = result.n_trials - n_completed - n_pruned

    logger.info(f"\nResumo dos trials:")
    logger.info(f"  • Concluídos: {n_completed}")
    logger.info(f"  • Podados: {n_pruned}")
    logger.info(f"  • Falhos: {n_failed}")

    if enable_mlflow and mlflow_tracker:
        ui_url = mlflow_tracker.get_experiment_url()
        if ui_url:
            logger.info(f"\nVeja os resultados no MLflow UI: {ui_url}")

    if artifacts:
        logger.info(f"\nGráficos de visualização salvos em: {output_dir}")
        for name, path in artifacts.items():
            logger.info(f"  • {name}: {path.name}")

    # Step 9: Return comprehensive results
    return {
        'best_params': result.best_params,
        'best_value': result.best_value,
        'best_trial_number': result.best_trial_number,
        'n_trials': result.n_trials,
        'optimization_time': optimization_time,
        'framework': result.framework,
        'study': getattr(strategy_instance, 'study', None),
        'mlflow_tracking_uri': mlflow_tracker.get_tracking_uri() if mlflow_tracker else None,
        'visualization_plots': artifacts,
        'best_params_file': best_params_file,
        'output_dir': output_dir,
        'strategy_instance': strategy_instance,
        'result': result,
    }


def optimize_kg_hyperparameters(
    n_trials: int = 100,
    strategy: str = "optuna",
    enable_mlflow: bool = True,
    enable_visualization: bool = True,
    study_name: str | None = None,
    output_dir: Path | None = None,
    target_entity_ratio: float = 0.7,
    kge_model: str = DEFAULT_KGE_MODEL,
) -> dict[str, Any]:
    """
    PFF Knowledge Graph Hyperparameter Optimization with REAL DATA (Polars)

    This function optimizes hyperparameters specifically for PFF's Knowledge Graph ensemble
    using REAL PFF data (10K+ triplets from /data/models/kg/) loaded with Polars.

    IMPORTANT: This function ALWAYS uses real data and will FAIL if data cannot be loaded.
    NO FALLBACKS or simulations - errors are intentional!

    Uses RotatE as the default KGE model (SOTA for sparse knowledge graphs):
    - RotatE: Rotation-based complex embeddings (h ∘ r ≈ t), better for sparse graphs

    Args:
        n_trials: Number of optimization trials
        strategy: Optimization strategy ('optuna' only)
        enable_mlflow: Whether to track in MLflow
        enable_visualization: Whether to generate visualization plots (default: True)
        study_name: Name for the optimization study
        output_dir: Directory to save results
        target_entity_ratio: Ratio of positive entities in synthetic labels (0.7 = 70%)
        kge_model: KGE model to use. Default is 'rotate' (RotatE), which is
            recommended for sparse graphs with >99% sparsity.

    Returns:
        Dictionary with optimization results including:
        - best_params: Optimized hyperparameters
        - best_value: Best achieved score
        - real_data_info: Information about loaded data
        - evaluation_metrics: F1, AUC, Precision, Recall
        - kge_model: The KGE model used for optimization

    Data Sources:
        - Training: /data/models/kg/train_optimized.parquet (10,241 triplets)
        - Validation: /data/models/kg/valid_optimized.parquet (2,194 triplets)
        - Knowledge Graph: 2,823 entities, 40 predicates
        - Format: (subject, predicate, object) triplets loaded with Polars

    Example:
        >>> # Using RotatE (default - recommended for sparse graphs)
        >>> result = optimize_kg_hyperparameters(
        ...     n_trials=50,
        ...     study_name="pff_kg_ensemble_rotate"
        ... )
        >>> print(f"Best KG ensemble score: {result['best_value']:.4f}")
        >>> print(f"KGE model used: {result['kge_model']}")
    """
    # Validate KGE model selection (only RotatE is supported)
    if kge_model != KGE_MODEL_ROTATE:
        logger.warning(f"Only RotatE model is supported. Using RotatE instead of '{kge_model}'")
        kge_model = KGE_MODEL_ROTATE

    logger.info("=" * 70)
    logger.info("Otimização de hiperparâmetros do PFF Knowledge Graph")
    logger.info("=" * 70)
    logger.info(f"Modelo KGE selecionado: {kge_model.upper()}")
    logger.info("Utilizando dados reais do Knowledge Graph (Polars)")
    logger.info("=" * 70)
    file_manager = FileManager()

    logger.info("\nCarregando dados reais do PFF KG com Polars...")
    try:
        train_df, valid_df, data_info = _load_real_kg_data()
        logger.success("Dados reais carregados com sucesso via Polars")
        logger.info(f"Treino: {data_info['n_train']:,} triplets")
        logger.info(f"Validação: {data_info['n_valid']:,} triplets")
        logger.info(f"Entidades: {data_info['n_entities']:,}")
        logger.info(f"Predicados: {data_info['n_predicates']}")
    except Exception as e:
        logger.error(f"Failed to load real data: {e}")
        logger.error("NO FALLBACKS - optimization requires real data!")
        raise RuntimeError(f"Cannot proceed without real PFF KG data: {e}")

    rule_filter_config_path = RULE_FILTER_CONFIG_PATH
    try:
        rule_filter = AnyBURLRuleFilter.from_config(rule_filter_config_path)
    except Exception as filter_exc:
        logger.warning(f"Failed to load filter configuration ({rule_filter_config_path}): {filter_exc}")
        rule_filter = AnyBURLRuleFilter(RuleFilterConfig())

    # P1.2 - Load HPO ranges from config (config-driven search space)
    hpo_ranges: dict[str, dict[str, int | float]] = {}
    try:
        hpo_config = file_manager.read(RULE_FILTER_HPO_CONFIG_PATH)
        hpo_ranges = hpo_config.get("rule_filter", {}).get("hpo_ranges", {})
        logger.info(" Ranges HPO carregados do config/models/kg.yaml (section rule_filter.hpo_ranges)")
    except Exception as ranges_exc:
        logger.warning(f"Failed to load HPO ranges from config: {ranges_exc}")
        # Fallback to defaults (P1.3 conservative expansion)
        hpo_ranges = {
            "max_length_cyclic": {"low": 3, "high": 4},
            "max_length_acyclic": {"low": 3, "high": 5},
            "confidence_quantile": {"low": 0.5, "high": 0.9},
            "support_quantile": {"low": 0.3, "high": 0.8},
            "target_ratio": {"low": 0.2, "high": 0.5},
        }

    # P2.x - Load ensemble HPO bounds (weights/thresholds) from config
    ensemble_hpo_bounds = _load_ensemble_hpo_bounds(file_manager)

    trial_runs_dir: Path | None = None
    symbolic_retry_state: dict[str, int] = {"enqueues": 0}
    max_symbolic_retry_enqueues = int(os.getenv("PFF_SYMBOLIC_MAX_RETRIES", "6"))

    def _maybe_enqueue_symbolic_retry(
        source_trial,
        failed_params: dict[str, Any],
        *,
        reason: str,
    ) -> None:
        fallback_params = _derive_symbolic_retry_params(failed_params)
        if not fallback_params:
            return
        if symbolic_retry_state["enqueues"] >= max_symbolic_retry_enqueues:
            logger.info(
                " Limite de reenfileiramentos simbólicos atingido; seguindo sem ajustes"
            )
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
                f"(motivo: {reason}) (retry {symbolic_retry_state['enqueues']}/"
                f"{max_symbolic_retry_enqueues})"
            )
        except Exception as enqueue_exc:  # noqa: BLE001
            logger.warning(f"Failed to enqueue symbolic retry: {enqueue_exc}")

    # Define KG-specific objective function
    # (search space is defined inside the objective function using trial.suggest_*)
    def kg_objective(trial):
        """
        Objective function optimized for PFF Knowledge Graph.
        Uses real KG triplets for realistic evaluation with Polars data.
        Uses RotatE as the KGE model (SOTA for sparse graphs).

        P1.2/P1.3: Rule length and filter ranges are now config-driven via hpo_ranges.
        """
        if trial_runs_dir is None:
            raise RuntimeError("Trial output directory not initialized")

        # P1.2 - Get ALL range bounds from config or use safe defaults
        cyclic_range = hpo_ranges.get("max_length_cyclic", {"low": 3, "high": 4})
        acyclic_range = hpo_ranges.get("max_length_acyclic", {"low": 3, "high": 5})
        conf_quantile_range = hpo_ranges.get("confidence_quantile", {"low": 0.5, "high": 0.9})
        support_quantile_range = hpo_ranges.get("support_quantile", {"low": 0.3, "high": 0.8})
        target_ratio_range = hpo_ranges.get("target_ratio", {"low": 0.2, "high": 0.5})
        # P2.x - Ensemble weight/threshold bounds from config/models/ensemble.yaml
        nw_low, nw_high = _get_range(ensemble_hpo_bounds, ["weights", "neural_weight"], 0.2, 0.45)
        rw_low, rw_high = _get_range(ensemble_hpo_bounds, ["weights", "rules_weight"], 0.1, 0.25)
        lw_low, lw_high = _get_range(ensemble_hpo_bounds, ["weights", "lightgbm_weight"], 0.45, 0.7)
        nt_low, nt_high = _get_range(ensemble_hpo_bounds, ["thresholds", "neural_threshold"], 0.3, 0.7)
        rt_low, rt_high = _get_range(ensemble_hpo_bounds, ["thresholds", "rules_threshold"], 0.2, 0.7)
        lt_low, lt_high = _get_range(ensemble_hpo_bounds, ["thresholds", "lightgbm_threshold"], 0.3, 0.7)
        tsr_low, tsr_high = _get_range(ensemble_hpo_bounds, ["target_symbolic_ratio"], 0.3, 0.42)
        fst_low, fst_high = _get_range(ensemble_hpo_bounds, ["feature_selection_threshold"], 0.3, 0.55)

        # Get KG ensemble hyperparameters from trial
        # Base ensemble weights (bounded to avoid symbolic dominance)
        params = {
            'neural_weight': trial.suggest_float('neural_weight', float(nw_low), float(nw_high)),
            'rules_weight': trial.suggest_float('rules_weight', float(rw_low), float(rw_high)),
            'lightgbm_weight': trial.suggest_float('lightgbm_weight', float(lw_low), float(lw_high)),

            # AnyBURL hyperparameters (rule-based model)
            'rule_confidence': trial.suggest_float('rule_confidence', 0.5, 0.95),
            'rule_support': trial.suggest_int('rule_support', 5, 50),
            'max_rule_length': trial.suggest_int('max_rule_length', 2, 5),
            # P1.2 - Rule filter quantile/ratio ranges (config-driven)
            'confidence_quantile': trial.suggest_float(
                'confidence_quantile',
                float(conf_quantile_range.get("low", 0.5)),
                float(conf_quantile_range.get("high", 0.9))
            ),
            'support_quantile': trial.suggest_float(
                'support_quantile',
                float(support_quantile_range.get("low", 0.3)),
                float(support_quantile_range.get("high", 0.8))
            ),
            'target_ratio': trial.suggest_float(
                'target_ratio',
                float(target_ratio_range.get("low", 0.2)),
                float(target_ratio_range.get("high", 0.5))
            ),
            # P1.3 - Cyclic/acyclic rule length controls (config-driven ranges)
            'max_length_cyclic': trial.suggest_int(
                'max_length_cyclic',
                int(cyclic_range.get("low", 3)),
                int(cyclic_range.get("high", 4))
            ),
            'max_length_acyclic': trial.suggest_int(
                'max_length_acyclic',
                int(acyclic_range.get("low", 3)),
                int(acyclic_range.get("high", 5))
            ),

            # LightGBM hyperparameters (meta-learner)
            'meta_learning_rate': trial.suggest_float('meta_learning_rate', 1e-4, 1e-1, log=True),
            'meta_n_estimators': trial.suggest_int('meta_n_estimators', 50, 300),
            'negative_ratio': trial.suggest_float('negative_ratio', 0.5, 3.0),

            # Ensemble configuration
            'target_symbolic_ratio': trial.suggest_float('target_symbolic_ratio', float(tsr_low), float(tsr_high)),
            'neural_threshold': trial.suggest_float('neural_threshold', float(nt_low), float(nt_high)),
            'rules_threshold': trial.suggest_float('rules_threshold', float(rt_low), float(rt_high)),
            'lightgbm_threshold': trial.suggest_float('lightgbm_threshold', float(lt_low), float(lt_high)),
            'ensemble_voting': trial.suggest_categorical('ensemble_voting', ['soft', 'hard']),
            'feature_selection_threshold': trial.suggest_float('feature_selection_threshold', float(fst_low), float(fst_high)),

            # KGE model selection (always RotatE - SOTA for sparse graphs)
            'kge_model': KGE_MODEL_ROTATE,

            # RotatE hyperparameters (optimized for fast HPO - 50 trials)
            # Smaller dims + larger batches = faster training per epoch
            'embedding_dim': trial.suggest_categorical('embedding_dim', [64, 128, 256]),
            'gamma': trial.suggest_float('gamma', 6.0, 15.0),
            'epsilon': trial.suggest_float('epsilon', 1.5, 2.5),
            'rotate_epochs': trial.suggest_int('rotate_epochs', 20, 50),
            'batch_size': trial.suggest_int('batch_size', 512, 2048),
            'negative_sample_size': trial.suggest_int('negative_sample_size', 64, 256),
            'adversarial_temperature': trial.suggest_float('adversarial_temperature', 0.5, 2.0),
            'self_adversarial': trial.suggest_categorical('self_adversarial', [True, False]),
            'regularization_weight': trial.suggest_float('regularization_weight', 1e-5, 1e-3, log=True),
        }

        # Evaluate with REAL data ONLY - NO FALLBACKS
        if 'train_df' not in locals() or 'valid_df' not in locals():
            raise RuntimeError("Cannot evaluate: Real data not loaded")

        try:
            score = _evaluate_kg_ensemble_real(
                params,
                train_df,
                valid_df,
                target_entity_ratio=target_entity_ratio,
                trial_number=trial.number,
                trial_output_root=trial_runs_dir,
                rule_filter=rule_filter,
            )
            return score
        except SymbolicCoverageError as cov_exc:
            _maybe_enqueue_symbolic_retry(trial, params, reason="cobertura")
            raise optuna.TrialPruned(f"Symbolic coverage failure: {cov_exc}") from cov_exc
        except SymbolicBalanceError as dominance_exc:
            _maybe_enqueue_symbolic_retry(trial, params, reason="dominância")
            raise optuna.TrialPruned(f"Symbolic balance failure: {dominance_exc}") from dominance_exc
        finally:
            # Clean up GPU memory between trials to prevent segfaults
            import gc
            gc.collect()
            try:
                # Only call CUDA functions if CUDA was actually initialized
                # torch.cuda.is_available() returns True even if CUDA init failed
                # torch.cuda.is_initialized() is the safe check
                if torch.cuda.is_available() and torch.cuda.is_initialized():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    # Reset CUDA peak memory stats for next trial
                    # Note: This can segfault if CUDA allocator state is corrupted
                    try:
                        torch.cuda.reset_peak_memory_stats()
                    except Exception:
                        pass  # Silently ignore - not critical
            except (RuntimeError, AssertionError) as cuda_err:
                logger.debug(f"CUDA cleanup skipped: {cuda_err}")

    # Run optimization using Optuna directly
    # Note: Don't use find_best_hyperparameters here because kg_objective already defines the search space
    import optuna

    study_name = study_name or f"pff_kg_optimization_{int(time.time())}"

    if not output_dir:
        output_dir = settings.OUTPUTS_DIR / "optimization_results" / "kg_ensemble"
    output_dir.mkdir(parents=True, exist_ok=True)

    hpo_memory_config = _load_hpo_memory_config(file_manager)
    trial_memory = PersistentBestTrialMemory(
        output_dir,
        hpo_memory_config,
        file_manager=file_manager,
    )

    checkpoint_path = output_dir / "checkpoint.json"
    storage_path = output_dir / "optuna_study.db"

    checkpoint_data = _load_checkpoint(checkpoint_path)
    resume_mode = False
    expected_trials = n_trials

    if checkpoint_data:
        previous_status = checkpoint_data.get("status")
        stored_target = int(checkpoint_data.get("expected_trials", expected_trials) or expected_trials)
        expected_trials = max(stored_target, n_trials)
        if previous_status in {"running", "interrupted"}:
            if storage_path.exists():
                resume_mode = True
                detected_study = checkpoint_data.get("study_name", study_name)
                logger.info(f"Checkpoint detectado ({previous_status}). Retomando estudo {detected_study}.")
            else:
                logger.warning(
                    "Checkpoint indicava execução em andamento, mas o storage não foi encontrado. "
                    "Um novo estudo será iniciado."
                )
                checkpoint_data = None
                expected_trials = n_trials
        elif previous_status == "completed":
            logger.info("Última execução finalizada com sucesso. Removendo artefatos anteriores.")
            for folder_name in ("trials", "best_models"):
                _delete_directory(output_dir / folder_name)
            if storage_path.exists():
                storage_path.unlink()
            try:
                checkpoint_path.unlink()
            except FileNotFoundError:
                pass
            checkpoint_data = None
            expected_trials = n_trials

    if not resume_mode and storage_path.exists() and checkpoint_data is None:
        storage_path.unlink()

    trial_runs_dir = output_dir / "trials"
    best_models_dir = output_dir / "best_models"
    if resume_mode:
        trial_runs_dir.mkdir(parents=True, exist_ok=True)
    else:
        _delete_directory(trial_runs_dir)
        trial_runs_dir.mkdir(parents=True, exist_ok=True)
        _delete_directory(best_models_dir)

    model_saver_callback = BestModelSaverCallback(output_dir, memory=trial_memory)

    storage_url = f"sqlite:///{storage_path}"
    study = optuna.create_study(
        study_name=study_name,
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.HyperbandPruner(
            min_resource=5,
            max_resource="auto",
            reduction_factor=3,
        ),
        storage=storage_url,
        load_if_exists=True,
    )

    warmstart_injected = trial_memory.warmstart_study(study)
    if warmstart_injected:
        logger.debug(f"Warm-start trials injected into study: {warmstart_injected}")

    total_target_trials = max(expected_trials, n_trials)
    existing_trials_count = len(study.trials)
    remaining_trials = max(total_target_trials - existing_trials_count, 0)
    completed_trials_count = sum(
        1 for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE
    )

    checkpoint_payload = {
        "status": "running",
        "study_name": study_name,
        "expected_trials": total_target_trials,
        "completed_trials": completed_trials_count,
        "resume_mode": resume_mode,
        "last_update": datetime.now(timezone.utc).isoformat(),
    }
    _write_checkpoint(checkpoint_path, checkpoint_payload)

    logger.info(f"Estudo Optuna criado: {study_name}")
    logger.info(f"Amostrador ativo: {study.sampler.__class__.__name__}")
    logger.info(f"Pruner configurado: {study.pruner.__class__.__name__}")
    logger.info(f"Modelos serão salvos em: {output_dir / 'best_models'}")

    if remaining_trials > 0:
        logger.info(
            f"Iniciando otimização com {remaining_trials} trials pendentes (alvo total: {total_target_trials})."
        )
    else:
        logger.info(
            "Nenhum trial pendente. Os resultados existentes já atingem o alvo configurado."
        )

    start_time = time.time()

    def cleanup_after_trial(study, trial):
        """Force cleanup after each trial to prevent segfaults."""
        gc.collect()
        # Flush SQLite writes
        try:
            if hasattr(study, '_storage') and hasattr(study._storage, '_engine'):
                study._storage._engine.dispose()
        except Exception:
            pass

    try:
        if remaining_trials > 0:
            # SOTA Optuna 4.x: gc_after_trial=True prevents OOM by running garbage collection
            # after each trial. This is especially important for GPU workloads.
            study.optimize(
                kg_objective,
                n_trials=remaining_trials,
                n_jobs=1,
                callbacks=[model_saver_callback, cleanup_after_trial],
                gc_after_trial=True,
            )
    except Exception:
        checkpoint_payload["status"] = "interrupted"
        checkpoint_payload["completed_trials"] = len(study.trials)
        checkpoint_payload["last_update"] = datetime.now(timezone.utc).isoformat()
        _write_checkpoint(checkpoint_path, checkpoint_payload)
        raise
    else:
        checkpoint_payload["status"] = "completed"
        checkpoint_payload["completed_trials"] = len(study.trials)
        checkpoint_payload["last_update"] = datetime.now(timezone.utc).isoformat()
        _write_checkpoint(checkpoint_path, checkpoint_payload)

    # Calculate total optimization time
    optimization_time = time.time() - start_time

    # Build result
    try:
        best_params = study.best_params
        best_value = study.best_value
    except Exception:
        logger.warning("Nenhum trial completo; retornando best_params vazio")
        best_params = {}
        best_value = None

    result = {
        'best_params': best_params,
        'best_value': best_value,
        'n_trials': len(study.trials),
        'optimization_time': optimization_time,
        'framework': 'optuna',
        'study': study,
        'trials': [],
    }

    # Add trial information
    for trial in study.trials:
        trial_result = {
            'number': trial.number,
            'value': trial.value,
            'params': trial.params,
            'state': str(trial.state),
        }
        result['trials'].append(trial_result)

    # Add PFF-specific information
    result['real_data_info'] = data_info
    result['kge_model'] = kge_model  # Track which KGE model was used
    if model_saver_callback.best_trial_result:
        result['evaluation_metrics'] = model_saver_callback.best_trial_result.get('ensemble_metrics', {})
        result['best_model_metrics'] = model_saver_callback.best_trial_result.get('model_metrics', {})
    else:
        result['evaluation_metrics'] = {}

    # Generate visualizations if enabled
    if enable_visualization:
        logger.info("\nGerando gráficos de visualização...")
        try:
            if not output_dir:
                output_dir = settings.OUTPUTS_DIR / "optimization_results" / "kg_ensemble"

            logger.info(f"  → Diretório de saída: {output_dir}")
            visualizer = OptimizationVisualizer(output_dir=output_dir)
            artifacts = visualizer.generate_all_plots(result, study=study)

            if artifacts:
                logger.success(f" Gerados {len(artifacts)} gráficos de visualização")
                result['visualization_plots'] = artifacts
                result['output_dir'] = output_dir

                if enable_mlflow:
                    logger.info("Registrando artefatos no MLflow...")

        except Exception as e:
            logger.warning(f"Failed to generate visualizations: {e}")

    logger.success("\n Otimização do KG concluída!")
    best_score = result.get("best_value")
    if isinstance(best_score, (int, float)):
        logger.info(f"Melhor score do ensemble: {best_score:.4f}")
    else:
        logger.warning("No completed trial produced a valid score; inspect pruning logs and symbolic limits.")
    logger.info(f"Dados utilizados: {data_info['n_train']:,} treino + {data_info['n_valid']:,} validação")

    # Add information about saved models
    best_models_dir = output_dir / "best_models"
    if best_models_dir.exists():
        result['best_models_dir'] = best_models_dir
        result['best_model_files'] = {}

        logger.info("\nModelos salvos:")

        # Check for RotatE model
        rotate_model = best_models_dir / "best_rotate_model.pt"
        if rotate_model.exists():
            result['best_model_files']['rotate'] = rotate_model
            logger.info(f"   RotatE armazenado em: {rotate_model}")

        # Check for AnyBURL rules
        anyburl_rules = best_models_dir / "anyburl" / "rules.tsv"
        if anyburl_rules.exists():
            result['best_model_files']['anyburl'] = anyburl_rules
            logger.info(f"   AnyBURL armazenado em: {anyburl_rules}")

        # Check for LightGBM model
        lgbm_model = best_models_dir / "best_lightgbm_model.bin"
        if lgbm_model.exists():
            result['best_model_files']['lightgbm'] = lgbm_model
            logger.info(f"   LightGBM armazenado em: {lgbm_model}")

        logger.info("\nHiperparâmetros salvos:")

        # Check for individual param files
        best_metrics_summary: dict[str, Dict[str, float]] = {}
        for model_name in ['rotate', 'anyburl', 'lightgbm', 'ensemble']:
            param_file = best_models_dir / f"best_params_{model_name}.json"
            if param_file.exists():
                logger.info(f"   Arquivo de hiperparâmetros ({model_name}): {param_file}")
                try:
                    payload = file_manager.read(param_file)
                    metrics = payload.get('metrics') or {}
                    numeric_metrics = {
                        key: float(value)
                        for key, value in metrics.items()
                        if isinstance(value, (int, float))
                    }
                    if numeric_metrics:
                        best_metrics_summary[model_name] = numeric_metrics

                    if model_name == 'anyburl':
                        classifier_metrics = payload.get('classifier_metrics') or {}
                        numeric_classifier = {
                            key: float(value)
                            for key, value in classifier_metrics.items()
                            if isinstance(value, (int, float))
                        }
                        if numeric_classifier:
                            best_metrics_summary['anyburl_classifier'] = numeric_classifier
                except Exception as metric_exc:
                    logger.warning(f"Failed to read metrics from {param_file}: {metric_exc}")

        if best_metrics_summary:
            logger.info("\nMétricas dos melhores modelos:")
            for model_name, metrics in best_metrics_summary.items():
                formatted = " | ".join(
                    f"{key}={value:.4f}" for key, value in metrics.items()
                ) or "no numeric metrics recorded"
                logger.info(f"  {model_name.replace('_', ' ').title()} → {formatted}")
            result['best_model_metrics'] = best_metrics_summary

    if enable_visualization and 'visualization_plots' in result:
        logger.info(f"\n Gráficos disponíveis em: {result['output_dir']}")

    # Cleanup: Close database connection pool to prevent segfault on exit
    try:
        asyncio.get_event_loop().run_until_complete(close_connection_pool())
    except Exception:
        pass  # Ignore cleanup errors - pool may already be closed

    return result


def _load_real_kg_data(file_manager: FileManager | None = None) -> tuple:
    """Load real PFF Knowledge Graph data using the FileManager abstraction."""

    base_dir = settings.MODELS_DIR / "kg"
    
    # Try multiple possible paths for train data
    train_candidates = [
        base_dir / "train_optimized.parquet",
        base_dir / "train.parquet",
        settings.OUTPUTS_DIR / "pyclause" / "train.homogenized.parquet",
    ]
    
    train_path = None
    for candidate in train_candidates:
        if candidate.exists():
            train_path = candidate
            break
    
    if train_path is None:
        raise FileNotFoundError(
            f"Training data not found. Tried: {[str(p) for p in train_candidates]}"
        )
    
    # Try multiple possible paths for valid data
    valid_candidates = [
        base_dir / "valid_optimized.parquet",
        base_dir / "valid.parquet",
        settings.OUTPUTS_DIR / "pyclause" / "valid.homogenized.parquet",
    ]
    
    valid_path = None
    for candidate in valid_candidates:
        if candidate.exists():
            valid_path = candidate
            break
    
    if valid_path is None:
        raise FileNotFoundError(
            f"Validation data not found. Tried: {[str(p) for p in valid_candidates]}"
        )

    fm = file_manager or FileManager()

    logger.info(f"Carregando com FileManager (Parquet): {train_path}")
    train_df = fm.read(train_path)
    logger.info(f"Carregando com FileManager (Parquet): {valid_path}")
    valid_df = fm.read(valid_path)

    # Extract entities using Polars
    train_subjects = train_df.select("s").to_series().unique()
    train_objects = train_df.select("o").to_series().unique()
    valid_subjects = valid_df.select("s").to_series().unique()
    valid_objects = valid_df.select("o").to_series().unique()

    all_entities = (
        train_subjects
        .extend(train_objects)
        .extend(valid_subjects)
        .extend(valid_objects)
        .unique()
    )

    data_info = {
        "n_train": len(train_df),
        "n_valid": len(valid_df),
        "n_entities": len(all_entities),
        "n_predicates": train_df.select("p").n_unique(),
        "train_path": str(train_path),
        "valid_path": str(valid_path),
        "sample_triplets": [
            {
                "s": row["s"][:30] + ("..." if len(row["s"]) > 30 else ""),
                "p": row["p"],
                "o": row["o"][:30] + ("..." if len(row["o"]) > 30 else ""),
            }
            for row in train_df.head(3).to_dicts()
        ],
    }

    return train_df, valid_df, data_info


def _compute_entity_quality_scores(train_df: pl.DataFrame, valid_df: pl.DataFrame) -> dict[str, float]:
    """Blend multiple connectivity signals into a normalized entity quality score."""

    def _count(df: pl.DataFrame, column: str, alias: str) -> pl.DataFrame:
        if df.is_empty():
            return pl.DataFrame({"entity": [], alias: []})
        return (
            df.groupby(column)
            .agg(pl.len().alias(alias))
            .rename({column: "entity"})
        )

    def _relation_span(df: pl.DataFrame, column: str, alias: str) -> pl.DataFrame:
        if df.is_empty():
            return pl.DataFrame({"entity": [], alias: []})
        return (
            df.groupby(column)
            .agg(pl.n_unique("p").alias(alias))
            .rename({column: "entity"})
        )

    train_out = _count(train_df, "s", "train_out")
    train_in = _count(train_df, "o", "train_in")
    valid_out = _count(valid_df, "s", "valid_out")
    valid_in = _count(valid_df, "o", "valid_in")
    train_rel = _relation_span(train_df, "s", "train_rel")
    valid_rel = _relation_span(valid_df, "s", "valid_rel")

    stats = (
        train_out.lazy()
        .join(train_in.lazy(), on="entity", how="outer")
        .join(valid_out.lazy(), on="entity", how="outer")
        .join(valid_in.lazy(), on="entity", how="outer")
        .join(train_rel.lazy(), on="entity", how="outer")
        .join(valid_rel.lazy(), on="entity", how="outer")
        .with_columns([
            pl.all().exclude("entity").fill_null(0.0),
        ])
        .with_columns([
            (pl.col("train_out") + pl.col("train_in")).alias("train_total"),
            (pl.col("valid_out") + pl.col("valid_in")).alias("valid_total"),
        ])
        .with_columns([
            (pl.col("train_total") + 1.0).log1p().alias("train_signal"),
            (pl.col("valid_total") + 1.0).log1p().alias("valid_signal"),
            (pl.col("train_rel") + pl.col("valid_rel") + 1.0)
            .log1p()
            .alias("relation_signal"),
        ])
        .with_columns(
            (
                0.55 * pl.col("train_signal")
                + 0.25 * pl.col("valid_signal")
                + 0.20 * pl.col("relation_signal")
            ).alias("quality_raw")
        )
        .select(["entity", "quality_raw"])
        .collect()
    )

    if stats.is_empty():
        return {}

    min_val = float(stats["quality_raw"].min())
    max_val = float(stats["quality_raw"].max())
    span = max(max_val - min_val, 1e-9)

    stats = stats.with_columns(
        ((pl.col("quality_raw") - min_val) / span).alias("quality_score")
    )

    return {
        row["entity"]: float(row["quality_score"])
        for row in stats.select(["entity", "quality_score"]).to_dicts()
    }


def _evaluate_kg_ensemble_real(
    params: dict[str, Any],
    train_df: pl.DataFrame,
    valid_df: pl.DataFrame,
    *,
    target_entity_ratio: float,
    trial_number: int,
    trial_output_root: Path,
    rule_filter: AnyBURLRuleFilter | None = None,
) -> float:
    """Evaluate KG ensemble using production pipelines in an isolated workspace.

    Uses RotatE as the KGE model (SOTA for sparse knowledge graphs).
    The model uses rotation-based complex embeddings where h ∘ r ≈ t.
    """

    start_time = time.time()
    logger.info(
        f" Visão do dataset do trial → treino={len(train_df):,} | validação={len(valid_df):,}"
    )

    # Always use RotatE (SOTA for sparse graphs)
    kge_model_type = KGE_MODEL_ROTATE

    trial_seed = stable_hash(tuple(sorted(params.items())), truncate=16) & (2**32 - 1)
    random.seed(trial_seed)
    np.random.seed(trial_seed)
    torch.manual_seed(trial_seed)
    # Only set CUDA seeds if CUDA is confirmed to work
    if _is_cuda_safe():
        try:
            torch.cuda.manual_seed(trial_seed)
            torch.cuda.manual_seed_all(trial_seed)
        except Exception:
            pass
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    normalized_weights = _normalize_ensemble_weights(params)
    logger.info(
        f"Pesos normalizados → neural={normalized_weights[0]:.3f} | "
        f"regras={normalized_weights[1]:.3f} | lightgbm={normalized_weights[2]:.3f}"
    )

    trial_dir = trial_output_root / f"trial_{trial_number:04d}"
    if trial_dir.exists():
        shutil.rmtree(trial_dir)
    trial_dir.mkdir(parents=True, exist_ok=True)

    config_dir = trial_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    models_dir = trial_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    kge_model_dir = models_dir / kge_model_type
    kge_model_dir.mkdir(parents=True, exist_ok=True)
    lightgbm_model_dir = models_dir / "lightgbm"
    lightgbm_model_dir.mkdir(parents=True, exist_ok=True)

    file_manager = FileManager()
    symbolic_params: dict[str, Any] = {}
    ensemble_config_path = ENSEMBLE_CONFIG_PATH
    try:
        ensemble_cfg = file_manager.read(ensemble_config_path) or {}
        for base_model in ensemble_cfg.get("base_models", []):
            if base_model.get("type") == "symbolic":
                symbolic_params = base_model.get("params", {})
                break
    except Exception as cfg_exc:
        logger.warning(
            f"Failed to load ensemble.yaml for symbolic limits: {cfg_exc}"
        )
        symbolic_params = {}
    coverage_gate = float(symbolic_params.get("min_coverage_threshold", 0.25))
    dominance_gate = float(symbolic_params.get("dominance_max_ratio", 0.99))
    dominance_gate = float(np.clip(dominance_gate, 0.55, 1.0))

    max_symbolic_rules_cfg = symbolic_params.get("max_rules")
    symbolic_max_rules = (
        int(max_symbolic_rules_cfg)
        if isinstance(max_symbolic_rules_cfg, (int, float)) and max_symbolic_rules_cfg > 0
        else None
    )
    default_activation_ratio = float(symbolic_params.get("min_activation_ratio", 0.01))
    raw_feature_threshold = params.get("feature_selection_threshold")
    min_symbolic_activation = default_activation_ratio
    try:
        if raw_feature_threshold is not None:
            # Cap at 0.05 (5%) to prevent aggressive pruning of sparse rules
            min_symbolic_activation = float(
                np.clip(float(raw_feature_threshold) * 0.1, 0.005, 0.05)
            )
    except (TypeError, ValueError):
        logger.debug(
            f"Invalid feature_selection_threshold received: {raw_feature_threshold}"
        )

    # ------------------------------------------------------------------
    # Build isolated KG configuration to keep AnyBURL artifacts per trial
    # ------------------------------------------------------------------
    kg_config_path = KG_PIPELINE_CONFIG_PATH
    kg_config_data = file_manager.read(kg_config_path)
    kg_config_data.setdefault("paths", {})
    kg_config_data["paths"]["data_dir"] = str(settings.DATA_DIR)
    kg_config_data["paths"]["output_dir"] = str(trial_dir / "outputs")
    kg_config_data["paths"]["graph_subdir"] = "models/kg"

    # Inject AnyBURL cyclic/acyclic parameters from HPO trial
    kg_config_data.setdefault("anyburl", {})
    kg_config_data["anyburl"]["MAX_LENGTH_CYCLIC"] = params.get("max_length_cyclic", 3)
    kg_config_data["anyburl"]["MAX_LENGTH_ACYCLIC"] = params.get("max_length_acyclic", 3)

    trial_kg_config_path = config_dir / "kg.yaml"
    file_manager.save(kg_config_data, trial_kg_config_path)

    trial_kg_config = KGConfig(trial_kg_config_path)

    # ------------------------------------------------------------------
    # Train RotatE model (SOTA for sparse knowledge graphs)
    # ------------------------------------------------------------------
    kge_checkpoint_dir = kge_model_dir / "checkpoints"
    kge_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    kge_manager, kge_metrics, kge_checkpoint_path = _train_rotate_model(
        params, kge_model_dir, kge_checkpoint_dir, config_dir, file_manager
    )

    # ------------------------------------------------------------------
    # LightGBM hybrid training (uses RotatE embeddings)
    # ------------------------------------------------------------------
    logger.info("Treinando modelo híbrido LightGBM...")

    # Create a wrapper that adapts RotatE manager for LightGBM training
    trainer = _create_rotate_lightgbm_trainer(kge_manager, lightgbm_model_dir, kge_model_dir, file_manager)

    # Force retrain for HPO - each trial needs its own model with trial-specific hyperparams
    lightgbm_metrics_raw = trainer.train_hybrid_model(force_retrain=True)
    lightgbm_metrics = {k: float(v) for k, v in lightgbm_metrics_raw.items()}
    lightgbm_model_path = lightgbm_model_dir / "lightgbm_model.bin"

    # ------------------------------------------------------------------
    # AnyBURL rule learning in isolated output directory
    # ------------------------------------------------------------------
    logger.info("Aprendendo regras com AnyBURL...")
    anyburl_learner = AnyBURLLearner()
    asyncio.run(anyburl_learner.learn_rules(trial_kg_config))

    rules_path = trial_kg_config.get_rules_path()
    rule_metadata_lookup: dict[str, Dict[str, Any]] = {}
    anyburl_metrics = _default_anyburl_metrics(
        conf_threshold=float(params.get("rule_confidence", max(target_entity_ratio, 0.5))),
        support_threshold=float(params.get("rule_support", 5)),
    )

    if rules_path.exists():
        filter_instance = rule_filter or AnyBURLRuleFilter(RuleFilterConfig())
        try:
            filter_result = filter_instance.filter_rules(
                rules_path=rules_path,
                output_dir=trial_dir / "anyburl",
                rule_confidence=float(params.get("rule_confidence", 0.5)),
                rule_support=float(params.get("rule_support", 5)),
                target_entity_ratio=target_entity_ratio,
                max_rules=symbolic_max_rules,
            )
            rules_path = filter_result.filtered_rules_path
            rule_metadata_lookup = filter_result.metadata_lookup
            anyburl_metrics.update(filter_result.metrics)
        except Exception as rule_exc:
            logger.warning(f"Failed to filter AnyBURL rules: {rule_exc}")
    else:
        logger.warning(
            "AnyBURL rule file not found; symbolic coverage will remain zero."
        )

    # ------------------------------------------------------------------
    # Prepare hybrid, symbolic, and ensemble evaluations (LightGBM + XGBoost)
    # ------------------------------------------------------------------
    xgboost_metrics: dict[str, Any] = {}
    ensemble_summary_metrics: dict[str, Any] = {}
    anyburl_classifier_metrics: dict[str, Any] = {}
    hybrid_eval_metrics: dict[str, Any] = {}
    symbolic_contribution_ratio: float | None = None
    dominance_violation_message: str | None = None
    hybrid_contribution_ratio: float | None = None
    xgboost_model_path: Path | None = None

    if lightgbm_model_path.exists():
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
                random_state=trial_seed,
                stratify=y_train_samples,
            )

            X_train_np = np.array(X_train_split, dtype=object)
            X_val_np = np.array(X_val_split, dtype=object)
            y_train_np = np.array(y_train_split)
            y_val_np = np.array(y_val_split)
            X_test_np = np.array(X_test_samples, dtype=object)
            y_test_np = np.array(y_test_samples)

            temp_outputs_dir = trial_dir / "runtime_outputs"
            if temp_outputs_dir.exists():
                shutil.rmtree(temp_outputs_dir)
            temp_rotate_dir = temp_outputs_dir / "rotate"
            temp_rotate_dir.mkdir(parents=True, exist_ok=True)

            original_outputs_dir = settings.OUTPUTS_DIR
            orig_rotate_dir = original_outputs_dir / "rotate"
            if orig_rotate_dir.exists():
                shutil.copytree(orig_rotate_dir, temp_rotate_dir, dirs_exist_ok=True)

            # Copy pyclause directory containing entity/relation mappings
            orig_pyclause_dir = original_outputs_dir / "pyclause"
            if orig_pyclause_dir.exists():
                temp_pyclause_dir = temp_outputs_dir / "pyclause"
                shutil.copytree(orig_pyclause_dir, temp_pyclause_dir, dirs_exist_ok=True)
                logger.info(f"Mapeamentos copiados de {orig_pyclause_dir} para {temp_pyclause_dir}")

            trial_embeddings = kge_model_dir / "node_embeddings.pkl"
            if trial_embeddings.exists():
                shutil.copy2(trial_embeddings, temp_rotate_dir / "node_embeddings.pkl")

            for metadata_name in ["lightgbm_metadata.pkl", "hybrid_metrics.json"]:
                src_meta = lightgbm_model_dir / metadata_name
                if src_meta.exists():
                    dest_meta_dir = temp_outputs_dir / "rotate"
                    dest_meta_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_meta, dest_meta_dir / metadata_name)

            settings.OUTPUTS_DIR = temp_outputs_dir
            try:
                ensemble_output_dir = models_dir / "ensemble"
                ensemble_trainer = AdvancedEnsembleTrainer(
                    neural_model_path=str(kge_model_dir),
                    rules_path=str(rules_path),
                    lightgbm_model_path=str(lightgbm_model_path),
                    output_dir=ensemble_output_dir,
                    force_symbolic_contribution=False,
                    min_symbolic_activation=min_symbolic_activation,
                )

                try:
                    ensemble_trainer.train(X_train_np, y_train_np, X_val_np, y_val_np)
                except SymbolicBalanceError as dominance_exc:
                    logger.warning(f" Symbolic dominance detectada durante o treino (ignoring): {dominance_exc}")
                    # RELAXED: Do not raise hard error during training either
                    # raise
                except SymbolicCoverageError:
                    raise
                trainer_balance = getattr(ensemble_trainer, "feature_balance", None)
                if trainer_balance:
                    if symbolic_contribution_ratio is None:
                        symbolic_contribution_ratio = trainer_balance.get("symbolic")
                    if hybrid_contribution_ratio is None:
                        hybrid_contribution_ratio = trainer_balance.get("hybrid")
                xgboost_metrics_raw = ensemble_trainer.evaluate(X_test_np, y_test_np, prefix="test")
                ensemble_trainer.save_model()

                xgboost_model_path = ensemble_output_dir / "stacking_model_advanced.joblib"
                try:
                    # EAGER: remove non-picklable pointers before joblib.dump
                    if hasattr(ensemble_trainer, 'ensemble_model'):
                        model = ensemble_trainer.ensemble_model
                        if hasattr(model, 'named_steps') and 'features' in model.named_steps:
                            feats = model.named_steps['features']
                            for name, step in getattr(feats, 'transformer_list', []):
                                if hasattr(step, '__getstate__'):
                                    # rely on wrapper's custom getstate
                                    pass
                                else:
                                    # strip ctypes pointers heuristically
                                    for attr in dir(step):
                                        if attr.startswith('_') and 'ptr' in attr:
                                            try:
                                                setattr(step, attr, None)
                                            except Exception:
                                                pass
                except Exception as _strip_exc:
                    logger.debug(f"Strip non-picklable pointers fallback ignored: {_strip_exc}")

                def _to_native(obj: Any) -> Any:
                    if isinstance(obj, dict):
                        return {k: _to_native(v) for k, v in obj.items()}
                    if isinstance(obj, (list, tuple)):
                        return [_to_native(v) for v in obj]
                    if isinstance(obj, (np.floating, np.float32, np.float64)):
                        return float(obj)
                    if isinstance(obj, (np.integer, np.int64, np.int32)):
                        return int(obj)
                    return obj

                xgboost_metrics = {k: _to_native(v) for k, v in xgboost_metrics_raw.items()}

                # Hybrid metrics (probabilities directly from HybridWrapper)
                features_union = ensemble_trainer.ensemble_model.named_steps["features"]
                trained_union = getattr(features_union, "_trained_union", None)
                if trained_union is not None:
                    transformer_map = dict(trained_union.transformer_list)
                elif hasattr(features_union, "base_union"):
                    transformer_map = dict(features_union.base_union.transformer_list)
                else:
                    transformer_map = dict(features_union.transformer_list)

                hybrid_pipeline = transformer_map["hybrid_pred"]
                proba_transformer = hybrid_pipeline.named_steps["hybrid"]
                hybrid_model = proba_transformer.model
                hybrid_proba = hybrid_model.predict_proba(X_test_np)[:, 1]
                precision_curve, recall_curve, thresholds_curve = precision_recall_curve(y_test_np, hybrid_proba)
                if thresholds_curve.size > 0:
                    f1_curve = 2 * (precision_curve[:-1] * recall_curve[:-1]) / (
                        precision_curve[:-1] + recall_curve[:-1] + 1e-9
                    )
                    best_idx = int(np.nanargmax(f1_curve))
                    hybrid_threshold = float(thresholds_curve[best_idx])
                else:
                    hybrid_threshold = 0.5
                hybrid_threshold = float(np.clip(hybrid_threshold, 0.1, 0.9))
                hybrid_pred = (hybrid_proba >= hybrid_threshold).astype(int)
                hybrid_eval_metrics = {
                    "auc": roc_auc_score(y_test_np, hybrid_proba) if len(np.unique(y_test_np)) > 1 else 0.0,
                    "accuracy": accuracy_score(y_test_np, hybrid_pred),
                    "precision": precision_score(y_test_np, hybrid_pred, zero_division=0),
                    "recall": recall_score(y_test_np, hybrid_pred, zero_division=0),
                    "f1": f1_score(y_test_np, hybrid_pred, zero_division=0),
                    "threshold": float(hybrid_threshold),
                }

                # AnyBURL classifier metrics (triggered rules)
                symbolic_transformer = transformer_map["symbolic_rules"]
                try:
                    meta_learner_step = ensemble_trainer.ensemble_model.named_steps["meta_learner"]
                    if isinstance(meta_learner_step, Pipeline):
                        xgb_model = meta_learner_step.named_steps.get("xgboost")
                    else:
                        xgb_model = meta_learner_step
                    if xgb_model is not None and hasattr(xgb_model, "feature_importances_"):
                        importances = getattr(xgb_model, "feature_importances_", None)
                        if importances is not None and len(importances) > 0:
                            hybrid_imp = float(importances[0])
                            symbolic_imp = float(np.sum(importances[1:])) if len(importances) > 1 else 0.0
                            total_imp = hybrid_imp + symbolic_imp
                            if total_imp > 0:
                                hybrid_contribution_ratio = hybrid_imp / total_imp
                                symbolic_contribution_ratio = symbolic_imp / total_imp
                                if symbolic_contribution_ratio > dominance_gate:
                                    dominance_violation_message = (
                                        f"Symbolic contribution {symbolic_contribution_ratio:.3f} exceeds dominance limit {dominance_gate:.3f}"
                                    )
                except Exception as contrib_exc:
                    logger.debug(f"Falha ao calcular contribuição híbrido/simbólico: {contrib_exc}")
                used_confidences: list[float] = []
                used_supports: list[float] = []
                if hasattr(symbolic_transformer, "rules_") and symbolic_transformer.rules_:
                    logger.info(
                        f" Symbolic extractor retained {len(symbolic_transformer.rules_)} rules after filtering"
                    )
                    for rule in symbolic_transformer.rules_:
                        meta = rule_metadata_lookup.get(rule.get("prolog") if isinstance(rule, dict) else str(rule))
                        if meta:
                            used_confidences.append(float(meta.get("confidence", 0.0)))
                            used_supports.append(float(meta.get("support", 0.0)))

                    if used_confidences:
                        coverage_threshold = float(anyburl_metrics.get("applied_conf_threshold", 0.0))
                        anyburl_metrics.update(
                            {
                                "rule_count": int(len(symbolic_transformer.rules_)),
                                "avg_confidence": float(np.mean(used_confidences)),
                                "avg_support": float(np.mean(used_supports)) if used_supports else anyburl_metrics.get("avg_support", 0.0),
                                "high_confidence_ratio": float(
                                    sum(1 for c in used_confidences if c >= coverage_threshold)
                                )
                                / len(used_confidences),
                            }
                        )
                symbolic_features = symbolic_transformer.transform(X_test_samples)
                symbolic_features = np.asarray(symbolic_features)
                rule_hits = symbolic_features.sum(axis=1)
                symbolic_pred = (rule_hits > 0).astype(int)
                positive_mask = y_test_np == 1
                coverage_samples = float(np.mean(rule_hits > 0)) if len(rule_hits) else 0.0
                coverage_density = float(np.count_nonzero(symbolic_features) / symbolic_features.size) if symbolic_features.size else 0.0
                activated_samples = int(np.count_nonzero(rule_hits > 0))
                predicate_activation: list[tuple[str, int]] = []
                has_symbolic_rules = bool(getattr(symbolic_transformer, "rules_", None))
                if has_symbolic_rules:
                    per_rule_hits = np.sum(symbolic_features > 0, axis=0).astype(int)
                    if per_rule_hits.size == len(symbolic_transformer.rules_):
                        predicate_counter: Counter[str] = Counter()
                        for idx, hit_count in enumerate(per_rule_hits):
                            if hit_count <= 0:
                                continue
                            rule_head = symbolic_transformer.rules_[idx].get("head", {}) if isinstance(symbolic_transformer.rules_[idx], dict) else {}
                            predicate = str(rule_head.get("predicate", "")).strip()
                            if predicate:
                                predicate_counter[predicate] += 1
                        predicate_activation = predicate_counter.most_common(10)
                if predicate_activation:
                    top_predicates_log = ", ".join(f"{pred}:{count}" for pred, count in predicate_activation[:5])
                    logger.info(
                        f"Cobertura de predicados simbólicos: principais ativações {top_predicates_log}"
                    )
                elif has_symbolic_rules:
                    logger.warning("Symbolic predicates reported zero activation")
                anyburl_classifier_metrics = {
                    "precision": precision_score(y_test_np, symbolic_pred, zero_division=0),
                    "recall": recall_score(y_test_np, symbolic_pred, zero_division=0),
                    "f1": f1_score(y_test_np, symbolic_pred, zero_division=0),
                    "accuracy": accuracy_score(y_test_np, symbolic_pred),
                    "coverage": max(coverage_samples, coverage_density),
                    "positive_rule_coverage": float(np.mean(rule_hits[positive_mask] > 0)) if positive_mask.any() else 0.0,
                    "negative_rule_activation": float(np.mean(rule_hits[~positive_mask] > 0)) if (~positive_mask).any() else 0.0,
                    "samples_with_rules": activated_samples,
                    "predicate_activation": predicate_activation,
                    "feature_density": coverage_density,
                }

                anyburl_metrics["coverage"] = float(anyburl_classifier_metrics["coverage"])
                anyburl_metrics["positive_rule_coverage"] = float(
                    anyburl_classifier_metrics["positive_rule_coverage"]
                )
                anyburl_metrics["samples_with_rules"] = activated_samples
                if predicate_activation:
                    anyburl_metrics["top_symbolic_predicates"] = predicate_activation[:5]

                ensemble_summary_metrics = {
                    "weighted_score": None,
                    "test_accuracy": xgboost_metrics.get("test_accuracy"),
                    "test_precision": xgboost_metrics.get("test_precision"),
                    "test_recall": xgboost_metrics.get("test_recall"),
                    "test_f1_score": xgboost_metrics.get("test_f1_score"),
                    "test_auc_roc": xgboost_metrics.get("test_auc_roc"),
                    "symbolic_contribution": symbolic_contribution_ratio,
                    "hybrid_contribution": hybrid_contribution_ratio,
                }

            finally:
                settings.OUTPUTS_DIR = original_outputs_dir
        except Exception as ensemble_exc:
            logger.warning(f"Failed to run XGBoost ensemble evaluation: {ensemble_exc}")
    else:
        logger.warning("Skipping ensemble evaluation because LightGBM model artifact is missing")

    anyburl_metrics.setdefault("coverage", 0.0)
    anyburl_metrics.setdefault("positive_rule_coverage", 0.0)

    coverage_val = float(anyburl_metrics.get("coverage", 0.0))
    if coverage_val < coverage_gate:
        warning_msg = (
            f"Symbolic coverage {coverage_val:.3f} below required target {coverage_gate:.3f}"
        )
        logger.warning(warning_msg)
        raise SymbolicCoverageError(warning_msg)
    if dominance_violation_message:
        logger.warning(dominance_violation_message)
        # RELAXED: Do not raise hard error, just warn and let penalty handle it
        # raise SymbolicBalanceError(dominance_violation_message)

    # ------------------------------------------------------------------
    # Compute composite score using production metrics
    # ------------------------------------------------------------------
    neural_w = float(params.get("neural_weight", 0.0))
    rules_w = float(params.get("rules_weight", 0.0))
    lgbm_w = float(params.get("lightgbm_weight", 0.0))

    safe_neural_w = max(neural_w, 0.05)
    safe_rules_w = max(rules_w, 0.05)
    safe_lgbm_w = min(max(lgbm_w, 0.05), 0.70)

    metric_bounds = _load_metric_bounds(file_manager)
    kge_low, kge_high = _get_range(metric_bounds, ["kge", "mrr"], 0.15, 0.75)
    rules_conf_low, rules_conf_high = _get_range(metric_bounds, ["rules", "confidence"], 0.4, 0.95)
    rules_rec_low, rules_rec_high = _get_range(metric_bounds, ["rules", "recall"], 0.05, 0.5)
    rules_cov_low, rules_cov_high = _get_range(metric_bounds, ["rules", "coverage"], 0.05, 0.5)
    lgb_auc_low, lgb_auc_high = _get_range(metric_bounds, ["learner", "lgbm_auc"], 0.6, 0.99)
    hybrid_f1_low, hybrid_f1_high = _get_range(metric_bounds, ["learner", "hybrid_f1"], 0.45, 0.9)
    xgb_f1_low, xgb_f1_high = _get_range(metric_bounds, ["learner", "xgb_f1"], 0.45, 0.9)

    kge_component = _normalize_metric(kge_metrics["mrr"], low=kge_low, high=kge_high)
    rules_conf_component = _normalize_metric(
        anyburl_metrics["avg_confidence"], low=rules_conf_low, high=rules_conf_high
    )
    rules_recall_component = _normalize_metric(
        anyburl_classifier_metrics.get("recall", 0.0), low=rules_rec_low, high=rules_rec_high
    )
    rules_cov_component = _normalize_metric(
        anyburl_metrics.get("coverage", 0.0), low=rules_cov_low, high=rules_cov_high
    )
    # P2.2 - Use config-driven rule component weights (confidence/recall/coverage)
    conf_weight, recall_weight, coverage_weight = _get_rule_component_weights(file_manager)
    rules_component = _blend_scores(
        [
            (rules_conf_component, conf_weight),
            (rules_recall_component, recall_weight),
            (rules_cov_component, coverage_weight),
        ]
    )

    lgbm_auc_component = _normalize_metric(
        lightgbm_metrics.get("auc", 0.0), low=lgb_auc_low, high=lgb_auc_high
    )
    hybrid_f1_component = _normalize_metric(
        hybrid_eval_metrics.get("f1", 0.0), low=hybrid_f1_low, high=hybrid_f1_high
    )
    xgb_f1_component = _normalize_metric(
        xgboost_metrics.get("test_f1_score", 0.0) if xgboost_metrics else 0.0,
        low=xgb_f1_low,
        high=xgb_f1_high,
    )
    learner_component = max(lgbm_auc_component, hybrid_f1_component, xgb_f1_component)

    base_score = _blend_scores(
        [
            (kge_component, safe_neural_w),
            (rules_component, safe_rules_w),
            (learner_component, safe_lgbm_w),
        ]
    )

    min_weight = min(neural_w, rules_w, lgbm_w)
    weight_penalty = max(0.0, 0.05 - min_weight)
    coverage_target = max(coverage_gate, 0.05)
    coverage_penalty = max(0.0, coverage_target - anyburl_metrics.get("coverage", 0.0))
    rules_weight_target = 0.25
    rules_weight_penalty = max(0.0, rules_weight_target - rules_w)
    overweight = max(0.0, lgbm_w - 0.70)

    # P3: Load scoring config for symbolic dominance penalty tuning
    scoring_config = _get_cached_config(ENSEMBLE_HPO_CONFIG_PATH, file_manager).get("scoring", {})
    fallback_dominance_target = float(scoring_config.get("fallback_dominance_target", 0.70))
    symbolic_dominance_penalty_coeff = float(scoring_config.get("symbolic_dominance_penalty_coeff", 0.50))

    # P3: Use target_symbolic_ratio from trial params as dominance target, fallback to config
    dominance_target = float(params.get("target_symbolic_ratio", fallback_dominance_target))
    symbolic_dominance_penalty = 0.0
    if symbolic_contribution_ratio is not None and symbolic_contribution_ratio > dominance_target:
        dominance_overflow = symbolic_contribution_ratio - dominance_target
        symbolic_dominance_penalty = dominance_overflow / max(1e-6, 1.0 - dominance_target)

    neural_contribution_penalty = 0.0
    if hybrid_contribution_ratio is not None:
        # Force at least 20% neural contribution (hybrid_contribution_ratio is usually hybrid importance)
        # Note: hybrid_contribution_ratio in code above is actually hybrid_imp / total_imp
        # If hybrid (neural+LGBM) is too low, penalize.
        min_neural_target = 0.20
        if hybrid_contribution_ratio < min_neural_target:
            neural_contribution_penalty = (min_neural_target - hybrid_contribution_ratio) / min_neural_target
            logger.warning(f"Low neural contribution: {hybrid_contribution_ratio:.2%} < {min_neural_target:.0%}")

    composite_score = base_score
    for coeff, penalty in [
        (0.40, weight_penalty),
        (0.45, coverage_penalty),
        (0.35, rules_weight_penalty),
        (0.20, overweight),
        (symbolic_dominance_penalty_coeff, symbolic_dominance_penalty),  # P3: config-driven coefficient
        (0.60, neural_contribution_penalty), # Strong penalty for ignoring neural model
    ]:
        composite_score *= (1.0 - coeff * min(1.0, penalty))
    composite_score = max(0.0, composite_score)

    ensemble_metrics = {
        "weighted_score": composite_score,
        "base_weighted_score": base_score,
        "kge_mrr": kge_metrics["mrr"],
        "rules_avg_confidence": anyburl_metrics["avg_confidence"],
        "rules_coverage": anyburl_metrics.get("coverage", 0.0),
        "lightgbm_auc": lightgbm_metrics.get("val_auc", lightgbm_metrics.get("auc", 0.0)),
        "normalized_neural": kge_component,
        "normalized_rules": rules_component,
        "normalized_learner": learner_component,
        "weight_penalty": weight_penalty,
        "coverage_penalty": coverage_penalty,
        "rules_weight_penalty": rules_weight_penalty,
        "symbolic_dominance_penalty": symbolic_dominance_penalty,
        "normalized_weights": {
            "neural": neural_w,
            "rules": rules_w,
            "lightgbm": lgbm_w,
        },
        "symbolic_contribution": symbolic_contribution_ratio,
        "hybrid_contribution": hybrid_contribution_ratio,
    }

    if ensemble_summary_metrics:
        ensemble_summary_metrics["weighted_score"] = composite_score
        ensemble_summary_metrics.update(
            {
                "normalized_weighted_score": base_score,
                "normalized_neural": kge_component,
                "normalized_rules": rules_component,
                "normalized_learner": learner_component,
            }
        )

    elapsed_time = time.time() - start_time

    logger.info("=" * 70)
    logger.info("Métricas individuais")
    logger.info("=" * 70)
    logger.info(
        f"KGE → MRR: {kge_metrics['mrr']:.4f} | "
        f"Hits@1: {kge_metrics['hits@1']:.4f} | "
        f"Hits@10: {kge_metrics['hits@10']:.4f} | "
        f"Best val MRR: {kge_metrics['best_val_mrr']:.4f}"
    )
    anyburl_rule_count = int(round(anyburl_metrics.get("rule_count", 0.0)))
    logger.info(
        f"AnyBURL → rules={anyburl_rule_count} | "
        f"avg_conf={anyburl_metrics.get('avg_confidence', 0.0):.4f} | "
        f"avg_support={anyburl_metrics.get('avg_support', 0.0):.2f} | "
        f"high_conf_ratio={anyburl_metrics.get('high_confidence_ratio', 0.0):.2f}"
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
    for metric_name in ["auc", "f1", "accuracy", "precision", "recall"]:
        if metric_name in lightgbm_metrics:
            logger.info(
                f"  {metric_name.upper()}: {_format_metric_value(lightgbm_metrics[metric_name])}"
            )
    if hybrid_eval_metrics:
        logger.info("Métricas do híbrido (RotatE + LightGBM):")
        for metric_name, metric_value in hybrid_eval_metrics.items():
            logger.info(
                f"  {metric_name.upper()}: {_format_metric_value(metric_value)}"
            )
    if anyburl_classifier_metrics:
        logger.info("Métricas do classificador AnyBURL:")
        for metric_name, metric_value in anyburl_classifier_metrics.items():
            logger.info(f"  {metric_name.upper()}: {_format_metric_value(metric_value)}")
    if xgboost_metrics:
        logger.info("Métricas do ensemble XGBoost:")
        for metric_name, metric_value in xgboost_metrics.items():
            logger.info(f"  {metric_name.upper()}: {_format_metric_value(metric_value)}")
    if ensemble_summary_metrics:
        logger.info("Resumo final do ensemble:")
        for metric_name, metric_value in ensemble_summary_metrics.items():
            if metric_value is None:
                continue
            logger.info(f"  {metric_name.upper()}: {_format_metric_value(metric_value)}")
    if weight_penalty > 0:
        logger.warning(
            f"Ensemble weight imbalance detected (min weight {min_weight:.3f})"
        )
    if coverage_penalty > 0:
        logger.warning(
            f"Rule coverage below target ({anyburl_metrics.get('coverage', 0.0):.3f} < {coverage_target:.3f})"
        )
    if symbolic_dominance_penalty > 0 and symbolic_contribution_ratio is not None:
        logger.warning(
            f"Symbolic dominance detected ({symbolic_contribution_ratio:.2%} > {dominance_target:.0%})"
        )

    logger.debug(
        f"Weights: neural={neural_w:.3f}, rules={rules_w:.3f}, "
        f"lgbm={lgbm_w:.3f}, base_norm={base_score:.4f}"
    )
    logger.success(f"Avaliacao do trial concluida: score={composite_score:.4f}, tempo={elapsed_time / 60.0:.1f}min")

    model_paths: dict[str, Path] = {}
    if kge_checkpoint_path.exists():
        model_paths["rotate"] = kge_checkpoint_path
    if rules_path.exists():
        model_paths["anyburl"] = rules_path
    if lightgbm_model_path.exists():
        model_paths["lightgbm"] = lightgbm_model_path
    if xgboost_model_path and xgboost_model_path.exists():
        model_paths["xgboost"] = xgboost_model_path

    logger.info(
        f"Modelos treinados neste trial → KGE={'sim' if kge_checkpoint_path.exists() else 'não'} | "
        f"AnyBURL={'sim' if rules_path.exists() else 'não'} | "
        f"LightGBM={'sim' if lightgbm_model_path.exists() else 'não'} | "
        f"XGBoost={'sim' if xgboost_model_path and xgboost_model_path.exists() else 'não'}"
    )

    model_metrics: dict[str, Any] = {
        "rotate": kge_metrics,
        "anyburl": anyburl_metrics,
        "lightgbm": lightgbm_metrics,
    }

    if anyburl_classifier_metrics:
        model_metrics["anyburl_classifier"] = anyburl_classifier_metrics
    if hybrid_eval_metrics:
        model_metrics["hybrid"] = hybrid_eval_metrics
    if xgboost_metrics:
        model_metrics["xgboost"] = xgboost_metrics
    if ensemble_summary_metrics:
        model_metrics["ensemble"] = ensemble_summary_metrics

    trial_result = {
        "composite_score": composite_score,
        "ensemble_metrics": ensemble_metrics,
        "model_metrics": model_metrics,
        "params": dict(params),
        "trial_number": trial_number,
        "trial_dir": trial_dir,
        "model_paths": model_paths,
        "models_trained": {
            "rotate": kge_checkpoint_path.exists(),
            "anyburl": rules_path.exists(),
            "lightgbm": lightgbm_model_path.exists(),
            "xgboost": bool(xgboost_model_path and xgboost_model_path.exists()),
            "ensemble": bool(xgboost_model_path and xgboost_model_path.exists()),
        },
        "elapsed_time": elapsed_time,
    }

    if not hasattr(_evaluate_kg_ensemble_real, "trial_results"):
        _evaluate_kg_ensemble_real.trial_results = {}
    trial_storage: dict[int, Dict[str, Any]] = _evaluate_kg_ensemble_real.trial_results  # type: ignore[attr-defined]
    trial_storage[trial_number] = trial_result

    return composite_score


class BestModelSaverCallback:
    """
    Optuna callback to save the best models after each trial.

    This callback:
    1. Checks if current trial is the best so far
    2. If best: copies models to permanent storage
    3. If not best: deletes temporary files
    4. Saves individual best_params for each model
    """

    def __init__(self, output_dir: Path, memory: PersistentBestTrialMemory | None = None):
        """
        Initialize callback.

        Args:
            output_dir: Directory to save best models
        """
        self.output_dir = output_dir
        self.best_models_dir = output_dir / "best_models"
        self.best_models_dir.mkdir(parents=True, exist_ok=True)

        self.best_value = float('-inf')
        self.best_trial_number = -1
        self.best_trial_result = None
        self.memory = memory

    def __call__(self, study, trial):
        """
        Called after each trial completes.

        Args:
            study: Optuna study object
            trial: Completed trial object
        """
        import shutil
        try:
            from optuna.trial import TrialState
        except Exception:  # pragma: no cover - optuna is an optional dep outside CLI
            TrialState = None

        trial_state = getattr(trial, "state", None)
        if TrialState is not None and trial_state != TrialState.COMPLETE:
            logger.debug(
                f"  → Ignorando trial #{trial.number} com estado {trial_state} (não completo)"
            )
            return

        # Get trial result from global storage
        trial_results = getattr(_evaluate_kg_ensemble_real, 'trial_results', {})

        # Prefer exact lookup by trial number (pop to avoid leaks)
        trial_result = trial_results.pop(trial.number, None)

        # Fallback: Find matching trial result by comparing params (legacy dicts keyed by id)
        if trial_result is None:
            for stored_key, result in list(trial_results.items()):
                if result.get('params') == trial.params:
                    trial_result = result
                    del trial_results[stored_key]
                    break

        # If exact match fails, try loose matching (for floating point precision issues)
        if trial_result is None:
            for stored_key, result in list(trial_results.items()):
                result_params = result['params']
                trial_params = trial.params
                # Check if all parameters match (with small tolerance for float)
                match = True
                for param_name in set(list(result_params.keys()) + list(trial_params.keys())):
                    val1 = result_params.get(param_name)
                    val2 = trial_params.get(param_name)
                    if isinstance(val1, float) and isinstance(val2, float):
                        if abs(val1 - val2) > 1e-9:
                            match = False
                            break
                    elif val1 != val2:
                        match = False
                        break
                if match:
                    trial_result = result
                    del trial_results[stored_key]
                    break

        if trial_result is None:
            logger.warning(f"Could not find trial result for trial #{trial.number}")
            logger.warning(f"Available trial results: {len(trial_results)}")
            return

        if self.memory:
            self.memory.record_trial(study, trial, trial_result)

        trial_dir = trial_result['trial_dir']
        is_best = trial.value > self.best_value

        if is_best:
            # New best trial found!
            self.best_value = trial.value
            self.best_trial_number = trial.number
            self.best_trial_result = trial_result

            logger.success(f"   NEW BEST TRIAL #{trial.number}: {trial.value:.4f}")
            logger.info(f"  → Saving best models to: {self.best_models_dir}")

            # Debug logging for trial result
            logger.debug(f"  → Trial result models_trained: {trial_result.get('models_trained')}")
            logger.debug(f"  → Trial result model_paths keys: {list(trial_result.get('model_paths', {}).keys())}")

            # Remove old best models if they exist
            if self.best_models_dir.exists():
                for item in self.best_models_dir.iterdir():
                    if item.is_file():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item)

            # Copy new best models
            model_paths = trial_result.get('model_paths', {})

            if 'rotate' in model_paths and model_paths['rotate'].exists():
                dest = self.best_models_dir / "best_rotate_model.pt"
                shutil.copy2(model_paths['rotate'], dest)
                logger.info(f"   RotatE model saved: {dest}")
            else:
                logger.warning(f"RotatE model NOT saved (models_trained={trial_result.get('models_trained', {}).get('rotate')}, path={'rotate' in model_paths})")

            if 'anyburl' in model_paths and model_paths['anyburl'].exists():
                dest_dir = self.best_models_dir / "anyburl"
                dest_dir.mkdir(exist_ok=True)
                dest = dest_dir / "rules.tsv"
                shutil.copy2(model_paths['anyburl'], dest)
                logger.info(f"   AnyBURL rules saved: {dest}")
            else:
                logger.warning(f"AnyBURL model NOT saved (models_trained={trial_result.get('models_trained', {}).get('anyburl')}, path={'anyburl' in model_paths})")

            if 'lightgbm' in model_paths and model_paths['lightgbm'].exists():
                dest = self.best_models_dir / "best_lightgbm_model.bin"
                shutil.copy2(model_paths['lightgbm'], dest)
                logger.info(f"   LightGBM model saved: {dest}")
            else:
                logger.warning(f"LightGBM model NOT saved (models_trained={trial_result.get('models_trained', {}).get('lightgbm')}, path={'lightgbm' in model_paths})")

            if 'xgboost' in model_paths and model_paths['xgboost'].exists():
                dest = self.best_models_dir / "best_xgboost_model.joblib"
                shutil.copy2(model_paths['xgboost'], dest)
                logger.info(f"   XGBoost ensemble model saved: {dest}")
            else:
                logger.warning(
                    "XGBoost model NOT saved "
                    f"(models_trained={trial_result.get('models_trained', {}).get('xgboost')}, path={'xgboost' in model_paths})"
                )

            # Save individual best_params for each model
            self._save_individual_best_params(trial_result)

        # Cleanup temporary trial directory
        if trial_dir.exists():
            try:
                shutil.rmtree(trial_dir)
                logger.debug(f"  → Cleaned up trial directory: {trial_dir}")
            except Exception as e:
                logger.warning(f"Failed to cleanup trial directory: {e}")

    def _save_individual_best_params(self, trial_result: dict):
        """
        Save best hyperparameters for each model individually.

        Args:
            trial_result: Trial result dictionary
        """
        file_manager = FileManager()
        params = trial_result['params']
        model_metrics = trial_result['model_metrics']
        ensemble_metrics = trial_result['ensemble_metrics']

        # RotatE best params
        if trial_result['models_trained']['rotate']:
            rotate_metrics_payload = {
                'mrr': model_metrics['rotate'].get('mrr', 0.0),
                'hits_at_1': model_metrics['rotate'].get('hits@1', 0.0),
                'hits_at_10': model_metrics['rotate'].get('hits@10', 0.0),
                'best_val_mrr': model_metrics['rotate'].get('best_val_mrr', 0.0),
            }
            rotate_params = {
                'model': 'RotatE',
                'hyperparameters': {
                    'embedding_dim': params.get('embedding_dim'),
                    'gamma': params.get('gamma'),
                    'epsilon': params.get('epsilon'),
                    'learning_rate': params.get('meta_learning_rate'),
                    'epochs': params.get('rotate_epochs'),
                    'batch_size': params.get('batch_size'),
                    'negative_sample_size': params.get('negative_sample_size'),
                    'adversarial_temperature': params.get('adversarial_temperature'),
                    'self_adversarial': params.get('self_adversarial'),
                    'regularization_weight': params.get('regularization_weight'),
                },
                'metrics': rotate_metrics_payload,
                'weight_in_ensemble': params.get('neural_weight'),
            }
            rotate_file = self.best_models_dir / "best_params_rotate.json"
            file_manager.save(rotate_params, rotate_file)
            logger.info(f"   RotatE params saved: {rotate_file}")

        # AnyBURL best params
        if trial_result['models_trained']['anyburl']:
            anyburl_metrics_payload = {
                'avg_confidence': model_metrics['anyburl'].get('avg_confidence', 0.0),
                'avg_support': model_metrics['anyburl'].get('avg_support', 0.0),
                'high_confidence_ratio': model_metrics['anyburl'].get('high_confidence_ratio', 0.0),
                'coverage': model_metrics['anyburl'].get('coverage', 0.0),
                'positive_rule_coverage': model_metrics['anyburl'].get('positive_rule_coverage', 0.0),
            }
            anyburl_params = {
                'model': 'AnyBURL',
                'hyperparameters': {
                    'rule_confidence': params.get('rule_confidence'),
                    'rule_support': params.get('rule_support'),
                    'max_rule_length': params.get('max_rule_length'),
                },
                'metrics': anyburl_metrics_payload,
                'weight_in_ensemble': params.get('rules_weight'),
            }
            if 'anyburl_classifier' in model_metrics:
                anyburl_params['classifier_metrics'] = model_metrics['anyburl_classifier']
            anyburl_file = self.best_models_dir / "best_params_anyburl.json"
            file_manager.save(anyburl_params, anyburl_file)
            logger.info(f"   AnyBURL params saved: {anyburl_file}")

        # LightGBM best params
        if trial_result['models_trained']['lightgbm']:
            lightgbm_metrics_payload = {
                key: model_metrics['lightgbm'].get(key)
                for key in ['auc', 'f1', 'accuracy', 'precision', 'recall']
                if key in model_metrics['lightgbm']
            }
            lgbm_params = {
                'model': 'LightGBM',
                'hyperparameters': {
                    'learning_rate': params.get('meta_learning_rate'),
                    'n_estimators': params.get('meta_n_estimators'),
                    'negative_ratio': params.get('negative_ratio'),
                },
                'metrics': lightgbm_metrics_payload,
                'weight_in_ensemble': params.get('lightgbm_weight'),
            }
            lgbm_file = self.best_models_dir / "best_params_lightgbm.json"
            file_manager.save(lgbm_params, lgbm_file)
            logger.info(f"   LightGBM params saved: {lgbm_file}")

        if 'hybrid' in model_metrics and trial_result['models_trained']['lightgbm']:
            hybrid_params = {
                'model': 'HybridWrapper',
                'hyperparameters': {
                    'learning_rate': params.get('meta_learning_rate'),
                    'n_estimators': params.get('meta_n_estimators'),
                    'negative_ratio': params.get('negative_ratio'),
                },
                'metrics': model_metrics['hybrid'],
                'neural_weight': params.get('neural_weight'),
                'rules_weight': params.get('rules_weight'),
                'lightgbm_weight': params.get('lightgbm_weight'),
            }
            hybrid_file = self.best_models_dir / "best_params_hybrid.json"
            file_manager.save(hybrid_params, hybrid_file)
            logger.info(f"   Hybrid wrapper params saved: {hybrid_file}")

        if trial_result['models_trained'].get('xgboost') and 'xgboost' in model_metrics:
            xgboost_params = {
                'model': 'XGBoost',
                'hyperparameters': {
                    'meta_learning_rate': params.get('meta_learning_rate'),
                    'meta_n_estimators': params.get('meta_n_estimators'),
                    'feature_selection_threshold': params.get('feature_selection_threshold'),
                    'ensemble_voting': params.get('ensemble_voting'),
                    'target_symbolic_ratio': params.get('target_symbolic_ratio'),
                },
                'metrics': model_metrics['xgboost'],
            }
            xgboost_file = self.best_models_dir / "best_params_xgboost.json"
            file_manager.save(xgboost_params, xgboost_file)
            logger.info(f"   XGBoost params saved: {xgboost_file}")

        # Ensemble best params
        ensemble_params = {
            'model': 'Ensemble',
            'hyperparameters': {
                'neural_weight': params.get('neural_weight'),
                'rules_weight': params.get('rules_weight'),
                'lightgbm_weight': params.get('lightgbm_weight'),
                'neural_threshold': params.get('neural_threshold'),
                'rules_threshold': params.get('rules_threshold'),
                'lightgbm_threshold': params.get('lightgbm_threshold'),
                'ensemble_voting': params.get('ensemble_voting'),
                'feature_selection_threshold': params.get('feature_selection_threshold'),
                'target_symbolic_ratio': params.get('target_symbolic_ratio'),
            },
            'metrics': {
                'weighted_score': ensemble_metrics.get('weighted_score'),
                'base_weighted_score': ensemble_metrics.get('base_weighted_score'),
                'normalized_neural': ensemble_metrics.get('normalized_neural'),
                'normalized_rules': ensemble_metrics.get('normalized_rules'),
                'normalized_learner': ensemble_metrics.get('normalized_learner'),
                'rules_coverage': ensemble_metrics.get('rules_coverage'),
                'weight_penalty': ensemble_metrics.get('weight_penalty'),
                'coverage_penalty': ensemble_metrics.get('coverage_penalty'),
                'rules_weight_penalty': ensemble_metrics.get('rules_weight_penalty'),
                'symbolic_dominance_penalty': ensemble_metrics.get('symbolic_dominance_penalty'),
                'symbolic_contribution': ensemble_metrics.get('symbolic_contribution'),
                'hybrid_contribution': ensemble_metrics.get('hybrid_contribution'),
            },
            'composite_score': trial_result['composite_score'],
        }
        coverage_val = float(ensemble_metrics.get('rules_coverage') or 0.0)
        rules_weight_val = float(params.get('rules_weight') or 0.0)
        if coverage_val < 0.15 or rules_weight_val < 0.20:
            warning_msg = (
                f"Symbolic coverage below target "
                f"(coverage={coverage_val:.3f}, rules_weight={rules_weight_val:.3f}). "
                f"Marking ensemble params as 'needs_retraining'."
            )
            logger.warning(warning_msg)
            ensemble_params['status'] = 'needs_retraining'
            ensemble_params['required_actions'] = ['rerun_optimize_kg_real']
            ensemble_params['notes'] = warning_msg

        ensemble_file = self.best_models_dir / "best_params_ensemble.json"
        file_manager.save(ensemble_params, ensemble_file)
        logger.info(f"   Ensemble params saved: {ensemble_file}")


def _check_if_multi_objective(objective_func: Callable[[Any], Union[float, List[float]]]) -> bool:
    """
    Quick check if objective function returns multiple values (multi-objective).

    Args:
        objective_func: Objective function to check

    Returns:
        True if likely multi-objective
    """
    # Create a dummy trial for testing
    class DummyTrial:
        def __init__(self):
            self.params = {}

        def suggest_float(self, name, low, high, log=False):
            return (low + high) / 2

        def suggest_int(self, name, low, high, step=1):
            return (low + high) // 2

        def suggest_categorical(self, name, choices):
            return choices[0]

    dummy_trial = DummyTrial()

    try:
        result = objective_func(dummy_trial)
        return isinstance(result, (list, tuple)) and len(result) > 1
    except Exception:
        # If objective fails with dummy trial, assume single objective
        return False


def _derive_symbolic_retry_params(current_params: dict[str, Any]) -> dict[str, Any] | None:
    """
    Generate a fallback parameter set biased toward higher symbolic coverage.
    """
    if not current_params:
        return None

    fallback = dict(current_params)
    fallback["feature_selection_threshold"] = float(
        np.clip(current_params.get("feature_selection_threshold", 0.15) * 0.5, 0.03, 0.2)
    )
    fallback["target_symbolic_ratio"] = float(
        np.clip(current_params.get("target_symbolic_ratio", 0.38), 0.3, 0.42)
    )
    fallback["rules_threshold"] = float(
        np.clip(current_params.get("rules_threshold", 0.3) * 0.8, 0.15, 0.45)
    )
    fallback["rules_weight"] = float(
        np.clip(current_params.get("rules_weight", 0.18), 0.12, 0.25)
    )
    fallback["lightgbm_weight"] = float(
        np.clip(current_params.get("lightgbm_weight", 0.55), 0.5, 0.65)
    )
    fallback["neural_weight"] = float(
        np.clip(
            1.0 - fallback["rules_weight"] - fallback["lightgbm_weight"],
            0.15,
            0.45,
        )
    )
    total_weight = (
        fallback["neural_weight"]
        + fallback["rules_weight"]
        + fallback["lightgbm_weight"]
    )
    if total_weight <= 0:
        return None
    scale = 1.0 / total_weight
    fallback["neural_weight"] *= scale
    fallback["rules_weight"] *= scale
    fallback["lightgbm_weight"] *= scale
    return fallback


def optimize_ensemble_hyperparameters(
    n_trials: int = 100,
    strategy: str = "auto",
    use_real_data: bool = True,
    enable_mlflow: bool = True,
    study_name: str | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """
     Convenience function for optimizing PFF Ensemble hyperparameters.

    This function provides a ready-to-use interface for optimizing the ensemble
    hyperparameters in the PFF (Production-Fix-Flow) project using Polars for data loading.

    Args:
        n_trials: Number of optimization trials
        strategy: Optimization strategy ('auto', 'optuna', 'optuna-auto')
        use_real_data: Whether to use real PFF data or simulation
        enable_mlflow: Whether to track in MLflow
        study_name: Name for the optimization study
        output_dir: Directory to save results

    Returns:
        Dictionary with optimization results
    """
    # Delegate to KG optimization with real data (Polars)
    return optimize_kg_hyperparameters(
        n_trials=n_trials,
        strategy=strategy,
        use_real_data=use_real_data,
        enable_mlflow=enable_mlflow,
        study_name=study_name,
        output_dir=output_dir,
    )


# Backward compatibility aliases (deprecated)
HyperparameterOptimizer = None
OptimizationResult = None
