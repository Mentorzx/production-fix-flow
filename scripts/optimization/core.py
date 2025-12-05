"""
Core Hyperparameter Optimization Module - SOTA "Zero-Touch" Implementation
with REAL PFF Data Integration (Using Polars)

Design Patterns Used:
- Template Method: Orchestrates complete optimization workflow
- Strategy Pattern: Different optimization algorithms (TPE, CMA-ES, Hyperband)
- Factory Pattern: Creates samplers, pruners, and strategies
- Observer Pattern: Monitors trial progress via callbacks
- Memento Pattern: PersistentBestTrialMemory saves/restores best trial state
- Callback Pattern: BestModelSaverCallback hooks into trial completion
- Registry Pattern: StrategyRegistry for auto-discovery of strategies
- Singleton Pattern: StrategyRegistry and config loaders
- Parameter Object: TrialEvaluationConfig encapsulates trial parameters

MLOps Integration:
- Automatically selects best framework (Optuna SOTA)
- Integrates with MLflow for complete experiment tracking
- Generates comprehensive visualizations
- Saves best parameters automatically
- USES REAL PFF DATA (Knowledge Graph triplets)

SOTA Features:
- TPESampler with n_ei_candidates=48, multivariate=True, group=True
- WilcoxonPruner for k-fold CV scenarios
- Embedding caching to avoid redundant training
- Lazy loading with Polars scan_parquet

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
import numbers
import os
import random
import time
import warnings

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable
from pff import settings
from pff.config import OPTIMIZATION_CONFIG_PATH, RULE_FILTER_CONFIG_PATH, RULE_FILTER_HPO_CONFIG_PATH
from pff.utils import logger

from .spaces import TuningConfig, SearchSpaceFactory
from .strategies import StrategyFactory
from .strategies.base import BaseOptimizerStrategy, OptimizationConfig
from .callbacks import CallbackManager, OptimizationObserver
from .extensions import ImportanceAnalyzer
from .tracker import MLflowTracker
from .visualizer import OptimizationVisualizer
from .trials.artifacts import TrialArtifactManager
from .trials.config_loader import load_ensemble_hpo_bounds
from .trials.data_loader import load_real_kg_data
from .trials.pipeline import evaluate_trial
from .trials.objective import kg_objective
from .trials.study import create_study_and_run

import atexit
import gc

from pff.utils.core.file_manager import FileManager
from pff.db.connection import close_connection_pool
from .trials.utils import is_cuda_safe, cleanup_resources
atexit.register(cleanup_resources)

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
from pff.validators.ensembles.data_loader import EnsembleDataLoader
from pff.validators.ensembles.advanced_trainer import AdvancedEnsembleTrainer, SymbolicBalanceError
from pff.validators.ensembles.ensemble_wrappers.transformers import SymbolicCoverageError
from .trials.constants import KGE_MODEL_ROTATE
DEFAULT_KGE_MODEL = KGE_MODEL_ROTATE

_checkpoint_file_manager = FileManager()


def _load_checkpoint(checkpoint_path: Path) -> dict[str, Any] | None:
    """Load checkpoint using FileManager (AGENTS.md §5).
    
    Args:
        checkpoint_path: Path to checkpoint JSON file.
        
    Returns:
        Checkpoint data or None if not found/invalid.
    """
    if not checkpoint_path.exists():
        return None
    try:
        return _checkpoint_file_manager.read(checkpoint_path)
    except Exception as exc:
        logger.warning(f"Could not read checkpoint {checkpoint_path}: {exc}")
        return None


def _write_checkpoint(checkpoint_path: Path, payload: dict[str, Any]) -> None:
    """Write checkpoint using FileManager (AGENTS.md §5).
    
    Args:
        checkpoint_path: Path to save checkpoint.
        payload: Data to persist.
    """
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    _checkpoint_file_manager.save(payload, checkpoint_path)


def _delete_directory(path: Path) -> None:
    """Delete directory tree safely using FileManager.
    
    Args:
        path: Directory path to remove.
    """
    _checkpoint_file_manager.delete_directory(path, ignore_errors=True)


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
                    system_attrs={"warmstart_seed": True},
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
        output_dir: Directory to save results (default: ./outputs/optimization/<study>)
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

    # Set output directory (save to outputs/optimization/<study>, not root)
    if not output_dir:
        output_dir = settings.OUTPUTS_DIR / "optimization" / study_name
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
        default_tracking = settings.OUTPUTS_DIR / "optimization" / "mlruns"
        mlflow_tracker = MLflowTracker(
            experiment_name=study_name,
            tracking_uri=storage_url or str(default_tracking),
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
        plots_dir = settings.OUTPUTS_DIR / "optimization" / "plots" / study_name
        visualizer = OptimizationVisualizer(output_dir=plots_dir)

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

        logger.success(f"Melhores parametros salvos em: {best_params_file}")

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
        train_df, valid_df, data_info = load_real_kg_data()
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
    ensemble_hpo_bounds = load_ensemble_hpo_bounds(file_manager)

    artifact_manager = TrialArtifactManager()
    symbolic_retry_state: dict[str, int] = {"enqueues": 0}
    max_symbolic_retry_enqueues = int(os.getenv("PFF_SYMBOLIC_MAX_RETRIES", "6"))

    # Create output directory and trial_runs_dir BEFORE defining objective_fn
    if not output_dir:
        output_dir = settings.OUTPUTS_DIR / "optimization" / "kg_ensemble"
    output_dir.mkdir(parents=True, exist_ok=True)
    trial_runs_dir = output_dir / "trials"
    trial_runs_dir.mkdir(parents=True, exist_ok=True)

    def objective_fn(trial):
        return kg_objective(
            trial,
            train_df=train_df,
            valid_df=valid_df,
            target_entity_ratio=target_entity_ratio,
            trial_runs_dir=trial_runs_dir,
            rule_filter=rule_filter,
            hpo_ranges=hpo_ranges,
            file_manager=file_manager,
            artifact_manager=artifact_manager,
            symbolic_retry_state=symbolic_retry_state,
            max_symbolic_retry_enqueues=max_symbolic_retry_enqueues,
        )

    # Run optimization using Optuna directly
    # Note: Don't use find_best_hyperparameters here because kg_objective already defines the search space
    import optuna

    study_name = study_name or f"pff_kg_optimization_{int(time.time())}"

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

    result = create_study_and_run(
        study_name=study_name,
        storage_path=storage_path,
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        n_trials=n_trials,
        expected_trials=expected_trials,
        resume_mode=resume_mode,
        checkpoint_data=checkpoint_data,
        hpo_memory_config=hpo_memory_config,
        trial_memory=trial_memory,
        warmstart_callback=trial_memory.warmstart_study,
        objective_fn=objective_fn,
        artifact_manager=artifact_manager,
        enable_mlflow=enable_mlflow,
        file_manager=file_manager,
    )

    result['real_data_info'] = data_info
    result['kge_model'] = kge_model

    if enable_visualization:
        logger.info("\nGerando gráficos de visualização...")
        try:
            vis_output_dir = output_dir or (settings.OUTPUTS_DIR / "optimization" / "plots" / "kg_ensemble")
            logger.info(f"  → Diretório de saída: {vis_output_dir}")
            visualizer = OptimizationVisualizer(output_dir=vis_output_dir)
            artifacts = visualizer.generate_all_plots(result, study=result.get('study'))
            if artifacts:
                logger.success(f" Gerados {len(artifacts)} gráficos de visualização")
                result['visualization_plots'] = artifacts
                result['output_dir'] = vis_output_dir
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

    best_models_dir = output_dir / "best_models"
    result.setdefault("best_model_files", {})

    anyburl_rules = best_models_dir / "anyburl" / "rules.tsv"
    if anyburl_rules.exists():
        result['best_model_files']['anyburl'] = anyburl_rules
        logger.info(f"   AnyBURL armazenado em: {anyburl_rules}")

    lgbm_model = best_models_dir / "best_lightgbm_model.bin"
    if lgbm_model.exists():
        result['best_model_files']['lightgbm'] = lgbm_model
        logger.info(f"   LightGBM armazenado em: {lgbm_model}")

    logger.info("\nHiperparâmetros salvos:")
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


class BestModelSaverCallback:
    """
    Optuna callback to save the best models after each trial.

    This callback:
    1. Checks if current trial is the best so far
    2. If best: copies models to permanent storage
    3. If not best: deletes temporary files
    4. Saves individual best_params for each model
    """

    def __init__(
        self,
        output_dir: Path,
        memory: PersistentBestTrialMemory | None = None,
        artifact_manager: TrialArtifactManager | None = None,
    ):
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
        self.artifact_manager = artifact_manager or TrialArtifactManager()

    def __call__(self, study, trial):
        """
        Called after each trial completes.

        Args:
            study: Optuna study object
            trial: Completed trial object
        """
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

        trial_result = self.artifact_manager.get_trial_result(trial)

        if trial_result is None:
            logger.warning(f"Could not find trial result for trial #{trial.number}")
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

            logger.success(f"   NOVO MELHOR TRIAL #{trial.number}: {trial.value:.4f}")
            logger.info(f"  → Salvando melhores modelos em: {self.best_models_dir}")

            # Debug logging for trial result
            logger.debug(f"  → Trial result models_trained: {trial_result.get('models_trained')}")
            logger.debug(f"  → Trial result model_paths keys: {list(trial_result.get('model_paths', {}).keys())}")

            # Delegate to artifact manager (DRY principle)
            self.artifact_manager.persist_best_models(self.best_models_dir, trial_result)
            self.artifact_manager.persist_best_params(self.best_models_dir, trial_result)

            # Save additional HPO-specific params
            self._save_additional_hpo_params(trial_result)

        # Cleanup temporary trial directory
        try:
            self.artifact_manager.cleanup_trial_dir(trial_dir)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Failed to cleanup trial directory: {e}")

    def _save_additional_hpo_params(self, trial_result: dict):
        """
        Save HPO-specific parameters not covered by TrialArtifactManager.

        Args:
            trial_result: Trial result dictionary
        """
        file_manager = FileManager()
        params = trial_result['params']
        model_metrics = trial_result['model_metrics']
        ensemble_metrics = trial_result.get('ensemble_metrics', {})

        # Hybrid wrapper params (HPO-specific combination)
        if 'hybrid' in model_metrics and trial_result['models_trained'].get('lightgbm'):
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
            logger.info(f"   Parametros hybrid wrapper salvos: {hybrid_file}")

        # XGBoost ensemble params (HPO-specific)
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
            logger.info(f"   Parametros XGBoost salvos: {xgboost_file}")

        # Full ensemble params with all HPO tuned values
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

        ensemble_file = self.best_models_dir / "best_params_full_ensemble.json"
        file_manager.save(ensemble_params, ensemble_file)
        logger.info(f"   Parametros full ensemble salvos: {ensemble_file}")


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
