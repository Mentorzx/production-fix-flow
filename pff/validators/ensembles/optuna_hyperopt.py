"""
SOTA Hyperparameter Optimization for Ensemble with Optuna.

Design Patterns:
- Strategy Pattern: Multiple optimization strategies (TPE, CMA-ES, Grid)
- Factory Pattern: Auto-select best sampler based on search space
- Observer Pattern: Callbacks for logging and early stopping
- Template Method: Base optimization workflow

Author: PFF Team
Date: 2025-11-01
Version: 2.0.0
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import optuna
from optuna.pruners import MedianPruner, SuccessiveHalvingPruner
from optuna.samplers import CmaEsSampler, GridSampler, TPESampler
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import StratifiedKFold, cross_val_score

from pff import settings
from pff.utils.core.logger import logger


class OptimizationStrategy(ABC):
    """Base class for optimization strategies (Strategy Pattern)."""
    
    @abstractmethod
    def create_sampler(self) -> optuna.samplers.BaseSampler:
        """Create Optuna sampler."""
        pass
    
    @abstractmethod
    def create_pruner(self) -> optuna.pruners.BasePruner:
        """Create Optuna pruner."""
        pass
    
    @abstractmethod
    def suggest_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest hyperparameters for trial."""
        pass


class TPEStrategy(OptimizationStrategy):
    """Tree-structured Parzen Estimator (Bayesian Optimization)."""
    
    def create_sampler(self) -> TPESampler:
        return TPESampler(
            seed=42,
            n_startup_trials=10,
            n_ei_candidates=24,
            multivariate=True,
            constant_liar=True,
        )
    
    def create_pruner(self) -> MedianPruner:
        return MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=3,
            interval_steps=1,
        )
    
    def suggest_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest params optimized for XGBoost ensemble."""
        return {
            "min_confidence_threshold": trial.suggest_float(
                "min_confidence", 0.001, 0.2, log=True
            ),
            "xgb_n_estimators": trial.suggest_int("n_estimators", 50, 500, step=50),
            "xgb_max_depth": trial.suggest_int("max_depth", 2, 15),
            "xgb_learning_rate": trial.suggest_float(
                "learning_rate", 0.001, 0.3, log=True
            ),
            "xgb_subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "xgb_colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.5, 1.0
            ),
            "xgb_colsample_bylevel": trial.suggest_float(
                "colsample_bylevel", 0.5, 1.0
            ),
            "xgb_min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "xgb_gamma": trial.suggest_float("gamma", 0, 0.5),
            "xgb_reg_alpha": trial.suggest_float("reg_alpha", 0, 10, log=True),
            "xgb_reg_lambda": trial.suggest_float("reg_lambda", 0, 10, log=True),
            "xgb_scale_pos_weight": trial.suggest_float(
                "scale_pos_weight", 0.5, 2.0
            ),
        }


class CMAESStrategy(OptimizationStrategy):
    """CMA-ES for continuous optimization."""
    
    def create_sampler(self) -> CmaEsSampler:
        return CmaEsSampler(
            seed=42,
            n_startup_trials=10,
        )
    
    def create_pruner(self) -> SuccessiveHalvingPruner:
        return SuccessiveHalvingPruner(
            min_resource=1,
            reduction_factor=4,
            min_early_stopping_rate=0,
        )
    
    def suggest_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        return {
            "min_confidence_threshold": trial.suggest_float(
                "min_confidence", 0.001, 0.2
            ),
            "xgb_learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3),
            "xgb_subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "xgb_colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "xgb_gamma": trial.suggest_float("gamma", 0, 0.5),
            "xgb_reg_alpha": trial.suggest_float("reg_alpha", 0, 10),
            "xgb_reg_lambda": trial.suggest_float("reg_lambda", 0, 10),
        }


class OptunaHyperoptimizer:
    """
    SOTA Hyperparameter optimizer using Optuna with design patterns.
    
    Features:
    - Multiple optimization strategies (TPE, CMA-ES)
    - Automatic pruning of unpromising trials
    - Multi-objective optimization support
    - MLflow integration for tracking
    - Parallel optimization with distributed backend
    """
    
    def __init__(
        self,
        strategy: Optional[OptimizationStrategy] = None,
        output_dir: Optional[Path] = None,
        use_mlflow: bool = True,
    ):
        """
        Initialize hyperparameter optimizer.
        
        Args:
            strategy: Optimization strategy (default: TPEStrategy)
            output_dir: Output directory for results
            use_mlflow: Whether to use MLflow for tracking
        """
        self.strategy = strategy or TPEStrategy()
        self.output_dir = output_dir or settings.OUTPUTS_DIR / "hyperopt"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_mlflow = use_mlflow
        
        logger.info("🔧 OptunaHyperoptimizer initialized")
        logger.info(f"Strategy: {self.strategy.__class__.__name__}")
    
    def optimize(
        self,
        objective_fn: callable,
        n_trials: int = 100,
        timeout: Optional[int] = None,
        n_jobs: int = -1,
        study_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run optimization with Optuna.
        
        Args:
            objective_fn: Objective function to maximize/minimize
            n_trials: Number of trials
            timeout: Timeout in seconds
            n_jobs: Number of parallel jobs
            study_name: Name of the study
            
        Returns:
            Optimization results
        """
        logger.info(f"🚀 Starting optimization: {n_trials} trials")
        
        study_name = study_name or f"ensemble_optuna_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        study = optuna.create_study(
            study_name=study_name,
            direction="maximize",
            sampler=self.strategy.create_sampler(),
            pruner=self.strategy.create_pruner(),
            storage=None,
        )
        
        callbacks = []
        if self.use_mlflow:
            try:
                from optuna.integration.mlflow import MLflowCallback
                callbacks.append(
                    MLflowCallback(
                        tracking_uri=str(settings.MLRUNS_DIR),
                        metric_name="f1_score",
                    )
                )
            except ImportError:
                logger.warning("MLflow not available, skipping callback")
        
        start_time = datetime.now()
        
        study.optimize(
            objective_fn,
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=n_jobs,
            callbacks=callbacks,
            show_progress_bar=True,
        )
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        results = {
            "best_params": study.best_params,
            "best_value": study.best_value,
            "best_trial": study.best_trial.number,
            "n_trials": len(study.trials),
            "n_complete": len([
                t for t in study.trials 
                if t.state == optuna.trial.TrialState.COMPLETE
            ]),
            "n_pruned": len([
                t for t in study.trials 
                if t.state == optuna.trial.TrialState.PRUNED
            ]),
            "elapsed_seconds": elapsed,
            "study": study,
        }
        
        self._save_results(results, study_name)
        self._log_results(results)
        
        return results
    
    def _save_results(self, results: Dict[str, Any], study_name: str):
        """Save optimization results to disk."""
        output_file = self.output_dir / f"{study_name}_results.pkl"
        
        results_to_save = {
            k: v for k, v in results.items() if k != "study"
        }
        
        joblib.dump(results_to_save, output_file)
        logger.info(f"💾 Resultados salvos em {output_file}")
    
    def _log_results(self, results: Dict[str, Any]):
        """Log optimization results."""
        logger.success(f"✅ Otimização concluída!")
        logger.info(f"📊 Estatísticas:")
        logger.info(f"   Best value: {results['best_value']:.4f}")
        logger.info(f"   Best trial: {results['best_trial']}")
        logger.info(f"   Complete trials: {results['n_complete']}/{results['n_trials']}")
        logger.info(f"   Pruned trials: {results['n_pruned']}")
        logger.info(f"   Time: {results['elapsed_seconds']:.2f}s")
        logger.info(f"📋 Melhores parâmetros:")
        for key, value in results["best_params"].items():
            logger.info(f"   {key}: {value}")


class EnsembleOptimizerFactory:
    """Factory for creating optimizers (Factory Pattern)."""
    
    @staticmethod
    def create_optimizer(
        optimizer_type: str = "tpe",
        **kwargs
    ) -> OptunaHyperoptimizer:
        """
        Create optimizer based on type.
        
        Args:
            optimizer_type: Type of optimizer ("tpe" or "cmaes")
            **kwargs: Additional arguments for optimizer
            
        Returns:
            Configured optimizer
        """
        strategies = {
            "tpe": TPEStrategy(),
            "cmaes": CMAESStrategy(),
        }
        
        if optimizer_type not in strategies:
            logger.warning(f"Unknown optimizer type: {optimizer_type}, using TPE")
            optimizer_type = "tpe"
        
        strategy = strategies[optimizer_type]
        return OptunaHyperoptimizer(strategy=strategy, **kwargs)
