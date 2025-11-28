#!/usr/bin/env python3
"""
Optuna Strategy Implementation - SOTA Framework

Implements optimization using Optuna, the state-of-the-art hyperparameter
optimization framework with advanced pruning, visualization, and integration.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .base import BaseOptimizerStrategy, OptimizationConfig, TrialResult, OptimizationResult
from pff.utils import logger


class OptunaStrategy(BaseOptimizerStrategy):
    """
    Optuna-based optimization strategy.

    Features:
    - Advanced pruning (Median, Hyperband, SuccessiveHalving)
    - Modern samplers (TPE, CMA-ES, NSGA-II)
    - Native visualizations
    - MLflow integration
    - Multi-objective optimization
    """

    def __init__(self, config: OptimizationConfig):
        """Initialize Optuna strategy."""
        super().__init__(config)
        self._framework_name = "optuna"
        self._study_name = config.study_name or f"optuna_study_{int(time.time())}"

        # Try to import optuna
        try:
            import optuna
            self.optuna = optuna
        except ImportError:
            raise ImportError(
                "Optuna not installed. Install with: pip install optuna"
            )

        # Storage backend
        self.storage = None
        if config.storage_url:
            self.storage = self.optuna.storages.RDBStorage(config.storage_url)

    def create_study(self) -> Any:
        """
        Create Optuna study with optimal configuration.

        Returns:
            Optuna study object
        """
        # Create study with direction
        direction = 'maximize' if self.config.direction == 'maximize' else 'minimize'

        if self.storage:
            self.study = self.optuna.create_study(
                study_name=self._study_name,
                storage=self.storage,
                direction=direction,
                load_if_exists=True,
            )
        else:
            self.study = self.optuna.create_study(
                study_name=self._study_name,
                direction=direction,
            )

        # Configure sampler (TPE with multivariate for best results)
        sampler = self.optuna.samplers.TPESampler(
            seed=self.config.random_state,
            n_startup_trials=10,
            n_ei_candidates=24,
            multivariate=True,  # Consider parameter interactions
        )
        self.study.sampler = sampler

        # Configure pruner based on config.pruner_type
        if self.config.enable_pruning:
            pruner = self._create_pruner()
            self.study.pruner = pruner

        logger.info(f"Estudo Optuna criado: {self._study_name}")
        logger.info(f"Sampler: {sampler.__class__.__name__}")
        if self.config.enable_pruning:
            logger.info(f"Pruner: {pruner.__class__.__name__}")

        return self.study

    def _create_pruner(self) -> Any:
        """
        Create pruner based on configuration.

        Supports:
        - "hyperband": Default, best for large search spaces
        - "median": Standard pruner
        - "wilcoxon": SOTA for k-fold cross-validation (Optuna v3.6.0+)

        Returns:
            Optuna pruner instance
        """
        pruner_type = getattr(self.config, 'pruner_type', 'hyperband')

        if pruner_type == "wilcoxon":
            # SOTA: WilcoxonPruner for k-fold cross-validation
            try:
                import warnings
                from optuna.pruners import WilcoxonPruner
                from optuna.exceptions import ExperimentalWarning
                p_threshold = getattr(self.config, 'wilcoxon_p_threshold', 0.1)
                n_startup_steps = getattr(
                    self.config, 'wilcoxon_n_startup_steps', 2
                )
                logger.info(
                    f"Usando WilcoxonPruner SOTA (p_threshold={p_threshold})"
                )
                # Suppress ExperimentalWarning from Optuna - we know it's experimental
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=ExperimentalWarning)
                    return WilcoxonPruner(
                        p_threshold=p_threshold,
                        n_startup_steps=n_startup_steps,
                    )
            except (ImportError, AttributeError):
                logger.warning(
                    "WilcoxonPruner not available (requires Optuna >= 3.6.0), "
                    "falling back to HyperbandPruner"
                )
                return self.optuna.pruners.HyperbandPruner(
                    min_resource=1,
                    max_resource=100,
                    reduction_factor=3,
                )
        elif pruner_type == "median":
            return self.optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=3,
                interval_steps=1,
            )
        else:
            # Default: Hyperband
            return self.optuna.pruners.HyperbandPruner(
                min_resource=1,
                max_resource=100,
                reduction_factor=3,
            )

        return self.study

    def suggest_params(self, trial: Any, search_space: dict[str, Any]) -> dict[str, Any]:
        """
        Suggest hyperparameters using Optuna's trial API.

        Args:
            trial: Optuna trial object
            search_space: Dictionary defining search space

        Returns:
            Dictionary of suggested parameters
        """
        params = {}

        for param_name, param_config in search_space.items():
            if isinstance(param_config, (list, tuple)):
                # Categorical parameter
                if len(param_config) > 0:
                    if all(isinstance(x, (int, float, str)) for x in param_config):
                        # Discrete categorical
                        params[param_name] = trial.suggest_categorical(
                            param_name, param_config
                        )
                    elif len(param_config) == 2:
                        # Numeric range
                        low, high = float(param_config[0]), float(param_config[1])
                        if low < 0 and high > 0 and abs(high / low) > 100:
                            # Likely log scale
                            params[param_name] = trial.suggest_float(
                                param_name, low, high, log=True
                            )
                        else:
                            # Regular float range
                            params[param_name] = trial.suggest_float(
                                param_name, low, high
                            )
                    else:
                        # Custom range specification
                        params[param_name] = trial.suggest_float(
                            param_name, param_config[0], param_config[1]
                        )
            elif isinstance(param_config, dict):
                # Detailed configuration
                param_type = param_config.get('type', 'float')
                if param_type == 'int':
                    if param_config.get('log', False):
                        params[param_name] = trial.suggest_int(
                            param_name,
                            param_config['low'],
                            param_config['high'],
                            log=True,
                        )
                    else:
                        step = param_config.get('step', 1)
                        params[param_name] = trial.suggest_int(
                            param_name,
                            param_config['low'],
                            param_config['high'],
                            step=step,
                        )
                elif param_type == 'float':
                    if param_config.get('log', False):
                        params[param_name] = trial.suggest_float(
                            param_name,
                            param_config['low'],
                            param_config['high'],
                            log=True,
                        )
                    else:
                        params[param_name] = trial.suggest_float(
                            param_name,
                            param_config['low'],
                            param_config['high'],
                        )
                elif param_type == 'categorical':
                    params[param_name] = trial.suggest_categorical(
                        param_name, param_config['choices']
                    )
            else:
                # Default: treat as float in range [0, 1]
                params[param_name] = trial.suggest_float(param_name, 0, 1)

        return params

    def run_optimization(
        self,
        objective_fn: Callable[[Any], Union[float, List[float]]],
        search_space: dict[str, Any],
    ) -> OptimizationResult:
        """
        Run optimization using Optuna.

        Args:
            objective_fn: Objective function
            search_space: Search space definition

        Returns:
            OptimizationResult with best parameters
        """
        if not self.study:
            self.create_study()

        # Create wrapper objective with pruning check
        start_time = time.time()

        def optuna_objective(trial):
            try:
                # Suggest parameters
                params = self.suggest_params(trial, search_space)

                # Log parameters to trial
                for key, value in params.items():
                    trial.set_user_attr(key, value)

                # Call user's objective function
                value = objective_fn(trial)

                # Report intermediate value for pruning
                if isinstance(value, (int, float)):
                    trial.report(value, step=0)

                    # Check if should prune
                    if self.should_prune(trial):
                        raise self.optuna.TrialPruned()

                return value

            except self.optuna.TrialPruned:
                # Re-raise pruning exception
                raise
            except Exception as e:
                logger.error(f"Trial {trial.number} failed: {e}")
                raise

        # Run optimization
        logger.info(
            f"Starting optimization with {self.config.n_trials} trials..."
        )

        try:
            self.study.optimize(
                optuna_objective,
                n_trials=self.config.n_trials,
                timeout=self.config.timeout_seconds,
                show_progress_bar=self.config.show_progress_bar,
                n_jobs=self.config.n_jobs,
            )
        except KeyboardInterrupt:
            logger.warning("Optimization interrupted by user")

        optimization_time = time.time() - start_time

        # Get results
        best_trial = self.study.best_trial

        result = OptimizationResult(
            best_params=best_trial.params,
            best_value=best_trial.value,
            best_trial_number=best_trial.number,
            n_trials=len(self.study.trials),
            trials=self.get_all_trials(),
            study_name=self._study_name,
            optimization_time=optimization_time,
            framework=self.framework_name,
        )

        logger.success(f"Optimization complete in {optimization_time:.2f}s")
        logger.info(f"Best value: {best_trial.value:.4f}")
        logger.info(f"Best params: {best_trial.params}")

        return result

    def get_best_trial(self) -> TrialResult:
        """Get best trial from optimization."""
        if not self.study:
            raise ValueError("No study created. Run optimization first.")

        trial = self.study.best_trial

        return TrialResult(
            params=trial.params,
            value=trial.value,
            trial_number=trial.number,
            state=str(trial.state),
            intermediate_values=trial.intermediate_values,
            user_attrs=trial.user_attrs,
        )

    def get_all_trials(self) -> list[TrialResult]:
        """Get all trials from optimization."""
        if not self.study:
            return []

        trials = []
        for trial in self.study.trials:
            trials.append(
                TrialResult(
                    params=trial.params,
                    value=trial.value if trial.value is not None else float('-inf'),
                    trial_number=trial.number,
                    state=str(trial.state),
                    intermediate_values=trial.intermediate_values,
                    user_attrs=trial.user_attrs,
                )
            )

        return trials

    def get_optimization_history(self) -> list[Tuple[int, float]]:
        """Get optimization history."""
        if not self.study:
            return []

        history = []
        for trial in self.study.trials:
            if trial.value is not None:
                history.append((trial.number, trial.value))

        return history

    def get_param_importances(self) -> dict[str, float]:
        """Get parameter importance scores."""
        if not self.study:
            return {}

        try:
            importances = self.optuna.importance.get_param_importances(
                self.study,
                evaluator=self.optuna.importance.FanovaImportanceEvaluator(),
            )
            return dict(importances)
        except Exception as e:
            logger.warning(f"Failed to calculate parameter importances: {e}")
            return {}

    def _check_pruning_condition(self, trial: Any) -> bool:
        """Check if trial should be pruned (Optuna-specific)."""
        try:
            return trial.should_prune()
        except Exception:
            return False

    def _save_study_impl(self, output_path: Path) -> None:
        """Save Optuna study."""
        if not self.study:
            return

        # Optuna studies with RDB storage are persistent by default
        # For file-based persistence, we can export study
        try:
            study_df = self.study.trials_dataframe()
            study_df.to_csv(output_path / f"{self._study_name}_trials.csv", index=False)
            logger.success(f"Study saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save study: {e}")

    def _load_study_impl(self, input_path: Path) -> None:
        """Load Optuna study."""
        # For RDB storage, just specify the same storage URL
        # For CSV, we would need to recreate trials
        logger.info("Optuna studies with RDB storage load automatically")
        logger.info(f"Use storage URL: {self.config.storage_url}")


class AutoOptunaStrategy(OptunaStrategy):
    """
    Auto-configuring Optuna strategy.

    Automatically selects the best sampler and pruner based on:
    - Search space characteristics
    - Number of trials
    - Multi-objective optimization
    """

    def __init__(self, config: OptimizationConfig, is_multi_objective: bool = False):
        """
        Initialize auto-configuring Optuna strategy.

        Args:
            config: Optimization configuration
            is_multi_objective: Whether using multi-objective optimization
        """
        super().__init__(config)
        self.is_multi_objective = is_multi_objective
        self._framework_name = "optuna-auto"

    def create_study(self) -> Any:
        """
        Create Optuna study with auto-configured sampler and pruner.

        Returns:
            Auto-configured Optuna study
        """
        # Create base study
        super().create_study()

        # Auto-select sampler based on characteristics
        sampler = self._auto_select_sampler()
        self.study.sampler = sampler

        # Auto-select pruner
        if self.config.enable_pruning:
            pruner = self._auto_select_pruner()
            self.study.pruner = pruner

        logger.info(f"Auto-selected sampler: {sampler.__class__.__name__}")
        if self.config.enable_pruning:
            logger.info(f"Auto-selected pruner: {pruner.__class__.__name__}")

        return self.study

    def _auto_select_sampler(self) -> Any:
        """Automatically select best sampler."""
        # For multi-objective, use NSGA-II
        if self.is_multi_objective:
            return self.optuna.samplers.NSGAIISampler(
                seed=self.config.random_state,
                population_size=50,
            )

        # For small search spaces, use CMA-ES
        if self.config.n_trials < 50:
            return self.optuna.samplers.CmaEsSampler(
                seed=self.config.random_state,
                n_startup_trials=5,
            )

        # Default: TPE with multivariate (best for most cases)
        return self.optuna.samplers.TPESampler(
            seed=self.config.random_state,
            n_startup_trials=10,
            n_ei_candidates=24,
            multivariate=True,
        )

    def _auto_select_pruner(self) -> Any:
        """Automatically select best pruner."""
        # For long trials, use Hyperband
        if self.config.timeout_seconds and self.config.timeout_seconds > 600:
            return self.optuna.pruners.HyperbandPruner(
                min_resource=1,
                max_resource=100,
                reduction_factor=3,
            )

        # Default: MedianPruner (robust)
        return self.optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=3,
            interval_steps=1,
        )
