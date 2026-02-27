#!/usr/bin/env python3
"""
Optuna Strategy Implementation - SOTA Framework

Implements optimization using Optuna, the state-of-the-art hyperparameter
optimization framework with advanced pruning, visualization, and integration.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pff.shared import load_config, logger
from pff.shared.core.config import OPTIMIZATION_CONFIG_PATH
from pff.shared.core.file_manager import FileManager
from pff.shared.ops.global_interrupt_manager import check_interruption

from .base import (
    BaseOptimizerStrategy,
    OptimizationConfig,
    OptimizationResult,
    TrialResult,
)


def _load_sampler_config() -> dict[str, Any]:
    """Load sampler config from YAML with caching."""
    cfg = load_config(OPTIMIZATION_CONFIG_PATH)
    return cfg.get("sampler", {})


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

        try:
            import optuna

            self.optuna: Any = optuna
        except ImportError:
            raise ImportError("Optuna not installed. Install with: pip install optuna")

        self.storage: Any = None
        self.study: Any = None
        if config.storage_url:
            self.storage = self.optuna.storages.RDBStorage(config.storage_url)

    def create_study(self) -> Any:
        """
        Create Optuna study with optimal configuration.

        Returns:
            Optuna study object
        """
        direction = "maximize" if self.config.direction == "maximize" else "minimize"

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

        sampler_config = _load_sampler_config()
        sampler = self.optuna.samplers.TPESampler(
            seed=self.config.random_state,
            n_startup_trials=sampler_config.get("n_startup_trials", 10),
            n_ei_candidates=sampler_config.get("n_ei_candidates", 48),
            multivariate=sampler_config.get("multivariate", True),
            group=sampler_config.get("group", True),
            constant_liar=sampler_config.get("constant_liar", True),
            consider_prior=sampler_config.get("consider_prior", True),
            consider_magic_clip=sampler_config.get("consider_magic_clip", True),
        )
        self.study.sampler = sampler

        pruner: Any = None
        if self.config.enable_pruning:
            pruner = self._create_pruner()
            self.study.pruner = pruner

        logger.info(f"Estudo Optuna criado: {self._study_name}")
        logger.info(f"Amostrador: {sampler.__class__.__name__}")
        if self.config.enable_pruning and pruner:
            logger.info(f"Podador: {pruner.__class__.__name__}")

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
        pruner_type = getattr(self.config, "pruner_type", "hyperband")

        if pruner_type == "wilcoxon":
            try:
                import warnings

                from optuna.exceptions import ExperimentalWarning
                from optuna.pruners import WilcoxonPruner

                p_threshold = getattr(self.config, "wilcoxon_p_threshold", 0.1)
                n_startup_steps = getattr(self.config, "wilcoxon_n_startup_steps", 2)
                logger.info(f"Usando WilcoxonPruner (p_threshold={p_threshold})")
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
            return self.optuna.pruners.HyperbandPruner(
                min_resource=1,
                max_resource=100,
                reduction_factor=3,
            )

        return self.study

    def suggest_params(
        self, trial: Any, search_space: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Suggest hyperparameters using Optuna's trial API.

        Args:
            trial: Optuna trial object
            search_space: Dictionary defining search space

        Returns:
            Dictionary of suggested parameters
        """
        params: dict[str, Any] = {}
        for param_name, param_config in search_space.items():
            params[param_name] = self._suggest_param(trial, param_name, param_config)
        return params

    def run_optimization(
        self,
        objective_fn: Callable[[Any], float | list[float]],
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

        start_time = time.time()
        interrupted = False
        optuna_objective = self._build_objective(objective_fn, search_space)

        logger.info(f"Iniciando otimizacao com {self.config.n_trials} trials...")

        try:
            self.study.optimize(
                optuna_objective,
                n_trials=self.config.n_trials,
                timeout=self.config.timeout_seconds,
                show_progress_bar=self.config.show_progress_bar,
                n_jobs=self.config.n_jobs,
            )
        except KeyboardInterrupt:
            interrupted = True
            logger.warning("Optimization interrupted by user")

        optimization_time = time.time() - start_time

        trials = self.get_all_trials()
        best_trial = self._resolve_best_trial(trials)
        best_params = best_trial.params if best_trial else {}
        best_value = (
            best_trial.value if best_trial and best_trial.value is not None else 0.0
        )
        best_trial_number = best_trial.trial_number if best_trial else -1

        result = OptimizationResult(
            best_params=best_params,
            best_value=best_value,
            best_trial_number=best_trial_number,
            n_trials=len(trials),
            trials=trials,
            study_name=self._study_name,
            optimization_time=optimization_time,
            framework=self.framework_name,
        )

        if interrupted:
            logger.info("Otimizacao interrompida graciosamente")
        else:
            logger.success(f"Otimização concluída em {optimization_time:.2f}s")
            logger.info(f"Melhor valor: {best_value:.4f}")
            logger.info(f"Melhores parametros: {best_params}")

        return result

    def _suggest_param(self, trial: Any, name: str, config: Any) -> Any:
        """Execute suggest param.



        Args:

            trial: Input value used by this callable.

            name: Input value used by this callable.

            config: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        if isinstance(config, (list, tuple)):
            return self._suggest_from_sequence(trial, name, config)
        if isinstance(config, dict):
            return self._suggest_from_dict(trial, name, config)
        return trial.suggest_float(name, 0, 1)

    def _suggest_from_sequence(
        self, trial: Any, name: str, config: list[Any] | tuple[Any, ...]
    ) -> Any:
        """Execute suggest from sequence.



        Args:

            trial: Input value used by this callable.

            name: Input value used by this callable.

            config: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        if len(config) == 2 and all(isinstance(x, (int, float)) for x in config):
            low, high = float(config[0]), float(config[1])
            return trial.suggest_float(name, low, high)
        if len(config) > 0:
            return trial.suggest_categorical(name, list(config))
        return trial.suggest_float(name, 0, 1)

    def _suggest_from_dict(self, trial: Any, name: str, config: dict[str, Any]) -> Any:
        """Execute suggest from dict.



        Args:

            trial: Input value used by this callable.

            name: Input value used by this callable.

            config: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        param_type = config.get("type", "float")
        if param_type == "int":
            return self._suggest_int(trial, name, config)
        if param_type == "float":
            return self._suggest_float(trial, name, config)
        if param_type == "categorical":
            return trial.suggest_categorical(name, config["choices"])
        return trial.suggest_float(name, 0, 1)

    def _suggest_int(self, trial: Any, name: str, config: dict[str, Any]) -> Any:
        """Execute suggest int.



        Args:

            trial: Input value used by this callable.

            name: Input value used by this callable.

            config: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        if config.get("log", False):
            return trial.suggest_int(name, config["low"], config["high"], log=True)
        step = config.get("step", 1)
        return trial.suggest_int(name, config["low"], config["high"], step=step)

    def _suggest_float(self, trial: Any, name: str, config: dict[str, Any]) -> Any:
        """Execute suggest float.



        Args:

            trial: Input value used by this callable.

            name: Input value used by this callable.

            config: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        if config.get("log", False):
            return trial.suggest_float(name, config["low"], config["high"], log=True)
        return trial.suggest_float(name, config["low"], config["high"])

    def _build_objective(
        self,
        objective_fn: Callable[[Any], float | list[float]],
        search_space: dict[str, Any],
    ) -> Callable[[Any], float | list[float]]:
        """Execute build objective.



        Args:

            objective_fn: Input value used by this callable.

            search_space: Input value used by this callable.



        Returns:

            Return value produced by the callable.



        Raises:

            Exception: Propagates domain-specific failures with context.

        """

        def optuna_objective(trial: Any) -> float | list[float]:
            """Execute optuna objective.



            Args:

                trial: Input value used by this callable.



            Returns:

                Return value produced by the callable.



            Raises:

                Exception: Propagates domain-specific failures with context.



            Notes:

                Keep behavior deterministic and free of hidden side effects.

            """

            try:
                check_interruption()
                params = self.suggest_params(trial, search_space)
                for key, value in params.items():
                    trial.set_user_attr(key, value)
                return objective_fn(trial)
            except self.optuna.TrialPruned:
                raise
            except Exception as exc:
                logger.error(f"Trial {trial.number} failed: {exc}")
                raise

        return optuna_objective

    def _resolve_best_trial(self, trials: list[TrialResult]) -> TrialResult | None:
        """Execute resolve best trial.



        Args:

            trials: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        if self.study and getattr(self.study, "trials", None):
            try:
                return self.get_best_trial()
            except Exception as exc:
                logger.warning(f"Failed to fetch best trial after interruption: {exc}")
                if trials:
                    return trials[0]
        return None

    def get_best_trial(self) -> TrialResult:
        """Get best trial from optimization."""
        if not self.study:
            raise ValueError("No study created. Run optimization first.")

        trial = self.study.best_trial

        return TrialResult(
            params=trial.params,
            value=trial.value or 0.0,
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
                    value=trial.value if trial.value is not None else float("-inf"),
                    trial_number=trial.number,
                    state=str(trial.state),
                    intermediate_values=trial.intermediate_values,
                    user_attrs=trial.user_attrs,
                )
            )

        return trials

    def get_optimization_history(self) -> list[tuple[int, float]]:
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

        try:
            study_df = self.study.trials_dataframe()
            import polars as pl

            FileManager.ensure_dir(output_path)
            output_file = output_path / f"{self._study_name}_trials.parquet"
            polars_df = pl.from_pandas(study_df, include_index=False)
            FileManager.save(polars_df, output_file)
            logger.success(f"Estudo salvo em {output_path}")
        except Exception as e:
            logger.error(f"Failed to save study: {e}")

    def _load_study_impl(self, input_path: Path) -> None:
        """Load Optuna study."""
        logger.info("Estudos Optuna com storage RDB sao carregados automaticamente")
        logger.info(f"Use a URL de storage: {self.config.storage_url}")


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
        super().create_study()

        sampler = self._auto_select_sampler()
        self.study.sampler = sampler

        pruner: Any = None
        if self.config.enable_pruning:
            pruner = self._auto_select_pruner()
            self.study.pruner = pruner

        logger.info(
            f"Amostrador selecionado automaticamente: {sampler.__class__.__name__}"
        )
        if self.config.enable_pruning and pruner:
            logger.info(
                f"Podador selecionado automaticamente: {pruner.__class__.__name__}"
            )

        return self.study

    def _auto_select_sampler(self) -> Any:
        """Automatically select best sampler."""
        if self.is_multi_objective:
            return self.optuna.samplers.NSGAIISampler(
                seed=self.config.random_state,
                population_size=50,
            )

        try:
            import importlib

            optuna_hub = importlib.import_module("optuna_hub")

            module = optuna_hub.load_module(package="samplers/auto_sampler")
            logger.info("AutoSampler optuna_hub habilitado")
            return module.AutoSampler()
        except Exception as exc:
            logger.debug(f"AutoSampler unavailable ({exc}); using default heuristic")

        if self.config.n_trials < 50:
            return self.optuna.samplers.CmaEsSampler(
                seed=self.config.random_state,
                n_startup_trials=5,
            )

        sampler_config = _load_sampler_config()
        return self.optuna.samplers.TPESampler(
            seed=self.config.random_state,
            multivariate=sampler_config.get("multivariate", True),
        )

    def _auto_select_pruner(self) -> Any:
        """Automatically select best pruner."""
        if self.config.timeout_seconds and self.config.timeout_seconds > 600:
            return self.optuna.pruners.HyperbandPruner(
                min_resource=1,
                max_resource=100,
                reduction_factor=3,
            )

        return self.optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=3,
            interval_steps=1,
        )
