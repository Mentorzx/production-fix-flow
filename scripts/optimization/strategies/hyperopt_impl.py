#!/usr/bin/env python3
"""
Hyperopt Strategy Implementation - Legacy Support

Implements optimization using Hyperopt for backward compatibility.
Note: Optuna is SOTA and recommended for new projects.
"""

from __future__ import annotations

import time
import pickle
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .base import BaseOptimizerStrategy, OptimizationConfig, TrialResult, OptimizationResult
from pff.utils import logger
from pff.utils.ops.global_interrupt_manager import check_interruption


class HyperoptStrategy(BaseOptimizerStrategy):
    """
    Hyperopt-based optimization strategy.

    Features:
    - TPE and Annealing samplers
    - Basic parallelization support
    - Limited visualization

    Note: Optuna is SOTA. Use Hyperopt only for legacy compatibility.
    """

    def __init__(self, config: OptimizationConfig):
        """Initialize Hyperopt strategy."""
        super().__init__(config)
        self._framework_name = "hyperopt"

        # Try to import hyperopt
        try:
            import hyperopt
            from hyperopt import fmin, tpe, hp, anneal, Trials, STATUS_OK, STATUS_FAIL
            self.hyperopt = hyperopt
            self.fmin = fmin
            self.tpe = tpe
            self.hp = hp
            self.anneal = anneal
            self.Trials = Trials
            self.STATUS_OK = STATUS_OK
            self.STATUS_FAIL = STATUS_FAIL
        except ImportError:
            raise ImportError(
                "Hyperopt not installed. Install with: pip install hyperopt"
            )

        self.trials = None
        self.best_params = None
        self.best_loss = None

    def create_study(self) -> Any:
        """
        Create Hyperopt trials object.

        Returns:
            Hyperopt Trials object
        """
        self.trials = self.Trials()

        logger.info("Objeto de trials do Hyperopt criado")
        logger.info("Observacao: Optuna e recomendado para mais recursos")

        return self.trials

    def suggest_params(self, trial: Any, search_space: dict[str, Any]) -> dict[str, Any]:
        """
        Note: In Hyperopt, parameter suggestion happens in the objective function.
        This method is not used in the standard Hyperopt workflow.

        Returns:
            Empty dict (parameters suggested in objective)
        """
        logger.warning(
            "Hyperopt suggests parameters in the objective function, "
            "not via trial.suggest_params()"
        )
        return {}

    def _convert_search_space(self, search_space: dict[str, Any]) -> dict[str, Any]:
        """
        Convert search space to Hyperopt format.

        Args:
            search_space: Optuna-style search space

        Returns:
            Hyperopt-style search space
        """
        hp_space = {}

        for param_name, param_config in search_space.items():
            if isinstance(param_config, (list, tuple)):
                if len(param_config) == 2:
                    low, high = float(param_config[0]), float(param_config[1])
                    if low < 0 and high > 0 and abs(high / low) > 100:
                        # Log scale
                        hp_space[param_name] = self.hp.lognormal(
                            param_name, (low + high) / 2, (high - low) / 4
                        )
                    else:
                        # Uniform
                        hp_space[param_name] = self.hp.uniform(
                            param_name, low, high
                        )
                elif len(param_config) > 2:
                    # Categorical
                    hp_space[param_name] = self.hp.choice(
                        param_name, param_config
                    )
            elif isinstance(param_config, dict):
                param_type = param_config.get('type', 'float')
                if param_type == 'int':
                    if param_config.get('log', False):
                        hp_space[param_name] = self.hp.lognormal(
                            param_name,
                            (param_config['low'] + param_config['high']) / 2,
                            (param_config['high'] - param_config['low']) / 4,
                        )
                    else:
                        step = param_config.get('step', 1)
                        if step == 1:
                            hp_space[param_name] = self.hp.randint(
                                param_name, param_config['low'], param_config['high'] + 1
                            )
                        else:
                            hp_space[param_name] = self.hp.uniform(
                                param_name, param_config['low'], param_config['high']
                            )
                elif param_type == 'float':
                    if param_config.get('log', False):
                        hp_space[param_name] = self.hp.lognormal(
                            param_name,
                            (param_config['low'] + param_config['high']) / 2,
                            (param_config['high'] - param_config['low']) / 4,
                        )
                    else:
                        hp_space[param_name] = self.hp.uniform(
                            param_name,
                            param_config['low'],
                            param_config['high'],
                        )
                elif param_type == 'categorical':
                    hp_space[param_name] = self.hp.choice(
                        param_name, param_config['choices']
                    )
            else:
                # Default: uniform [0, 1]
                hp_space[param_name] = self.hp.uniform(param_name, 0, 1)

        return hp_space

    def run_optimization(
        self,
        objective_fn: Callable[[Any], Union[float, List[float]]],
        search_space: dict[str, Any],
    ) -> OptimizationResult:
        """
        Run optimization using Hyperopt.

        Args:
            objective_fn: Objective function (should use Hyperopt's API)
            search_space: Search space definition

        Returns:
            OptimizationResult with best parameters
        """
        if not self.trials:
            self.create_study()

        start_time = time.time()
        interrupted = False

        # Convert search space to Hyperopt format
        hp_space = self._convert_search_space(search_space)

        # Auto-select algorithm
        if self.config.n_trials < 50:
            algo = self.anneal.suggest
            logger.info("Usando algoritmo Annealing")
        else:
            algo = self.tpe.suggest
            logger.info("Usando algoritmo TPE")

        logger.info(f"Iniciando otimizacao Hyperopt com {self.config.n_trials} trials...")

        try:
            # Hyperopt requires objective to return {'loss': value, 'status': status}
            def hyperopt_objective(params):
                try:
                    check_interruption()
                    # Convert to Optuna-style trial-like object
                    class HyperoptTrial:
                        def __init__(self, params):
                            self.params = params

                        def suggest_float(self, name, low, high, log=False):
                            return params[name]

                        def suggest_int(self, name, low, high, step=1):
                            return int(params[name])

                        def suggest_categorical(self, name, choices):
                            idx = params[name]
                            return choices[idx] if isinstance(idx, int) else idx

                    trial = HyperoptTrial(params)

                    value = objective_fn(trial)
                    if not isinstance(value, (int, float)):
                        raise ValueError("Multi-objective not supported in Hyperopt")
                    loss = -value  # Hyperopt minimizes, Optuna maximizes

                    return {
                        'loss': loss,
                        'status': self.STATUS_OK,
                    }

                except Exception as e:
                    logger.error(f"Trial failed: {e}")
                    return {
                        'loss': float('inf'),
                        'status': self.STATUS_FAIL,
                    }

            # Run optimization
            best_params = self.fmin(
                fn=hyperopt_objective,
                space=hp_space,
                algo=algo,
                max_evals=self.config.n_trials,
                trials=self.trials,
                rstate=np.random.RandomState(self.config.random_state),
            )

        except KeyboardInterrupt:
            interrupted = True
            logger.warning("Optimization interrupted by user")

        optimization_time = time.time() - start_time

        trials_result = self.get_all_trials() if self.trials else []
        best_params_result: dict[str, Any] = {}
        best_value_result = 0.0
        best_trial_number = -1

        if self.trials and getattr(self.trials, "trials", None):
            losses = [t["result"]["loss"] for t in self.trials.trials] if self.trials.trials else []
            if losses:
                best_idx = int(np.argmin(losses))
                best_trial_data = self.trials.trials[best_idx]
                best_params_result = best_params if "best_params" in locals() else {}
                best_value_result = -float(best_trial_data["result"]["loss"])
                best_trial_number = best_idx

        result = OptimizationResult(
            best_params=best_params_result,
            best_value=best_value_result,
            best_trial_number=best_trial_number,
            n_trials=len(trials_result),
            trials=trials_result,
            study_name=self._study_name,
            optimization_time=optimization_time,
            framework=self.framework_name,
        )

        if interrupted:
            logger.info("Otimizacao interrompida graciosamente")
        else:
            logger.success(f"Otimizacao concluida em {optimization_time:.2f}s")
            logger.info(f"Melhor valor: {best_value_result:.4f}")
            logger.info(f"Melhores parametros: {best_params_result}")

        return result

    def get_best_trial(self) -> TrialResult:
        """Get best trial from optimization."""
        if not self.trials or self.best_params is None:
            raise ValueError("No optimization run. Run optimization first.")

        return TrialResult(
            params=self.best_params,
            best_value=-self.best_loss,
            best_trial_number=0,
            state='COMPLETE',
        )

    def get_all_trials(self) -> list[TrialResult]:
        """Get all trials from optimization."""
        if not self.trials:
            return []

        trials = []
        for i, trial_data in enumerate(self.trials.trials):
            try:
                trials.append(
                    TrialResult(
                        params=trial_data['misc']['vals'],
                        value=-trial_data['result']['loss'],
                        trial_number=i,
                        state='COMPLETE' if trial_data['result']['status'] == self.STATUS_OK else 'FAIL',
                    )
                )
            except Exception:
                continue

        return trials

    def get_optimization_history(self) -> list[Tuple[int, float]]:
        """Get optimization history."""
        if not self.trials:
            return []

        history = []
        for i, trial_data in enumerate(self.trials.trials):
            try:
                history.append((i, -trial_data['result']['loss']))
            except Exception:
                continue

        return history

    def get_param_importances(self) -> dict[str, float]:
        """
        Get parameter importance scores.

        Note: Hyperopt doesn't have built-in importance calculation.
        Returns empty dict (would need custom implementation).
        """
        logger.warning(
            "Parameter importance not natively supported in Hyperopt. "
            "Consider using Optuna for this feature."
        )
        return {}

    def _check_pruning_condition(self, trial: Any) -> bool:
        """Hyperopt doesn't have pruning."""
        return False

    def _save_study_impl(self, output_path: Path) -> None:
        """Save Hyperopt trials."""
        if not self.trials:
            return

        try:
            self.file_manager.save(self.trials, output_path / f"{self._study_name}_trials.pkl")
            logger.success(f"Estudo salvo em {output_path}")
        except Exception as e:
            logger.error(f"Failed to save Hyperopt study: {e}")

    def _load_study_impl(self, input_path: Path) -> None:
        """Load Hyperopt trials."""
        try:
            self.trials = self.file_manager.read(input_path / f"{self._study_name}_trials.pkl")
            logger.success(f"Estudo carregado de {input_path}")
        except Exception as e:
            logger.error(f"Failed to load Hyperopt study: {e}")
