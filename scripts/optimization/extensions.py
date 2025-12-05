#!/usr/bin/env python3
"""
Advanced Optimization Extensions Module

Implements cutting-edge optimization features:
1. Multi-objective optimization (Pareto front)
2. Neural Architecture Search (NAS) integration
3. Distributed optimization with Ray Tune
4. Automated reporting with Optuna Dashboard
5. Hyperparameter importance analysis
6. Transfer learning from previous optimizations

Design Patterns:
- Strategy Pattern: Different optimization approaches
- Observer Pattern: Monitoring and reporting
- Template Method: Common optimization workflow
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import polars as pl
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from pff import settings
from pff.utils import logger
from pff.utils.core.file_manager import FileManager

# Core dependencies
try:
    import optuna
    from optuna.importance import get_param_importances
    from optuna.visualization import (
        plot_optimization_history,
        plot_param_importances,
        plot_pareto_front,
    )
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logger.warning("Optuna not available")

# Optional: Ray Tune for distributed optimization
try:
    import ray
    from ray import tune
    from ray.tune import CLIReporter
    from ray.tune.schedulers import ASHAScheduler
    from ray.tune.search.optuna import OptunaSearch
    RAY_TUNE_AVAILABLE = True
except ImportError:
    RAY_TUNE_AVAILABLE = False
    logger.debug("Ray Tune not available (optional)")

# Optional: Optuna Dashboard
try:
    import optuna_dashboard
    OPTUNA_DASHBOARD_AVAILABLE = True
except ImportError:
    OPTUNA_DASHBOARD_AVAILABLE = False
    logger.debug("Optuna Dashboard not available (optional)")


# ═══════════════════════════════════════════════════════════════════════════
# 1. Multi-Objective Optimization (Pareto Front)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class MultiObjectiveConfig:
    """Configuration for multi-objective optimization."""
    objectives: list[str] = field(default_factory=lambda: ['f1', 'roc_auc', 'precision'])
    weights: list[float] = field(default_factory=lambda: [0.5, 0.3, 0.2])
    direction: list[str] = field(default_factory=lambda: ['maximize', 'maximize', 'maximize'])
    enable_pareto_front: bool = True
    save_pareto_solutions: bool = True


def _get_multi_objective_config() -> dict[str, Any]:
    """Load multi-objective config from optimization.yaml."""
    try:
        fm = FileManager()
        config_path = Path("config/hpo/optimization.yaml")
        if config_path.exists():
            cfg = fm.read(config_path)
            return cfg.get("multi_objective", {})
    except Exception:
        pass
    return {}


class MultiObjectiveOptimizer:
    """
    Multi-objective optimization with Pareto front analysis.

    Uses NSGA-II (Non-dominated Sorting Genetic Algorithm II) for finding
    the Pareto-optimal solutions.

    Design Patterns:
    - Strategy Pattern: Multi-objective optimization strategy
    - Repository Pattern: Pareto front storage
    """

    def __init__(self, config: MultiObjectiveConfig = None):
        self.config = config or MultiObjectiveConfig()
        self.file_manager = FileManager()

        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna required for multi-objective optimization")

    def create_study(self, study_name: str = None) -> optuna.Study:
        """
        Create multi-objective study.

        Args:
            study_name: Optional name for the study

        Returns:
            Optuna study configured for multi-objective optimization
        """
        study_name = study_name or f"multi_obj_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Load config-driven NSGA-II parameters
        mo_config = _get_multi_objective_config()

        # Create study with multiple objectives
        study = optuna.create_study(
            study_name=study_name,
            directions=self.config.direction,
            sampler=optuna.samplers.NSGAIISampler(
                population_size=mo_config.get("population_size", 50),
                mutation_prob=mo_config.get("mutation_prob", 0.1),
                crossover_prob=mo_config.get("crossover_prob", 0.9),
                seed=42,
            ),
        )

        logger.info(f"Estudo multiobjetivo criado: {study_name}")
        logger.info(f"Objetivos: {self.config.objectives}")

        return study

    def extract_pareto_front(
        self,
        study: optuna.Study,
    ) -> tuple[list[optuna.trial.FrozenTrial], pl.DataFrame]:
        """
        Extract Pareto-optimal solutions from study.

        Args:
            study: Optuna study

        Returns:
            Tuple of (pareto trials, pareto DataFrame)
        """
        # Get Pareto-optimal trials
        pareto_trials = study.best_trials

        # Create DataFrame with Pareto solutions
        pareto_data = []
        for trial in pareto_trials:
            row = {
                'trial_number': trial.number,
                **{f'objective_{i}': val for i, val in enumerate(trial.values)},
                **trial.params,
            }
            pareto_data.append(row)

        pareto_df = pl.DataFrame(pareto_data)

        logger.success(f"Encontradas {len(pareto_trials)} soluções Pareto-ótimas")

        return pareto_trials, pareto_df

    def save_pareto_front(
        self,
        study: optuna.Study,
        output_dir: str = None,
    ) -> Path:
        """Save Pareto front visualization and data."""
        if output_dir is None:
            output_dir = settings.OUTPUTS_DIR / "hyperopt" / "pareto"

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save Pareto trials
        pareto_trials, pareto_df = self.extract_pareto_front(study)

        # Save as CSV
        csv_file = output_path / f"pareto_front_{study.study_name}.csv"
        pareto_df.to_csv(csv_file, index=False)

        # Save visualization (if plotly available)
        try:
            fig = plot_pareto_front(study, target_names=self.config.objectives)
            html_file = output_path / f"pareto_front_{study.study_name}.html"
            fig.write_html(str(html_file))
            logger.success(f"Visualizacao da frente de Pareto salva: {html_file}")
        except Exception as e:
            logger.warning(f"Could not save Pareto visualization: {e}")

        logger.success(f"Dados da frente de Pareto salvos: {csv_file}")

        return csv_file


# ═══════════════════════════════════════════════════════════════════════════
# 2. Neural Architecture Search (NAS) Integration
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class NASConfig:
    """Configuration for Neural Architecture Search."""
    search_space: dict[str, Any] = field(default_factory=lambda: {
        'n_layers': (1, 5),
        'hidden_units': (32, 512),
        'dropout_rate': (0.0, 0.5),
        'activation': ['relu', 'tanh', 'sigmoid'],
        'optimizer': ['adam', 'sgd', 'rmsprop'],
        'learning_rate': (1e-4, 1e-2),
    })
    max_epochs: int = 50
    early_stopping_patience: int = 10


class NeuralArchitectureSearch:
    """
    Neural Architecture Search integration.

    Searches for optimal neural network architectures using Optuna.
    """

    def __init__(self, config: NASConfig = None):
        self.config = config or NASConfig()

        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna required for NAS")

    def suggest_architecture(self, trial: optuna.Trial) -> dict[str, Any]:
        """
        Suggest neural network architecture.

        Args:
            trial: Optuna trial

        Returns:
            Dictionary with architecture configuration
        """
        n_layers = trial.suggest_int('n_layers', *self.config.search_space['n_layers'])

        architecture = {
            'n_layers': n_layers,
            'layers': [],
        }

        # Suggest layers
        for i in range(n_layers):
            layer_config = {
                'hidden_units': trial.suggest_int(
                    f'layer_{i}_units',
                    *self.config.search_space['hidden_units'],
                    log=True,
                ),
                'dropout_rate': trial.suggest_float(
                    f'layer_{i}_dropout',
                    *self.config.search_space['dropout_rate'],
                ),
                'activation': trial.suggest_categorical(
                    f'layer_{i}_activation',
                    self.config.search_space['activation'],
                ),
            }
            architecture['layers'].append(layer_config)

        # Suggest optimizer
        architecture['optimizer'] = trial.suggest_categorical(
            'optimizer',
            self.config.search_space['optimizer'],
        )

        architecture['learning_rate'] = trial.suggest_float(
            'learning_rate',
            *self.config.search_space['learning_rate'],
            log=True,
        )

        return architecture

    def build_model(self, architecture: dict[str, Any]) -> Any:
        """
        Build neural network from architecture specification.

        Note: This is a placeholder. Actual implementation would use
        TensorFlow/PyTorch to build the model.

        Args:
            architecture: Architecture configuration

        Returns:
            Model object
        """
        # Placeholder - would create actual model
        logger.info(f"Construindo modelo com {architecture['n_layers']} camadas")
        return architecture


# ═══════════════════════════════════════════════════════════════════════════
# 3. Distributed Optimization with Ray Tune
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class DistributedOptimizationConfig:
    """Configuration for distributed optimization."""
    num_samples: int = 100
    max_concurrent_trials: int = 4
    cpu_per_trial: int = 2
    gpu_per_trial: float = 0.0
    use_ray_tune: bool = True


class DistributedOptimizer:
    """
    Distributed hyperparameter optimization using Ray Tune.

    Enables parallel optimization across multiple CPUs/GPUs.
    """

    def __init__(self, config: DistributedOptimizationConfig = None):
        self.config = config or DistributedOptimizationConfig()

        if self.config.use_ray_tune and not RAY_TUNE_AVAILABLE:
            logger.warning("Ray Tune not available, falling back to sequential")
            self.config.use_ray_tune = False

    def setup_ray(self) -> None:
        """Initialize Ray cluster."""
        if not self.config.use_ray_tune:
            return

        try:
            ray.init(ignore_reinit_error=True)
            logger.success("Cluster Ray inicializado")
        except Exception as e:
            logger.error(f"Failed to initialize Ray: {e}")
            self.config.use_ray_tune = False

    def create_search_space(self) -> dict[str, Any]:
        """Create Ray Tune search space."""
        return {
            'min_confidence_threshold': tune.loguniform(0.01, 0.20),
            'max_violation_percentage': tune.uniform(50.0, 300.0),
            'xgb_max_depth': tune.randint(2, 7),
            'xgb_learning_rate': tune.loguniform(0.01, 0.3),
            'xgb_n_estimators': tune.choice([50, 100, 150, 200, 250, 300]),
            'xgb_subsample': tune.uniform(0.6, 1.0),
            'xgb_colsample_bytree': tune.uniform(0.3, 0.8),
        }

    def run_distributed(
        self,
        objective_fn: Callable,
        search_space: dict[str, Any] = None,
    ) -> tune.ResultGrid:
        """
        Run distributed optimization.

        Args:
            objective_fn: Function to optimize
            search_space: Hyperparameter search space

        Returns:
            Ray Tune results
        """
        if not self.config.use_ray_tune:
            raise RuntimeError("Ray Tune not available")

        self.setup_ray()

        search_space = search_space or self.create_search_space()

        # Create Optuna search algorithm
        search_alg = OptunaSearch(
            metric="score",
            mode="max",
        )

        # Create ASHA scheduler for early stopping
        scheduler = ASHAScheduler(
            max_t=100,
            grace_period=10,
            reduction_factor=3,
        )

        # Setup reporter
        reporter = CLIReporter(
            metric_columns=["score", "f1", "roc_auc", "training_iteration"],
        )

        # Run optimization
        logger.info(f"Iniciando otimização distribuída com {self.config.num_samples} trials")

        tuner = tune.Tuner(
            tune.with_resources(
                objective_fn,
                resources={
                    "cpu": self.config.cpu_per_trial,
                    "gpu": self.config.gpu_per_trial,
                },
            ),
            tune_config=tune.TuneConfig(
                metric="score",
                mode="max",
                search_alg=search_alg,
                scheduler=scheduler,
                num_samples=self.config.num_samples,
                max_concurrent_trials=self.config.max_concurrent_trials,
            ),
            run_config=tune.RunConfig(
                progress_reporter=reporter,
                verbose=1,
            ),
            param_space=search_space,
        )

        results = tuner.fit()

        logger.success("Otimizacao distribuida concluida!")

        return results


# ═══════════════════════════════════════════════════════════════════════════
# 4. Automated Reporting with Optuna Dashboard
# ═══════════════════════════════════════════════════════════════════════════

class OptunaReporting:
    """
    Automated reporting and visualization with Optuna Dashboard.

    Provides interactive web interface for monitoring optimization.
    """

    def __init__(self):
        self.file_manager = FileManager()

    def start_dashboard(
        self,
        storage_url: str = None,
        port: int = 8080,
        host: str = "127.0.0.1",
    ) -> None:
        """
        Start Optuna Dashboard server.

        Args:
            storage_url: SQLite/MySQL/PostgreSQL URL for study storage
            port: Dashboard port
            host: Dashboard host
        """
        if not OPTUNA_DASHBOARD_AVAILABLE:
            logger.error("Optuna Dashboard not installed. Install with: pip install optuna-dashboard")
            return

        # Default to SQLite storage
        if storage_url is None:
            db_path = settings.OUTPUTS_DIR / "hyperopt" / "optuna_studies.db"
            db_path.parent.mkdir(parents=True, exist_ok=True)
            storage_url = f"sqlite:///{db_path}"

        logger.info(f"Iniciando Optuna Dashboard em http://{host}:{port}")
        logger.info(f"Armazenamento: {storage_url}")

        try:
            optuna_dashboard.run_server(
                storage=storage_url,
                host=host,
                port=port,
            )
        except Exception as e:
            logger.error(f"Failed to start dashboard: {e}")

    def generate_report(
        self,
        study: optuna.Study,
        output_dir: str = None,
    ) -> Path:
        """
        Generate comprehensive optimization report.

        Args:
            study: Optuna study
            output_dir: Output directory

        Returns:
            Path to report HTML
        """
        if output_dir is None:
            output_dir = settings.OUTPUTS_DIR / "hyperopt" / "reports"

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        report_file = output_path / f"report_{study.study_name}.html"

        # Generate visualizations
        try:
            from optuna.visualization import (
                plot_contour,
                plot_edf,
                plot_intermediate_values,
                plot_optimization_history,
                plot_parallel_coordinate,
                plot_param_importances,
                plot_slice,
            )

            # Create HTML report
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Optimization Report - {study.study_name}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .plot {{ margin: 20px 0; }}
                    h1, h2 {{ color: #333; }}
                    .metrics {{ background: #f0f0f0; padding: 15px; border-radius: 5px; }}
                </style>
            </head>
            <body>
                <h1>Hyperparameter Optimization Report</h1>
                <p><strong>Study:</strong> {study.study_name}</p>
                <p><strong>Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

                <div class="metrics">
                    <h2>Best Trial</h2>
                    <p><strong>Number:</strong> {study.best_trial.number}</p>
                    <p><strong>Value:</strong> {study.best_trial.value:.4f}</p>
                    <p><strong>Parameters:</strong></p>
                    <ul>
            """

            for key, value in study.best_trial.params.items():
                html_content += f"<li><strong>{key}:</strong> {value}</li>\n"

            html_content += """
                    </ul>
                </div>

                <div class="plot">
                    <h2>Optimization History</h2>
                    <div id="history"></div>
                </div>

                <div class="plot">
                    <h2>Parameter Importances</h2>
                    <div id="importances"></div>
                </div>
            </body>
            </html>
            """

            # Save report (AGENTS.md §4.1)
            self.file_manager.save(html_content, report_file)

            logger.success(f"Relatório gerado: {report_file}")

        except Exception as e:
            logger.error(f"Failed to generate report: {e}")

        return report_file


# ═══════════════════════════════════════════════════════════════════════════
# 5. Hyperparameter Importance Analysis
# ═══════════════════════════════════════════════════════════════════════════

class ImportanceAnalyzer:
    """
    Hyperparameter importance analysis using fANOVA.

    Identifies which hyperparameters have the most impact on performance.
    """

    def __init__(self):
        self.file_manager = FileManager()

    def analyze_importance(
        self,
        study: optuna.Study,
        top_k: int = 10,
    ) -> pl.DataFrame:
        """
        Analyze hyperparameter importance.

        Args:
            study: Optuna study
            top_k: Number of top parameters to analyze

        Returns:
            DataFrame with importance scores
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna required for importance analysis")

        try:
            # Calculate importances using fANOVA
            importances = get_param_importances(
                study,
                evaluator=optuna.importance.FanovaImportanceEvaluator(),
            )

            # Create DataFrame
            importance_df = pl.DataFrame([
                {'parameter': param, 'importance': score}
                for param, score in importances.items()
            ]).sort_values('importance', ascending=False)

            logger.info(f"Top {top_k} hiperparametros mais importantes:")
            for _, row in importance_df.head(top_k).iterrows():
                logger.info(f"  {row['parameter']}: {row['importance']:.4f}")

            return importance_df

        except Exception as e:
            logger.error(f"Failed to analyze importance: {e}")
            return pl.DataFrame()

    def save_importance_analysis(
        self,
        study: optuna.Study,
        output_dir: str = None,
    ) -> Path:
        """Save importance analysis results."""
        if output_dir is None:
            output_dir = settings.OUTPUTS_DIR / "hyperopt" / "importance"

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Analyze importance
        importance_df = self.analyze_importance(study)

        # Save as CSV
        csv_file = output_path / f"importance_{study.study_name}.csv"
        importance_df.to_csv(csv_file, index=False)

        # Save visualization
        try:
            fig = plot_param_importances(study)
            html_file = output_path / f"importance_{study.study_name}.html"
            fig.write_html(str(html_file))
            logger.success(f"Visualizacao de importancia salva: {html_file}")
        except Exception as e:
            logger.warning(f"Could not save visualization: {e}")

        logger.success(f"Analise de importancia salva: {csv_file}")

        return csv_file


# ═══════════════════════════════════════════════════════════════════════════
# 6. Transfer Learning from Previous Optimizations
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TransferLearningConfig:
    """Configuration for transfer learning."""
    enable_transfer: bool = True
    n_previous_trials: int = 20
    similarity_threshold: float = 0.7
    warmstart_trials: int = 10


class TransferLearningOptimizer:
    """
    Transfer learning from previous optimization runs.

    Reuses knowledge from past optimizations to speed up new ones.
    """

    def __init__(self, config: TransferLearningConfig = None):
        self.config = config or TransferLearningConfig()
        self.file_manager = FileManager()
        self.history_file = settings.OUTPUTS_DIR / "hyperopt" / "optimization_history.pkl"
        self.history_file.parent.mkdir(parents=True, exist_ok=True)

    def load_history(self) -> list[dict[str, Any]]:
        """Load optimization history from previous runs."""
        if not self.history_file.exists():
            logger.info("Nenhum historico de otimizacao anterior encontrado")
            return []

        try:
            history = self.file_manager.read(self.history_file)
            logger.info(f"Carregados {len(history)} execuções de otimização anteriores")
            return history
        except Exception as e:
            logger.error(f"Failed to load optimization history: {e}")
            return []

    def save_history(self, study: optuna.Study) -> None:
        """Save optimization results to history."""
        history = self.load_history()

        # Extract study data
        study_data = {
            'study_name': study.study_name,
            'timestamp': datetime.now(),
            'best_params': study.best_trial.params,
            'best_value': study.best_trial.value,
            'n_trials': len(study.trials),
            'trials': [
                {
                    'number': t.number,
                    'params': t.params,
                    'value': t.value,
                    'state': str(t.state),
                }
                for t in study.trials
            ],
        }

        history.append(study_data)

        # Save updated history (AGENTS.md §4.1)
        self.file_manager.save(history, self.history_file)

        logger.success(f"Histórico de otimização salvo: {self.history_file}")

    def get_warmstart_params(
        self,
        search_space: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """
        Get warm-start parameters from previous optimizations.

        Args:
            search_space: Current search space

        Returns:
            List of promising parameter configurations
        """
        history = self.load_history()

        if not history:
            return []

        # Extract best trials from previous runs
        warmstart_params = []

        for study_data in history[-self.config.n_previous_trials:]:
            # Get top trials from this study
            trials = sorted(
                study_data['trials'],
                key=lambda t: t['value'],
                reverse=True,
            )[:self.config.warmstart_trials]

            for trial in trials:
                # Filter params to match current search space
                params = {
                    k: v for k, v in trial['params'].items()
                    if k in search_space
                }

                if params:
                    warmstart_params.append(params)

        logger.info(f"Encontradas {len(warmstart_params)} configurações de warm-start")

        return warmstart_params[:self.config.warmstart_trials]

    def create_warmstart_sampler(
        self,
        warmstart_params: list[dict[str, Any]],
    ) -> optuna.samplers.BaseSampler:
        """
        Create sampler with warm-start from previous optimizations.

        Args:
            warmstart_params: Parameter configurations to warm-start with

        Returns:
            Optuna sampler
        """
        if not warmstart_params:
            return optuna.samplers.TPESampler(seed=42)

        # Create TPE sampler with warm-start
        sampler = optuna.samplers.TPESampler(
            seed=42,
            n_startup_trials=len(warmstart_params),
            multivariate=True,
        )

        logger.success(f"Sampler de warm-start criado com {len(warmstart_params)} trials")

        return sampler
