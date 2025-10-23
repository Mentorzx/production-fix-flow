from __future__ import annotations

import gc
import time
from pathlib import Path
from typing import Any

import mlflow
import optuna
import torch
from optuna.samplers import TPESampler

from pff import settings
from pff.utils import FileManager, logger
from pff.validators.kg.config import KGConfig
from pff.validators.transe.core import TransEManager

"""
TransE Hyperparameter Optimization

This module provides hyperparameter optimization for TransE models
using Optuna framework with MLflow integration.
"""


def objective(
    trial: optuna.Trial, kg_config: KGConfig, base_config_path: Path
) -> float:
    """
    Objective function for Optuna optimization.

    Args:
        trial: Optuna trial object
        kg_config: Knowledge graph configuration
        base_config_path: Base TransE configuration file path

    Returns:
        Validation MRR (higher is better)
    """
    # Suggest hyperparameters
    params = {
        "model": {
            "embedding_dim": trial.suggest_int("embedding_dim", 64, 256, step=32),
            "margin": trial.suggest_float("margin", 0.5, 5.0, step=0.5),
            "norm": trial.suggest_int("norm", 1, 2),
        },
        "training": {
            "epochs": 100,  # Fixed for fair comparison
            "batch_size": trial.suggest_int("batch_size", 128, 1024, step=128),
            "optimizer": {
                "type": trial.suggest_categorical(
                    "optimizer", ["adam", "adamw", "sgd"]
                ),
                "params": {},
            },
            "scheduler": {
                "type": "reduce_on_plateau",
                "params": {"patience": 5, "factor": 0.5},
            },
            "patience": 15,
            "validate_every": 5,
            "num_negatives": trial.suggest_int("num_negatives", 1, 10),
            "normalize_embeddings": trial.suggest_categorical(
                "normalize_embeddings", [True, False]
            ),
            "max_grad_norm": trial.suggest_float("max_grad_norm", 0.5, 5.0),
        },
        "mlflow": {"enabled": True, "experiment_name": "TransE_Hyperopt"},
        "checkpointing": {"save_dir": f"checkpoints/transe/trial_{trial.number}"},
    }

    # Optimizer-specific parameters
    if params["training"]["optimizer"]["type"] == "adam":
        params["training"]["optimizer"]["params"] = {
            "lr": trial.suggest_float("lr_adam", 1e-4, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay_adam", 0.0, 0.01),
        }
    elif params["training"]["optimizer"]["type"] == "adamw":
        params["training"]["optimizer"]["params"] = {
            "lr": trial.suggest_float("lr_adamw", 1e-4, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay_adamw", 0.01, 0.1),
        }
    else:  # sgd
        params["training"]["optimizer"]["params"] = {
            "lr": trial.suggest_float("lr_sgd", 1e-3, 1e-1, log=True),
            "momentum": trial.suggest_float("momentum", 0.0, 0.99),
            "weight_decay": trial.suggest_float("weight_decay_sgd", 0.0, 0.01),
        }

    # Create temporary config file
    trial_config_path = Path(f"config/transe_trial_{trial.number}.yaml")
    file_manager = FileManager()

    # Load base config and update with trial parameters
    base_config = file_manager.read(base_config_path)
    base_config.update(params)

    # Save trial config
    file_manager.save(base_config, trial_config_path)

    try:
        # Initialize manager
        manager = TransEManager(
            transe_config_path=trial_config_path,
            kg_config_path=kg_config.configuration_path,
        )

        # Setup data
        manager._setup_data()

        # Train model
        logger.info(f"\n🔄 Trial {trial.number}: Training with parameters:")
        logger.info(f"   Embedding dim: {params['model']['embedding_dim']}")
        logger.info(f"   Margin: {params['model']['margin']}")
        logger.info(f"   Batch size: {params['training']['batch_size']}")
        logger.info(f"   Optimizer: {params['training']['optimizer']['type']}")

        start_time = time.time()
        training_stats = manager.train()
        training_time = time.time() - start_time

        # Get validation MRR
        val_mrr = training_stats.get("best_val_mrr", 0.0)

        # Log to MLflow
        if mlflow.active_run():
            mlflow.log_metric("val_mrr", val_mrr)
            mlflow.log_metric("training_time", training_time)
            mlflow.log_params(trial.params)

        val_hit1 = training_stats.get("best_val_hit@1", 0.0)
        val_hit3 = training_stats.get("best_val_hit@3", 0.0)
        val_hit10 = training_stats.get("best_val_hit@10", 0.0)

        logger.info(
            f"✅ Trial {trial.number}: Val MRR = {val_mrr:.4f}, "
            f"Hit@1 = {val_hit1:.4f}, Hit@3 = {val_hit3:.4f}, Hit@10 = {val_hit10:.4f}"
        )

        # Log to MLflow
        if mlflow.active_run():
            mlflow.log_metric("val_mrr", val_mrr)
            mlflow.log_metric("val_hit@1", val_hit1)
            mlflow.log_metric("val_hit@3", val_hit3)
            mlflow.log_metric("val_hit@10", val_hit10)
            mlflow.log_metric("training_time", training_time)
            mlflow.log_params(trial.params)

        # Clean up memory
        del manager
        gc.collect()
        torch.cuda.empty_cache()

        # Clean up trial config
        trial_config_path.unlink(missing_ok=True)

        return val_mrr

    except Exception as e:
        logger.error(f"❌ Trial {trial.number} failed: {e}")

        # Clean up
        if trial_config_path.exists():
            trial_config_path.unlink()

        return 0.0


def run_transe_optimization(
    kg_config: KGConfig,
    base_config_path: Path,
    n_trials: int = 50,
    n_jobs: int = 1,
    study_name: str = "transe_optimization",
    storage: str | None = None,
) -> optuna.Study:
    """
    Run hyperparameter optimization for TransE.

    Args:
        kg_config: Knowledge graph configuration
        base_config_path: Base TransE configuration file
        n_trials: Number of optimization trials
        n_jobs: Number of parallel jobs
        study_name: Name of the Optuna study
        storage: Optuna storage URL (uses in-memory if None)

    Returns:
        Optuna study object
    """
    logger.info("🚀 Iniciando otimização de hiperparâmetros TransE")
    logger.info(f"   Trials: {n_trials}")
    logger.info(f"   Jobs paralelos: {n_jobs}")

    # Create sampler
    sampler = TPESampler(seed=42)

    # Create or load study
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=sampler,
        storage=storage,
        load_if_exists=True,
    )

    # Set MLflow experiment
    mlflow.set_experiment("TransE_Hyperopt")

    # Run optimization
    study.optimize(
        lambda trial: objective(trial, kg_config, base_config_path),
        n_trials=n_trials,
        n_jobs=n_jobs,
        gc_after_trial=True,
    )

    # Log best results
    logger.info("\n" + "=" * 60)
    logger.info("🏆 MELHORES RESULTADOS")
    logger.info("=" * 60)
    logger.info(f"Melhor trial: #{study.best_trial.number}")
    logger.info(f"Melhor MRR: {study.best_value:.4f}")
    logger.info("\n📊 Melhores parâmetros:")

    for param, value in study.best_params.items():
        logger.info(f"   {param}: {value}")

    # Save best configuration
    save_best_config(study, base_config_path)

    # Create visualization
    create_optimization_plots(study)

    return study


def save_best_config(study: optuna.Study, base_config_path: Path) -> Path:
    """
    Save the best configuration found during optimization.

    Args:
        study: Completed Optuna study
        base_config_path: Base configuration file path

    Returns:
        Path to the saved optimized configuration
    """
    file_manager = FileManager()
    base_config = file_manager.read(base_config_path)

    # Extract best parameters
    best_params = study.best_params

    # Update configuration
    optimized_config = base_config.copy()

    # Model parameters
    optimized_config["model"]["embedding_dim"] = best_params["embedding_dim"]
    optimized_config["model"]["margin"] = best_params["margin"]
    optimized_config["model"]["norm"] = best_params["norm"]

    # Training parameters
    optimized_config["training"]["batch_size"] = best_params["batch_size"]
    optimized_config["training"]["num_negatives"] = best_params["num_negatives"]
    optimized_config["training"]["normalize_embeddings"] = best_params[
        "normalize_embeddings"
    ]
    optimized_config["training"]["max_grad_norm"] = best_params["max_grad_norm"]

    # Optimizer parameters
    optimizer_type = best_params["optimizer"]
    optimized_config["training"]["optimizer"]["type"] = optimizer_type

    if optimizer_type == "adam":
        optimized_config["training"]["optimizer"]["params"] = {
            "lr": best_params["lr_adam"],
            "weight_decay": best_params["weight_decay_adam"],
        }
    elif optimizer_type == "adamw":
        optimized_config["training"]["optimizer"]["params"] = {
            "lr": best_params["lr_adamw"],
            "weight_decay": best_params["weight_decay_adamw"],
        }
    else:  # sgd
        optimized_config["training"]["optimizer"]["params"] = {
            "lr": best_params["lr_sgd"],
            "momentum": best_params["momentum"],
            "weight_decay": best_params["weight_decay_sgd"],
        }

    # Save optimized configuration
    output_path = settings.CONFIG_DIR / "transe_optimized.yaml"
    file_manager.save(optimized_config, output_path)

    logger.info(f"✅ Configuração otimizada salva em: {output_path}")

    return output_path


def create_optimization_plots(study: optuna.Study) -> None:
    """
    Create and save optimization visualization plots.

    Args:
        study: Completed Optuna study
    """
    try:
        import optuna.visualization as vis

        output_dir = settings.OUTPUTS_DIR / "transe" / "hyperopt"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Optimization history
        fig = vis.plot_optimization_history(study)
        fig.write_html(str(output_dir / "optimization_history.html"))

        # Parameter importance
        fig = vis.plot_param_importances(study)
        fig.write_html(str(output_dir / "param_importance.html"))

        # Parallel coordinate plot
        fig = vis.plot_parallel_coordinate(study)
        fig.write_html(str(output_dir / "parallel_coordinate.html"))

        # Slice plot
        fig = vis.plot_slice(study)
        fig.write_html(str(output_dir / "slice_plot.html"))

        logger.info(f"✅ Visualizações salvas em: {output_dir}")

    except Exception as e:
        logger.warning(f"⚠️ Erro ao criar visualizações: {e}")


def analyze_study_results(study: optuna.Study) -> dict[str, Any]:
    """
    Analyze and summarize study results.

    Args:
        study: Completed Optuna study

    Returns:
        Dictionary with analysis results
    """
    trials_df = study.trials_dataframe()

    # Basic statistics
    completed_trials = len(study.get_trials(states=[optuna.trial.TrialState.COMPLETE]))
    failed_trials = len(study.get_trials(states=[optuna.trial.TrialState.FAIL]))

    # Parameter statistics
    param_stats = {}
    for param in study.best_params.keys():
        if param in trials_df.columns:
            param_stats[param] = {
                "mean": trials_df[param].mean(),
                "std": trials_df[param].std(),
                "best": study.best_params[param],
            }

    # Performance statistics
    values = [t.value for t in study.get_trials() if t.value is not None]

    results = {
        "n_trials": len(study.trials),
        "completed_trials": completed_trials,
        "failed_trials": failed_trials,
        "best_value": study.best_value,
        "best_trial": study.best_trial.number,
        "mean_value": sum(values) / len(values) if values else 0,
        "std_value": torch.std(torch.tensor(values)).item() if values else 0,
        "param_stats": param_stats,
    }

    # Log analysis
    logger.info("\n📊 ANÁLISE DO ESTUDO")
    logger.info("=" * 50)
    logger.info(f"Trials completados: {completed_trials}/{len(study.trials)}")
    logger.info(f"Trials falhados: {failed_trials}")
    logger.info(f"MRR médio: {results['mean_value']:.4f} ± {results['std_value']:.4f}")

    return results


def resume_optimization(
    study_name: str,
    kg_config: KGConfig,
    base_config_path: Path,
    additional_trials: int = 20,
) -> optuna.Study:
    """
    Resume a previous optimization study.

    Args:
        study_name: Name of the existing study
        kg_config: Knowledge graph configuration
        base_config_path: Base TransE configuration
        additional_trials: Number of additional trials to run

    Returns:
        Updated study object
    """
    logger.info(f"📂 Resumindo estudo: {study_name}")

    # Load existing study
    study = optuna.load_study(
        study_name=study_name,
        storage="",  # Uses in-memory storage
    )

    logger.info(f"   Trials existentes: {len(study.trials)}")
    logger.info(f"   Melhor MRR atual: {study.best_value:.4f}")

    # Run additional trials
    study = run_transe_optimization(
        kg_config=kg_config,
        base_config_path=base_config_path,
        n_trials=additional_trials,
        study_name=study_name,
    )

    return study
