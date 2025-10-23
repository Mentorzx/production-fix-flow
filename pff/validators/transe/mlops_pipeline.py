from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import mlflow
import optuna
import polars as pl

from pff import settings
from pff.utils import logger
from pff.validators.kg.config import KGConfig
from pff.validators.transe.core import compare_mlflow_experiments
from pff.validators.transe.transe_hyperopt import run_transe_optimization
from pff.validators.transe.transe_pipeline import TransEPipeline

"""
MLOps Pipeline for TransE

This module provides MLOps capabilities for TransE model training,
including experiment tracking, model versioning, and deployment.
"""


async def train_with_mlflow(
    kg_config_path: Path | None = None, experiment_name: str = "TransE_KGC"
) -> dict[str, Any]:
    """
    Train TransE model with MLflow tracking.

    Args:
        kg_config_path: Path to KG configuration
        experiment_name: MLflow experiment name

    Returns:
        Dictionary with training results
    """
    logger.info("=" * 80)
    logger.info("TREINAMENTO COM MLFLOW")
    logger.info("=" * 80)

    if kg_config_path is None:
        kg_config_path = settings.CONFIG_DIR / "kg.yaml"

    # Set MLflow experiment
    mlflow.set_experiment(experiment_name)

    # Initialize pipeline
    pipeline = TransEPipeline(kg_config_path)

    # Train model
    with mlflow.start_run():
        # Log configuration
        mlflow.log_param("kg_config", str(kg_config_path))

        # Train
        results = await pipeline.train_transe()

        # Log metrics
        if "best_val_mrr" in results:
            mlflow.log_metric("best_val_mrr", results["best_val_mrr"])
            mlflow.log_metric("best_epoch", results["best_epoch"])
            mlflow.log_metric("training_time", results["training_time"])

        # Log model artifacts
        checkpoint_path = Path("checkpoints/transe/best_model.pt")
        if checkpoint_path.exists():
            mlflow.log_artifact(str(checkpoint_path))

    logger.info("✅ Treino com MLflow concluído!")
    logger.info("Para visualizar: mlflow ui")

    return results


def optimize_hyperparameters(
    kg_config_path: Path | None = None, n_trials: int = 20, n_jobs: int = 1
) -> optuna.Study:
    """
    Run hyperparameter optimization.

    Args:
        kg_config_path: Path to KG configuration
        n_trials: Number of optimization trials
        n_jobs: Number of parallel jobs

    Returns:
        Optuna study object
    """
    logger.info("=" * 80)
    logger.info("OTIMIZAÇÃO DE HIPERPARÂMETROS")
    logger.info("=" * 80)

    if kg_config_path is None:
        kg_config_path = settings.CONFIG_DIR / "kg.yaml"

    kg_config = KGConfig(kg_config_path)
    base_config_path = settings.CONFIG_DIR / "transe.yaml"

    study = run_transe_optimization(
        kg_config=kg_config,
        base_config_path=base_config_path,
        n_trials=n_trials,
        n_jobs=n_jobs,
    )

    logger.info(f"\nMelhor MRR: {study.best_value:.4f}")
    logger.info(f"Melhor trial: #{study.best_trial.number}")

    return study


def compare_experiments(
    experiment_name: str = "TransE_KGC", metric: str = "val_mrr"
) -> pl.DataFrame | None:
    """
    Compare MLflow experiments.

    Args:
        experiment_name: Name of MLflow experiment
        metric: Metric to sort by

    Returns:
        DataFrame with experiment comparison
    """
    logger.info("=" * 80)
    logger.info("COMPARAÇÃO DE EXPERIMENTOS")
    logger.info("=" * 80)

    df = compare_mlflow_experiments(experiment_name, metric)

    if df is not None and df.height > 0:
        logger.info(f"\nTop 5 runs por {metric}:")
        display_cols = ["run_name", metric, "training_time", "embedding_dim"]
        display_cols = [col for col in display_cols if col in df.columns]
        print(df[display_cols].head())
    else:
        logger.info("Nenhum experimento encontrado.")

    return df


async def train_with_optimized_config(
    kg_config_path: Path | None = None,
) -> dict[str, Any]:
    """
    Train with optimized configuration.

    Args:
        kg_config_path: Path to KG configuration

    Returns:
        Dictionary with training results
    """
    logger.info("=" * 80)
    logger.info("TREINO COM CONFIG OTIMIZADA")
    logger.info("=" * 80)

    optimized_config_path = settings.CONFIG_DIR / "transe_optimized.yaml"

    if not optimized_config_path.exists():
        logger.error("Config otimizada não encontrada! Execute otimização primeiro.")
        return {"error": "optimized config not found"}

    if kg_config_path is None:
        kg_config_path = settings.CONFIG_DIR / "kg.yaml"

    pipeline = TransEPipeline(kg_config_path)
    results = await pipeline.train_transe(transe_config_path=optimized_config_path)

    logger.info("✅ Treino com config otimizada concluído!")

    return results


def deploy_model(
    model_path: Path | None = None, deployment_config: dict[str, Any] | None = None
) -> dict[str, Any]:
    """
    Deploy TransE model.

    Args:
        model_path: Path to model checkpoint
        deployment_config: Deployment configuration

    Returns:
        Deployment status
    """
    logger.info("=" * 80)
    logger.info("DEPLOY DO MODELO")
    logger.info("=" * 80)

    if model_path is None:
        model_path = Path("checkpoints/transe/best_model.pt")

    if not model_path.exists():
        logger.error(f"Modelo não encontrado: {model_path}")
        return {"status": "error", "message": "model not found"}

    # Here you would implement actual deployment logic
    # For example: Docker, Kubernetes, cloud services, etc.

    logger.info(f"🚀 Deploying model from: {model_path}")

    # Placeholder for deployment logic
    deployment_status = {
        "status": "deployed",
        "model_path": str(model_path),
        "deployment_config": deployment_config or {},
        "endpoint": "http://localhost:8000/predict",
    }

    logger.success("✅ Modelo deployed com sucesso!")

    return deployment_status


async def run_complete_mlops_pipeline(
    kg_config_path: Path | None = None, optimize: bool = True, n_trials: int = 20
) -> dict[str, Any]:
    """
    Run complete MLOps pipeline.

    Args:
        kg_config_path: Path to KG configuration
        optimize: Whether to run hyperparameter optimization
        n_trials: Number of optimization trials

    Returns:
        Dictionary with pipeline results
    """
    logger.info("🚀 EXECUTANDO PIPELINE MLOPS COMPLETO")
    logger.info("=" * 80)

    results = {}

    # Step 1: Hyperparameter optimization
    if optimize:
        study = optimize_hyperparameters(kg_config_path, n_trials=n_trials)
        results["optimization"] = {
            "best_value": study.best_value,
            "best_params": study.best_params,
        }

    # Step 2: Train with best config
    if optimize:
        train_results = await train_with_optimized_config(kg_config_path)
    else:
        train_results = await train_with_mlflow(kg_config_path)

    results["training"] = train_results

    # Step 3: Compare experiments
    comparison = compare_experiments()
    results["comparison"] = comparison

    # Step 4: Deploy best model
    deployment = deploy_model()
    results["deployment"] = deployment

    logger.success("✅ Pipeline MLOps concluído com sucesso!")

    return results


if __name__ == "__main__":
    # Example usage
    asyncio.run(run_complete_mlops_pipeline())
