"""
TransE Knowledge Graph Embedding Module

This module provides a complete implementation of TransE (Translating Embeddings)
for knowledge graph completion, including:

- Data preprocessing and optimization
- TransE model training and inference
- Hybrid TransE + LightGBM models
- Hyperparameter optimization
- Evaluation and ranking

The module is designed to be a drop-in replacement for the HGT implementation,
maintaining API compatibility while providing improved performance for
telecommunications domain data.
"""

from __future__ import annotations

__version__ = "4.0.0"
__author__ = "Alex Lira"

# Core components
from typing import Any

from .core import TransEManager, TransEModel, compare_mlflow_experiments
from .lightgbm_trainer import TransELightGBMTrainer
from .mapping_utils import (
    convert_graph_to_indices,
    create_raw_mappings,
    create_reverse_mappings,
    load_mappings,
    merge_mappings,
    save_mappings_to_checkpoint,
    validate_mappings,
)
from .transe_hyperopt import (
    analyze_study_results,
    create_optimization_plots,
    resume_optimization,
    run_transe_optimization,
    save_best_config,
)
from .transe_pipeline import TransEPipeline
from .transe_preprocessor import TransEPreprocessor
from .transe_service import TransEScorerService

# Backward compatibility aliases (HGT -> TransE)
HGTManager = TransEManager
HGTPipeline = TransEPipeline
HGTScorerService = TransEScorerService
HGTLightGBMTrainer = TransELightGBMTrainer


__all__ = [
    # Version info
    "__version__",
    "__author__",
    # Core classes
    "TransEModel",
    "TransEManager",
    "TransEPipeline",
    "TransEPreprocessor",
    "TransEScorerService",
    "TransELightGBMTrainer",
    # Utility functions
    "create_raw_mappings",
    "load_mappings",
    "validate_mappings",
    "convert_graph_to_indices",
    "merge_mappings",
    "create_reverse_mappings",
    "save_mappings_to_checkpoint",
    "compare_mlflow_experiments",
    # Hyperopt functions
    "run_transe_optimization",
    "save_best_config",
    "create_optimization_plots",
    "analyze_study_results",
    "resume_optimization",
    # Compatibility aliases
    "HGTManager",
    "HGTPipeline",
    "HGTScorerService",
    "HGTLightGBMTrainer",
]


def get_version() -> str:
    """Get the version string of the TransE module."""
    return __version__


def get_config_template() -> dict[str, Any]:
    """
    Get a template configuration for TransE.

    Returns:
        Dictionary with default TransE configuration
    """
    return {
        "model": {"embedding_dim": 128, "margin": 2.0, "norm": 2},
        "training": {
            "epochs": 100,
            "batch_size": 512,
            "optimizer": {"type": "adam", "params": {"lr": 0.001, "weight_decay": 0.0}},
            "scheduler": {
                "type": "reduce_on_plateau",
                "params": {"patience": 5, "factor": 0.5},
            },
            "patience": 15,
            "validate_every": 5,
            "num_negatives": 5,
            "normalize_embeddings": True,
            "max_grad_norm": 1.0,
            "seed": 42,
            "num_workers": 0,
        },
        "mlflow": {
            "enabled": False,
            "experiment_name": "TransE_KGC",
            "tracking_uri": None,
        },
        "checkpointing": {"save_dir": "checkpoints/transe"},
    }


def validate_environment() -> dict[str, bool]:
    """
    Validate that all required dependencies are available.

    Returns:
        Dictionary with validation results
    """
    validations = {}

    # Check required packages
    required_packages = [
        "torch",
        "numpy",
        "polars",
        "lightgbm",
        "sklearn",
        "optuna",
        "mlflow",
    ]

    for package in required_packages:
        try:
            __import__(package)
            validations[package] = True
        except ImportError:
            validations[package] = False

    # Check CUDA availability
    try:
        import torch

        validations["cuda"] = torch.cuda.is_available()
        if validations["cuda"]:
            validations["cuda_device"] = torch.cuda.get_device_name(0)
    except Exception:
        validations["cuda"] = False

    return validations


# Module initialization
def _initialize_module():
    """Initialize the TransE module."""
    import logging

    # Set up logging if not already configured
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    # Validate environment
    validations = validate_environment()
    missing = [
        pkg for pkg, available in validations.items() if not available and pkg != "cuda"
    ]

    if missing:
        import warnings

        warnings.warn(
            f"Missing optional dependencies: {', '.join(missing)}. "
            "Some features may not be available.",
            RuntimeWarning,
        )


# Initialize module on import
_initialize_module()
