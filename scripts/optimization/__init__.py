"""
Optimization Module - SOTA "Zero-Touch" Hyperparameter Optimization
with REAL PFF Data Integration

State-of-the-Art Features:
 Automatic framework selection (Optuna SOTA by default)
 Complete MLflow integration (experiments, params, metrics, artifacts)
 Automatic visualization generation (optimization history, param importances, etc.)
 Zero-touch experience (just define objective + search space)
 Advanced optimization algorithms (TPE, CMA-ES, Hyperband, NSGA-II)
 Smart pruning (Median, Hyperband, SuccessiveHalving)
 Multi-objective optimization support
 Distributed optimization ready
 Comprehensive result persistence (best_params.json)
 REAL PFF DATA INTEGRATION (10K+ Knowledge Graph triplets)

Main Exports:
- find_best_hyperparameters():  Primary SOTA entry point (ZERO-TOUCH)
- optimize_kg_hyperparameters():  PFF KG optimization with REAL DATA
- optimize_ensemble_hyperparameters():  PFF Ensemble convenience function
- StrategyFactory: Automatic framework selection
- MLflowTracker: Complete experiment tracking
- OptimizationVisualizer: Automatic plot generation

Quick Start - General Optimization:
>>> from scripts.optimization import find_best_hyperparameters
>>>
>>> def objective(trial):
...     lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
...     return train_model(lr)  # Your model training
>>>
>>> result = find_best_hyperparameters(
...     objective_func=objective,
...     search_space={'lr': (1e-5, 1e-1)},
...     n_trials=100
... )
>>>
>>> print(f"Best: {result['best_value']:.4f}")
>>> print(f"Params: {result['best_params']}")

Quick Start - PFF Knowledge Graph (with REAL DATA):
>>> from scripts.optimization import optimize_kg_hyperparameters
>>>
>>> result = optimize_kg_hyperparameters(
...     n_trials=100,
...     use_real_data=True,  # Uses REAL PFF KG data!
...     study_name="pff_kg_v1"
... )
>>>
>>> print(f"Best KG score: {result['best_value']:.4f}")
>>> print(f"Data: {result['real_data_info']}")

PFF Real Data Statistics:
- Knowledge Graph: 10,241 training + 2,194 validation triplets
- Entities: 2,823 unique entities
- Predicates: 40 different relationship types
- Data format: (subject, predicate, object) triplets
- Source: /data/models/kg/*.parquet
"""

# Core SOTA API
from .core import (
    find_best_hyperparameters,  # Primary SOTA entry point
    optimize_kg_hyperparameters,  # PFF KG with REAL data
    optimize_ensemble_hyperparameters,  # PFF ensemble convenience
)

# Configuration
from .spaces import TuningConfig, SearchSpaceFactory
from .strategies.base import OptimizationConfig

# Strategy Pattern - Framework Abstraction
from .strategies import StrategyFactory, BaseOptimizerStrategy
from .strategies import OptunaStrategy, AutoOptunaStrategy, HyperoptStrategy

# MLflow Integration
from .tracker import MLflowTracker

# Visualization
from .visualizer import OptimizationVisualizer

# Callbacks and Observers
from .callbacks import (
    OptimizationObserver,
    CallbackManager,
    LoggingObserver,
    BestScoreObserver,
    RealTimeVisualizer,
    LivePlotCallback,
)

# Advanced Extensions
from .extensions import (
    MultiObjectiveOptimizer,
    DistributedOptimizer,
    OptunaReporting,
    ImportanceAnalyzer,
    TransferLearningOptimizer,
    NeuralArchitectureSearch,
)

# Auto Threshold Tuning
from .threshold import AutoThresholdTuner, ThresholdConfig

#  Advanced SOTA Features
from .advanced import (
    DistributedOptimizer,  # Ray Tune distributed optimization
    OptunaDashboard,  # Optuna Dashboard integration
    BayesianOptimizer,  # BoTorch Bayesian optimization
    EarlyStoppingOptimizer,  # Optuna Terminator early stopping
    ImportanceAnalyzer,  # fANOVA hyperparameter importance
    PDFReportGenerator,  # Automated PDF reports
    ModelRegistry,  # MLflow model registry
    AdvancedOptimizer,  # Unified advanced optimizer wrapper
)

__all__ = [
    #  Core SOTA API
    'find_best_hyperparameters',  # Primary entry point - ZERO-TOUCH
    'optimize_kg_hyperparameters',  # PFF KG with REAL data
    'optimize_ensemble_hyperparameters',  # PFF convenience function

    #  Configuration
    'TuningConfig',
    'SearchSpaceFactory',
    'OptimizationConfig',

    #  Strategy Pattern - Framework Abstraction
    'StrategyFactory',  # Auto-selects best framework
    'BaseOptimizerStrategy',  # Abstract base
    'OptunaStrategy',  # SOTA framework
    'AutoOptunaStrategy',  # Auto-configured Optuna
    'HyperoptStrategy',  # Legacy support

    #  MLOps Integration
    'MLflowTracker',  # Complete experiment tracking
    'OptimizationVisualizer',  # Automatic plot generation

    #  Observers & Callbacks
    'OptimizationObserver',
    'CallbackManager',
    'LoggingObserver',
    'BestScoreObserver',
    'RealTimeVisualizer',
    'LivePlotCallback',

    #  Advanced Extensions
    'MultiObjectiveOptimizer',  # Pareto front optimization
    'DistributedOptimizer',  # Ray Tune distributed
    'OptunaReporting',  # Automated reporting
    'ImportanceAnalyzer',  # fANOVA importance
    'TransferLearningOptimizer',  # Warm-start from history
    'NeuralArchitectureSearch',  # NAS integration

    #  Auto Threshold Tuning
    'AutoThresholdTuner',
    'ThresholdConfig',

    #  Advanced SOTA Features (7 Features)
    'DistributedOptimizer',  # Ray Tune distributed optimization
    'OptunaDashboard',  # Optuna Dashboard integration
    'BayesianOptimizer',  # BoTorch Bayesian optimization
    'EarlyStoppingOptimizer',  # Optuna Terminator early stopping
    'ImportanceAnalyzer',  # fANOVA hyperparameter importance
    'PDFReportGenerator',  # Automated PDF reports
    'ModelRegistry',  # MLflow model registry
    'AdvancedOptimizer',  # Unified advanced optimizer wrapper
]
