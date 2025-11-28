"""
Optimization Strategies Module - SOTA Implementation

This module provides abstraction for different hyperparameter optimization frameworks
with automatic selection of the best available framework (Optuna SOTA by default).

SOTA Features:
- WilcoxonPruner for k-fold cross-validation (Optuna v3.6.0+)
- HyperbandPruner for large search spaces
- TPE sampler with multivariate support
"""

from .base import BaseOptimizerStrategy, OptimizationConfig, TrialResult, OptimizationResult
from .optuna_impl import OptunaStrategy, AutoOptunaStrategy
from .hyperopt_impl import HyperoptStrategy
from .factory import StrategyFactory

__all__ = [
    'BaseOptimizerStrategy',
    'OptimizationConfig',
    'TrialResult',
    'OptimizationResult',
    'OptunaStrategy',
    'AutoOptunaStrategy',
    'HyperoptStrategy',
    'StrategyFactory',
]
