"""
Optimization Strategies Module - SOTA Implementation

This module provides abstraction for different hyperparameter optimization frameworks
with automatic selection of the best available framework (Optuna SOTA by default).
"""

from .base import BaseOptimizerStrategy
from .optuna_impl import OptunaStrategy, AutoOptunaStrategy
from .hyperopt_impl import HyperoptStrategy
from .factory import StrategyFactory

__all__ = [
    'BaseOptimizerStrategy',
    'OptunaStrategy',
    'AutoOptunaStrategy',
    'HyperoptStrategy',
    'StrategyFactory',
]
