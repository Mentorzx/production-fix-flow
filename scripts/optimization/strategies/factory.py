#!/usr/bin/env python3
"""
Strategy Factory - Automatic Framework Selection

Factory Pattern: Creates the appropriate optimization strategy based on:
- Available libraries (Optuna SOTA preferred)
- User preference
- Optimization characteristics
"""

from __future__ import annotations

from typing import Optional, Dict, Any
from dataclasses import asdict

from .base import BaseOptimizerStrategy, OptimizationConfig
from .optuna_impl import OptunaStrategy, AutoOptunaStrategy
from .hyperopt_impl import HyperoptStrategy
from pff.utils import logger


class StrategyFactory:
    """
    Factory for creating optimization strategies.

    Factory Pattern: Abstracts framework selection from user.
    """

    _strategies = {
        'optuna': OptunaStrategy,
        'optuna-auto': AutoOptunaStrategy,
        'hyperopt': HyperoptStrategy,
    }

    _framework_availability = {}

    @classmethod
    def _check_availability(cls) -> Dict[str, bool]:
        """
        Check which frameworks are available.

        Returns:
            Dictionary mapping framework names to availability
        """
        if not cls._framework_availability:
            cls._framework_availability = {
                'optuna': _check_optuna_availability(),
                'hyperopt': _check_hyperopt_availability(),
            }

        return cls._framework_availability

    @classmethod
    def create_strategy(
        cls,
        strategy_name: str = "auto",
        config: Optional[OptimizationConfig] = None,
        is_multi_objective: bool = False,
    ) -> BaseOptimizerStrategy:
        """
        Create optimization strategy with automatic selection.

        Args:
            strategy_name: Strategy name ('auto', 'optuna', 'optuna-auto', 'hyperopt')
            config: Optimization configuration
            is_multi_objective: Whether using multi-objective optimization

        Returns:
            Configured strategy instance

        Raises:
            ImportError: If requested framework not available
        """
        config = config or OptimizationConfig()
        available = cls._check_availability()

        # Auto-select best available framework
        if strategy_name == "auto":
            strategy_name = cls._select_best_framework(available, is_multi_objective)

        # Validate strategy exists
        if strategy_name not in cls._strategies:
            logger.warning(
                f"Unknown strategy '{strategy_name}'. "
                f"Available: {list(cls._strategies.keys())}. Using 'optuna'."
            )
            strategy_name = 'optuna'

        # Check availability
        if not available.get(strategy_name, False):
            available_strategies = [
                k for k, v in available.items() if v
            ]
            raise ImportError(
                f"Framework '{strategy_name}' not available.\n"
                f"Available: {available_strategies}\n"
                f"Install with: pip install {strategy_name}"
            )

        # Create strategy
        strategy_class = cls._strategies[strategy_name]

        if strategy_name == 'optuna-auto':
            strategy = strategy_class(config, is_multi_objective=is_multi_objective)
        else:
            strategy = strategy_class(config)

        logger.info(f"Created strategy: {strategy_name}")
        logger.info(f"Framework: {strategy.framework_name}")

        return strategy

    @classmethod
    def _select_best_framework(
        cls,
        available: Dict[str, bool],
        is_multi_objective: bool = False,
    ) -> str:
        """
        Automatically select the best available framework.

        Priority:
        1. Optuna (SOTA - best features)
        2. Hyperopt (legacy support)

        Args:
            available: Available frameworks
            is_multi_objective: Multi-objective optimization flag

        Returns:
            Best framework name
        """
        if available.get('optuna', False):
            logger.info("Auto-selected Optuna (SOTA framework)")
            if is_multi_objective:
                logger.info("Usando AutoOptunaStrategy para multiobjetivo")
            return 'optuna-auto'
        elif available.get('hyperopt', False):
            logger.warning("Using Hyperopt (Optuna recommended for better features)")
            return 'hyperopt'
        else:
            raise ImportError(
                "No optimization framework available.\n"
                "Install at least one: pip install optuna"
            )

    @classmethod
    def get_available_strategies(cls) -> Dict[str, bool]:
        """Get all available strategy names and their availability."""
        return cls._check_availability()

    @classmethod
    def get_strategy_info(cls, strategy_name: str) -> Dict[str, Any]:
        """
        Get information about a strategy.

        Args:
            strategy_name: Strategy name

        Returns:
            Dictionary with strategy information
        """
        info = {
            'optuna': {
                'name': 'Optuna',
                'description': 'State-of-the-art hyperparameter optimization framework',
                'features': [
                    'Advanced pruning (Median, Hyperband, SuccessiveHalving)',
                    'Modern samplers (TPE, CMA-ES, NSGA-II)',
                    'Native visualizations',
                    'MLflow integration',
                    'Multi-objective optimization',
                    'Distributed optimization',
                ],
                'recommended': True,
            },
            'optuna-auto': {
                'name': 'Auto-Optuna',
                'description': 'Optuna with automatic configuration',
                'features': [
                    'Auto-selects best sampler based on search space',
                    'Auto-selects best pruner based on trial characteristics',
                    'Optimized for different scenarios',
                ],
                'recommended': True,
            },
            'hyperopt': {
                'name': 'Hyperopt',
                'description': 'Legacy hyperparameter optimization framework',
                'features': [
                    'TPE and Annealing samplers',
                    'Basic parallelization',
                    'Limited visualization',
                ],
                'recommended': False,
            },
        }

        return info.get(strategy_name, {})

    @classmethod
    def register_strategy(cls, name: str, strategy_class: type):
        """
        Register a new strategy.

        Args:
            name: Strategy name
            strategy_class: Strategy class (must inherit from BaseOptimizerStrategy)
        """
        if not issubclass(strategy_class, BaseOptimizerStrategy):
            raise ValueError(
                "Strategy must inherit from BaseOptimizerStrategy"
            )

        cls._strategies[name] = strategy_class
        logger.info(f"Registered new strategy: {name}")


def _check_optuna_availability() -> bool:
    """Check if Optuna is installed and importable."""
    try:
        import optuna
        return True
    except ImportError:
        return False


def _check_hyperopt_availability() -> bool:
    """Check if Hyperopt is installed and importable."""
    try:
        import hyperopt
        return True
    except ImportError:
        return False
