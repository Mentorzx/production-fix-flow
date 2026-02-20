#!/usr/bin/env python3
"""
Strategy Factory - Automatic Framework Selection

Factory Pattern: Creates the appropriate optimization strategy based on:
- Available libraries (Optuna SOTA preferred)
- User preference
- Optimization characteristics
"""

from __future__ import annotations

from typing import Any

from pff.shared import logger

from .base import BaseOptimizerStrategy, OptimizationConfig
from .optuna_impl import AutoOptunaStrategy, OptunaStrategy


class StrategyFactory:
    """
    Factory for creating optimization strategies.

    Factory Pattern: Abstracts framework selection from user.
    """

    _strategies = {
        "optuna": OptunaStrategy,
        "optuna-auto": AutoOptunaStrategy,
    }

    _framework_availability = {}

    @classmethod
    def _check_availability(cls) -> dict[str, bool]:
        """
        Check which frameworks are available.

        Returns:
            Dictionary mapping framework names to availability
        """
        if not cls._framework_availability:
            cls._framework_availability = {
                "optuna": _check_optuna_availability(),
            }

        return cls._framework_availability

    @classmethod
    def create_strategy(
        cls,
        strategy_name: str = "auto",
        config: OptimizationConfig | None = None,
        is_multi_objective: bool = False,
    ) -> BaseOptimizerStrategy:
        """
        Create optimization strategy with automatic selection.

        Args:
            strategy_name: Strategy name ('auto', 'optuna', 'optuna-auto')
            config: Optimization configuration
            is_multi_objective: Whether using multi-objective optimization

        Returns:
            Configured strategy instance

        Raises:
            ImportError: If requested framework not available
        """
        config = config or OptimizationConfig()
        available = cls._check_availability()

        if strategy_name == "auto":
            strategy_name = cls._select_best_framework(available, is_multi_objective)

        if strategy_name not in cls._strategies:
            logger.warning(
                f"Unknown strategy '{strategy_name}'. "
                f"Available: {list(cls._strategies.keys())}. Using 'optuna'."
            )
            strategy_name = "optuna"

        if not available.get(strategy_name, False):
            available_strategies = [k for k, v in available.items() if v]
            raise ImportError(
                f"Framework '{strategy_name}' not available.\n"
                f"Available: {available_strategies}\n"
                f"Install with: pip install {strategy_name}"
            )

        strategy_class = cls._strategies[strategy_name]

        if strategy_name == "optuna-auto":
            strategy = strategy_class(config, is_multi_objective=is_multi_objective)
        else:
            strategy = strategy_class(config)

        logger.info(f"Estrategia criada: {strategy_name}")
        logger.info(f"Framework: {strategy.framework_name}")

        return strategy

    @classmethod
    def _select_best_framework(
        cls,
        available: dict[str, bool],
        is_multi_objective: bool = False,
    ) -> str:
        """
        Automatically select the best available framework.

        Priority:
        1. Optuna (SOTA - best features)

        Args:
            available: Available frameworks
            is_multi_objective: Multi-objective optimization flag

        Returns:
            Best framework name
        """
        if available.get("optuna", False):
            logger.info("Optuna selecionado automaticamente (framework SOTA)")
            if is_multi_objective:
                logger.info("Usando AutoOptunaStrategy para multiobjetivo")
            return "optuna-auto"
        raise ImportError(
            "No optimization framework available.\nInstall at least one: pip install optuna"
        )

    @classmethod
    def get_available_strategies(cls) -> dict[str, bool]:
        """Get all available strategy names and their availability."""
        return cls._check_availability()

    @classmethod
    def get_strategy_info(cls, strategy_name: str) -> dict[str, Any]:
        """
        Get information about a strategy.

        Args:
            strategy_name: Strategy name

        Returns:
            Dictionary with strategy information
        """
        info = {
            "optuna": {
                "name": "Optuna",
                "description": "State-of-the-art hyperparameter optimization framework",
                "features": [
                    "Advanced pruning (Median, Hyperband, SuccessiveHalving)",
                    "Modern samplers (TPE, CMA-ES, NSGA-II)",
                    "Native visualizations",
                    "MLflow integration",
                    "Multi-objective optimization",
                    "Distributed optimization",
                ],
                "recommended": True,
            },
            "optuna-auto": {
                "name": "Auto-Optuna",
                "description": "Optuna with automatic configuration",
                "features": [
                    "Auto-selects best sampler based on search space",
                    "Auto-selects best pruner based on trial characteristics",
                    "Optimized for different scenarios",
                ],
                "recommended": True,
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
            raise ValueError("Strategy must inherit from BaseOptimizerStrategy")

        cls._strategies[name] = strategy_class
        logger.info(f"Nova estratégia registrada: {name}")


def _check_optuna_availability() -> bool:
    """Check if Optuna is installed and importable."""
    try:
        __import__("optuna")

        return True
    except ImportError:
        return False
