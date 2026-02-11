"""Dependency injection container for application services."""

from __future__ import annotations

from pathlib import Path

from dependency_injector import containers, providers

from pff.application.audit_use_case import AuditUseCase
from pff.application.learn_use_case import LearnUseCase
from pff.application.optimize_use_case import OptimizeUseCase
from pff.application.ports.hpo import HpoRunnerPort
from pff.application.strategy_registry import get_strategy_registry


class ApplicationContainer(containers.DeclarativeContainer):
    """Application-level DI container."""

    config_path: providers.Object[Path | None] = providers.Object(None)
    strategy_registry = providers.Singleton(get_strategy_registry)
    learn_use_case = providers.Factory(
        LearnUseCase,
        config_path=config_path,
        strategy_registry=strategy_registry,
    )
    hpo_runner: providers.Dependency[HpoRunnerPort] = providers.Dependency()
    optimize_use_case = providers.Factory(OptimizeUseCase, runner=hpo_runner)
    audit_use_case = providers.Factory(AuditUseCase)
