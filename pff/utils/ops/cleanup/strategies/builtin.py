"""Built-in cleanup strategy implementations.

This module provides concrete implementations of CleanupStrategy for
different cleanup scenarios: standard, deep, ML-focused, and shutdown.

Design Patterns:
    - Strategy: Each class implements CleanupStrategy protocol.
    - Template Method: DeepCleanup extends StandardCleanup's commands.
    - Composite: Commands are assembled into executable lists.
"""
from __future__ import annotations

from pff import settings
from pff.utils.ops.cleanup.commands.base import CompositeCommand
from pff.utils.ops.cleanup.commands.database import (
    DatabaseCleanCommand,
    KGDataCleanCommand,
    KGRulesCleanCommand,
    PipelineCheckpointsCleanCommand,
)
from pff.utils.ops.cleanup.commands.filesystem import (
    DirCleanCommand,
    NestedDirCleanCommand,
    OptunaDatabaseCleanCommand,
    PyCacheCleanCommand,
    TrainingArtifactsCleanCommand,
)
from pff.utils.ops.cleanup.commands.memory import CloseLoggerCommand, FlushMemoryCommand
from pff.utils.ops.cleanup.commands.ml import MLTrainingCleanCommand
from pff.utils.ops.cleanup.strategies.base import CleanupStrategy


class StandardCleanup(CleanupStrategy):
    """Default cleanup strategy for routine cache and log cleanup.

    Cleans Python caches, output directories, disk caches, logs,
    database tables, and common development artifacts like pytest
    cache, mypy cache, and node_modules.
    """

    def build_commands(self) -> list[CompositeCommand]:
        """Build standard cleanup commands.

        Returns:
            List of commands for routine cleanup operations.
        """
        return [
            PyCacheCleanCommand(),
            DirCleanCommand("Limpando outputs", settings.OUTPUTS_DIR),
            DirCleanCommand("Limpando cache em disco", settings.ROOT_DIR / ".cache"),
            FlushMemoryCommand(),
            CloseLoggerCommand(),
            DirCleanCommand("Limpando logs", settings.LOGS_DIR, "*.log"),
            DatabaseCleanCommand(),
            NestedDirCleanCommand(".cache", "Limpando todos os .cache"),
            DirCleanCommand(
                "Limpando pytest cache",
                settings.ROOT_DIR / ".pytest_cache",
                recursive=True,
            ),
            DirCleanCommand(
                "Limpando mypy cache", settings.ROOT_DIR / ".mypy_cache", recursive=True
            ),
            DirCleanCommand(
                "Limpando checkpoints Jupyter",
                settings.ROOT_DIR,
                "**/.ipynb_checkpoints",
                recursive=True,
            ),
            NestedDirCleanCommand(".pytest_cache", "Limpando todos os .pytest_cache"),
            NestedDirCleanCommand(".mypy_cache", "Limpando todos os .mypy_cache"),
            NestedDirCleanCommand("node_modules", "Limpando todos os node_modules"),
            NestedDirCleanCommand("dist", "Limpando todos os dist"),
            NestedDirCleanCommand(".coverage", "Limpando todos os .coverage"),
            NestedDirCleanCommand("htmlcov", "Limpando todos os htmlcov"),
            DirCleanCommand("Limpando mlruns", settings.ROOT_DIR / "mlruns"),
            DirCleanCommand(
                "Limpando pip cache", settings.PIP_CACHE_DIR, recursive=True
            ),
        ]


class DeepCleanup(StandardCleanup):
    """Aggressive cleanup including ML artifacts and model caches.

    Extends StandardCleanup with additional commands for cleaning
    KG data, PyTorch/HuggingFace caches, and training logs.
    """

    def build_commands(self) -> list[CompositeCommand]:
        """Build deep cleanup commands with ML artifact cleanup.

        Returns:
            Extended list of commands including ML-specific cleanup.
        """
        base = super().build_commands()
        ml_commands = [
            MLTrainingCleanCommand(),
            KGDataCleanCommand(),
            KGRulesCleanCommand(),
            DirCleanCommand(
                "Limpando dados KG processados",
                settings.OUTPUTS_DIR / "kg",
                recursive=True,
            ),
            DirCleanCommand(
                "Limpando cache PyTorch",
                settings.ROOT_DIR / ".cache" / "torch",
                recursive=True,
            ),
            DirCleanCommand(
                "Limpando cache HuggingFace",
                settings.ROOT_DIR / ".cache" / "huggingface",
                recursive=True,
            ),
            DirCleanCommand(
                "Limpando logs de treinamento", settings.LOGS_DIR, "training_*.log"
            ),
            DirCleanCommand("Limpando logs MLflow", settings.LOGS_DIR, "mlflow_*.log"),
        ]
        base[-2:-2] = ml_commands
        return base


class MLCleanup(CleanupStrategy):
    """Cleanup strategy focused on ML training artifacts.

    Targets ML-specific resources: training checkpoints, pipeline
    checkpoints, and ML-related logs. Suitable for reclaiming space
    after training runs.
    """

    def build_commands(self) -> list[CompositeCommand]:
        """Build ML-focused cleanup commands.

        Returns:
            List of commands targeting ML training artifacts.
        """
        return [
            FlushMemoryCommand(),
            MLTrainingCleanCommand(),
            PipelineCheckpointsCleanCommand(),
            DirCleanCommand("Limpando logs ML", settings.LOGS_DIR, "*training*.log"),
            DirCleanCommand("Limpando logs MLflow", settings.LOGS_DIR, "*mlflow*.log"),
            CloseLoggerCommand(),
        ]


class ShutdownCleanup(CleanupStrategy):
    """Selective cleanup strategy for graceful application shutdown.

    Performs minimal, fast cleanup suitable for shutdown scenarios.
    Focuses on memory flush, disk cache, and Python bytecode caches.
    """

    def build_commands(self) -> list[CompositeCommand]:
        """Build lightweight shutdown cleanup commands.

        Returns:
            Minimal list of commands for fast graceful shutdown.
        """
        logger = __import__("pff.utils.core.logger", fromlist=["logger"]).logger
        logger.info("Construindo comandos seletivos para shutdown gracioso...")
        return [
            FlushMemoryCommand(),
            DirCleanCommand("Limpando cache em disco", settings.ROOT_DIR / ".cache"),
            PyCacheCleanCommand(),
        ]


__all__ = ["StandardCleanup", "DeepCleanup", "MLCleanup", "ShutdownCleanup"]
