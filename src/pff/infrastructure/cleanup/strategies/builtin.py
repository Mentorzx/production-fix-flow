"""Built-in cleanup strategy implementations.

This module provides concrete implementations of CleanupStrategy for
different cleanup scenarios: standard, deep, ML-focused, and shutdown.

Design Patterns:
    - Strategy: Each class implements CleanupStrategy protocol.
    - Template Method: DeepCleanup extends StandardCleanup's commands.
    - Composite: Commands are assembled into executable lists.
"""

from __future__ import annotations

from pathlib import Path

from pff.infrastructure.cleanup.commands.base import CleanupCommand
from pff.infrastructure.cleanup.commands.database import (
    DatabaseCleanCommand,
    HpoCheckpointsCleanCommand,
    HpoTrialResultsCleanCommand,
    KGDataCleanCommand,
    KGEmbeddingsCleanCommand,
    KGMappingsCleanCommand,
    KGPreprocessedSplitsCleanCommand,
    KGRulesCleanCommand,
    LanceDBOptimizeCommand,
    OptunaTablesCleanCommand,
    PipelineCheckpointsCleanCommand,
    TrainingMetricsCleanCommand,
)
from pff.infrastructure.cleanup.commands.filesystem import (
    DirCleanCommand,
    LintCacheCleanCommand,
    LogArchiverCommand,
    NestedDirCleanCommand,
    OptunaDatabaseCleanCommand,
    TrainingArtifactsCleanCommand,
)
from pff.infrastructure.cleanup.commands.memory import (
    CloseLoggerCommand,
    FlushMemoryCommand,
)
from pff.infrastructure.cleanup.commands.ml import (
    DSLFMCheckpointsCleanCommand,
    MLTrainingCleanCommand,
)
from pff.infrastructure.cleanup.strategies.base import CleanupStrategy
from pff.shared.core.config import settings


class StandardCleanup(CleanupStrategy):
    """Default cleanup strategy for routine cache and log cleanup.

    Cleans Python caches, output directories, disk caches, logs,
    database tables, and common development artifacts like pytest
    cache, mypy cache, and node_modules.
    """

    def build_commands(self, collector=None) -> list[CleanupCommand]:
        """Build standard cleanup commands.

        Args:
            collector: Optional shared scan collector.

        Returns:
            List of commands for routine cleanup operations.
        """
        return [
            DirCleanCommand(
                "Limpando outputs",
                settings.OUTPUTS_DIR,
                exclude_dirs=[settings.ROOT_DIR / "outputs" / ".cache"],
            ),
            DirCleanCommand(
                "Limpando cache centralizado",
                settings.CACHE_DIR,
            ),
            NestedDirCleanCommand(
                "__pycache__",
                "Limpando todos os __pycache__",
                collector=collector,
                exclude_roots=[settings.OUTPUTS_DIR],
            ),
            FlushMemoryCommand(),
            CloseLoggerCommand(),
            LogArchiverCommand(settings.LOGS_DIR),
            DirCleanCommand("Limpando logs", settings.LOGS_DIR),
            DatabaseCleanCommand(),
            DirCleanCommand(
                "Limpando checkpoints Jupyter",
                settings.ROOT_DIR,
                "**/.ipynb_checkpoints",
                recursive=True,
            ),
            NestedDirCleanCommand(
                "node_modules", "Limpando todos os node_modules", collector=collector
            ),
            NestedDirCleanCommand("dist", "Limpando todos os dist", collector=collector),
            NestedDirCleanCommand(".coverage", "Limpando todos os .coverage", collector=collector),
            NestedDirCleanCommand("htmlcov", "Limpando todos os htmlcov", collector=collector),
            DirCleanCommand("Limpando mlruns", settings.ROOT_DIR / "mlruns"),
            DirCleanCommand("Limpando pip cache", settings.PIP_CACHE_DIR, recursive=True),
        ]


class DeepCleanup(StandardCleanup):
    """Aggressive cleanup including ML artifacts and model caches.

    Extends StandardCleanup with additional commands for cleaning
    KG data, PyTorch/HuggingFace caches, and training logs.
    """

    def build_commands(self, collector=None) -> list[CleanupCommand]:
        """Build deep cleanup commands with ML artifact cleanup.

        Args:
            collector: Optional shared scan collector.

        Returns:
            Extended list of commands including ML-specific cleanup.
        """
        base = super().build_commands(collector=collector)

        # Override outputs/ and mlruns/ commands to remove the dirs entirely
        for i, cmd in enumerate(base):
            if isinstance(cmd, DirCleanCommand) and cmd._dir == settings.OUTPUTS_DIR:
                base[i] = DirCleanCommand(
                    "Limpando outputs (remover pasta)",
                    settings.OUTPUTS_DIR,
                    remove_dir=True,
                )
            elif isinstance(cmd, DirCleanCommand) and cmd._dir == settings.ROOT_DIR / "mlruns":
                base[i] = DirCleanCommand(
                    "Limpando mlruns (remover pasta)",
                    settings.ROOT_DIR / "mlruns",
                    remove_dir=True,
                )

        ml_commands: list[CleanupCommand] = [
            LintCacheCleanCommand(),
            DSLFMCheckpointsCleanCommand(),
            TrainingArtifactsCleanCommand(),
            OptunaDatabaseCleanCommand(),
            KGDataCleanCommand(),
            KGPreprocessedSplitsCleanCommand(),
            KGMappingsCleanCommand(),
            KGEmbeddingsCleanCommand(),
            KGRulesCleanCommand(),
            TrainingMetricsCleanCommand(),
            OptunaTablesCleanCommand(),
            HpoTrialResultsCleanCommand(),
            HpoCheckpointsCleanCommand(),
            LanceDBOptimizeCommand(),
            DirCleanCommand(
                "Limpando dados LanceDB",
                settings.ROOT_DIR / "data" / "lancedb",
                recursive=True,
            ),
            DirCleanCommand(
                "Limpando cache PyTorch (Home)",
                Path.home() / ".cache" / "torch",
                recursive=True,
            ),
            DirCleanCommand(
                "Limpando cache HuggingFace (Home)",
                Path.home() / ".cache" / "huggingface",
                recursive=True,
            ),
            DirCleanCommand(
                "Limpando target Rust (remover pasta)",
                settings.ROOT_DIR / "target",
                remove_dir=True,
            ),
        ]
        base[-2:-2] = ml_commands
        return base


class MLCleanup(CleanupStrategy):
    """Cleanup strategy focused on ML training artifacts.

    Targets ML-specific resources: training checkpoints, pipeline
    checkpoints, and ML-related logs. Suitable for reclaiming space
    after training runs.
    """

    def build_commands(self, collector=None) -> list[CleanupCommand]:
        """Build ML-focused cleanup commands.

        Args:
            collector: Optional shared scan collector.

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

    def build_commands(self, collector=None) -> list[CleanupCommand]:
        """Build lightweight shutdown cleanup commands.

        Args:
            collector: Optional shared scan collector.

        Returns:
            Minimal list of commands for fast graceful shutdown.
        """
        logger = __import__("pff.shared.core.logging", fromlist=["logger"]).logger
        logger.info("Construindo comandos seletivos para shutdown gracioso...")
        return [
            FlushMemoryCommand(),
            DirCleanCommand("Limpando cache centralizado", settings.CACHE_DIR),
        ]


__all__ = ["StandardCleanup", "DeepCleanup", "MLCleanup", "ShutdownCleanup"]
