"""Cleanup utilities package.

Provides commands, strategies, engine, observers, and config loader with
backward-compatible exports.
"""

from .commands.base import CleanupCommand, CompositeCommand, TransparentCompositeCommand
from .commands.database import (
    DatabaseCleanCommand,
    KGDataCleanCommand,
    KGPreprocessedSplitsCleanCommand,
    KGRulesCleanCommand,
    PipelineCheckpointsCleanCommand,
)
from .commands.filesystem import (
    DirCleanCommand,
    LintCacheCleanCommand,
    ModelCacheCleanCommand,
    NestedDirCleanCommand,
    OptunaDatabaseCleanCommand,
    TrainingArtifactsCleanCommand,
)
from .commands.memory import (
    CloseLoggerCommand,
    ConcurrencyShutdownCommand,
    FlushMemoryCommand,
)
from .commands.ml import (
    DSLFMCheckpointsCleanCommand,
    MLFlowCleanCommand,
    MLTrainingCleanCommand,
)
from .config import CLEANUP_CONFIG, CLEANUP_CONFIG_PATH, load_cleanup_config
from .engine import CleanupEngine, build_engine, main
from .observer import CompositeCleanupObserver, LoggingCleanupObserver
from .strategies.base import CleanupStrategy
from .strategies.builtin import DeepCleanup, MLCleanup, ShutdownCleanup, StandardCleanup

__all__ = [
    "CLEANUP_CONFIG_PATH",
    "CLEANUP_CONFIG",
    "load_cleanup_config",
    "_load_cleanup_config",
    "CleanupCommand",
    "CompositeCommand",
    "TransparentCompositeCommand",
    "DatabaseCleanCommand",
    "KGDataCleanCommand",
    "KGPreprocessedSplitsCleanCommand",
    "KGRulesCleanCommand",
    "PipelineCheckpointsCleanCommand",
    "DirCleanCommand",
    "NestedDirCleanCommand",
    "LintCacheCleanCommand",
    "ModelCacheCleanCommand",
    "TrainingArtifactsCleanCommand",
    "OptunaDatabaseCleanCommand",
    "MLFlowCleanCommand",
    "DSLFMCheckpointsCleanCommand",
    "MLTrainingCleanCommand",
    "CloseLoggerCommand",
    "ConcurrencyShutdownCommand",
    "FlushMemoryCommand",
    "CleanupEngine",
    "CleanupStrategy",
    "StandardCleanup",
    "DeepCleanup",
    "MLCleanup",
    "ShutdownCleanup",
    "build_engine",
    "main",
    "CompositeCleanupObserver",
    "LoggingCleanupObserver",
]

_load_cleanup_config = load_cleanup_config
