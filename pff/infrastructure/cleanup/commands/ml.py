from __future__ import annotations

from pathlib import Path

from pff import settings
from pff.infrastructure.cleanup.file_ops import FileOps
from pff.shared.core.logger import logger

from .base import CleanupCommand, CompositeCommand
from .filesystem import (
    DirCleanCommand,
    ModelCacheCleanCommand,
    OptunaDatabaseCleanCommand,
    TrainingArtifactsCleanCommand,
)


class MLFlowCleanCommand(CleanupCommand):
    """Remove MLflow experiments.

    Deletes the `mlruns` directory containing MLflow experiment tracking data,
    metrics, artifacts, and run metadata.

    Attributes:
        label: Display label for UI.
    """

    label = "Limpando experimentos MLflow"

    def execute(self) -> None:
        """Execute the cleanup operation.

        Removes the mlruns directory if it exists.
        """
        mlruns_dir = settings.ROOT_DIR / "mlruns"
        if mlruns_dir.exists():
            FileOps.rmtree_sync(mlruns_dir, ignore_errors=True)
        else:
            logger.debug("MLflow directory not found")


class DSLFMCheckpointsCleanCommand(CleanupCommand):
    """Remove DSLFM checkpoints from common locations."""

    label = "Limpando checkpoints DSLFM"

    def execute(self) -> None:
        """Execute the cleanup operation.

        Removes checkpoint files and empty directories from known locations.
        """
        locations: list[Path] = [
            settings.ROOT_DIR / "checkpoints",
            settings.OUTPUTS_DIR / "dslfm",
            Path.cwd() / "checkpoints",
        ]
        file_patterns = [
            "*.pt",
            "*.pth",
            "checkpoint_*.pt",
            "checkpoint_*.pth",
            "best_model.pt",
            "latest_checkpoint.pt",
        ]
        for location in locations:
            if not location.exists():
                continue
            for pattern in file_patterns:
                for fp in location.rglob(pattern):
                    try:
                        fp.unlink(missing_ok=True)
                        logger.debug(f"Removed checkpoint file: {fp}")
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(f"Could not remove {fp}: {exc}")
            try:
                if not any(location.iterdir()):
                    FileOps.rmtree_sync(location, ignore_errors=True)
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Could not remove directory {location}: {exc}")


class MLTrainingCleanCommand(CompositeCommand):
    """Composite cleanup for ML artifacts.

    Aggregates multiple ML-related cleanup commands into a single composite
    operation, including DSLFM checkpoints, MLflow experiments, model caches,
    training artifacts, Optuna databases, and output directories.

    Design Pattern: Composite.
    """

    def __init__(self):
        super().__init__(
            "Limpeza completa de ML/DSLFM",
            [
                DSLFMCheckpointsCleanCommand(),
                MLFlowCleanCommand(),
                ModelCacheCleanCommand(),
                TrainingArtifactsCleanCommand(),
                OptunaDatabaseCleanCommand(),
                DirCleanCommand(
                    "Limpando outputs DSLFM", settings.OUTPUTS_DIR / "dslfm"
                ),
            ],
        )


__all__ = [
    "MLFlowCleanCommand",
    "DSLFMCheckpointsCleanCommand",
    "MLTrainingCleanCommand",
]
