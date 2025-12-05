from __future__ import annotations

from pathlib import Path

from pff import settings
from pff.utils.core.file_ops import FileOps
from pff.utils.core.logger import logger

from .base import CleanupCommand, CompositeCommand
from .filesystem import DirCleanCommand, ModelCacheCleanCommand, OptunaDatabaseCleanCommand, TrainingArtifactsCleanCommand


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
            logger.info(f"Removendo MLflow experiments: {mlruns_dir}")
            FileOps.rmtree_sync(mlruns_dir, ignore_errors=True)
            logger.info("Experimentos MLflow removidos")
        else:
            logger.debug("MLflow directory not found")


class RotatECheckpointsCleanCommand(CleanupCommand):
    """Remove RotatE checkpoints from common locations.

    Cleans PyTorch checkpoint files (`.pt`, `.pth`) from checkpoint directories
    used by RotatE training, including `checkpoints/`, `outputs/rotate/`, and CWD.

    Attributes:
        label: Display label for UI.
    """

    label = "Limpando checkpoints RotatE"

    def execute(self) -> None:
        """Execute the cleanup operation.

        Removes checkpoint files and empty directories from known locations.
        """
        locations: list[Path] = [
            settings.ROOT_DIR / "checkpoints",
            settings.OUTPUTS_DIR / "rotate",
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
            logger.info(f"Limpando checkpoints em: {location}")
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
                    logger.info(f" Diretório de checkpoints removido: {location}")
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Could not remove directory {location}: {exc}")

        logger.info(" Checkpoints RotatE removidos")


class MLTrainingCleanCommand(CompositeCommand):
    """Composite cleanup for ML artifacts.

    Aggregates multiple ML-related cleanup commands into a single composite
    operation, including RotatE checkpoints, MLflow experiments, model caches,
    training artifacts, Optuna databases, and output directories.

    Design Pattern: Composite.
    """

    def __init__(self):
        super().__init__(
            "Limpeza completa de ML/RotatE",
            [
                RotatECheckpointsCleanCommand(),
                MLFlowCleanCommand(),
                ModelCacheCleanCommand(),
                TrainingArtifactsCleanCommand(),
                OptunaDatabaseCleanCommand(),
                DirCleanCommand(
                    "Limpando outputs RotatE", settings.OUTPUTS_DIR / "rotate"
                ),
                DirCleanCommand("Limpando PyClause outputs", settings.PYCLAUSE_DIR),
            ],
        )


__all__ = [
    "MLFlowCleanCommand",
    "RotatECheckpointsCleanCommand",
    "MLTrainingCleanCommand",
]
