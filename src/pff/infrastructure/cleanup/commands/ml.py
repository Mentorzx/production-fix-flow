"""Provide module-level functionality for the PFF codebase.



Notes:

    File: src/pff/infrastructure/cleanup/commands/ml.py

"""

from __future__ import annotations

from pathlib import Path

import os
import re
import signal
import subprocess

from pff.infrastructure.cleanup.file_ops import FileOps
from pff.shared.core.config import settings
from pff.shared.core.logging import logger

from .base import CleanupCommand, CompositeCommand
from .filesystem import (
    DirCleanCommand,
    ModelCacheCleanCommand,
    OptunaDatabaseCleanCommand,
    TrainingArtifactsCleanCommand,
)


class DashboardResetCommand(CleanupCommand):
    """Reset the HPO Dashboard server and its memory cache.

    Finds and terminates any running dashboard server process on port 8766
    to ensure its in-memory 'LOOKBACK_MEMORY' is fully cleared.

    Attributes:
        label: Display label for UI.
    """

    label = "Resetando servidor de Dashboard HPO (limpeza de RAM)"

    def execute(self) -> None:
        """Execute the dashboard reset operation.

        Uses system commands to find and kill processes listening on port 8766.
        """
        try:
            result = subprocess.run(
                ["ss", "-lptn", "sport = :8766"],
                check=False,
                capture_output=True,
                text=True,
            )
            output = result.stdout.strip()
            pid_matches = re.findall(r"pid=(\d+)", output)

            if pid_matches:
                pids = set(pid_matches)
                for pid_str in pids:
                    pid = int(pid_str)
                    logger.info(f"Finalizando servidor dashboard antigo (PID={pid})")
                    os.kill(pid, signal.SIGKILL)
            else:
                subprocess.run(
                    ["pkill", "-9", "-f", "server.py --port 8766"],
                    check=False,
                    capture_output=True,
                    text=True,
                )

        except Exception as exc:
            logger.debug(f"Failed to reset dashboard server: {exc}")


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
                    except Exception as exc:
                        logger.warning(f"Could not remove {fp}: {exc}")
            try:
                if not any(location.iterdir()):
                    FileOps.rmtree_sync(location, ignore_errors=True)
            except Exception as exc:
                logger.warning(f"Could not remove directory {location}: {exc}")


class MLTrainingCleanCommand(CompositeCommand):
    """Composite cleanup for ML artifacts.

    Aggregates multiple ML-related cleanup commands into a single composite
    operation, including DSLFM checkpoints, MLflow experiments, model caches,
    training artifacts, Optuna databases, and output directories.

    Design Pattern: Composite.
    """

    def __init__(self):
        """Execute init.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        super().__init__(
            "Limpeza completa de ML/DSLFM",
            [
                DSLFMCheckpointsCleanCommand(),
                MLFlowCleanCommand(),
                ModelCacheCleanCommand(),
                TrainingArtifactsCleanCommand(),
                OptunaDatabaseCleanCommand(),
                DashboardResetCommand(),
                DirCleanCommand("Limpando outputs DSLFM", settings.OUTPUTS_DIR / "dslfm"),
            ],
        )


__all__ = [
    "MLFlowCleanCommand",
    "DSLFMCheckpointsCleanCommand",
    "MLTrainingCleanCommand",
    "DashboardResetCommand",
]
