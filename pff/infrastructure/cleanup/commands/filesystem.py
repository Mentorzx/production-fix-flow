from __future__ import annotations

from pathlib import Path

from pff import settings
from pff.infrastructure.cleanup.file_ops import FileOps
from pff.shared.core.logger import logger

from .base import CleanupCommand


class DirCleanCommand(CleanupCommand):
    """Clean files and directories based on a glob pattern.

    Removes files and subdirectories matching a pattern within the target
    directory. Uses `FileOps.rmtree_sync` for interrupt-aware removal.

    Args:
        label: Human-readable description for UI display.
        directory: Target directory to clean.
        pattern: Glob pattern to match (default: "*" matches all).
        recursive: If True, uses `rglob` for recursive matching.

    Attributes:
        label: Display label for UI.
    """

    def __init__(
        self,
        label: str,
        directory: Path,
        pattern: str | None = None,
        recursive: bool = False,
    ):
        self.label = label
        self._dir = directory
        self._pattern = pattern
        self._recursive = recursive

    def execute(self) -> None:
        """Execute the cleanup operation.

        Iterates through matching files/directories and removes them.
        Permission errors on non-log files are logged as warnings.
        """
        if not self._dir.exists():
            return
        iterator = (
            self._dir.rglob(self._pattern or "*")
            if self._recursive
            else self._dir.glob(self._pattern or "*")
        )
        for item in iterator:
            if item.is_dir():
                FileOps.rmtree_sync(item, ignore_errors=True)
            else:
                try:
                    item.unlink(missing_ok=True)
                except PermissionError:
                    try:
                        item.open("w").close()
                        item.unlink(missing_ok=True)
                    except Exception as exc:  # noqa: BLE001
                        if not item.suffix == ".log":
                            logger.warning(f"Could not remove {item}: {exc}")


class LogArchiverCommand(CleanupCommand):
    """Compress log files using Zstandard before they are potentially removed.

    Attributes:
        label: Display label for UI.
    """

    label = "Arquivando logs com Zstandard"

    def __init__(self, logs_dir: Path):
        self._logs_dir = logs_dir

    def execute(self) -> None:
        """Execute archiving of .log files."""
        if not self._logs_dir.exists():
            return

        for log_file in self._logs_dir.glob("*.log"):
            # Only archive files that are not being currently written to (simple heuristic)
            if log_file.stat().st_size > 0:
                FileOps.archive_with_zstd(log_file, delete_original=True)


class NestedDirCleanCommand(CleanupCommand):
    """Clean all nested directories with a specific name.

    Recursively finds and removes all directories matching `dirname` under
    the project root.

    Args:
        dirname: Name of directories to remove (e.g., "__pycache__").
        label: Human-readable description for UI display.

    Attributes:
        dirname: Directory name pattern to match.
        label: Display label for UI.
    """

    def __init__(self, dirname: str, label: str, collector=None):
        self.dirname = dirname
        self.label = label
        self.collector = collector

    def execute(self) -> None:
        """Execute the cleanup operation synchronously."""
        from pff.infrastructure.cleanup.collector import CleanupScanCollector

        collector = self.collector or CleanupScanCollector()
        collector.scan({self.dirname})
        paths = collector.get_paths(self.dirname)

        for p in paths:
            FileOps.rmtree_sync(p, ignore_errors=True)


class PyCacheCleanCommand(CleanupCommand):
    """Remove all __pycache__ directories.

    Recursively finds and removes Python bytecode cache directories from
    the entire project tree.

    Attributes:
        label: Display label for UI.
    """

    label = "Removendo __pycache__"

    def __init__(self, collector=None):
        self.collector = collector

    def execute(self) -> None:
        """Execute the cleanup operation synchronously."""
        from pff.infrastructure.cleanup.collector import CleanupScanCollector

        collector = self.collector or CleanupScanCollector()
        collector.scan({"__pycache__"})
        paths = collector.get_paths("__pycache__")

        for p in paths:
            FileOps.rmtree_sync(p, ignore_errors=True)


class ModelCacheCleanCommand(CleanupCommand):
    """Remove model caches from common locations.

    Cleans PyTorch and HuggingFace cache directories from both project-local
    and user home locations.

    Attributes:
        label: Display label for UI.
    """

    label = "Limpando cache de modelos"

    def execute(self) -> None:
        """Execute the cleanup operation.

        Removes cache directories for torch and huggingface from multiple locations.
        """
        cache_locations = [
            settings.OUTPUTS_DIR / "dslfm" / "temp_models",
            settings.CACHE_DIR / "torch",
            settings.CACHE_DIR / "huggingface",
            Path.home() / ".cache" / "torch",
            Path.home() / ".cache" / "huggingface",
        ]

        for cache_dir in cache_locations:
            if cache_dir.exists():
                logger.debug(f"Removendo cache: {cache_dir}")
                FileOps.rmtree_sync(cache_dir, ignore_errors=True)


class TrainingArtifactsCleanCommand(CleanupCommand):
    """Remove temporary training artifacts.

    Cleans temporary files generated during ML training, including Optuna
    trial configs, temporary YAML files, and training state snapshots.

    Attributes:
        label: Display label for UI.
    """

    label = "Limpando artefatos de treinamento"

    def execute(self) -> None:
        """Execute the cleanup operation.

        Removes temporary training files matching predefined patterns.
        """
        artifacts_patterns = [
            settings.OUTPUTS_DIR / "dslfm" / "temp_*",
            settings.OUTPUTS_DIR / "dslfm" / "*_temp.yaml",
            settings.OUTPUTS_DIR / "temp_config_trial_*.yaml",
            settings.ROOT_DIR / "temp_config_trial_*.yaml",
            settings.OUTPUTS_DIR / "**" / "*.tmp",
            settings.OUTPUTS_DIR / "**" / "training_state_*.json",
        ]

        for pattern in artifacts_patterns:
            if "*" in str(pattern):
                parent = pattern.parent
                pattern_name = pattern.name
                if parent.exists():
                    for item in parent.glob(pattern_name):
                        try:
                            if item.is_file():
                                item.unlink(missing_ok=True)
                            elif item.is_dir():
                                FileOps.rmtree_sync(item, ignore_errors=True)
                        except Exception as exc:  # noqa: BLE001
                            logger.warning(f"Could not remove {item}: {exc}")
            elif pattern.exists():
                if pattern.is_file():
                    pattern.unlink(missing_ok=True)
                elif pattern.is_dir():
                    FileOps.rmtree_sync(pattern, ignore_errors=True)

        logger.info(" Artefatos de treinamento removidos")


class OptunaDatabaseCleanCommand(CleanupCommand):
    """Remove Optuna SQLite databases.

    Cleans SQLite database files created by Optuna for hyperparameter
    optimization studies.

    Attributes:
        label: Display label for UI.
    """

    label = "Limpando bancos Optuna"

    def execute(self) -> None:
        """Execute the cleanup operation.

        Removes `.db` files from ROOT_DIR and OUTPUTS_DIR.
        """
        optuna_files = [
            settings.ROOT_DIR / "optuna.db",
            settings.ROOT_DIR / "**/*.db",
            settings.OUTPUTS_DIR / "**/*.db",
        ]

        for pattern in optuna_files:
            if "*" in str(pattern):
                parent = pattern.parent
                pattern_name = pattern.name
                if parent.exists():
                    for item in parent.rglob(pattern_name):
                        try:
                            item.unlink(missing_ok=True)
                        except Exception as exc:  # noqa: BLE001
                            logger.warning(f"Could not remove {item}: {exc}")
            elif pattern.exists():
                pattern.unlink(missing_ok=True)

        logger.info(" Bancos de dados Optuna removidos")


__all__ = [
    "DirCleanCommand",
    "NestedDirCleanCommand",
    "PyCacheCleanCommand",
    "ModelCacheCleanCommand",
    "TrainingArtifactsCleanCommand",
    "OptunaDatabaseCleanCommand",
]
