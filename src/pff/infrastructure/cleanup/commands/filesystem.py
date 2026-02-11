from __future__ import annotations

from pathlib import Path

from pff.infrastructure.cleanup.file_ops import FileOps
from pff.shared.core.config import settings
from pff.shared.core.logging import logger

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
        exclude_dirs: list[Path] | None = None,
        remove_dir: bool = False,
    ):
        self.label = label
        self._dir = directory
        self._pattern = pattern
        self._recursive = recursive
        self._exclude_dirs = list(exclude_dirs or [])
        self._remove_dir = remove_dir

    def _is_excluded(self, path: Path) -> bool:
        for root in self._exclude_dirs:
            try:
                if path.is_relative_to(root):
                    return True
            except AttributeError:
                if str(path).startswith(str(root)):
                    return True
        return False

    def execute(self) -> None:
        """Execute the cleanup operation.

        Iterates through matching files/directories and removes them.
        Permission errors on non-log files are logged as warnings.
        """
        if not self._dir.exists():
            return
        if not self._pattern and not self._recursive:
            for item in self._dir.iterdir():
                if self._is_excluded(item):
                    continue
                if item.is_dir():
                    FileOps.rmtree_sync(item, ignore_errors=True)
                else:
                    try:
                        item.unlink(missing_ok=True)
                    except PermissionError:
                        try:
                            from pff.shared.core.file_manager import FileManager

                            FileManager().save(b"", item)
                            item.unlink(missing_ok=True)
                        except Exception as exc:
                            if not item.suffix == ".log":
                                logger.warning(f"Could not remove {item}: {exc}")
            return
        iterator = (
            self._dir.rglob(self._pattern or "*")
            if self._recursive
            else self._dir.glob(self._pattern or "*")
        )
        for item in iterator:
            if self._is_excluded(item):
                continue
            if item.is_dir():
                FileOps.rmtree_sync(item, ignore_errors=True)
            else:
                try:
                    item.unlink(missing_ok=True)
                except PermissionError:
                    try:
                        from pff.shared.core.file_manager import FileManager

                        FileManager().save(b"", item)
                        item.unlink(missing_ok=True)
                    except Exception as exc:
                        if not item.suffix == ".log":
                            logger.warning(f"Could not remove {item}: {exc}")

        if self._remove_dir and self._dir.exists():
            FileOps.rmtree_sync(self._dir, ignore_errors=True)


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

    def __init__(self, dirname: str, label: str, collector=None, exclude_roots=None):
        self.dirname = dirname
        self.label = label
        self.collector = collector
        self.exclude_roots = list(exclude_roots or [])

    def _is_excluded(self, path: Path) -> bool:
        for root in self.exclude_roots:
            try:
                if path.is_relative_to(root):
                    return True
            except AttributeError:
                if str(path).startswith(str(root)):
                    return True
        return False

    def _filtered_paths(self, collector) -> list[Path]:
        paths = collector.get_paths(self.dirname)
        if not self.exclude_roots:
            return paths
        return [p for p in paths if not self._is_excluded(p)]

    def execute(self) -> None:
        """Execute the cleanup operation synchronously."""
        from pff.infrastructure.cleanup.collector import CleanupScanCollector

        collector = self.collector or CleanupScanCollector()
        collector.scan({self.dirname})
        paths = self._filtered_paths(collector)

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
                logger.debug(f"Removing cache: {cache_dir}")
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
                        except Exception as exc:
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
                        except Exception as exc:
                            logger.warning(f"Could not remove {item}: {exc}")
            elif pattern.exists():
                pattern.unlink(missing_ok=True)

        logger.info(" Bancos de dados Optuna removidos")


class LintCacheCleanCommand(CleanupCommand):
    """Remove lint and tooling caches scattered across the project.

    Targets: .mypy_cache, .ruff_cache, .pytest_cache, .pyright,
    .pylint.d, .eslintcache, and the guardrail DiskCache under
    outputs/.cache/guardrail.

    Attributes:
        label: Display label for UI.
    """

    label = "Limpando caches de lint/tooling"

    LINT_CACHE_DIRS: list[str] = [
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
        ".pyright",
        ".pylint.d",
        ".eslintcache",
    ]

    def _iter_cache_paths(self) -> list[Path]:
        """Collect all existing lint cache directories, skipping .venv/node_modules."""
        root = settings.ROOT_DIR
        paths: list[Path] = []

        for cache_name in self.LINT_CACHE_DIRS:
            for cache_path in root.rglob(cache_name):
                if not cache_path.is_dir():
                    continue
                if any(skip in cache_path.parts for skip in (".venv", "node_modules")):
                    continue
                paths.append(cache_path)

        guardrail_cache = root / "outputs" / ".cache" / "guardrail"
        if guardrail_cache.is_dir():
            paths.append(guardrail_cache)

        return paths

    def calculate_size(self) -> int:
        """Calculate total size of lint cache directories."""
        return sum(FileOps.calculate_size(p) for p in self._iter_cache_paths())

    def execute(self) -> None:
        """Remove lint cache directories and the guardrail DiskCache."""
        paths = self._iter_cache_paths()

        for cache_path in paths:
            FileOps.rmtree_sync(cache_path, ignore_errors=True)

        if paths:
            logger.info(f"Caches de lint removidos: {len(paths)} diretorios")


__all__ = [
    "DirCleanCommand",
    "NestedDirCleanCommand",
    "ModelCacheCleanCommand",
    "TrainingArtifactsCleanCommand",
    "OptunaDatabaseCleanCommand",
    "LintCacheCleanCommand",
]
