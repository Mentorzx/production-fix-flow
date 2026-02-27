"""Cleanup engine module providing orchestration for cleanup operations.

This module implements the Template Method pattern to orchestrate cleanup
execution with support for graceful shutdown, dry-run mode, and observer
notifications. Integrates with GlobalInterruptManager for safe interruption.

Design Patterns:
    - Template Method: Defines the skeleton of cleanup execution in run().
    - Observer: Notifies observers of command start/complete/error events.
    - Strategy: Accepts CleanupStrategy to build commands dynamically.
    - Factory: build_engine() creates engines with named strategies.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import time
from collections.abc import Iterable
from typing import Any, cast

from rich.console import Console

from pff.infrastructure.cleanup.collector import CleanupScanCollector
from pff.infrastructure.cleanup.commands.base import (
    CleanupCommand,
    CompositeCommand,
    TransparentCompositeCommand,
)
from pff.infrastructure.cleanup.commands.database import (
    DatabaseCleanCommand,
    KGDataCleanCommand,
    KGEmbeddingsCleanCommand,
    KGMappingsCleanCommand,
    KGRulesCleanCommand,
    LanceDBOptimizeCommand,
    OptunaTablesCleanCommand,
    PipelineCheckpointsCleanCommand,
    TrainingMetricsCleanCommand,
)
from pff.infrastructure.cleanup.file_ops import FileOps
from pff.infrastructure.cleanup.observer import CleanupObserver, LoggingCleanupObserver
from pff.infrastructure.cleanup.presenter import CleanupPresenter
from pff.infrastructure.cleanup.strategies.base import CleanupStrategy
from pff.infrastructure.cleanup.strategies.builtin import (
    DeepCleanup,
    MLCleanup,
    ShutdownCleanup,
    StandardCleanup,
)
from pff.shared.core.logging import logger
from pff.shared.acceleration.asyncio_runner import run_coroutine_sync
from pff.shared.ops.global_interrupt_manager import (
    PRIORITY_HIGH,
    get_interrupt_manager,
    should_stop,
)


class CleanupEngine:
    """Engine orchestrating cleanup execution with interrupt handling.

    Coordinates cleanup commands built from a strategy, provides user
    confirmation flow, calculates target sizes, and notifies observers
    of execution progress. Supports dry-run mode and graceful shutdown.

    Attributes:
        _commands: List of cleanup commands from the strategy.
        _console: Rich console for user interaction.
        _auto_yes: Whether to skip confirmation prompts.
        _dry_run: Whether to simulate execution without deleting.
        _interrupt_manager: GlobalInterruptManager for graceful shutdown.
        _presenter: CleanupPresenter for displaying previews and targets.
        _observers: List of observers to notify during execution.
    """

    def __init__(
        self,
        strategy: CleanupStrategy,
        auto_yes: bool = False,
        dry_run: bool = False,
        observers: Iterable[CleanupObserver] | None = None,
    ) -> None:
        """Initialize the cleanup engine with a strategy.

        Args:
            strategy: The cleanup strategy that builds commands.
            auto_yes: If True, skip confirmation prompts.
            dry_run: If True, simulate execution without deleting.
            observers: Optional observers to notify during execution.
                Defaults to LoggingCleanupObserver if not provided.
        """
        self.collector = CleanupScanCollector()
        self._commands = strategy.build_commands(collector=self.collector)
        self._console = Console()
        self._auto_yes = auto_yes
        self._dry_run = dry_run
        self._interrupt_manager = get_interrupt_manager()
        self._should_stop = should_stop
        self._interrupt_callback_label = self._interrupt_manager.register_callback(
            self._emergency_stop,
            priority=PRIORITY_HIGH,
            label="cleanup_engine_emergency",
        )
        self._presenter = CleanupPresenter(self._console)
        self._observers = (
            list(observers) if observers is not None else [LoggingCleanupObserver()]
        )

    def _emergency_stop(self) -> None:
        """Handle emergency interrupts triggered externally.

        Called by GlobalInterruptManager when SIGINT/SIGTERM is received.
        Logs a warning and allows the run loop to exit gracefully.
        """
        logger.warning("Cleanup interrupted - stopping gracefully")

    def __del__(self) -> None:
        """Cleanup destructor that unregisters interrupt callback.

        Safely unregisters the emergency stop callback from the
        interrupt manager to prevent memory leaks.
        """
        try:
            self._interrupt_manager.unregister_callback(self._interrupt_callback_label)
        except Exception:
            return

    def _flatten_commands(self, commands: list[CleanupCommand]) -> list[CleanupCommand]:
        """Flatten nested composite commands into leaf commands.

        Args:
            commands: List of commands that may contain composites.

        Returns:
            Flat list of leaf cleanup commands.
        """
        flattened = []
        for cmd in commands:
            if isinstance(cmd, TransparentCompositeCommand):
                flattened.extend(cmd.get_all_leaf_commands())
            elif isinstance(cmd, CompositeCommand):
                flattened.extend(self._flatten_commands(cmd.children))
            else:
                flattened.append(cmd)
        return flattened

    def _is_db_command(self, cmd: CleanupCommand) -> bool:
        return isinstance(
            cmd,
            (
                DatabaseCleanCommand,
                PipelineCheckpointsCleanCommand,
                KGDataCleanCommand,
                KGMappingsCleanCommand,
                KGEmbeddingsCleanCommand,
                KGRulesCleanCommand,
                LanceDBOptimizeCommand,
                TrainingMetricsCleanCommand,
                OptunaTablesCleanCommand,
            ),
        )

    def _calculate_target_size(self, cmd: CleanupCommand) -> int:
        """Calculate the total size of files targeted by a command.

        Args:
            cmd: The cleanup command to calculate size for.

        Returns:
            Total size in bytes of files that would be cleaned.
        """
        from pff.infrastructure.cleanup.commands.filesystem import (
            DirCleanCommand,
            NestedDirCleanCommand,
        )

        total_size = self._calculate_target_size_by_command_type(
            cmd=cmd,
            dir_clean_cls=DirCleanCommand,
            nested_dir_clean_cls=NestedDirCleanCommand,
        )
        if isinstance(cmd, DirCleanCommand) and cmd._dir.name == ".cache":
            logger.debug(f"Cache size computed: {cmd._dir} size={total_size}")
        return total_size

    def _calculate_target_size_by_command_type(
        self,
        *,
        cmd: CleanupCommand,
        dir_clean_cls: type,
        nested_dir_clean_cls: type,
    ) -> int:
        """Execute calculate target size by command type.



        Args:

            cmd: Input value used by this callable.

            dir_clean_cls: Input value used by this callable.

            nested_dir_clean_cls: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        if isinstance(cmd, dir_clean_cls):
            return self._calculate_dir_clean_target_size(cmd)
        if isinstance(cmd, nested_dir_clean_cls):
            nested_cmd = cast(Any, cmd)
            collector = nested_cmd.collector or self.collector
            collector.scan({nested_cmd.dirname})
            paths = nested_cmd._filtered_paths(collector)
            return sum(FileOps.calculate_size(p) for p in paths)
        if isinstance(cmd, CompositeCommand):
            return sum(self._calculate_target_size(c) for c in cmd.children)
        cmd_any = cast(Any, cmd)
        if hasattr(cmd_any, "calculate_size") and callable(cmd_any.calculate_size):
            size_value = cmd_any.calculate_size()
            return int(size_value) if isinstance(size_value, (int, float)) else 0
        return 0

    @staticmethod
    def _safe_path_size(path: Path) -> int:
        """Execute safe path size.



        Args:

            path: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        try:
            if path.is_file():
                return path.stat().st_size
            if path.is_dir():
                return FileOps.calculate_size(path)
        except FileNotFoundError:
            return 0
        return 0

    def _calculate_dir_clean_target_size(self, cmd: Any) -> int:
        """Execute calculate dir clean target size.



        Args:

            cmd: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        if not cmd._dir.exists():
            return 0
        if cmd._recursive:
            return self._calculate_dir_clean_recursive_size(cmd)
        return self._calculate_dir_clean_non_recursive_size(cmd)

    def _calculate_dir_clean_non_recursive_size(self, cmd: Any) -> int:
        """Execute calculate dir clean non recursive size.



        Args:

            cmd: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        total_size = 0
        if not cmd._pattern and not cmd._recursive:
            iterator = cmd._dir.iterdir()
        else:
            iterator = cmd._dir.glob(cmd._pattern or "*")
        for item in iterator:
            if cmd._is_excluded(item):
                continue
            total_size += self._safe_path_size(item)
        return total_size

    def _calculate_dir_clean_recursive_size(self, cmd: Any) -> int:
        """Execute calculate dir clean recursive size.



        Args:

            cmd: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        ignored_dirs = {
            ".git",
            ".venv",
            "node_modules",
            "__pycache__",
            ".mypy_cache",
            ".pytest_cache",
        }
        pattern = cmd._pattern or "*"
        dir_pattern = pattern[3:] if pattern.startswith("**/") else pattern
        total_size = 0
        for root, dirs, files in os.walk(cmd._dir):
            root_path = Path(root)
            dirs[:] = [d for d in dirs if d not in ignored_dirs]
            if cmd._exclude_dirs:
                dirs[:] = [d for d in dirs if not cmd._is_excluded(root_path / d)]
            total_size += self._sum_matching_files(root_path, files, pattern, cmd)
            total_size += self._sum_matching_dirs(root_path, dirs, dir_pattern, cmd)
        return total_size

    def _sum_matching_files(
        self,
        root_path: Path,
        files: list[str],
        pattern: str,
        cmd: Any,
    ) -> int:
        """Execute sum matching files.



        Args:

            root_path: Input value used by this callable.

            files: Input value used by this callable.

            pattern: Input value used by this callable.

            cmd: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        import fnmatch

        total_size = 0
        file_pattern = pattern[3:] if pattern.startswith("**/") else pattern
        for filename in files:
            file_path = root_path / filename
            if cmd._is_excluded(file_path):
                continue
            if fnmatch.fnmatch(filename, pattern) or fnmatch.fnmatch(
                filename, file_pattern
            ):
                total_size += self._safe_path_size(file_path)
        return total_size

    def _sum_matching_dirs(
        self,
        root_path: Path,
        dirs: list[str],
        dir_pattern: str,
        cmd: Any,
    ) -> int:
        """Execute sum matching dirs.



        Args:

            root_path: Input value used by this callable.

            dirs: Input value used by this callable.

            dir_pattern: Input value used by this callable.

            cmd: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        import fnmatch

        total_size = 0
        matched_dirs: list[str] = []
        for dirname in dirs:
            if not fnmatch.fnmatch(dirname, dir_pattern):
                continue
            full_path = root_path / dirname
            if cmd._exclude_dirs and cmd._is_excluded(full_path):
                continue
            total_size += self._safe_path_size(full_path)
            matched_dirs.append(dirname)
        for dirname in matched_dirs:
            if dirname in dirs:
                dirs.remove(dirname)
        return total_size

    async def _filter_commands(self) -> list[tuple[CleanupCommand, int]]:
        """Filter commands to only include those with targets to clean.

        Calculates sizes concurrently and filters out commands with
        zero-size targets (except database commands).

        Returns:
            List of (command, size) tuples for commands with work to do.
        """
        flat_commands = self._flatten_commands(self._commands)

        from pff.infrastructure.cleanup.commands.filesystem import (
            NestedDirCleanCommand,
        )

        nested_targets = set()
        for cmd in flat_commands:
            if isinstance(cmd, NestedDirCleanCommand):
                nested_targets.add(cmd.dirname)

        if nested_targets:
            self.collector.scan(nested_targets)

        def _get_size(cmd):
            return self._calculate_target_size(cmd)

        command_sizes = [_get_size(cmd) for cmd in flat_commands]

        return [
            (cmd, size)
            for cmd, size in zip(flat_commands, command_sizes)
            if size > 0 or self._is_db_command(cmd)
        ]

    async def _confirm(self) -> list[tuple[CleanupCommand, int]]:
        """Display confirmation prompt and get user approval.

        Shows database previews and target list, then prompts for
        confirmation unless auto_yes or dry_run is enabled.

        Returns:
            List of confirmed (command, size) tuples, or empty if aborted.
        """
        visible_commands_with_sizes = await self._filter_commands()

        if not visible_commands_with_sizes:
            logger.info("Nenhum arquivo ou diretório para limpar.")
            return []

        await self._presenter.display_database_previews(visible_commands_with_sizes)

        display_commands_with_sizes = [
            (cmd, size)
            for cmd, size in visible_commands_with_sizes
            if size > 0
            or getattr(cmd, "size_bytes", 0) > 0
            or getattr(cmd, "total_rows", 0) > 0
        ]

        display_commands_with_sizes = [
            (cmd, size)
            for cmd, size in display_commands_with_sizes
            if not self._is_db_command(cmd) or getattr(cmd, "total_rows", 0) > 0
        ]

        if not display_commands_with_sizes and not self._auto_yes:
            logger.info("Nenhum arquivo ou diretório para limpar.")
            return []

        if display_commands_with_sizes:
            self._presenter.confirm_targets(display_commands_with_sizes)

        adjusted_commands: list[tuple[CleanupCommand, int]] = []
        for cmd, size in visible_commands_with_sizes:
            preview_size = getattr(cmd, "size_bytes", 0) or 0
            adjusted_commands.append((cmd, max(size, int(preview_size))))

        if self._auto_yes:
            return adjusted_commands

        if self._dry_run:
            return []

        response = self._console.input("Prosseguir? (y/N): ")
        if response.lower() != "y":
            logger.info("Abortado.")
            return []

        return adjusted_commands

    async def run(self, confirm: bool = True) -> None:
        """Execute the cleanup commands with optional confirmation.

        Orchestrates the full cleanup flow: filtering commands,
        confirming with user, executing database and file commands,
        and notifying observers. Respects interrupt signals throughout.

        Args:
            confirm: If True, show confirmation prompt before execution.
                Ignored if auto_yes is True.
        """
        if self._should_stop():
            logger.warning("Cleanup aborted due to interrupt signal")
            return

        visible_commands_with_sizes = await self._resolve_visible_commands(
            confirm=confirm
        )
        if self._handle_dry_run(visible_commands_with_sizes):
            return
        if not visible_commands_with_sizes:
            logger.info("Nenhuma tarefa de limpeza a ser executada.")
            return

        if self._should_stop():
            logger.warning("Cleanup aborted due to interrupt signal")
            return

        db_commands, file_commands = self._split_db_and_file_commands(
            visible_commands_with_sizes
        )
        freed_bytes = 0
        should_return = await self._execute_db_commands(db_commands)
        if should_return:
            return
        self._execute_file_commands(file_commands)

        total_size = sum(size for _, size in visible_commands_with_sizes)
        freed_bytes += total_size

        for obs in self._observers:
            obs.on_cleanup_complete(freed_bytes)
        logger.success("Limpeza finalizada com sucesso.")

    async def _resolve_visible_commands(
        self, *, confirm: bool
    ) -> list[tuple[CleanupCommand, int]]:
        """Execute resolve visible commands.



        Args:

            confirm: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        if confirm:
            return await self._confirm()
        return await self._filter_commands()

    def _handle_dry_run(
        self, visible_commands_with_sizes: list[tuple[CleanupCommand, int]]
    ) -> bool:
        """Execute handle dry run.



        Args:

            visible_commands_with_sizes: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        if not self._dry_run:
            return False
        self._console.print(  # noqa: T201
            "[bold yellow]Execução simulada: Os seguintes comandos seriam executados:[/]"
        )
        for cmd, _ in visible_commands_with_sizes:
            self._console.print(f" • {cmd.label}")  # noqa: T201
        return True

    def _split_db_and_file_commands(
        self, visible_commands_with_sizes: list[tuple[CleanupCommand, int]]
    ) -> tuple[list[tuple[CleanupCommand, int]], list[tuple[CleanupCommand, int]]]:
        """Execute split db and file commands.



        Args:

            visible_commands_with_sizes: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        db_commands = [
            (cmd, size)
            for cmd, size in visible_commands_with_sizes
            if self._is_db_command(cmd)
        ]
        file_commands = [
            (cmd, size)
            for cmd, size in visible_commands_with_sizes
            if not self._is_db_command(cmd)
        ]
        return db_commands, file_commands

    async def _execute_db_commands(
        self, db_commands: list[tuple[CleanupCommand, int]]
    ) -> bool:
        """Execute execute db commands.



        Args:

            db_commands: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        from pff.infrastructure.cleanup.commands.database import (
            AbstractDatabaseCleanCommand,
        )

        for cmd, _ in db_commands:
            if self._should_stop():
                logger.warning("Cleanup aborted due to interrupt signal")
                return True
            try:
                if isinstance(cmd, AbstractDatabaseCleanCommand):
                    await cmd.execute_async()
                else:
                    cmd.execute()
            except Exception as exc:
                for obs in self._observers:
                    obs.on_command_error(cmd, exc)
            else:
                for obs in self._observers:
                    obs.on_command_complete(cmd, 0.0)
        return False

    def _execute_file_commands(
        self, file_commands: list[tuple[CleanupCommand, int]]
    ) -> None:
        """Execute execute file commands.



        Args:

            file_commands: Input value used by this callable.

        """

        for cmd_tuple in file_commands:
            self._run_file_command(cmd_tuple)

    def _run_file_command(self, cmd_tuple: tuple[CleanupCommand, int]) -> int:
        """Execute run file command.



        Args:

            cmd_tuple: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        cmd, _ = cmd_tuple
        if self._should_stop():
            return 0
        for obs in self._observers:
            obs.on_command_start(cmd)
        start_time = time.perf_counter()
        try:
            cmd.execute()
        except Exception as exc:
            for obs in self._observers:
                obs.on_command_error(cmd, exc)
            return 0
        duration = (time.perf_counter() - start_time) * 1000
        for obs in self._observers:
            obs.on_command_complete(cmd, duration)
        return 1


def build_engine(strategy_name: str, **kwargs) -> CleanupEngine:
    """Build a CleanupEngine with the specified strategy name.

    Factory function that creates a CleanupEngine configured with one
    of the built-in cleanup strategies.

    Args:
        strategy_name: Name of the strategy ('standard', 'deep', 'ml', 'shutdown').
        **kwargs: Additional arguments passed to CleanupEngine constructor
            (auto_yes, dry_run, observers).

    Returns:
        Configured CleanupEngine instance.

    Raises:
        ValueError: If strategy_name is not recognized.
    """
    strategies = {
        "standard": StandardCleanup,
        "deep": DeepCleanup,
        "ml": MLCleanup,
        "shutdown": ShutdownCleanup,
    }
    strategy_class = strategies.get(strategy_name.lower())
    if not strategy_class:
        available = ", ".join(strategies.keys())
        raise ValueError(
            f"Estratégia de limpeza desconhecida: {strategy_name}. Disponíveis: {available}"
        )
    return CleanupEngine(strategy_class(), **kwargs)


def main() -> None:
    """CLI entry point for the cleanup utility.

    Parses command-line arguments and runs the cleanup engine with
    the specified strategy and options.
    """
    try:
        import uvloop

        uvloop.install()
    except ImportError:
        pass

    parser = argparse.ArgumentParser(
        description="Limpa caches antigos, logs e outputs."
    )
    parser.add_argument(
        "strategy",
        choices=["standard", "deep", "ml", "shutdown"],
        nargs="?",
        default="standard",
        help="A estratégia de limpeza a ser utilizada.",
    )
    parser.add_argument(
        "-y", "--yes", action="store_true", help="Não pedir confirmação."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Simular execução sem deletar."
    )
    ns = parser.parse_args()
    engine = build_engine(ns.strategy, auto_yes=ns.yes, dry_run=ns.dry_run)
    run_coroutine_sync(engine.run())


__all__ = ["CleanupEngine", "build_engine", "main"]
