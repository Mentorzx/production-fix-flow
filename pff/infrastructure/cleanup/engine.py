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
import asyncio
import os
import time
from collections.abc import Iterable

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
from pff.shared.acceleration.concurrency import ConcurrencyManager
from pff.shared.core.logging import logger
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
        self._observers = list(observers) if observers is not None else [LoggingCleanupObserver()]

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

    def _calculate_target_size(self, cmd: CleanupCommand) -> int:
        """Calculate the total size of files targeted by a command.

        Args:
            cmd: The cleanup command to calculate size for.

        Returns:
            Total size in bytes of files that would be cleaned.
        """
        from pff.infrastructure.cleanup.commands.filesystem import (
            DirCleanCommand,
        )

        total_size = 0

        ignored_dirs = {
            ".git",
            ".venv",
            "node_modules",
            "__pycache__",
            ".mypy_cache",
            ".pytest_cache",
        }

        if isinstance(cmd, DirCleanCommand):
            if cmd._dir.exists():
                if not cmd._recursive:
                    for item in cmd._dir.glob(cmd._pattern or "*"):
                        try:
                            if item.is_file():
                                total_size += item.stat().st_size
                            elif item.is_dir():
                                total_size += FileOps.calculate_size(item)
                        except FileNotFoundError:
                            continue
                else:
                    import fnmatch

                    for root, dirs, files in os.walk(cmd._dir):
                        dirs[:] = [d for d in dirs if d not in ignored_dirs]

                        pattern = cmd._pattern or "*"

                        for f in files:
                            if fnmatch.fnmatch(f, pattern) or (
                                pattern.startswith("**/") and fnmatch.fnmatch(f, pattern[3:])
                            ):
                                total_size += os.path.getsize(os.path.join(root, f))

                        matched_dirs = []
                        for d in dirs:
                            check_pattern = pattern[3:] if pattern.startswith("**/") else pattern
                            if fnmatch.fnmatch(d, check_pattern):
                                from pathlib import Path

                                full_path = Path(root) / d
                                total_size += FileOps.calculate_size(full_path)
                                matched_dirs.append(d)

                        for d in matched_dirs:
                            dirs.remove(d)

        elif isinstance(cmd, CompositeCommand):
            total_size += sum(self._calculate_target_size(c) for c in cmd.children)

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
            PyCacheCleanCommand,
        )

        nested_targets = set()
        for cmd in flat_commands:
            if isinstance(cmd, NestedDirCleanCommand):
                nested_targets.add(cmd.dirname)
            elif isinstance(cmd, PyCacheCleanCommand):
                nested_targets.add("__pycache__")

        if nested_targets:
            self.collector.scan(nested_targets)

        def _get_size(cmd):
            if isinstance(cmd, NestedDirCleanCommand):
                return self.collector.get_size(cmd.dirname)
            if isinstance(cmd, PyCacheCleanCommand):
                return self.collector.get_size("__pycache__")
            return self._calculate_target_size(cmd)

        cm = ConcurrencyManager()
        command_sizes = await cm.execute(
            _get_size,
            [(cmd,) for cmd in flat_commands],
            task_type="thread",
            desc="Scanning alvos de limpeza",
        )

        def is_db_command(cmd):
            return isinstance(
                cmd,
                (
                    DatabaseCleanCommand,
                    PipelineCheckpointsCleanCommand,
                    KGDataCleanCommand,
                    KGMappingsCleanCommand,
                    KGEmbeddingsCleanCommand,
                    KGRulesCleanCommand,
                    TrainingMetricsCleanCommand,
                    OptunaTablesCleanCommand,
                ),
            )

        return [
            (cmd, size)
            for cmd, size in zip(flat_commands, command_sizes)
            if size > 0 or is_db_command(cmd)
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

        visible_commands_with_sizes = [
            (cmd, size)
            for cmd, size in visible_commands_with_sizes
            if size > 0 or getattr(cmd, "size_bytes", 0) > 0 or getattr(cmd, "total_rows", 0) > 0
        ]

        def is_db_command(cmd):
            return isinstance(
                cmd,
                (
                    DatabaseCleanCommand,
                    PipelineCheckpointsCleanCommand,
                    KGDataCleanCommand,
                    KGMappingsCleanCommand,
                    KGEmbeddingsCleanCommand,
                    KGRulesCleanCommand,
                    TrainingMetricsCleanCommand,
                    OptunaTablesCleanCommand,
                ),
            )

        visible_commands_with_sizes = [
            (cmd, size)
            for cmd, size in visible_commands_with_sizes
            if not is_db_command(cmd) or getattr(cmd, "total_rows", 0) > 0
        ]

        if not visible_commands_with_sizes:
            logger.info("Nenhum arquivo ou diretório para limpar.")
            return []

        self._presenter.confirm_targets(visible_commands_with_sizes)

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

        if confirm and not self._auto_yes:
            visible_commands_with_sizes = await self._confirm()
        else:
            visible_commands_with_sizes = await self._filter_commands()

        if self._dry_run:
            self._console.print(
                "[bold yellow]Execução simulada: Os seguintes comandos seriam executados:[/]"
            )
            for cmd, _ in visible_commands_with_sizes:
                self._console.print(f" • {cmd.label}")
            return

        if not visible_commands_with_sizes:
            logger.info("Nenhuma tarefa de limpeza a ser executada.")
            return

        if self._should_stop():
            logger.warning("Cleanup aborted due to interrupt signal")
            return

        def is_db_command(cmd):
            return isinstance(
                cmd,
                (
                    DatabaseCleanCommand,
                    PipelineCheckpointsCleanCommand,
                    KGDataCleanCommand,
                    KGMappingsCleanCommand,
                    KGEmbeddingsCleanCommand,
                    KGRulesCleanCommand,
                    TrainingMetricsCleanCommand,
                    OptunaTablesCleanCommand,
                ),
            )

        db_commands = [
            (cmd, size) for cmd, size in visible_commands_with_sizes if is_db_command(cmd)
        ]
        file_commands = [
            (cmd, size)
            for cmd, size in visible_commands_with_sizes
            if not isinstance(
                cmd,
                (
                    DatabaseCleanCommand,
                    PipelineCheckpointsCleanCommand,
                    KGDataCleanCommand,
                    KGMappingsCleanCommand,
                    KGEmbeddingsCleanCommand,
                    KGRulesCleanCommand,
                    TrainingMetricsCleanCommand,
                    OptunaTablesCleanCommand,
                ),
            )
        ]

        freed_bytes = 0
        for cmd, _ in db_commands:
            if self._should_stop():
                logger.warning("Cleanup aborted due to interrupt signal")
                return
            try:
                if hasattr(cmd, "execute_async"):
                    await cmd.execute_async()
                else:
                    cmd.execute()
            except Exception as exc:
                for obs in self._observers:
                    obs.on_command_error(cmd, exc)
            else:
                for obs in self._observers:
                    obs.on_command_complete(cmd, 0.0)

        if file_commands:
            cm = ConcurrencyManager()

            def _run_cmd(cmd_tuple):
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
                else:
                    duration = (time.perf_counter() - start_time) * 1000
                    for obs in self._observers:
                        obs.on_command_complete(cmd, duration)
                    return 1

            await cm.execute(
                _run_cmd,
                [(c,) for c in file_commands],
                task_type="thread",
                desc="Executando limpeza de arquivos",
            )

        total_size = sum(size for _, size in visible_commands_with_sizes)
        freed_bytes += total_size

        for obs in self._observers:
            obs.on_cleanup_complete(freed_bytes)
        logger.success("Limpeza finalizada com sucesso.")


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

    parser = argparse.ArgumentParser(description="Limpa caches antigos, logs e outputs.")
    parser.add_argument(
        "strategy",
        choices=["standard", "deep", "ml", "shutdown"],
        nargs="?",
        default="standard",
        help="A estratégia de limpeza a ser utilizada.",
    )
    parser.add_argument("-y", "--yes", action="store_true", help="Não pedir confirmação.")
    parser.add_argument("--dry-run", action="store_true", help="Simular execução sem deletar.")
    ns = parser.parse_args()
    engine = build_engine(ns.strategy, auto_yes=ns.yes, dry_run=ns.dry_run)
    asyncio.run(engine.run())


__all__ = ["CleanupEngine", "build_engine", "main"]
