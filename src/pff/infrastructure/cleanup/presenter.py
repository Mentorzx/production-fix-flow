"""Cleanup presenter module for console output and user interaction.

This module implements the Single Responsibility Principle by extracting
all console presentation logic from the cleanup engine into a dedicated
presenter class.

Design Patterns:
    - SRP: Dedicated class for presentation concerns only.
    - Presenter: Formats and displays cleanup information to users.
"""

from __future__ import annotations

from typing import Any, cast

from rich.console import Console
from rich.table import Table

from .commands.base import CleanupCommand
from .commands.database import (
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
from .utils import format_size


class CleanupPresenter:
    """Presenter responsible for console output during cleanup operations.

    Handles formatting and display of database previews, target lists,
    and size summaries using Rich console for enhanced output.

    Attributes:
        _console: Rich Console instance for formatted output.
    """

    def __init__(self, console: Console) -> None:
        """Initialize the presenter with a console.

        Args:
            console: Rich Console instance for output.
        """
        self._console = console

    async def display_database_previews(
        self, commands_with_sizes: list[tuple[CleanupCommand, int]]
    ) -> None:
        """Display preview tables for database cleanup commands.

        Shows sample rows and row counts for PostgreSQL tables that
        will be affected by database cleanup commands.

        Args:
            commands_with_sizes: List of (command, size) tuples to preview.
        """
        db_commands = self._filter_database_commands(commands_with_sizes)
        if not db_commands:
            return

        self._console.print(  # noqa: T201
            "\n[bold magenta] Preview das tabelas PostgreSQL que serão limpas:[/]\n"
        )

        previews_with_tables: list[tuple[dict, Table]] = []
        previews_without_tables: list[dict] = []

        for cmd in db_commands:
            preview = await self._get_command_preview(cmd)
            if not self._has_visible_rows(preview):
                continue
            sample_rows = preview.get("sample_rows")
            if sample_rows:
                table = self._build_preview_table(sample_rows)
                previews_with_tables.append((preview, table))
                continue
            previews_without_tables.append(preview)

        for preview, table in previews_with_tables:
            self._print_preview_header(preview, with_newline=True)
            self._console.print(table)  # noqa: T201
            self._console.print("")  # noqa: T201

        for preview in previews_without_tables:
            self._print_preview_header(preview, with_newline=False)

        if previews_without_tables:
            self._console.print("")  # noqa: T201

    @staticmethod
    def _is_database_command(cmd: CleanupCommand) -> bool:
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
                LanceDBOptimizeCommand,
            ),
        )

    def _filter_database_commands(
        self, commands_with_sizes: list[tuple[CleanupCommand, int]]
    ) -> list[CleanupCommand]:
        return [cmd for cmd, _ in commands_with_sizes if self._is_database_command(cmd)]

    async def _get_command_preview(self, cmd: CleanupCommand) -> dict:
        """Execute get command preview.



        Args:

            cmd: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        cmd_any = cast(Any, cmd)
        if not hasattr(cmd_any, "get_preview"):
            return {}
        preview = await cmd_any.get_preview()
        if not preview:
            return {}
        if "size_bytes" in preview:
            cmd_any.size_bytes = preview["size_bytes"]
            if hasattr(cmd_any, "total_rows"):
                cmd_any.total_rows = preview.get("total_rows", 0)
        return preview

    @staticmethod
    def _has_visible_rows(preview: dict) -> bool:
        """Execute has visible rows.



        Args:

            preview: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        if not preview:
            return False
        total_rows = preview.get("total_rows", 0)
        return total_rows > 0

    @staticmethod
    def _build_preview_table(sample_rows: list[dict]) -> Table:
        """Execute build preview table.



        Args:

            sample_rows: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        table = Table(show_header=True, header_style="bold green")
        for column in sample_rows[0].keys():
            table.add_column(column, style="dim")
        for row in sample_rows:
            table.add_row(*CleanupPresenter._format_preview_row(row))
        return table

    @staticmethod
    def _format_preview_row(row: dict) -> list[str]:
        """Execute format preview row.



        Args:

            row: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        formatted_row: list[str] = []
        for value in row.values():
            if value is None:
                formatted_row.append("[dim]NULL[/dim]")
            elif isinstance(value, (int, float)):
                formatted_row.append(str(value))
            else:
                str_value = str(value)
                if len(str_value) > 50:
                    str_value = f"{str_value[:47]}..."
                formatted_row.append(str_value)
        return formatted_row

    def _print_preview_header(self, preview: dict, *, with_newline: bool) -> None:
        """Execute print preview header.



        Args:

            preview: Input value used by this callable.

            with_newline: Input value used by this callable.

        """

        total_rows = preview.get("total_rows", 0)
        size_bytes = preview.get("size_bytes", 0)
        size_str = format_size(size_bytes) if size_bytes > 0 else "tamanho indisponível"
        total_str = f"Total: [bold yellow]{total_rows}[/] registros"
        template = (
            "[bold cyan]  {desc}[/] ({total}, Espaço alocado: {size})\n"
            if with_newline
            else "[bold cyan]  {desc}[/] ({total}, Espaço alocado: {size})"
        )
        self._console.print(  # noqa: T201
            template.format(
                desc=preview["description"],
                total=total_str,
                size=size_str,
            )
        )

    def confirm_targets(
        self, visible_commands_with_sizes: list[tuple[CleanupCommand, int]]
    ) -> int:
        """Display confirmation list of targets to be cleaned.

        Shows all files and directories that will be deleted along with
        their sizes, and calculates the total space to be reclaimed.

        Args:
            visible_commands_with_sizes: List of (command, size) tuples
                representing cleanup targets.

        Returns:
            Total size in bytes that will be freed by cleanup.
        """
        self._console.print(  # noqa: T201
            "[bold yellow]Os diretórios/arquivos a seguir serão apagados:[/]"
        )
        total_size_to_delete = 0
        for cmd, size in visible_commands_with_sizes:
            if hasattr(cmd, "size_bytes") and getattr(cmd, "size_bytes") > 0:
                display_size = getattr(cmd, "size_bytes")
            else:
                display_size = size

            if display_size <= 0:
                continue

            if display_size > 0:
                total_size_to_delete += display_size

            if isinstance(
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
                    LanceDBOptimizeCommand,
                ),
            ):
                if display_size > 0:
                    size_str = f"[bold magenta]({format_size(display_size)})[/]"
                else:
                    size_str = "[bold magenta](tamanho indisponível)[/]"
            else:
                size_str = (
                    f"({format_size(display_size)})"
                    if display_size > 0
                    else "(tamanho indisponível)"
                )

            target_path = getattr(cmd, "_dir", None)
            if not target_path and hasattr(cmd, "dirname"):
                target_path = f"**/{getattr(cmd, 'dirname')}"
            size_suffix = f" [bold cyan]{size_str}[/]" if size_str else ""
            if target_path:
                self._console.print(  # noqa: T201
                    f" • {cmd.label}: {target_path}{size_suffix}"
                )
            else:
                self._console.print(f" • {cmd.label}{size_suffix}")  # noqa: T201

        self._console.print("-" * 30)  # noqa: T201
        if total_size_to_delete > 0:
            self._console.print(  # noqa: T201
                f"Total a ser liberado: [bold green]{format_size(total_size_to_delete)}[/]"
            )
        return total_size_to_delete


__all__ = ["CleanupPresenter"]
