"""Cleanup presenter module for console output and user interaction.

This module implements the Single Responsibility Principle by extracting
all console presentation logic from the cleanup engine into a dedicated
presenter class.

Design Patterns:
    - SRP: Dedicated class for presentation concerns only.
    - Presenter: Formats and displays cleanup information to users.
"""

from __future__ import annotations

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
        db_commands = [
            cmd
            for cmd, _ in commands_with_sizes
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
            )
        ]

        if not db_commands:
            return

        self._console.print(
            "\n[bold magenta] Preview das tabelas PostgreSQL que serão limpas:[/]\n"
        )

        previews_with_tables: list[tuple[dict, Table]] = []
        previews_without_tables: list[dict] = []

        for cmd in db_commands:
            if hasattr(cmd, "get_preview"):
                preview = await cmd.get_preview()

                if preview and "size_bytes" in preview:
                    cmd.size_bytes = preview["size_bytes"]
                    if hasattr(cmd, "total_rows"):
                        cmd.total_rows = preview.get("total_rows", 0)

                if preview:
                    total_rows = preview.get("total_rows", 0)
                    size_bytes = preview.get("size_bytes", 0)
                    if total_rows <= 0:
                        continue

                    sample_rows = preview.get("sample_rows")
                    if sample_rows:
                        table = Table(show_header=True, header_style="bold green")
                        for column in sample_rows[0].keys():
                            table.add_column(column, style="dim")
                        for row in sample_rows:
                            formatted_row = []
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
                            table.add_row(*formatted_row)
                        previews_with_tables.append((preview, table))
                    else:
                        previews_without_tables.append(preview)

        for preview, table in previews_with_tables:
            total_rows = preview.get("total_rows", 0)
            size_bytes = preview.get("size_bytes", 0)
            size_str = format_size(size_bytes) if size_bytes > 0 else "tamanho indisponível"
            total_str = f"Total: [bold yellow]{total_rows}[/] registros"
            self._console.print(
                "[bold cyan]  {desc}[/] ({total}, Espaço alocado: {size})\n".format(
                    desc=preview["description"],
                    total=total_str,
                    size=size_str,
                )
            )
            self._console.print(table)
            self._console.print("")

        for preview in previews_without_tables:
            total_rows = preview.get("total_rows", 0)
            size_bytes = preview.get("size_bytes", 0)
            size_str = format_size(size_bytes) if size_bytes > 0 else "tamanho indisponível"
            total_str = f"Total: [bold yellow]{total_rows}[/] registros"
            self._console.print(
                "[bold cyan]  {desc}[/] ({total}, Espaço alocado: {size})".format(
                    desc=preview["description"],
                    total=total_str,
                    size=size_str,
                )
            )

        if previews_without_tables:
            self._console.print("")

    def confirm_targets(self, visible_commands_with_sizes: list[tuple[CleanupCommand, int]]) -> int:
        """Display confirmation list of targets to be cleaned.

        Shows all files and directories that will be deleted along with
        their sizes, and calculates the total space to be reclaimed.

        Args:
            visible_commands_with_sizes: List of (command, size) tuples
                representing cleanup targets.

        Returns:
            Total size in bytes that will be freed by cleanup.
        """
        self._console.print("[bold yellow]Os diretórios/arquivos a seguir serão apagados:[/]")
        total_size_to_delete = 0
        for cmd, size in visible_commands_with_sizes:
            if hasattr(cmd, "size_bytes") and getattr(cmd, "size_bytes") > 0:
                display_size = getattr(cmd, "size_bytes")
            else:
                display_size = size

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
                self._console.print(f" • {cmd.label}: {target_path}{size_suffix}")
            else:
                self._console.print(f" • {cmd.label}{size_suffix}")

        self._console.print("-" * 30)
        if total_size_to_delete > 0:
            self._console.print(
                f"Total a ser liberado: [bold green]{format_size(total_size_to_delete)}[/]"
            )
        return total_size_to_delete


__all__ = ["CleanupPresenter"]
