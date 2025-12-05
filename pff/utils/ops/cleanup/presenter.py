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
    KGRulesCleanCommand,
    PipelineCheckpointsCleanCommand,
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
                    KGRulesCleanCommand,
                ),
            )
        ]

        if not db_commands:
            return

        self._console.print("\n[bold magenta] Preview das tabelas PostgreSQL que serão limpas:[/]\n")

        for cmd in db_commands:
            if hasattr(cmd, "get_preview"):
                preview = await cmd.get_preview()

                if preview and "size_bytes" in preview:
                    cmd.size_bytes = preview["size_bytes"]

                if preview and preview.get("total_rows", 0) > 0:
                    size_str = format_size(preview.get("size_bytes", 0))
                    self._console.print(
                        f"[bold cyan]  {preview['description']}[/] (Total: [bold yellow]{preview['total_rows']}[/] registros, {size_str})\n"
                    )

                    if preview.get("sample_rows"):
                        table = Table(show_header=True, header_style="bold green")

                        sample_rows = preview["sample_rows"]
                        if sample_rows:
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

                            self._console.print(table)
                            self._console.print("")
                else:
                    desc = preview["description"] if preview else getattr(cmd, "label", "Tabela desconhecida")
                    self._console.print(f"[dim]  {desc}: 0 registros (0B)[/]\n")

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
        self._console.print("[bold yellow]Os diretórios/arquivos a seguir serão apagados:[/]")
        total_size_to_delete = 0
        for cmd, size in visible_commands_with_sizes:
            if hasattr(cmd, "size_bytes") and getattr(cmd, "size_bytes") > 0:
                display_size = getattr(cmd, "size_bytes")
            else:
                display_size = size

            total_size_to_delete += display_size

            from .commands.database import (
                DatabaseCleanCommand,
                KGDataCleanCommand,
                KGRulesCleanCommand,
                PipelineCheckpointsCleanCommand,
            )

            if isinstance(cmd, (DatabaseCleanCommand, PipelineCheckpointsCleanCommand, KGDataCleanCommand, KGRulesCleanCommand)):
                size_str = f"[bold magenta]({format_size(display_size)})[/]"
            else:
                size_str = f"({format_size(display_size)})"

            target_path = getattr(cmd, "_dir", None)
            if not target_path and hasattr(cmd, "dirname"):
                target_path = f"**/{getattr(cmd, 'dirname')}"
            if target_path:
                self._console.print(f" • {cmd.label}: {target_path} [bold cyan]{size_str}[/]")
            else:
                self._console.print(f" • {cmd.label} [bold cyan]{size_str}[/]")

        self._console.print("-" * 30)
        self._console.print(
            f"Total a ser liberado: [bold green]{format_size(total_size_to_delete)}[/]"
        )
        return total_size_to_delete


__all__ = ["CleanupPresenter"]
