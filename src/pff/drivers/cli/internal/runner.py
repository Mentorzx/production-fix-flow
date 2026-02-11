"""CLI runner facade."""

from __future__ import annotations

import argparse
import sys
from typing import TYPE_CHECKING

from pff.shared.core.logging import logger

from .factory import CommandFactory
from .parser import CLIParserBuilder

if TYPE_CHECKING:
    from pff.__main__ import AppLauncher


class CLIRunner:
    """
    Main CLI runner class.

    Pattern: Facade Pattern (simplifies complex subsystems)
    """

    def __init__(self, launcher: AppLauncher | None = None):
        self.launcher = launcher
        self.parser = self._build_parser()

    def _build_parser(self) -> argparse.ArgumentParser:
        """Build argument parser using builder."""
        builder = CLIParserBuilder()
        return builder.add_version().create_subparsers().add_commands().build()

    async def run(self, argv: list[str] | None = None) -> None:
        """
        Run CLI application.

        Args:
            argv: Optional list of arguments (defaults to sys.argv)
        """
        args = self.parser.parse_args(argv)

        try:
            command = CommandFactory.create(
                args.command,
                args,
                launcher=self.launcher,
            )

            await command.execute()

        except KeyboardInterrupt:
            logger.warning(" Application interrupted by user")
            sys.exit(128)
        except Exception as e:
            logger.exception(f"Critical application error: {e}")
            sys.exit(1)
