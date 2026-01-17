"""CLI runner facade."""

from __future__ import annotations

import argparse
import sys

from pff.__main__ import AppLauncher
from pff.shared import logger

from .factory import CommandFactory
from .parser import CLIParserBuilder


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
            # Create command using factory
            command = CommandFactory.create(
                args.command,
                args,
                launcher=self.launcher,
            )

            # Execute command
            await command.execute()

        except KeyboardInterrupt:
            logger.warning(" Application interrupted by user")
            sys.exit(128)
        except Exception as e:
            logger.exception(f"Critical application error: {e}")
            sys.exit(1)
