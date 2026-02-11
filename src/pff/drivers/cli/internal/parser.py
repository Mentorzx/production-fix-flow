"""Argument parser builder for the CLI."""

from __future__ import annotations

import argparse

from pff import __version__

from .factory import CommandFactory


class CLIParserBuilder:
    """
    Builder class for constructing the CLI argument parser.

    Pattern: Builder Pattern
    """

    def __init__(self):
        self.parser = argparse.ArgumentParser(
            prog="pff",
            description=f"PFF - Production Fix Flow v{__version__}",
            formatter_class=argparse.RawTextHelpFormatter,
        )
        self.subparsers = None

    def add_version(self) -> CLIParserBuilder:
        """Add version argument."""
        self.parser.add_argument(
            "--version",
            action="version",
            version=f"PFF v{__version__}",
        )
        return self

    def create_subparsers(self) -> CLIParserBuilder:
        """Create subparsers for commands."""
        self.subparsers = self.parser.add_subparsers(
            dest="command",
            help="Comandos principais",
        )
        self.subparsers.required = True
        return self

    def add_commands(self) -> CLIParserBuilder:
        """Add all registered commands."""
        if not self.subparsers:
            raise RuntimeError("Must call create_subparsers() first")

        for command_name in CommandFactory.get_all_commands():
            command_class = CommandFactory._command_registry[command_name]
            command_class.configure_parser(self.subparsers)

        return self

    def build(self) -> argparse.ArgumentParser:
        """Build and return the parser."""
        return self.parser
