"""
PFF CLI - Production Fix Flow Command Line Interface

Thin wrapper around internal CLI modules while preserving the public API.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pff.__main__ import AppLauncher


from pff.drivers.cli.internal.commands import (
    APICommand,
    CleanCommand,
    Command,
    GenerateCommand,
    HpoCommand,
    LearnCommand,
    LogsCommand,
    ResetMLCommand,
    RunCommand,
    SyncCommand,
    WorkerCommand,
)
from pff.drivers.cli.internal.factory import CommandFactory
from pff.drivers.cli.internal.parser import CLIParserBuilder
from pff.drivers.cli.internal.runner import CLIRunner
from pff.drivers.cli.internal.strategies import (
    FullPipelineStrategy,
    KGCTrainingStrategy,
    KGTrainingStrategy,
    TrainingStrategy,
)

__all__ = [
    "Command",
    "SyncCommand",
    "RunCommand",
    "GenerateCommand",
    "WorkerCommand",
    "APICommand",
    "CleanCommand",
    "ResetMLCommand",
    "LogsCommand",
    "LearnCommand",
    "HpoCommand",
    "TrainingStrategy",
    "KGTrainingStrategy",
    "KGCTrainingStrategy",
    "FullPipelineStrategy",
    "CommandFactory",
    "CLIParserBuilder",
    "CLIRunner",
    "main",
    "cli_entrypoint",
]


async def main(launcher: AppLauncher | None = None, argv: list[str] | None = None):
    """
    Main entry point for the PFF CLI application.

    Args:
        launcher: Optional AppLauncher instance
        argv: Optional list of command-line arguments
    """
    runner = CLIRunner(launcher)
    await runner.run(argv)


def cli_entrypoint() -> None:
    """Entry point for poetry script."""
    from pff import __version__
    from pff.shared.core.logging import logger
    from pff.shared.determinism import (
        configure_numba_threads,
        configure_torch_determinism,
    )
    from pff.shared.system.runtime import initialize_runtime

    configure_torch_determinism(enforce=True)
    configure_numba_threads()
    initialize_runtime(__version__)
    try:
        asyncio.run(main(launcher=None))
    finally:
        logger.complete()


if __name__ == "__main__":
    cli_entrypoint()
