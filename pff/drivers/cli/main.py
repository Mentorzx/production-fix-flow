"""
PFF CLI - Production Fix Flow Command Line Interface

Thin wrapper around internal CLI modules while preserving the public API.
"""

from __future__ import annotations

import asyncio
import os
import sys
import warnings

# Suppress Transformers deprecation warnings and set unified cache
if "TRANSFORMERS_CACHE" in os.environ and "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = os.environ["TRANSFORMERS_CACHE"]

# Performance optimizations for PyTorch memory management
if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

warnings.filterwarnings(
    "ignore", category=FutureWarning, module="transformers.utils.hub"
)

try:
    import _xxsubinterpreters  # noqa: F401
except ModuleNotFoundError:
    from pff.shared.compat import xxsubinterpreters_stub as _xxsubinterpreters_stub

# Register stub in sys.modules for re-export
sys.modules.setdefault("_xxsubinterpreters", _xxsubinterpreters_stub)

from pff.__main__ import AppLauncher  # noqa: E402
from pff.drivers.cli.internal.commands import (  # noqa: E402
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
from pff.drivers.cli.internal.factory import CommandFactory  # noqa: E402
from pff.drivers.cli.internal.parser import CLIParserBuilder  # noqa: E402
from pff.drivers.cli.internal.runner import CLIRunner  # noqa: E402
from pff.drivers.cli.internal.strategies import (  # noqa: E402
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
    from pff.shared.determinism import (
        configure_numba_threads,
        configure_torch_determinism,
    )
    from pff.shared.system.runtime import initialize_runtime
    from pff.shared.core.logger import logger
    from pff import __version__

    configure_torch_determinism(enforce=True)
    configure_numba_threads()
    initialize_runtime(__version__)
    try:
        asyncio.run(main(launcher=None))
    finally:
        logger.complete()


if __name__ == "__main__":
    cli_entrypoint()
