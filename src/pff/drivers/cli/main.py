"""
PFF CLI - Production Fix Flow Command Line Interface

Thin wrapper around internal CLI modules while preserving the public API.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pff.__main__ import AppLauncher
    from pff.drivers.cli.internal.commands import Command

from pff.shared.acceleration.asyncio_runner import run_coroutine_sync


def _is_clean_command(argv: list[str]) -> bool:
    for arg in argv[1:]:
        if arg.startswith("-"):
            continue
        return arg == "clean"
    return False


if _is_clean_command(sys.argv):
    os.environ.setdefault("PFF_CLEAN_MODE", "1")
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    os.environ.setdefault("FILEMANAGER_DISABLE_CONFIG_CACHE", "1")
    os.environ.setdefault("CACHE_DIR", str(Path(tempfile.gettempdir()) / "pff_clean_cache"))
    os.environ.setdefault("LOG_DIR", str(Path(tempfile.gettempdir()) / "pff_clean_logs"))


__all__ = [  # noqa: F822
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

_LAZY_ATTRS = {
    "APICommand": "pff.drivers.cli.internal.commands",
    "CleanCommand": "pff.drivers.cli.internal.commands",
    "Command": "pff.drivers.cli.internal.commands",
    "GenerateCommand": "pff.drivers.cli.internal.commands",
    "HpoCommand": "pff.drivers.cli.internal.commands",
    "LearnCommand": "pff.drivers.cli.internal.commands",
    "LogsCommand": "pff.drivers.cli.internal.commands",
    "ResetMLCommand": "pff.drivers.cli.internal.commands",
    "RunCommand": "pff.drivers.cli.internal.commands",
    "SyncCommand": "pff.drivers.cli.internal.commands",
    "WorkerCommand": "pff.drivers.cli.internal.commands",
    "CommandFactory": "pff.drivers.cli.internal.factory",
    "CLIParserBuilder": "pff.drivers.cli.internal.parser",
    "CLIRunner": "pff.drivers.cli.internal.runner",
    "TrainingStrategy": "pff.drivers.cli.internal.strategies",
    "KGTrainingStrategy": "pff.drivers.cli.internal.strategies",
    "KGCTrainingStrategy": "pff.drivers.cli.internal.strategies",
    "FullPipelineStrategy": "pff.drivers.cli.internal.strategies",
}


def __getattr__(name: str):
    module_path = _LAZY_ATTRS.get(name)
    if not module_path:
        raise AttributeError(name)
    module = importlib.import_module(module_path)
    return getattr(module, name)


async def main(launcher: AppLauncher | None = None, argv: list[str] | None = None):
    """
    Main entry point for the PFF CLI application.

    Args:
        launcher: Optional AppLauncher instance
        argv: Optional list of command-line arguments
    """
    from pff.drivers.cli.internal.runner import CLIRunner

    runner = CLIRunner(launcher)
    await runner.run(argv)


def _run_clean_command(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="pff clean")
    parser.add_argument(
        "strategy",
        choices=["standard", "deep", "ml", "shutdown"],
        nargs="?",
        default="standard",
    )
    parser.add_argument("-y", "--yes", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv or sys.argv[2:])

    import importlib

    importlib.reload(importlib.import_module("pff.shared.core.config"))

    from pff.infrastructure.cleanup.engine import build_engine
    from pff.shared.core.logging import logger

    logger.info(f"Iniciando limpeza: estrategia={args.strategy} dry_run={args.dry_run}")
    engine = build_engine(args.strategy, auto_yes=args.yes, dry_run=args.dry_run)
    run_coroutine_sync(engine.run())


def cli_entrypoint() -> None:
    """Entry point for poetry script."""
    from pff import __version__
    from pff.shared.core.logging import logger
    from pff.shared.system.runtime import initialize_runtime

    if _is_clean_command(sys.argv):
        _run_clean_command()
        return
    initialize_runtime(__version__)
    try:
        run_coroutine_sync(main(launcher=None))
    finally:
        logger.complete()


if __name__ == "__main__":
    cli_entrypoint()
