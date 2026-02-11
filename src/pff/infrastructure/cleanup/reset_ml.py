"""ML reset helper using the cleanup engine."""

from __future__ import annotations

from pff.infrastructure.cleanup.engine import build_engine


async def run_reset_ml(*, auto_yes: bool = True, dry_run: bool = False) -> None:
    """Run the ML cleanup strategy via the cleanup engine.

    Args:
        auto_yes: Skip confirmation prompts.
        dry_run: Enable dry-run mode.
    """
    engine = build_engine("ml", auto_yes=auto_yes, dry_run=dry_run)
    await engine.run()


__all__ = ["run_reset_ml"]
