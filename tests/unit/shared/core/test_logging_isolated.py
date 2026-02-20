"""Provide module-level functionality for the PFF codebase.



Notes:

    File: tests/unit/shared/core/test_logging_isolated.py

"""

from __future__ import annotations

from pathlib import Path

from pff.shared.core.logging import create_isolated_logger


def test_create_isolated_logger_writes_files(tmp_path: Path) -> None:
    """Execute test create isolated logger writes files.



    Args:

        tmp_path: Input value used by this callable.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    log_dir = tmp_path / "dashboard"
    log = create_isolated_logger("hpo_dashboard", log_dir=log_dir)
    log.info("component_name=hpo_dashboard message='teste isolamento'")
    log.complete()

    assert any(log_dir.glob("*.log"))
    readable_dir = log_dir / "readable"
    if readable_dir.exists():
        assert any(readable_dir.glob("*.log"))
