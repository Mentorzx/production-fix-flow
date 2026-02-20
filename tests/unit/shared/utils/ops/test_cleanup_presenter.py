"""Provide module-level functionality for the PFF codebase.



Notes:

    File: tests/unit/shared/utils/ops/test_cleanup_presenter.py

"""

import pytest
from rich.console import Console

from pff.infrastructure.cleanup.commands.database import DatabaseCleanCommand
from pff.infrastructure.cleanup.presenter import CleanupPresenter


class DummyPreviewCommand(DatabaseCleanCommand):
    """Represent DummyPreviewCommand."""

    label = "Limpando teste (PostgreSQL)"

    def __init__(self, preview: dict) -> None:
        """Execute init.



        Args:

            preview: Input value used by this callable.

        """

        self._preview = preview

    async def get_preview(self) -> dict:
        """Execute get preview.



        Returns:

            Return value produced by the callable.

        """

        return self._preview


@pytest.mark.asyncio
async def test_presenter_skips_zero_previews_and_sizes():
    """Execute test presenter skips zero previews and sizes.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    console = Console(record=True, width=120)
    presenter = CleanupPresenter(console)

    preview_zero = {
        "description": "Preview Zero",
        "total_rows": 0,
        "size_bytes": 0,
        "sample_rows": [],
    }
    preview_rows_no_size = {
        "description": "Preview Rows",
        "total_rows": 5,
        "size_bytes": 0,
        "sample_rows": [],
    }
    preview_size_only = {
        "description": "Preview Size",
        "total_rows": 0,
        "size_bytes": 1024,
        "sample_rows": [],
    }

    commands = [
        (DummyPreviewCommand(preview_zero), 0),
        (DummyPreviewCommand(preview_rows_no_size), 0),
        (DummyPreviewCommand(preview_size_only), 0),
    ]

    await presenter.display_database_previews(commands)

    output = console.export_text()
    assert "Preview Zero" not in output
    assert "Preview Rows" in output
    assert "Preview Size" not in output  # Skipped because total_rows=0 (line 99 in presenter)
    assert "Espaço alocado" in output
    assert "0B" not in output


def test_confirm_targets_omits_zero_size_suffix():
    """Execute test confirm targets omits zero size suffix.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    console = Console(record=True, width=120)
    presenter = CleanupPresenter(console)

    cmd = DummyPreviewCommand(
        {
            "description": "Preview Rows",
            "total_rows": 5,
            "size_bytes": 0,
            "sample_rows": [],
        }
    )
    cmd.total_rows = 5
    cmd.size_bytes = 0

    presenter.confirm_targets([(cmd, 0)])
    output = console.export_text()

    # Production code skips commands with display_size <= 0
    assert cmd.label not in output
    assert "0B" not in output
