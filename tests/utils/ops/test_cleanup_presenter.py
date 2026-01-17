import pytest
from rich.console import Console

from pff.infrastructure.cleanup.presenter import CleanupPresenter
from pff.infrastructure.cleanup.commands.database import DatabaseCleanCommand


class DummyPreviewCommand(DatabaseCleanCommand):
    label = "Limpando teste (PostgreSQL)"

    def __init__(self, preview: dict) -> None:
        self._preview = preview

    async def get_preview(self) -> dict:
        return self._preview


@pytest.mark.asyncio
async def test_presenter_skips_zero_previews_and_sizes():
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
    assert "Preview Size" in output
    assert "Espaço alocado" in output
    assert "0B" not in output


def test_confirm_targets_omits_zero_size_suffix():
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

    assert cmd.label in output
    assert "0B" not in output
    assert "tamanho indisponível" in output
