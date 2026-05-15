from __future__ import annotations

from dataclasses import dataclass

import pytest

from pff.infrastructure.cleanup.commands.database import DatabaseCleanCommand
from pff.infrastructure.cleanup.engine import CleanupEngine


@dataclass
class _DummyFileCommand:
    label: str = "Limpando arquivos temporários"

    def execute(self) -> None:
        return None


class _DummyDatabaseCommand(DatabaseCleanCommand):
    label = "Limpando tabela de teste (PostgreSQL)"

    async def get_preview(self) -> dict | None:
        return None

    async def _execute(self) -> int:
        return 0


class _DummyStrategy:
    def build_commands(self, collector):  # noqa: ANN001
        del collector
        return [_DummyDatabaseCommand(), _DummyFileCommand()]


@pytest.mark.asyncio
async def test_filter_commands_skips_db_when_postgres_is_unreachable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Skip PostgreSQL cleanup commands when connectivity probe fails."""

    engine = CleanupEngine(strategy=_DummyStrategy(), auto_yes=True)

    async def _probe_unreachable() -> tuple[bool, str]:
        return False, "[Errno 111] Connect call failed ('127.0.0.1', 5432)"

    monkeypatch.setattr(
        "pff.infrastructure.cleanup.engine.probe_postgres_reachability",
        _probe_unreachable,
    )
    monkeypatch.setattr(
        engine,
        "_calculate_target_size",
        lambda cmd: 1024 if not engine._is_db_command(cmd) else 0,
    )

    filtered_commands = await engine._filter_commands()

    assert [cmd.label for cmd, _ in filtered_commands] == [
        "Limpando arquivos temporários"
    ]
