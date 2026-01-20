from __future__ import annotations

import pytest

from pff.infrastructure.cleanup.commands.database import (
    KGDataCleanCommand,
    KGRulesCleanCommand,
)


@pytest.mark.asyncio
async def test_kg_data_clean_uses_truncate(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    class FakeRepo:
        async def truncate_all(self) -> int:
            calls.append("truncate")
            return 3

        async def delete_all(self) -> int:
            calls.append("delete")
            return 1

    monkeypatch.setattr(
        "pff.infrastructure.persistence.db.repositories.KGSplitsRepository",
        lambda: FakeRepo(),
    )

    cmd = KGDataCleanCommand()
    deleted = await cmd._execute()

    assert deleted == 3
    assert calls == ["truncate"]


@pytest.mark.asyncio
async def test_kg_rules_clean_uses_truncate(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    class FakeRepo:
        async def truncate_all(self) -> int:
            calls.append("truncate")
            return 2

        async def delete_all(self) -> int:
            calls.append("delete")
            return 1

    monkeypatch.setattr(
        "pff.infrastructure.persistence.db.repositories.kg_rules.KGRulesRepository",
        lambda: FakeRepo(),
    )

    cmd = KGRulesCleanCommand()
    deleted = await cmd._execute()

    assert deleted == 2
    assert calls == ["truncate"]
