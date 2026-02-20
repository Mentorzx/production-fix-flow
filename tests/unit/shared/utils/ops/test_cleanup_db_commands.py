"""Provide module-level functionality for the PFF codebase.



Notes:

    File: tests/unit/shared/utils/ops/test_cleanup_db_commands.py

"""

from __future__ import annotations

import pytest

from pff.infrastructure.cleanup.commands.database import (
    KGDataCleanCommand,
    KGRulesCleanCommand,
)


@pytest.mark.asyncio
async def test_kg_data_clean_uses_truncate(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute test kg data clean uses truncate.



    Args:

        monkeypatch: Input value used by this callable.



    Returns:

        Return value produced by the callable.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    calls: list[str] = []

    class FakeRepo:
        """Represent FakeRepo."""

        async def truncate_all(self) -> int:
            """Execute truncate all.



            Returns:

                Return value produced by the callable.

            """

            calls.append("truncate")
            return 3

        async def delete_all(self) -> int:
            """Execute delete all.



            Returns:

                Return value produced by the callable.

            """

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
    """Execute test kg rules clean uses truncate.



    Args:

        monkeypatch: Input value used by this callable.



    Returns:

        Return value produced by the callable.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    calls: list[str] = []

    class FakeRepo:
        """Represent FakeRepo."""

        async def truncate_all(self) -> int:
            """Execute truncate all.



            Returns:

                Return value produced by the callable.

            """

            calls.append("truncate")
            return 2

        async def delete_all(self) -> int:
            """Execute delete all.



            Returns:

                Return value produced by the callable.

            """

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
