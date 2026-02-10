from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pff.infrastructure.persistence.db.repositories.kg_mappings import (
    KGMappingsRepository,
)


class AsyncContext:
    def __init__(self, result):
        self._result = result

    async def __aenter__(self):
        return self._result

    async def __aexit__(self, exc_type, exc, tb):
        return False


@pytest.mark.asyncio
class TestKGMappingsRepository:
    @pytest.fixture
    def mock_pool(self):
        pool = MagicMock()
        conn = AsyncMock()
        pool.acquire.return_value = AsyncContext(conn)
        conn.transaction = MagicMock(return_value=AsyncContext(None))
        return pool, conn

    async def test_ensure_schema(self, mock_pool):
        pool, conn = mock_pool

        with patch(
            "pff.infrastructure.persistence.db.repositories.base.get_connection_pool",
            return_value=pool,
        ):
            repo = KGMappingsRepository()
            repo.pool = pool

            await repo._ensure_schema()

            assert conn.execute.call_count >= 1
            assert (
                "CREATE TABLE IF NOT EXISTS kg_mappings"
                in conn.execute.call_args_list[0][0][0]
            )

    async def test_save_mappings_uses_copy(self, mock_pool):
        pool, conn = mock_pool

        with patch(
            "pff.infrastructure.persistence.db.repositories.base.get_connection_pool",
            return_value=pool,
        ):
            repo = KGMappingsRepository()
            repo.pool = pool
            repo._schema_ready = True

            mappings = {"user_1": 1, "user_2": 2}
            conn.copy_records_to_table.return_value = None

            inserted = await repo.save_mappings(
                "entity", mappings, batch_size=1, source="test"
            )

            assert inserted == 2
            assert conn.copy_records_to_table.call_count == 2
            args, kwargs = conn.copy_records_to_table.call_args
            assert args[0] == "kg_mappings"
            assert kwargs["columns"] == ("mapping_type", "key", "value", "source")

    async def test_save_mappings_clears_on_empty(self, mock_pool):
        pool, conn = mock_pool
        conn.execute.return_value = "DELETE 3"

        with patch(
            "pff.infrastructure.persistence.db.repositories.base.get_connection_pool",
            return_value=pool,
        ):
            repo = KGMappingsRepository()
            repo.pool = pool
            repo._schema_ready = True

            inserted = await repo.save_mappings("relation", {})

            assert inserted == 0
            conn.execute.assert_called_once()
            assert "DELETE FROM kg_mappings" in conn.execute.call_args[0][0]

    async def test_load_mappings_returns_dict(self, mock_pool):
        pool, conn = mock_pool

        with patch(
            "pff.infrastructure.persistence.db.repositories.base.get_connection_pool",
            return_value=pool,
        ):
            repo = KGMappingsRepository()
            repo.pool = pool
            repo._schema_ready = True

            conn.fetch.return_value = [
                {"key": "u1", "value": 1},
                {"key": "u2", "value": 2},
            ]

            mappings = await repo.load_mappings("entity", use_cache=False)

            assert mappings == {"u1": 1, "u2": 2}
            conn.fetch.assert_called_once()
