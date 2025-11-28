import numpy as np
import pytest
from unittest.mock import AsyncMock, MagicMock

import pff.db.repositories.embeddings as embeddings_module
from pff.db.repositories.embeddings import EmbeddingsRepository


class _DummyAcquire:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _DummyPool:
    def __init__(self, conn):
        self._conn = conn

    def acquire(self):
        return _DummyAcquire(self._conn)


@pytest.mark.asyncio
async def test_search_similar_uses_cache(monkeypatch):
    monkeypatch.setattr(embeddings_module, "register_postgres_listener", AsyncMock())

    repo = EmbeddingsRepository()
    repo._cache.clear()

    fetch_mock = AsyncMock(
        return_value=[{"entity": "E1", "distance": 0.12}, {"entity": "E2", "distance": 0.34}]
    )
    conn = MagicMock()
    conn.fetch = fetch_mock

    repo.pool = _DummyPool(conn)
    monkeypatch.setattr(repo, "_ensure_pool", AsyncMock())

    query_vec = np.array([0.1, 0.2, 0.3], dtype=np.float32)

    first = await repo.search_similar(query_vec, top_k=2, model_version="v1", entity_type="entity")
    assert first == [
        {"entity": "E1", "distance": 0.12, "score": pytest.approx(0.892857)},
        {"entity": "E2", "distance": 0.34, "score": pytest.approx(0.746269)},
    ]
    assert fetch_mock.call_count == 1

    # Second call should hit cache (no additional fetch)
    second = await repo.search_similar(query_vec, top_k=2, model_version="v1", entity_type="entity")
    assert second == first
    assert fetch_mock.call_count == 1


@pytest.mark.asyncio
async def test_search_similar_latest_version_query(monkeypatch):
    monkeypatch.setattr(embeddings_module, "register_postgres_listener", AsyncMock())

    repo = EmbeddingsRepository()
    repo._cache.clear()

    fetch_mock = AsyncMock(return_value=[{"entity": "E3", "distance": 0.56}])
    conn = MagicMock()
    conn.fetch = fetch_mock

    repo.pool = _DummyPool(conn)
    monkeypatch.setattr(repo, "_ensure_pool", AsyncMock())

    query_vec = np.array([0.4, 0.5, 0.6], dtype=np.float32)
    await repo.search_similar(query_vec, top_k=1, model_version=None, entity_type="relation")

    assert fetch_mock.call_count == 1
    sql = fetch_mock.call_args.args[0]
    assert "SELECT model_version" in sql  # ensures latest-version subquery path is used
