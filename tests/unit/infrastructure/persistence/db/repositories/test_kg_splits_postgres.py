from __future__ import annotations

import asyncio

import polars as pl
import pytest

from pff.infrastructure.persistence.db.repositories.kg_splits_postgres import (
    KGSplitsRepository,
)


class _FakeTransaction:
    async def __aenter__(self) -> None:
        return None

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False


class _LegacySchemaConnection:
    def __init__(self) -> None:
        self.insert_query: str | None = None
        self.insert_records: list[tuple[str, ...]] | None = None
        self.load_query: str | None = None
        self.execute_queries: list[str] = []

    def transaction(self) -> _FakeTransaction:
        return _FakeTransaction()

    async def fetch(self, query: str, *args):
        if "information_schema.columns" in query:
            return [
                {"column_name": "id"},
                {"column_name": "split_name"},
                {"column_name": "split_type"},
                {"column_name": "s"},
                {"column_name": "p"},
                {"column_name": "o"},
                {"column_name": "sample_id"},
            ]

        self.load_query = query
        return [{"s": "10", "p": "20", "o": "30", "sample_id": None}]

    async def execute(self, query: str, *args) -> str:
        self.execute_queries.append(query)
        return "DELETE 0"

    async def executemany(self, query: str, records) -> None:
        self.insert_query = query
        self.insert_records = list(records)


@pytest.mark.asyncio
async def test_save_split_uses_legacy_triple_columns(monkeypatch: pytest.MonkeyPatch) -> None:
    repo = KGSplitsRepository()
    conn = _LegacySchemaConnection()

    async def _fake_execute_with_schema(operation):
        return await operation(conn)

    monkeypatch.setattr(repo, "_execute_with_schema", _fake_execute_with_schema)

    inserted = await repo.save_split(
        "train",
        "preprocessed",
        pl.DataFrame({"s": [1], "p": [2], "o": [3]}),
    )

    assert inserted == 1
    assert conn.insert_query is not None
    assert "(split_name, split_type, s, p, o, sample_id, source)" in conn.insert_query
    assert "ON CONFLICT (split_name, split_type, s, p, o)" in conn.insert_query
    assert conn.insert_records == [("train", "preprocessed", "1", "2", "3", None, "correct.parquet")]


@pytest.mark.asyncio
async def test_load_split_uses_legacy_triple_columns(monkeypatch: pytest.MonkeyPatch) -> None:
    repo = KGSplitsRepository()
    conn = _LegacySchemaConnection()

    async def _fake_execute_with_schema(operation):
        return await operation(conn)

    async def _raise_from_to_thread(func, *args, **kwargs):
        raise RuntimeError("force asyncpg fallback")

    monkeypatch.setattr(repo, "_execute_with_schema", _fake_execute_with_schema)
    monkeypatch.setattr(asyncio, "to_thread", _raise_from_to_thread)

    split = await repo.load_split("train", "preprocessed", map_to_ints=False)

    assert split is not None
    assert split.to_dict(as_series=False) == {"s": ["10"], "p": ["20"], "o": ["30"], "sample_id": [None]}
    assert conn.load_query is not None
    assert "SELECT s as s, p as p, o as o, sample_id" in conn.load_query


@pytest.mark.asyncio
async def test_create_schema_upgrades_legacy_columns() -> None:
    repo = KGSplitsRepository()
    conn = _LegacySchemaConnection()

    await repo._create_schema(conn)

    assert any("ADD COLUMN IF NOT EXISTS source" in query for query in conn.execute_queries)
    assert any(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_kg_splits_unique" in query
        and "(split_name, split_type, s, p, o)" in query
        for query in conn.execute_queries
    )
