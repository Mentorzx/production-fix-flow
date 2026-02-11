"""
Repository Pattern for KG entity/relation ID mappings.

Manages the mapping aggregate used during ingestion and validation.
"""

from __future__ import annotations

from typing import Any

import polars as pl

import asyncpg

from pff.infrastructure.persistence.db.repositories.base import PostgresRepository
from pff.shared.core.file_manager import FileManager
from pff.shared.core.logging import logger


class KGMappingsRepository(PostgresRepository):
    """
    Repository for managing entity and relation ID mappings.
    """

    def __init__(
        self, pool: Any | None = None, file_manager: FileManager | None = None
    ):
        """Initialize repository with optional injected pool and file manager."""
        super().__init__(pool=pool, file_manager=file_manager)
        self._cache: dict[str, dict[str, int]] = {}

    async def _create_schema(self, conn: asyncpg.Connection) -> None:
        """Create kg_mappings table and indexes."""
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS kg_mappings (
                mapping_type VARCHAR(32) NOT NULL,
                key TEXT NOT NULL,
                value INTEGER NOT NULL,
                source VARCHAR(64),
                created_at TIMESTAMPTZ DEFAULT NOW(),
                PRIMARY KEY (mapping_type, key)
            )
            """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_kg_mappings_value
            ON kg_mappings (value)
            """)

    async def save_mappings(
        self,
        mapping_type: str,
        mappings: dict[str, int],
        batch_size: int = 5000,
        source: str | None = None,
    ) -> int:
        """
        Save entity or relation mappings to PostgreSQL.
        """
        await self._ensure_pool()

        total = len(mappings)
        logger.debug(f"mappings_saving type={mapping_type} n={total:,}")

        columns = ("mapping_type", "key", "value", "source")

        if self.pool is None:
            raise RuntimeError("Database pool not initialized")
        pool = self.pool

        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    "DELETE FROM kg_mappings WHERE mapping_type = $1",
                    mapping_type,
                )

                if total == 0:
                    inserted = 0
                else:
                    records = [
                        (mapping_type, key, value, source)
                        for key, value in mappings.items()
                    ]

                    inserted = 0
                    for batch_start in range(0, total, batch_size):
                        batch_end = min(batch_start + batch_size, total)
                        batch = records[batch_start:batch_end]
                        await conn.copy_records_to_table(
                            "kg_mappings", records=batch, columns=columns
                        )
                        inserted += len(batch)

        self._cache.pop(mapping_type, None)

        logger.debug(f"mappings_saved type={mapping_type} n={inserted:,}")
        return inserted

    async def load_mappings(
        self, mapping_type: str, use_cache: bool = True
    ) -> dict[str, int] | None:
        """
        Load entity or relation mappings from PostgreSQL.
        """
        if use_cache and mapping_type in self._cache:
            return self._cache[mapping_type]

        await self._ensure_pool()

        if self.pool is None:
            raise RuntimeError("Database pool not initialized")
        pool = self.pool

        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT key, value FROM kg_mappings WHERE mapping_type = $1",
                mapping_type,
            )

        if not rows:
            return None

        mappings = {row["key"]: row["value"] for row in rows}
        self._cache[mapping_type] = mappings

        return mappings

    async def load_mappings_as_dataframe(
        self, mapping_type: str
    ) -> pl.DataFrame | None:
        """
        Load mappings as Polars DataFrame.
        """
        mappings = await self.load_mappings(mapping_type, use_cache=True)

        if mappings is None:
            return None

        data = {"id": list(mappings.values()), "label": list(mappings.keys())}

        return pl.DataFrame(data).sort("id")

    async def save_mappings_from_dataframe(
        self,
        mapping_type: str,
        df: pl.DataFrame,
        source: str | None = None,
    ) -> int:
        """
        Save mappings from a Polars DataFrame with id and label columns.
        """
        if "label" not in df.columns or "id" not in df.columns:
            logger.warning(
                "Invalid mappings DataFrame; expected columns ['id', 'label']"
            )
            return 0

        mapping_dict = {
            str(label): int(idx)
            for idx, label in zip(df["id"].to_list(), df["label"].to_list())
        }

        return await self.save_mappings(mapping_type, mapping_dict, source=source)

    async def get_id(self, mapping_type: str, key: str) -> int | None:
        """
        Get ID for a specific key.
        """
        mappings = await self.load_mappings(mapping_type, use_cache=True)

        if mappings is None:
            return None

        return mappings.get(key)

    async def get_label(self, mapping_type: str, value: int) -> str | None:
        """
        Get label for a specific ID (reverse lookup).
        """
        mappings = await self.load_mappings(mapping_type, use_cache=True)

        if mappings is None:
            return None

        reverse = {v: k for k, v in mappings.items()}
        return reverse.get(value)

    async def mapping_exists(self, mapping_type: str) -> bool:
        """
        Check if mappings exist for a type.
        """
        await self._ensure_pool()

        if self.pool is None:
            raise RuntimeError("Database pool not initialized")
        pool = self.pool

        async with pool.acquire() as conn:
            count = await conn.fetchval(
                "SELECT COUNT(*) FROM kg_mappings WHERE mapping_type = $1", mapping_type
            )

        return bool(count and count > 0)

    async def delete_mappings(self, mapping_type: str) -> int:
        """
        Delete mappings for a specific type.
        """
        await self._ensure_pool()

        if self.pool is None:
            raise RuntimeError("Database pool not initialized")
        pool = self.pool

        async with pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM kg_mappings WHERE mapping_type = $1", mapping_type
            )

        self._cache.pop(mapping_type, None)

        deleted = int(result.split()[-1]) if result else 0

        if deleted > 0:
            logger.info(f"{deleted:,} {mapping_type} mappings deletados")

        return deleted

    async def delete_all(self) -> int:
        """
        Delete all mappings from PostgreSQL.
        """
        await self._ensure_pool()

        if self.pool is None:
            raise RuntimeError("Database pool not initialized")
        pool = self.pool

        async with pool.acquire() as conn:
            result = await conn.execute("DELETE FROM kg_mappings")

        self._cache.clear()

        deleted = int(result.split()[-1]) if result else 0

        if deleted > 0:
            logger.info(f"Todos os {deleted:,} mappings deletados")

        return deleted

    async def get_statistics(self) -> dict[str, int]:
        """
        Get statistics about stored mappings.
        """
        await self._ensure_pool()

        if self.pool is None:
            raise RuntimeError("Database pool not initialized")
        pool = self.pool

        async with pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT mapping_type, COUNT(*) as count
                FROM kg_mappings
                GROUP BY mapping_type
                ORDER BY mapping_type
                """)

        stats = {row["mapping_type"]: row["count"] for row in rows}
        return stats

    def clear_cache(self):
        """Clear in-memory cache."""
        self._cache.clear()
        logger.debug("Mapping cache cleared")
