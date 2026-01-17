"""
Repository Pattern for KG entity/relation ID mappings.

Manages the mapping aggregate used during ingestion and validation.
"""

from __future__ import annotations

import asyncio
from typing import Any

import polars as pl

from pff.infrastructure.persistence.db.connection import get_connection_pool
from pff.shared.core.file_manager import FileManager
from pff.shared.core.logger import logger


class KGMappingsRepository:
    """
    Repository for managing entity and relation ID mappings.

    Pattern: Repository + Cache-Aside.
    """

    def __init__(
        self, pool: Any | None = None, file_manager: FileManager | None = None
    ):
        """Initialize repository with optional injected pool and file manager."""
        self.pool = pool
        self._file_manager = file_manager or FileManager()
        self._cache: dict[str, dict[str, int]] = {}
        self._schema_ready = False
        self._schema_lock = asyncio.Lock()

    async def _ensure_pool(self) -> None:
        """Lazy initialization of connection pool and schema."""
        if self.pool is None:
            self.pool = await get_connection_pool()
        await self._ensure_schema()

    async def _ensure_schema(self) -> None:
        """Ensure the kg_mappings table exists."""
        if self._schema_ready or self.pool is None:
            return

        async with self._schema_lock:
            if self._schema_ready:
                return

            async with self.pool.acquire() as conn:
                await conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS kg_mappings (
                        mapping_type VARCHAR(32) NOT NULL,
                        key TEXT NOT NULL,
                        value INTEGER NOT NULL,
                        source VARCHAR(64),
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        PRIMARY KEY (mapping_type, key)
                    )
                    """
                )
                await conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_kg_mappings_value
                    ON kg_mappings (value)
                    """
                )

            self._schema_ready = True

    async def save_mappings(
        self,
        mapping_type: str,
        mappings: dict[str, int],
        batch_size: int = 5000,
        source: str | None = None,
    ) -> int:
        """
        Save entity or relation mappings to PostgreSQL.

        Args:
            mapping_type: 'entity' or 'relation'
            mappings: Dictionary {label: id}
            batch_size: Records per batch
            source: Optional source identifier for auditing

        Returns:
            Number of mappings inserted

        Pattern: Batch Processing.
        """
        await self._ensure_pool()

        total = len(mappings)
        logger.debug(f"mappings_saving type={mapping_type} n={total:,}")

        columns = ("mapping_type", "key", "value", "source")

        async with self.pool.acquire() as conn:
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
                        if batch_end < total:
                            logger.debug(
                                f"Mapping batch inserted: {batch_start:,}-{batch_end:,}"
                            )

        self._cache.pop(mapping_type, None)

        logger.debug(f"mappings_saved type={mapping_type} n={inserted:,}")
        return inserted

    async def load_mappings(
        self, mapping_type: str, use_cache: bool = True
    ) -> dict[str, int] | None:
        """
        Load entity or relation mappings from PostgreSQL.

        Args:
            mapping_type: 'entity' or 'relation'
            use_cache: Use in-memory cache if available

        Returns:
            Dictionary {label: id} or None if not found

        Pattern: Cache-Aside
        """
        if use_cache and mapping_type in self._cache:
            logger.debug(f"{mapping_type} mappings loaded from cache")
            return self._cache[mapping_type]

        await self._ensure_pool()

        logger.debug(f"mappings_loading type={mapping_type}")

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT key, value FROM kg_mappings WHERE mapping_type = $1",
                mapping_type,
            )

        if not rows:
            logger.warning(f"No {mapping_type} mappings found in database")
            return None

        mappings = {row["key"]: row["value"] for row in rows}

        # Update cache
        self._cache[mapping_type] = mappings

        logger.debug(f"mappings_loaded type={mapping_type} n={len(mappings):,}")
        return mappings

    async def load_mappings_as_dataframe(
        self, mapping_type: str
    ) -> pl.DataFrame | None:
        """
        Load mappings as Polars DataFrame (for compatibility).

        Args:
            mapping_type: 'entity' or 'relation'

        Returns:
            DataFrame with columns [id, label] or None
        """
        mappings = await self.load_mappings(mapping_type, use_cache=True)

        if mappings is None:
            return None

        # Create DataFrame matching expected format
        data = {"id": list(mappings.values()), "label": list(mappings.keys())}

        return pl.DataFrame(data).sort("id")

    async def save_mappings_from_dataframe(
        self,
        mapping_type: str,
        df: pl.DataFrame,
        source: str | None = None,
    ) -> int:
        """
        Save mappings from a Polars DataFrame with ``id`` and ``label`` columns.

        Args:
            mapping_type: 'entity' or 'relation'
            df: DataFrame containing columns ``id`` and ``label``
            source: Optional source identifier

        Returns:
            Number of mappings inserted.
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

        Args:
            mapping_type: 'entity' or 'relation'
            key: Entity or relation label

        Returns:
            ID or None if not found
        """
        mappings = await self.load_mappings(mapping_type, use_cache=True)

        if mappings is None:
            return None

        return mappings.get(key)

    async def get_label(self, mapping_type: str, value: int) -> str | None:
        """
        Get label for a specific ID (reverse lookup).

        Args:
            mapping_type: 'entity' or 'relation'
            value: ID to look up

        Returns:
            Label or None if not found
        """
        mappings = await self.load_mappings(mapping_type, use_cache=True)

        if mappings is None:
            return None

        # Reverse lookup
        reverse = {v: k for k, v in mappings.items()}
        return reverse.get(value)

    async def mapping_exists(self, mapping_type: str) -> bool:
        """
        Check if mappings exist for a type.

        Args:
            mapping_type: 'entity' or 'relation'

        Returns:
            True if mappings exist
        """
        await self._ensure_pool()

        async with self.pool.acquire() as conn:
            count = await conn.fetchval(
                "SELECT COUNT(*) FROM kg_mappings WHERE mapping_type = $1", mapping_type
            )

        return count > 0

    async def delete_mappings(self, mapping_type: str) -> int:
        """
        Delete mappings for a specific type.

        Args:
            mapping_type: 'entity' or 'relation'

        Returns:
            Number of records deleted
        """
        await self._ensure_pool()

        async with self.pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM kg_mappings WHERE mapping_type = $1", mapping_type
            )

        # Invalidate cache
        self._cache.pop(mapping_type, None)

        deleted = int(result.split()[-1]) if result else 0

        if deleted > 0:
            logger.info(f"{deleted:,} {mapping_type} mappings deletados")

        return deleted

    async def delete_all(self) -> int:
        """
        Delete all mappings from PostgreSQL.

        Returns:
            Number of records deleted
        """
        await self._ensure_pool()

        async with self.pool.acquire() as conn:
            result = await conn.execute("DELETE FROM kg_mappings")

        # Clear cache
        self._cache.clear()

        deleted = int(result.split()[-1]) if result else 0

        if deleted > 0:
            logger.info(f"Todos os {deleted:,} mappings deletados")

        return deleted

    async def get_statistics(self) -> dict:
        """
        Get statistics about stored mappings.

        Returns:
            Dictionary with mapping statistics
        """
        await self._ensure_pool()

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT mapping_type, COUNT(*) as count
                FROM kg_mappings
                GROUP BY mapping_type
                ORDER BY mapping_type
                """
            )

        stats = {row["mapping_type"]: row["count"] for row in rows}
        return stats

    def clear_cache(self):
        """Clear in-memory cache."""
        self._cache.clear()
        logger.debug("Mapping cache cleared")
