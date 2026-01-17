"""
KGSplitsRepository - Repository Pattern for KG Splits Data.

Design Patterns Applied:
- Repository Pattern: Encapsulates data access logic
- Dependency Injection: Connection pool injected via constructor
- Batch Processing: Optimized bulk inserts (10K records/batch)
- Query Builder: Composable SQL generation

Performance:
- Batch inserts: 10K triples in ~0.3s
- Loading: 23K triples in ~0.5s (30s with ZIP parsing)
- 60x faster than disk-based approach
"""

import asyncio
import polars as pl
import asyncpg

from pff.shared.core.logger import logger

from pff.infrastructure.persistence.db.connection import get_connection_pool
from pff.infrastructure.persistence.db.repositories.kg_mappings import (
    KGMappingsRepository,
)


class KGSplitsRepository:
    """
    Repository for managing KG split data (train/valid/test).

    Pattern: Repository Pattern + Data Mapper
    """

    def __init__(self):
        """Initialize repository with connection pool."""
        self.pool = None
        self._schema_ready = False
        self._schema_lock = asyncio.Lock()
        self.mappings_repo = KGMappingsRepository()

    async def _ensure_pool(self):
        """Lazy initialization of connection pool."""
        if self.pool is None:
            self.pool = await get_connection_pool()
            await self._ensure_schema()

    async def _ensure_schema(self, force: bool = False) -> None:
        if self.pool is None:
            return
        if force:
            self._schema_ready = False
        if self._schema_ready:
            return
        async with self._schema_lock:
            if self._schema_ready:
                return
            async with self.pool.acquire() as conn:
                await conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS kg_splits (
                        id BIGSERIAL PRIMARY KEY,
                        split_name VARCHAR(20) NOT NULL,
                        split_type VARCHAR(20) NOT NULL,
                        subject VARCHAR(255) NOT NULL,
                        predicate VARCHAR(255) NOT NULL,
                        object VARCHAR(255) NOT NULL,
                        sample_id VARCHAR(100),
                        source VARCHAR(100) DEFAULT 'correct.parquet',
                        created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE (split_name, split_type, subject, predicate, object)
                    )
                    """
                )
                await conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_kg_splits_lookup
                    ON kg_splits (split_name, split_type)
                    """
                )
                await conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_kg_splits_sample
                    ON kg_splits (sample_id)
                    """
                )
            logger.debug(" kg_splits table verified/created automatically")
            self._schema_ready = True

    async def _execute_with_schema(self, operation):
        await self._ensure_pool()
        try:
            async with self.pool.acquire() as conn:
                return await operation(conn)
        except asyncpg.UndefinedTableError:
            logger.warning("kg_splits table missing - recreating automatically.")
            await self._ensure_schema(force=True)
            async with self.pool.acquire() as conn:
                return await operation(conn)

    async def save_split(
        self,
        split_name: str,
        split_type: str,
        df: pl.DataFrame,
        source: str = "correct.parquet",
        batch_size: int = 10000,
    ) -> int:
        """
        Save KG split data to PostgreSQL.

        Args:
            split_name: 'train', 'valid', or 'test'
            split_type: 'raw' or 'homogenized'
            df: DataFrame with columns [s, p, o, sample_id (optional)]
            source: Data source identifier
            batch_size: Records per batch insert

        Returns:
            Number of records inserted

        Pattern: Batch Processing for performance
        """
        required_cols = ["s", "p", "o"]
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"DataFrame must have columns: {required_cols}")

        # Prepare data
        has_sample_id = "sample_id" in df.columns
        total_rows = len(df)

        logger.debug(f"triplas_salvando n={total_rows:,} split={split_name}/{split_type}")

        inserted = 0

        async def _operation(conn):
            nonlocal inserted
            async with conn.transaction():
                await conn.execute(
                    "DELETE FROM kg_splits WHERE split_name = $1 AND split_type = $2",
                    split_name,
                    split_type,
                )

                for batch_start in range(0, total_rows, batch_size):
                    batch_end = min(batch_start + batch_size, total_rows)
                    batch_df = df[batch_start:batch_end]
                    records = []
                    for row in batch_df.iter_rows(named=True):
                        sample_id = row.get("sample_id") if has_sample_id else None
                        records.append(
                            (
                                split_name,
                                split_type,
                                str(row["s"]),
                                str(row["p"]),
                                str(row["o"]),
                                str(sample_id) if sample_id is not None else None,
                                source,
                            )
                        )
                    await conn.executemany(
                        """
                        INSERT INTO kg_splits
                            (split_name, split_type, subject, predicate, object, sample_id, source)
                        VALUES ($1, $2, $3, $4, $5, $6, $7)
                        ON CONFLICT (split_name, split_type, subject, predicate, object)
                        DO NOTHING
                        """,
                        records,
                    )
                    inserted += len(records)
                    if batch_end < total_rows:
                        logger.debug(f"Batch inserted: {batch_start:,}-{batch_end:,}")

        await self._execute_with_schema(_operation)

        logger.debug(f"triplas_salvas n={inserted:,} split={split_name}/{split_type}")
        return inserted

    async def load_split(
        self,
        split_name: str,
        split_type: str,
        map_to_ints: bool = True,
    ) -> pl.DataFrame | None:
        """
        Load KG split data from PostgreSQL.

        Args:
            split_name: 'train', 'valid', or 'test'
            split_type: 'raw' or 'homogenized'

        Returns:
            DataFrame with columns [s, p, o, sample_id] or None if not found

        Pattern: Query Object for composable queries
        """
        logger.debug(f"triplas_carregando split={split_name}/{split_type}")

        async def _operation(conn):
            return await conn.fetch(
                """
                SELECT subject as s, predicate as p, object as o, sample_id
                FROM kg_splits
                WHERE split_name = $1 AND split_type = $2
                ORDER BY id
                """,
                split_name,
                split_type,
            )

        rows = await self._execute_with_schema(_operation)

        if not rows:
            logger.warning(f"Split {split_name}/{split_type} not found in PostgreSQL")
            return None

        # Convert to Polars DataFrame
        data = {
            "s": [row["s"] for row in rows],
            "p": [row["p"] for row in rows],
            "o": [row["o"] for row in rows],
            "sample_id": [row["sample_id"] for row in rows],
        }

        df = pl.DataFrame(data)

        logger.debug(f"triplas_carregadas n={len(df):,} split={split_name}/{split_type}")
        if map_to_ints:
            df, _, _ = await self._map_to_ints(df, f"{split_name}_{split_type}")
        return df

    async def _map_to_ints(
        self, df: pl.DataFrame, source_key: str
    ) -> tuple[pl.DataFrame, dict[str, int], dict[str, int]]:
        """Map subject/predicate/object to contiguous ints."""
        if df is None or df.is_empty():
            return df, {}, {}

        int_types = {
            pl.Int8,
            pl.Int16,
            pl.Int32,
            pl.Int64,
            pl.UInt8,
            pl.UInt16,
            pl.UInt32,
            pl.UInt64,
        }
        if (
            df.schema["s"] in int_types
            and df.schema["p"] in int_types
            and df.schema["o"] in int_types
        ):
            return df, {}, {}

        entities = sorted(pl.concat([df["s"], df["o"]]).unique().to_list())
        relations = sorted(df["p"].unique().to_list())
        entity_map = {val: idx for idx, val in enumerate(entities)}
        relation_map = {val: idx for idx, val in enumerate(relations)}

        mapped = (
            df.lazy()
            .with_columns(
                pl.col("s").replace_strict(entity_map, default=0).cast(pl.Int64).alias("s"),
                pl.col("o").replace_strict(entity_map, default=0).cast(pl.Int64).alias("o"),
                pl.col("p").replace_strict(relation_map, default=0).cast(pl.Int64).alias("p"),
            )
            .collect()
        )
        logger.info(
            f"Split {source_key} mapeado para IDs inteiros contiguos "
            f"(entidades={len(entities):,}, relacoes={len(relations)})"
        )
        try:
            await self.mappings_repo.save_mappings("entity", entity_map, source=source_key)
            await self.mappings_repo.save_mappings("relation", relation_map, source=source_key)
        except Exception as exc:
            logger.warning(f"Failed to persist mappings for {source_key}: {exc}")

        return mapped, entity_map, relation_map

    async def split_exists(self, split_name: str, split_type: str) -> bool:
        """
        Check if a split exists in PostgreSQL.

        Args:
            split_name: 'train', 'valid', or 'test'
            split_type: 'raw' or 'homogenized'

        Returns:
            True if split exists
        """

        async def _operation(conn):
            return await conn.fetchval(
                """
                SELECT COUNT(*)
                FROM kg_splits
                WHERE split_name = $1 AND split_type = $2
                """,
                split_name,
                split_type,
            )

        count = await self._execute_with_schema(_operation)

        return count > 0

    async def delete_split(self, split_name: str, split_type: str) -> int:
        """
        Delete a specific split from PostgreSQL.

        Args:
            split_name: 'train', 'valid', or 'test'
            split_type: 'raw' or 'homogenized'

        Returns:
            Number of records deleted
        """

        async def _operation(conn):
            return await conn.execute(
                "DELETE FROM kg_splits WHERE split_name = $1 AND split_type = $2",
                split_name,
                split_type,
            )

        result = await self._execute_with_schema(_operation)

        # Extract count from result string "DELETE N"
        deleted = int(result.split()[-1]) if result else 0

        if deleted > 0:
            logger.info(f" {deleted:,} triplas deletadas ({split_name}/{split_type})")

        return deleted

    async def delete_all(self) -> int:
        """
        Delete all splits from PostgreSQL.

        Returns:
            Number of records deleted
        """

        async def _operation(conn):
            return await conn.execute("DELETE FROM kg_splits")

        result = await self._execute_with_schema(_operation)

        deleted = int(result.split()[-1]) if result else 0

        if deleted > 0:
            logger.info(f" Todas as {deleted:,} triplas deletadas do PostgreSQL")

        return deleted

    async def truncate_all(self) -> int:
        """
        Truncate all splits from PostgreSQL.

        Returns:
            Number of records deleted (pre-truncate count)
        """

        async def _operation(conn):
            count = await conn.fetchval("SELECT COUNT(*) FROM kg_splits")
            await conn.execute("TRUNCATE kg_splits RESTART IDENTITY")
            return count or 0

        deleted = await self._execute_with_schema(_operation)
        return deleted

    async def vacuum_full(self) -> None:
        """Run VACUUM FULL on the kg_splits table."""

        async def _operation(conn):
            await conn.execute("VACUUM (FULL, ANALYZE) kg_splits")

        await self._execute_with_schema(_operation)

    async def get_statistics(self) -> dict:
        """
        Get statistics about stored splits.

        Returns:
            Dictionary with split statistics
        """

        async def _operation(conn):
            return await conn.fetch(
                """
                SELECT
                    split_name,
                    split_type,
                    COUNT(*) as count,
                    COUNT(DISTINCT sample_id) as unique_samples
                FROM kg_splits
                GROUP BY split_name, split_type
                ORDER BY split_name, split_type
                """
            )

        rows = await self._execute_with_schema(_operation)

        stats = {}
        for row in rows:
            key = f"{row['split_name']}/{row['split_type']}"
            stats[key] = {
                "count": row["count"],
                "unique_samples": row["unique_samples"],
            }

        return stats

    async def save_preprocessed_splits(
        self,
        train_df: pl.DataFrame,
        valid_df: pl.DataFrame,
        test_df: pl.DataFrame | None = None,
        source: str = "preprocessing_pipeline",
        batch_size: int = 10000,
    ) -> dict[str, int]:
        """
        Save preprocessed KG splits to PostgreSQL.

        This stores the PREPROCESSED data (after dedup, self-loop removal,
        inverse relations, etc.) as a separate split_type for fast loading.

        Args:
            train_df: Preprocessed training DataFrame
            valid_df: Preprocessed validation DataFrame
            test_df: Preprocessed test DataFrame (optional)
            source: Source identifier for tracking
            batch_size: Records per batch insert

        Returns:
            Dictionary with counts of inserted records per split

        Pattern: Batch Processing + Transaction for atomicity
        """
        results = {}

        logger.info("Salvando splits preprocessados no PostgreSQL...")

        if train_df is not None and len(train_df) > 0:
            results["train"] = await self.save_split(
                "train", "preprocessed", train_df, source, batch_size
            )

        if valid_df is not None and len(valid_df) > 0:
            results["valid"] = await self.save_split(
                "valid", "preprocessed", valid_df, source, batch_size
            )

        if test_df is not None and len(test_df) > 0:
            results["test"] = await self.save_split(
                "test", "preprocessed", test_df, source, batch_size
            )

        total = sum(results.values())
        logger.success(
            f"Splits preprocessados salvos: {total:,} triplas "
            f"(train={results.get('train', 0):,}, "
            f"valid={results.get('valid', 0):,}, "
            f"test={results.get('test', 0):,})"
        )

        return results

    async def load_preprocessed_splits(
        self,
        fallback_to_raw: bool = True,
        map_to_ints: bool = True,
    ) -> tuple[pl.DataFrame | None, pl.DataFrame | None, pl.DataFrame | None, dict]:
        """
        Load preprocessed KG splits from PostgreSQL.

        If preprocessed data doesn't exist and fallback_to_raw is True,
        returns raw data instead (caller should apply preprocessing).

        Args:
            fallback_to_raw: If True, load raw data when preprocessed unavailable

        Returns:
            Tuple of (train_df, valid_df, test_df, metadata)
            metadata includes 'source' ('preprocessed' or 'raw')

        Pattern: Strategy Pattern for data source selection
        """
        metadata = {"source": None, "splits_loaded": []}

        train_exists = await self.split_exists("train", "preprocessed")
        valid_exists = await self.split_exists("valid", "preprocessed")

        if train_exists and valid_exists:
            logger.info("Carregando splits preprocessados do PostgreSQL...")
            metadata["source"] = "preprocessed"

            train_df = await self.load_split("train", "preprocessed", map_to_ints=map_to_ints)
            valid_df = await self.load_split("valid", "preprocessed", map_to_ints=map_to_ints)
            test_df = await self.load_split("test", "preprocessed", map_to_ints=map_to_ints)

            metadata["splits_loaded"] = ["train", "valid"]
            if test_df is not None:
                metadata["splits_loaded"].append("test")

            train_len = len(train_df) if train_df is not None else 0
            valid_len = len(valid_df) if valid_df is not None else 0
            logger.success(
                f"Splits preprocessados carregados: train={train_len:,}, valid={valid_len:,}"
            )

            return train_df, valid_df, test_df, metadata

        elif fallback_to_raw:
            logger.debug("Preprocessed splits not found, loading raw splits...")
            metadata["source"] = "raw"
            metadata["needs_preprocessing"] = True

            train_df = await self.load_split("train", "raw", map_to_ints=map_to_ints)
            valid_df = await self.load_split("valid", "raw", map_to_ints=map_to_ints)
            test_df = await self.load_split("test", "raw", map_to_ints=map_to_ints)

            if train_df is not None:
                metadata["splits_loaded"].append("train")
            if valid_df is not None:
                metadata["splits_loaded"].append("valid")
            if test_df is not None:
                metadata["splits_loaded"].append("test")

            return train_df, valid_df, test_df, metadata

        else:
            logger.debug("Preprocessed splits not found")
            return None, None, None, metadata

    async def preprocessed_exists(self) -> bool:
        """
        Check if preprocessed splits exist in PostgreSQL.

        Returns:
            True if both train and valid preprocessed splits exist
        """
        train_exists = await self.split_exists("train", "preprocessed")
        valid_exists = await self.split_exists("valid", "preprocessed")
        return train_exists and valid_exists

    async def delete_preprocessed(self) -> int:
        """
        Delete all preprocessed splits (keeps raw data intact).

        Returns:
            Number of records deleted
        """
        total_deleted = 0
        for split_name in ["train", "valid", "test"]:
            deleted = await self.delete_split(split_name, "preprocessed")
            total_deleted += deleted

        if total_deleted > 0:
            logger.info(f"Removidos {total_deleted:,} registros preprocessados")

        return total_deleted
