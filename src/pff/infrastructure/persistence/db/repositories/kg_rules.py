"""
Repository Pattern for KG rules and autofeeding iterations.

Handles rule aggregates for reuse across pipelines (legacy rule miners disabled).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import polars as pl

from pff.infrastructure.persistence.db.config import get_postgres_config
from pff.infrastructure.persistence.db.repositories.base import PostgresRepository
from pff.shared.core.file_manager import FileManager, ParquetBundle
from pff.shared.core.logging import logger


class KGRulesRepository(PostgresRepository):
    """
    Repository for managing rules and autofeeding iterations.

    Pattern: Repository + Iterator for streaming.
    """

    def __init__(self, pool: Any | None = None, file_manager: FileManager | None = None) -> None:
        """Initialize repository with optional injected pool and file manager."""
        super().__init__(pool=pool, file_manager=file_manager)

    async def _create_schema(self, conn: asyncpg.Connection) -> None:
        """Create kg_rules table and indexes."""
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS kg_rules (
                id BIGSERIAL PRIMARY KEY,
                rule_text TEXT NOT NULL,
                confidence DOUBLE PRECISION,
                support INTEGER,
                num_predictions INTEGER,
                source VARCHAR(50),
                iteration INTEGER,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_kg_rules_source ON kg_rules(source)")
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_kg_rules_confidence ON kg_rules(confidence)"
        )

    async def save_rules(
        self,
        rules: list[dict[str, Any]],
        source: str = "dslfm",
        iteration: int | None = None,
        batch_size: int = 5000,
    ) -> int:
        """
        Save rules to PostgreSQL.

        Args:
            rules: List of rule dictionaries (must contain 'rule', optional 'confidence', 'support', 'num_predictions')
            source: Rule source identifier (e.g., 'dslfm', 'manual', 'ensemble')
            iteration: Autofeeding iteration number (None for initial)
            batch_size: Rules per batch insert

        Returns:
            Number of rules inserted
        """
        await self._ensure_pool()

        if not rules:
            return 0

        total = len(rules)
        logger.debug(f"Saving {total:,} rules ({source}) to PostgreSQL")

        inserted = 0
        columns = (
            "rule_text",
            "confidence",
            "support",
            "num_predictions",
            "source",
            "iteration",
        )

        assert self.pool is not None
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                for batch_start in range(0, total, batch_size):
                    batch_end = min(batch_start + batch_size, total)
                    batch_data = rules[batch_start:batch_end]
                    records = []
                    for r in batch_data:
                        records.append(
                            (
                                r.get("rule", r.get("rule_text", "")),
                                r.get("confidence"),
                                r.get("support"),
                                r.get("num_predictions"),
                                source,
                                iteration,
                            )
                        )

                    await conn.copy_records_to_table(
                        "kg_rules",
                        records=records,
                        columns=columns,
                    )

                    inserted += len(records)

                    if batch_end < total:
                        logger.debug(f"Batch {batch_start:,}-{batch_end:,} inserted")

        logger.success(f"{inserted:,} regras salvas no PostgreSQL")
        return inserted

    async def load_rules(
        self,
        source: str | None = None,
        iteration: int | None = None,
        min_confidence: float | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Load rules from PostgreSQL.

        Args:
            source: Filter by source identifier (e.g., 'dslfm', 'manual', 'ensemble')
            iteration: Filter by iteration number
            min_confidence: Filter by minimum confidence
            limit: Optional maximum number of rules to return

        Returns:
            List of rule dictionaries
        """
        await self._ensure_pool()

        logger.debug(f"Loading rules from PostgreSQL (source={source})")

        import os

        if os.environ.get("PYTEST_CURRENT_TEST") is None:
            try:
                rules = await self._load_rules_connectorx(
                    source=source,
                    iteration=iteration,
                    min_confidence=min_confidence,
                    limit=limit,
                )
                if rules is not None:
                    logger.success(f"{len(rules):,} regras carregadas (via connectorx)")
                    return rules

            except Exception as e:
                logger.debug(f"ConnectorX rule load failed, falling back: {e}")

        query, params = self._build_rules_query(
            source=source,
            iteration=iteration,
            min_confidence=min_confidence,
            limit=limit,
        )

        logger.debug(f"Loading rules from PostgreSQL (source={source})")

        assert self.pool is not None
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *params)

        if not rows:
            return []

        rules = self._rows_to_rule_dicts(rows)

        logger.success(f"{len(rules):,} regras carregadas")
        return rules

    async def _load_rules_connectorx(
        self,
        *,
        source: str | None,
        iteration: int | None,
        min_confidence: float | None,
        limit: int | None,
    ) -> list[dict[str, Any]] | None:
        """Execute load rules connectorx.



        Args:

            source: Input value used by this callable.

            iteration: Input value used by this callable.

            min_confidence: Input value used by this callable.

            limit: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        def _cx_load():
            """Execute cx load.



            Returns:

                Return value produced by the callable.

            """

            config = get_postgres_config()
            query = """
                SELECT rule_text, confidence, support, num_predictions, source
                FROM kg_rules
                WHERE 1=1
            """
            if source is not None:
                safe_source = source.replace("'", "''")
                query += f" AND source = '{safe_source}'"
            if iteration is not None:
                query += f" AND iteration = {int(iteration)}"
            if min_confidence is not None:
                query += f" AND confidence >= {float(min_confidence)}"
            query += " ORDER BY confidence DESC NULLS LAST, id"
            if limit is not None and limit > 0:
                query += f" LIMIT {int(limit)}"
            return pl.read_database_uri(query, config.dsn_asyncpg, engine="connectorx")

        df = await asyncio.to_thread(_cx_load)
        if df is None:
            return None
        if df.is_empty():
            return []
        return df.rename({"rule_text": "rule"}).to_dicts()

    def _build_rules_query(
        self,
        *,
        source: str | None,
        iteration: int | None,
        min_confidence: float | None,
        limit: int | None,
    ) -> tuple[str, list[Any]]:
        """Execute build rules query.



        Args:

            source: Input value used by this callable.

            iteration: Input value used by this callable.

            min_confidence: Input value used by this callable.

            limit: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        query = """
            SELECT rule_text, confidence, support, num_predictions, source
            FROM kg_rules
            WHERE 1=1
        """
        params: list[Any] = []
        if source is not None:
            params.append(source)
            query += f" AND source = ${len(params)}"
        if iteration is not None:
            params.append(iteration)
            query += f" AND iteration = ${len(params)}"
        if min_confidence is not None:
            params.append(min_confidence)
            query += f" AND confidence >= ${len(params)}"
        query += " ORDER BY confidence DESC NULLS LAST, id"
        if limit is not None and limit > 0:
            params.append(limit)
            query += f" LIMIT ${len(params)}"
        return query, params

    def _rows_to_rule_dicts(self, rows: list[Any]) -> list[dict[str, Any]]:
        return [
            {
                "rule": row["rule_text"],
                "confidence": row["confidence"],
                "support": row["support"],
                "num_predictions": row["num_predictions"],
                "source": row["source"],
            }
            for row in rows
        ]

    async def stream_rules(
        self,
        source: str | None = None,
        iteration: int | None = None,
        batch_size: int = 1000,
    ) -> AsyncIterator[list[str]]:
        """
        Stream rules in batches (for large datasets).

        Args:
            source: Filter by source
            iteration: Filter by iteration
            batch_size: Rules per batch

        Yields:
            Batches of rule strings

        Pattern: Iterator Pattern for memory efficiency
        """
        await self._ensure_pool()

        query = "SELECT rule_text, confidence FROM kg_rules WHERE 1=1"
        params: list[Any] = []

        if source is not None:
            params.append(source)
            query += f" AND source = ${len(params)}"

        if iteration is not None:
            params.append(iteration)
            query += f" AND iteration = ${len(params)}"

        query += " ORDER BY confidence DESC NULLS LAST, id"

        logger.info("Iniciando streaming de regras do PostgreSQL")

        assert self.pool is not None
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                cursor = await conn.cursor(query, *params)

                while True:
                    rows = await cursor.fetch(batch_size)
                    if not rows:
                        break

                    batch = [
                        (
                            f"{row['rule_text']}\t{row['confidence']}"
                            if row["confidence"] is not None
                            else row["rule_text"]
                        )
                        for row in rows
                    ]

                    yield batch

    async def stream_rules_dict(
        self,
        source: str | None = None,
        iteration: int | None = None,
        batch_size: int = 1000,
    ) -> AsyncIterator[list[dict[str, Any]]]:
        """
        Stream rules as dictionaries in batches (memory efficient).

        Args:
            source: Filter by source
            iteration: Filter by iteration
            batch_size: Rules per batch

        Yields:
            Batches of rule dictionaries
        """
        await self._ensure_pool()

        query = (
            "SELECT rule_text, confidence, support, num_predictions, source FROM kg_rules WHERE 1=1"
        )
        params: list[Any] = []

        if source is not None:
            params.append(source)
            query += f" AND source = ${len(params)}"

        if iteration is not None:
            params.append(iteration)
            query += f" AND iteration = ${len(params)}"

        query += " ORDER BY confidence DESC NULLS LAST, id"

        assert self.pool is not None
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                cursor = await conn.cursor(query, *params)
                while True:
                    rows = await cursor.fetch(batch_size)
                    if not rows:
                        break
                    yield [
                        {
                            "rule": row["rule_text"],
                            "confidence": row["confidence"],
                            "support": row["support"],
                            "num_predictions": row["num_predictions"],
                            "source": row["source"],
                        }
                        for row in rows
                    ]

    async def count_rules(self, source: str | None = None, iteration: int | None = None) -> int:
        """
        Count rules matching filters.

        Args:
            source: Filter by source
            iteration: Filter by iteration

        Returns:
            Number of rules
        """
        await self._ensure_pool()

        query = "SELECT COUNT(*) FROM kg_rules WHERE 1=1"
        params: list[Any] = []

        if source is not None:
            params.append(source)
            query += f" AND source = ${len(params)}"

        if iteration is not None:
            params.append(iteration)
            query += f" AND iteration = ${len(params)}"

        assert self.pool is not None
        async with self.pool.acquire() as conn:
            count = await conn.fetchval(query, *params)

        return count

    async def delete_rules(self, source: str | None = None, iteration: int | None = None) -> int:
        """
        Delete rules matching filters.

        Args:
            source: Filter by source
            iteration: Filter by iteration

        Returns:
            Number of rules deleted
        """
        await self._ensure_pool()

        query = "DELETE FROM kg_rules WHERE 1=1"
        params: list[Any] = []

        if source is not None:
            params.append(source)
            query += f" AND source = ${len(params)}"

        if iteration is not None:
            params.append(iteration)
            query += f" AND iteration = ${len(params)}"

        assert self.pool is not None
        async with self.pool.acquire() as conn:
            result = await conn.execute(query, *params)

        deleted = int(result.split()[-1]) if result else 0

        if deleted > 0:
            logger.info(f"{deleted:,} regras deletadas (source={source}, iteration={iteration})")

        return deleted

    async def delete_all(self) -> int:
        """
        Delete all rules from PostgreSQL.

        Returns:
            Number of rules deleted
        """
        await self._ensure_pool()

        assert self.pool is not None
        async with self.pool.acquire() as conn:
            result = await conn.execute("DELETE FROM kg_rules")

        deleted = int(result.split()[-1]) if result else 0

        if deleted > 0:
            logger.info(f"Todas as {deleted:,} regras deletadas")

        return deleted

    async def truncate_all(self) -> int:
        """
        Truncate all rules from PostgreSQL.

        Returns:
            Number of records deleted (pre-truncate count)
        """
        await self._ensure_pool()

        assert self.pool is not None
        async with self.pool.acquire() as conn:
            count = await conn.fetchval("SELECT COUNT(*) FROM kg_rules")
            await conn.execute("TRUNCATE kg_rules RESTART IDENTITY")

        deleted = count or 0

        if deleted > 0:
            logger.info(f"Todas as {deleted:,} regras truncadas")

        return deleted

    async def vacuum_full(self) -> None:
        """Run VACUUM FULL on the kg_rules table."""
        assert self.pool is not None
        async with self.pool.acquire() as conn:
            await conn.execute("VACUUM (FULL, ANALYZE) kg_rules")

    async def get_statistics(self) -> dict:
        """
        Get statistics about stored rules.

        Returns:
            Dictionary with rule statistics
        """
        await self._ensure_pool()

        assert self.pool is not None
        async with self.pool.acquire() as conn:
            total = await conn.fetchval("SELECT COUNT(*) FROM kg_rules")

            source_rows = await conn.fetch("""
                SELECT source, COUNT(*) as count, AVG(confidence) as avg_conf
                FROM kg_rules
                GROUP BY source
                ORDER BY count DESC
                """)

            iteration_rows = await conn.fetch("""
                SELECT iteration, COUNT(*) as count
                FROM kg_rules
                WHERE iteration IS NOT NULL
                GROUP BY iteration
                ORDER BY iteration
                """)

        stats = {
            "total": total,
            "by_source": {
                row["source"]: {
                    "count": row["count"],
                    "avg_confidence": (float(row["avg_conf"]) if row["avg_conf"] else None),
                }
                for row in source_rows
            },
            "by_iteration": {row["iteration"]: row["count"] for row in iteration_rows},
        }

        return stats

    async def save_rules_from_file(
        self, file_path: str, source: str = "manual", iteration: int | None = None
    ) -> int:
        """
        Load rules from TSV file and save to PostgreSQL.

        Args:
            file_path: Path to TSV file
            source: Rule source identifier
            iteration: Iteration number

        Returns:
            Number of rules saved

        Pattern: Adapter Pattern (file → database)
        """
        logger.info(f"Carregando regras de {file_path}...")

        try:
            bundle = self._file_manager.read(file_path, separator="\t", has_header=False)
            df = (
                bundle.lazyframe().collect(engine="streaming")
                if isinstance(bundle, ParquetBundle)
                else bundle
            )
            if df is None:
                logger.warning("No data found in rules file; skipping import")
                return 0

            rules_data = []
            if df.shape[1] >= 4:
                for row in df.iter_rows():
                    rules_data.append(
                        {
                            "num_predictions": int(row[0]),
                            "support": int(row[1]),
                            "confidence": float(row[2]),
                            "rule": str(row[3]),
                        }
                    )
            else:
                for row in df.iter_rows():
                    rules_data.append({"rule": str(row[0]), "confidence": None})

            return await self.save_rules(rules_data, source=source, iteration=iteration)

        except Exception as exc:
            logger.error(f"Failed to read rules file {file_path}: {exc}")
            return 0
