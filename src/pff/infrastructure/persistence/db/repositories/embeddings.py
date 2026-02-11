"""
EmbeddingsRepository - Repository Pattern for KG Embeddings.

Design Patterns Applied:
- Repository Pattern: Encapsulates data access logic
- Dependency Injection: Connection pool injected via constructor
- Batch Processing: Optimized bulk inserts (1K records/batch)
- Cache-Aside: In-memory cache for frequently accessed embeddings

Performance:
- Batch inserts: 1K embeddings in ~0.1s
- Loading: All embeddings in ~0.2s (vs 2-3s pickle)
- Similarity search: HNSW index (100x faster than numpy)
"""

from __future__ import annotations

import asyncio
import weakref
from collections.abc import Awaitable, Callable
from typing import Any

import numpy as np
import polars as pl

from pff.infrastructure.persistence.db.config import get_postgres_config
from pff.infrastructure.persistence.db.connection import get_connection_pool
from pff.infrastructure.persistence.db.repositories.base import PostgresRepository
from pff.shared.core.logging import logger
from pff_rust import stable_hash

PayloadHandler = Callable[[str | None], Awaitable[None]]


async def notify_postgres(channel: str, payload: str | None = None) -> None:
    """Publish a notification on the given PostgreSQL channel."""
    pool = await get_connection_pool()
    async with pool.acquire() as conn:
        await conn.execute("SELECT pg_notify($1, $2)", channel, payload or "")

    logger.debug(f" Notified channel '{channel}'", extra={"payload": payload})


async def register_postgres_listener(
    channel: str,
    handler: PayloadHandler,
) -> Any | None:
    """Register a coroutine handler for PostgreSQL LISTEN notifications.

    Returns:
        Connection kept open to receive notifications (caller owns lifecycle).
    """
    pool = await get_connection_pool()

    async def _invoke_handler(payload: str | None) -> None:
        try:
            await handler(payload)
        except Exception as exc:
            logger.error(f"Listener for channel '{channel}' failed: {exc}")

    def _listener(connection, pid, ch, payload) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.get_event_loop()
        loop.create_task(_invoke_handler(payload))

    conn = await pool.acquire()
    await conn.add_listener(channel, _listener)
    logger.debug(f"Registered listener for channel '{channel}'")
    return conn


class EmbeddingsRepository(PostgresRepository):
    """
    Repository for managing KG embeddings with pgvector.

    Pattern: Repository Pattern + Cache-Aside
    """

    _instances: "weakref.WeakSet[EmbeddingsRepository]" = weakref.WeakSet()
    _listener_registered: bool = False
    _listener_conn: Any | None = None

    def __init__(self, *, register_listener: bool = True):
        """Initialize repository with connection pool."""
        super().__init__()
        self._cache: dict[str, Any] = {}
        EmbeddingsRepository._instances.add(self)
        if register_listener:
            self._ensure_listener_task()

    @classmethod
    def _ensure_listener_task(cls) -> None:
        if cls._listener_registered:
            return
        cls._listener_registered = True

        async def _subscribe() -> None:
            if cls._listener_conn is None:
                cls._listener_conn = await register_postgres_listener(
                    "kg_embeddings_changed", cls._handle_event
                )

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.get_event_loop()
        loop.create_task(_subscribe())

    @classmethod
    async def _handle_event(cls, payload: str | None) -> None:
        logger.debug(
            " Received embeddings invalidation event", extra={"payload": payload}
        )
        for repo in list(cls._instances):
            repo._cache.clear()

    async def save_embeddings(
        self,
        entity_ids: list[str],
        embeddings: np.ndarray,
        model_version: str,
        entity_type: str = "entity",
        batch_size: int = 1000,
    ) -> int:
        """
        Save embeddings to PostgreSQL with pgvector.

        Args:
            entity_ids: List of entity identifiers
            embeddings: NumPy array of shape (N, D) where D is embedding dimension
            model_version: Model version (e.g., 'dslfm_v1.0', timestamp)
            entity_type: Type of entity ('entity', 'relation', 'node')
            batch_size: Records per batch insert

        Returns:
            Number of embeddings inserted

        Pattern: Batch Processing for performance
        """
        await self._ensure_pool()

        if len(entity_ids) != len(embeddings):
            raise ValueError(
                f"entity_ids ({len(entity_ids)}) and embeddings ({len(embeddings)}) length mismatch"
            )

        dimension = embeddings.shape[1]
        total_rows = len(entity_ids)

        logger.debug(
            f"Saving {total_rows:,} embeddings ({entity_type}, dim={dimension}) to PostgreSQL"
        )

        inserted = 0

        if self.pool is None:
            raise RuntimeError("Database pool not initialized")
        pool = self.pool

        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    "DELETE FROM kg_embeddings WHERE model_version = $1 AND entity_type = $2",
                    model_version,
                    entity_type,
                )

                await conn.execute(
                    """
                    CREATE TEMP TABLE IF NOT EXISTS tmp_kg_embeddings (
                        entity TEXT,
                        entity_type TEXT,
                        embedding vector,
                        dimension INTEGER,
                        model_version TEXT
                    ) ON COMMIT DROP
                    """
                )

                for batch_start in range(0, total_rows, batch_size):
                    batch_end = min(batch_start + batch_size, total_rows)
                    batch_ids = entity_ids[batch_start:batch_end]
                    batch_embeddings = embeddings[batch_start:batch_end]

                    records = []
                    for entity_id, embedding in zip(batch_ids, batch_embeddings):
                        embedding_str = (
                            "[" + ",".join(map(str, embedding.tolist())) + "]"
                        )
                        records.append(
                            (
                                str(entity_id),
                                entity_type,
                                embedding_str,
                                dimension,
                                model_version,
                            )
                        )

                    await conn.copy_records_to_table(
                        table_name="tmp_kg_embeddings",
                        columns=(
                            "entity",
                            "entity_type",
                            "embedding",
                            "dimension",
                            "model_version",
                        ),
                        records=records,
                    )

                    await conn.execute(
                        """
                        INSERT INTO kg_embeddings
                            (entity, entity_type, embedding, dimension, model_version, created_at, updated_at)
                        SELECT entity, entity_type, embedding::vector, dimension, model_version, NOW(), NOW()
                        FROM tmp_kg_embeddings
                        """
                    )

                    await conn.execute("TRUNCATE tmp_kg_embeddings")

                    inserted += len(records)

                    if batch_start % 5000 == 0 and batch_start > 0:
                        logger.info(
                            f"   {inserted:,}/{total_rows:,} embeddings inseridos..."
                        )

        logger.success(f" {inserted:,} embeddings salvos no PostgreSQL")

        self._cache.clear()
        await notify_postgres("kg_embeddings_changed", model_version)

        return inserted

    async def load_embeddings(
        self,
        entity_ids: list[str] | None = None,
        model_version: str | None = None,
        entity_type: str = "entity",
    ) -> dict[str, np.ndarray]:
        """
        Load embeddings from PostgreSQL.

        Args:
            entity_ids: Optional list of specific entities to load (None = all)
            model_version: Optional model version filter (None = latest)
            entity_type: Type of entity to load

        Returns:
            Dictionary mapping entity_id -> embedding (numpy array)

        Pattern: Cache-Aside (check cache first, then DB)
        """
        await self._ensure_pool()

        cache_key = f"{entity_type}_{model_version or 'latest'}_{len(entity_ids) if entity_ids else 'all'}"

        if cache_key in self._cache and entity_ids is None:
            logger.debug(f"Loading embeddings from cache ({entity_type})")
            cached_embeddings: dict[str, np.ndarray] = self._cache[cache_key]
            return cached_embeddings

        logger.debug(f"Loading embeddings from PostgreSQL ({entity_type})")

        try:

            def _cx_load():
                config = get_postgres_config()
                safe_type = entity_type.replace("'", "''")

                if entity_ids:
                    return None

                if model_version:
                    safe_version = model_version.replace("'", "''")
                    query = f"""
                        SELECT entity, embedding::text
                        FROM kg_embeddings
                        WHERE entity_type = '{safe_type}'
                        AND model_version = '{safe_version}'
                    """
                else:
                    query = f"""
                        SELECT entity, embedding::text
                        FROM kg_embeddings
                        WHERE entity_type = '{safe_type}'
                        AND model_version = (
                            SELECT model_version
                            FROM kg_embeddings
                            WHERE entity_type = '{safe_type}'
                            ORDER BY created_at DESC
                            LIMIT 1
                        )
                    """

                return pl.read_database_uri(
                    query, config.dsn_asyncpg, engine="connectorx"
                )

            if entity_ids is None:
                df = await asyncio.to_thread(_cx_load)

                if df is not None:
                    df = df.with_columns(
                        pl.col("embedding")
                        .str.strip_chars("[]")
                        .str.split(",")
                        .list.eval(pl.element().cast(pl.Float32))
                        .alias("vector")
                    )

                    embeddings = {
                        row["entity"]: np.array(row["vector"], dtype=np.float32)
                        for row in df.select(["entity", "vector"]).iter_rows(named=True)
                    }

                    logger.success(
                        f" {len(embeddings):,} embeddings carregados via connectorx/polars"
                    )

                    self._cache[cache_key] = embeddings
                    return embeddings

        except Exception as e:
            logger.debug(f"ConnectorX embedding load failed, falling back: {e}")

        if self.pool is None:
            raise RuntimeError("Database pool not initialized")
        pool = self.pool

        async with pool.acquire() as conn:
            if entity_ids:
                query = "SELECT entity, embedding FROM kg_embeddings WHERE entity_type = $1 AND model_version = $2 AND entity = ANY($3)"
                rows = await conn.fetch(query, entity_type, model_version, entity_ids)
            elif model_version:
                query = "SELECT entity, embedding FROM kg_embeddings WHERE entity_type = $1 AND model_version = $2"
                rows = await conn.fetch(query, entity_type, model_version)
            else:
                query = "SELECT entity, embedding FROM kg_embeddings WHERE entity_type = $1 AND model_version = (SELECT model_version FROM kg_embeddings WHERE entity_type = $1 ORDER BY created_at DESC LIMIT 1)"
                rows = await conn.fetch(query, entity_type)

        fallback_embeddings: dict[str, np.ndarray] = {}
        for row in rows:
            entity = row["entity"]
            embedding_str = str(row["embedding"])
            embedding_list = [float(x) for x in embedding_str.strip("[]").split(",")]
            embedding = np.array(embedding_list, dtype=np.float32)
            fallback_embeddings[entity] = embedding

        logger.success(
            f" {len(fallback_embeddings):,} embeddings carregados do PostgreSQL"
        )

        if entity_ids is None:
            self._cache[cache_key] = fallback_embeddings

        return fallback_embeddings

    async def search_similar(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        model_version: str | None = None,
        entity_type: str = "entity",
    ) -> list[dict[str, Any]]:
        """Find nearest neighbors using pgvector similarity search."""

        if top_k <= 0:
            return []

        await self._ensure_pool()

        rounded = tuple(np.round(query_embedding.astype(float), 6))
        cache_key = f"similarity:{entity_type}:{model_version or 'latest'}:{top_k}:{stable_hash(rounded)}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            cached_results: list[dict[str, Any]] = cached
            return cached_results

        vector_str = "[" + ",".join(map(str, query_embedding.tolist())) + "]"

        if self.pool is None:
            raise RuntimeError("Database pool not initialized")
        pool = self.pool

        async with pool.acquire() as conn:
            if model_version:
                query = """
                    SELECT entity,
                           embedding <-> ($1)::vector AS distance
                    FROM kg_embeddings
                    WHERE entity_type = $2
                      AND model_version = $3
                    ORDER BY embedding <-> ($1)::vector
                    LIMIT $4
                """
                rows = await conn.fetch(
                    query, vector_str, entity_type, model_version, top_k
                )
            else:
                query = """
                    SELECT entity,
                           embedding <-> ($1)::vector AS distance
                    FROM kg_embeddings
                    WHERE entity_type = $2
                      AND model_version = (
                        SELECT model_version
                        FROM kg_embeddings
                        WHERE entity_type = $2
                        ORDER BY created_at DESC
                        LIMIT 1
                      )
                    ORDER BY embedding <-> ($1)::vector
                    LIMIT $3
                """
                rows = await conn.fetch(query, vector_str, entity_type, top_k)

        results: list[dict[str, Any]] = [
            {
                "entity": row["entity"],
                "distance": float(row["distance"]),
                "score": 1.0 / (1.0 + float(row["distance"])),
            }
            for row in rows
        ]
        self._cache[cache_key] = results
        return results

    async def find_similar(
        self,
        embedding: np.ndarray,
        top_k: int = 10,
        model_version: str | None = None,
        entity_type: str = "entity",
    ) -> list[tuple[str, float]]:
        """
        Find top-k most similar entities using pgvector HNSW index.

        Performance: HNSW index ~100x faster than brute-force numpy
        """
        await self._ensure_pool()

        embedding_list = embedding.tolist()

        logger.info(f" Buscando top-{top_k} embeddings similares ({entity_type})...")

        if self.pool is None:
            raise RuntimeError("Database pool not initialized")
        pool = self.pool

        async with pool.acquire() as conn:
            if model_version:
                query = """
                    SELECT entity, 1 - (embedding <=> $1::vector) AS similarity
                    FROM kg_embeddings
                    WHERE entity_type = $2
                    AND model_version = $3
                    ORDER BY embedding <=> $1::vector
                    LIMIT $4
                """
                rows = await conn.fetch(
                    query, embedding_list, entity_type, model_version, top_k
                )
            else:
                query = """
                    SELECT entity, 1 - (embedding <=> $1::vector) AS similarity
                    FROM kg_embeddings
                    WHERE entity_type = $2
                    ORDER BY embedding <=> $1::vector
                    LIMIT $3
                """
                rows = await conn.fetch(query, embedding_list, entity_type, top_k)

        results = [(row["entity"], float(row["similarity"])) for row in rows]

        logger.success(f" {len(results)} embeddings similares encontrados")

        return results

    async def delete_embeddings(
        self, model_version: str | None = None, entity_type: str | None = None
    ) -> int:
        """
        Delete embeddings from PostgreSQL.
        """
        await self._ensure_pool()

        if self.pool is None:
            raise RuntimeError("Database pool not initialized")
        pool = self.pool

        async with pool.acquire() as conn:
            if model_version and entity_type:
                query = "DELETE FROM kg_embeddings WHERE model_version = $1 AND entity_type = $2"
                result = await conn.execute(query, model_version, entity_type)
            elif model_version:
                query = "DELETE FROM kg_embeddings WHERE model_version = $1"
                result = await conn.execute(query, model_version)
            elif entity_type:
                query = "DELETE FROM kg_embeddings WHERE entity_type = $1"
                result = await conn.execute(query, entity_type)
            else:
                query = "DELETE FROM kg_embeddings"
                result = await conn.execute(query)

        deleted = int(result.split()[-1]) if result else 0

        logger.info(f"  {deleted:,} embeddings deletados do PostgreSQL")

        self._cache.clear()

        return deleted

    async def get_statistics(self) -> dict[str, Any]:
        """
        Get statistics about stored embeddings.
        """
        await self._ensure_pool()

        if self.pool is None:
            raise RuntimeError("Database pool not initialized")
        pool = self.pool

        async with pool.acquire() as conn:
            stats_query = """
                SELECT
                    entity_type,
                    model_version,
                    dimension,
                    COUNT(*) as count
                FROM kg_embeddings
                GROUP BY entity_type, model_version, dimension
                ORDER BY entity_type, model_version
            """
            rows = await conn.fetch(stats_query)

        stats = {"total": sum(row["count"] for row in rows), "by_type": {}}

        for row in rows:
            entity_type = row["entity_type"]
            if entity_type not in stats["by_type"]:
                stats["by_type"][entity_type] = []

            stats["by_type"][entity_type].append(
                {
                    "model_version": row["model_version"],
                    "dimension": row["dimension"],
                    "count": row["count"],
                }
            )

        return stats
