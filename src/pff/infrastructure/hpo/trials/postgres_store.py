"""PostgreSQL-backed storage for HPO trial artifacts and state."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta, timezone
from typing import Any

import asyncpg

from pff.infrastructure.persistence.db.repositories.base import PostgresRepository
from pff.shared.core.file_manager import FileManager
from pff.shared.core.logging import logger


class HpoPostgresStore(PostgresRepository):
    """Persist HPO artifacts (trials, checkpoints, best params) in Postgres."""

    def __init__(
        self, pool: Any | None = None, file_manager: FileManager | None = None
    ) -> None:
        """Execute init.



        Args:

            pool: Optional input value.

            file_manager: Optional input value.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        super().__init__(pool=pool, file_manager=file_manager)

    async def _create_schema(self, conn: asyncpg.Connection) -> None:
        """Create HPO storage tables and indexes."""
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS hpo_trial_results (
                id BIGSERIAL PRIMARY KEY,
                study_name TEXT NOT NULL,
                trial_number INTEGER NOT NULL,
                payload JSONB NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
            """)
        await conn.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS hpo_trial_results_unique
            ON hpo_trial_results (study_name, trial_number)
            """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS hpo_trial_results_study
            ON hpo_trial_results (study_name)
            """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS hpo_checkpoints (
                checkpoint_key TEXT PRIMARY KEY,
                payload JSONB NOT NULL,
                updated_at TIMESTAMPTZ DEFAULT NOW()
            )
            """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS hpo_best_params (
                study_name TEXT PRIMARY KEY,
                best_value DOUBLE PRECISION,
                best_params JSONB,
                updated_at TIMESTAMPTZ DEFAULT NOW()
            )
            """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS hpo_memory_entries (
                study_name TEXT PRIMARY KEY,
                entries JSONB NOT NULL,
                updated_at TIMESTAMPTZ DEFAULT NOW()
            )
            """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS hpo_advisor_cache (
                id BIGSERIAL PRIMARY KEY,
                study_name TEXT NOT NULL,
                dataset_fingerprint TEXT NOT NULL,
                direction TEXT NOT NULL,
                advisor_version TEXT NOT NULL,
                last_trial INTEGER NOT NULL,
                search_space_hash TEXT NOT NULL,
                objective_schema_hash TEXT NOT NULL,
                payload JSONB NOT NULL,
                expires_at TIMESTAMPTZ,
                updated_at TIMESTAMPTZ DEFAULT NOW(),
                UNIQUE (
                    study_name,
                    dataset_fingerprint,
                    direction,
                    advisor_version,
                    last_trial,
                    search_space_hash,
                    objective_schema_hash
                )
            )
            """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS hpo_advisor_cache_study_idx
            ON hpo_advisor_cache (study_name, updated_at DESC)
            """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS hpo_advisor_cache_expires_idx
            ON hpo_advisor_cache (expires_at)
            """)

    async def _execute_with_schema(
        self, operation: Callable[[Any], Awaitable[Any]]
    ) -> Any:
        """Execute execute with schema.



        Args:

            operation: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        await self._ensure_pool()
        assert self.pool is not None
        async with self.pool.acquire() as conn:
            return await operation(conn)

    async def upsert_trial_result(
        self,
        study_name: str,
        trial_number: int,
        payload: dict[str, Any],
    ) -> None:
        """Execute upsert trial result.



        Args:

            study_name: Input value used by this callable.

            trial_number: Input value used by this callable.

            payload: Input value used by this callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        await self._ensure_pool()
        payload_json = self._file_manager.json_dumps(payload)

        async def _op(conn):
            await conn.execute(
                """
                INSERT INTO hpo_trial_results (study_name, trial_number, payload, created_at)
                VALUES ($1, $2, $3::jsonb, NOW())
                ON CONFLICT (study_name, trial_number)
                DO UPDATE SET payload = EXCLUDED.payload, created_at = NOW()
                """,
                study_name,
                int(trial_number),
                payload_json,
            )

        await self._execute_with_schema(_op)

    async def list_trial_metrics(self, study_name: str) -> list[dict[str, Any]]:
        """Execute list trial metrics.



        Args:

            study_name: Input value used by this callable.



        Returns:

            Return value produced by the callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        await self._ensure_pool()

        async def _op(conn):
            return await conn.fetch(
                """
                SELECT payload
                FROM hpo_trial_results
                WHERE study_name = $1
                ORDER BY trial_number
                """,
                study_name,
            )

        rows = await self._execute_with_schema(_op)
        metrics: list[dict[str, Any]] = []
        for row in rows:
            payload = row["payload"]
            if isinstance(payload, str):
                payload = self._file_manager.json_loads(payload)
            data = payload or {}
            metrics_payload = data.get("metrics") or data.get("kge_metrics") or {}
            if (
                "duration" not in metrics_payload
                and data.get("elapsed_time") is not None
            ):
                try:
                    metrics_payload["duration"] = float(data["elapsed_time"])
                except Exception as exc:
                    logger.debug(f"Failed to coerce elapsed_time into duration: {exc}")
            metrics.append(metrics_payload)
        return metrics

    async def load_all_results(self, study_name: str) -> list[dict[str, Any]]:
        """Execute load all results.



        Args:

            study_name: Input value used by this callable.



        Returns:

            Return value produced by the callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        await self._ensure_pool()

        async def _op(conn):
            return await conn.fetch(
                """
                SELECT payload
                FROM hpo_trial_results
                WHERE study_name = $1
                ORDER BY trial_number
                """,
                study_name,
            )

        rows = await self._execute_with_schema(_op)
        results: list[dict[str, Any]] = []
        for row in rows:
            payload = row["payload"]
            if isinstance(payload, str):
                payload = self._file_manager.json_loads(payload)
            if payload is not None:
                results.append(payload)
        return results

    async def upsert_checkpoint(
        self, checkpoint_key: str, payload: dict[str, Any]
    ) -> None:
        """Execute upsert checkpoint.



        Args:

            checkpoint_key: Input value used by this callable.

            payload: Input value used by this callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        await self._ensure_pool()
        payload_json = self._file_manager.json_dumps(payload)

        async def _op(conn):
            await conn.execute(
                """
                INSERT INTO hpo_checkpoints (checkpoint_key, payload, updated_at)
                VALUES ($1, $2::jsonb, NOW())
                ON CONFLICT (checkpoint_key)
                DO UPDATE SET payload = EXCLUDED.payload, updated_at = NOW()
                """,
                checkpoint_key,
                payload_json,
            )

        await self._execute_with_schema(_op)

    async def load_checkpoint(self, checkpoint_key: str) -> dict[str, Any] | None:
        """Execute load checkpoint.



        Args:

            checkpoint_key: Input value used by this callable.



        Returns:

            Return value produced by the callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        await self._ensure_pool()

        async def _op(conn):
            return await conn.fetchrow(
                """
                SELECT payload
                FROM hpo_checkpoints
                WHERE checkpoint_key = $1
                """,
                checkpoint_key,
            )

        row = await self._execute_with_schema(_op)
        if row is None:
            return None
        payload = row["payload"]
        if isinstance(payload, str):
            payload = self._file_manager.json_loads(payload)
        return payload or None

    async def delete_checkpoint(self, checkpoint_key: str) -> None:
        """Execute delete checkpoint.



        Args:

            checkpoint_key: Input value used by this callable.

        """

        await self._ensure_pool()

        async def _op(conn):
            await conn.execute(
                "DELETE FROM hpo_checkpoints WHERE checkpoint_key = $1",
                checkpoint_key,
            )

        await self._execute_with_schema(_op)

    async def upsert_best_params(
        self,
        study_name: str,
        best_params: dict[str, Any],
        best_value: float,
    ) -> None:
        """Execute upsert best params.



        Args:

            study_name: Input value used by this callable.

            best_params: Input value used by this callable.

            best_value: Input value used by this callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        await self._ensure_pool()
        params_json = self._file_manager.json_dumps(best_params)

        async def _op(conn):
            await conn.execute(
                """
                INSERT INTO hpo_best_params (study_name, best_value, best_params, updated_at)
                VALUES ($1, $2, $3::jsonb, NOW())
                ON CONFLICT (study_name)
                DO UPDATE SET best_value = EXCLUDED.best_value,
                              best_params = EXCLUDED.best_params,
                              updated_at = NOW()
                """,
                study_name,
                float(best_value),
                params_json,
            )

        await self._execute_with_schema(_op)

    async def load_best_params(self, study_name: str) -> dict[str, Any] | None:
        """Execute load best params.



        Args:

            study_name: Input value used by this callable.



        Returns:

            Return value produced by the callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        await self._ensure_pool()

        async def _op(conn):
            return await conn.fetchrow(
                """
                SELECT best_value, best_params
                FROM hpo_best_params
                WHERE study_name = $1
                """,
                study_name,
            )

        row = await self._execute_with_schema(_op)
        if row is None:
            return None
        payload = row["best_params"]
        if isinstance(payload, str):
            payload = self._file_manager.json_loads(payload)
        return {
            "best_value": (
                float(row["best_value"]) if row["best_value"] is not None else None
            ),
            "best_params": payload or {},
        }

    async def upsert_memory_entries(
        self, study_name: str, entries: list[dict[str, Any]]
    ) -> None:
        """Execute upsert memory entries.



        Args:

            study_name: Input value used by this callable.

            entries: Input value used by this callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        await self._ensure_pool()
        entries_json = self._file_manager.json_dumps({"entries": entries})

        async def _op(conn):
            await conn.execute(
                """
                INSERT INTO hpo_memory_entries (study_name, entries, updated_at)
                VALUES ($1, $2::jsonb, NOW())
                ON CONFLICT (study_name)
                DO UPDATE SET entries = EXCLUDED.entries, updated_at = NOW()
                """,
                study_name,
                entries_json,
            )

        await self._execute_with_schema(_op)

    async def load_memory_entries(self, study_name: str) -> list[dict[str, Any]]:
        """Execute load memory entries.



        Args:

            study_name: Input value used by this callable.



        Returns:

            Return value produced by the callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        await self._ensure_pool()

        async def _op(conn):
            return await conn.fetchrow(
                """
                SELECT entries
                FROM hpo_memory_entries
                WHERE study_name = $1
                """,
                study_name,
            )

        row = await self._execute_with_schema(_op)
        if row is None:
            return []
        payload = row["entries"]
        if isinstance(payload, str):
            payload = self._file_manager.json_loads(payload)
        data = payload or {}
        return data.get("entries", []) if isinstance(data, dict) else []

    async def clear_study(self, study_name: str) -> None:
        """Execute clear study.



        Args:

            study_name: Input value used by this callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        await self._ensure_pool()

        async def _op(conn):
            """Execute op.



            Args:

                conn: Input value used by this callable.

            """

            await conn.execute(
                "DELETE FROM hpo_trial_results WHERE study_name = $1",
                study_name,
            )
            await conn.execute(
                "DELETE FROM hpo_best_params WHERE study_name = $1",
                study_name,
            )
            await conn.execute(
                "DELETE FROM hpo_memory_entries WHERE study_name = $1",
                study_name,
            )
            await conn.execute(
                "DELETE FROM hpo_advisor_cache WHERE study_name = $1",
                study_name,
            )

        await self._execute_with_schema(_op)

    async def upsert_advisor_cache(
        self,
        *,
        study_name: str,
        dataset_fingerprint: str,
        direction: str,
        advisor_version: str,
        last_trial: int,
        search_space_hash: str,
        objective_schema_hash: str,
        payload: dict[str, Any],
        ttl_seconds: int | None = None,
    ) -> None:
        """Upsert search-space advisor cache payload."""
        await self._ensure_pool()
        payload_json = self._file_manager.json_dumps(payload)
        expires_at = (
            datetime.now(timezone.utc) + timedelta(seconds=max(1, int(ttl_seconds)))
            if isinstance(ttl_seconds, int) and ttl_seconds > 0
            else None
        )

        async def _op(conn):
            await conn.execute(
                """
                INSERT INTO hpo_advisor_cache (
                    study_name,
                    dataset_fingerprint,
                    direction,
                    advisor_version,
                    last_trial,
                    search_space_hash,
                    objective_schema_hash,
                    payload,
                    expires_at,
                    updated_at
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb, $9, NOW())
                ON CONFLICT (
                    study_name,
                    dataset_fingerprint,
                    direction,
                    advisor_version,
                    last_trial,
                    search_space_hash,
                    objective_schema_hash
                )
                DO UPDATE SET
                    payload = EXCLUDED.payload,
                    expires_at = EXCLUDED.expires_at,
                    updated_at = NOW()
                """,
                study_name,
                dataset_fingerprint,
                direction,
                advisor_version,
                int(last_trial),
                search_space_hash,
                objective_schema_hash,
                payload_json,
                expires_at,
            )

        await self._execute_with_schema(_op)

    async def load_advisor_cache(
        self,
        *,
        study_name: str,
        dataset_fingerprint: str,
        direction: str,
        advisor_version: str,
        last_trial: int,
        search_space_hash: str,
        objective_schema_hash: str,
    ) -> dict[str, Any] | None:
        """Load search-space advisor cache payload when not expired."""
        await self._ensure_pool()

        async def _op(conn):
            return await conn.fetchrow(
                """
                SELECT payload, expires_at
                FROM hpo_advisor_cache
                WHERE study_name = $1
                  AND dataset_fingerprint = $2
                  AND direction = $3
                  AND advisor_version = $4
                  AND last_trial = $5
                  AND search_space_hash = $6
                  AND objective_schema_hash = $7
                LIMIT 1
                """,
                study_name,
                dataset_fingerprint,
                direction,
                advisor_version,
                int(last_trial),
                search_space_hash,
                objective_schema_hash,
            )

        row = await self._execute_with_schema(_op)
        if row is None:
            return None

        expires_at = row.get("expires_at")
        if expires_at is not None:
            now_utc = datetime.now(timezone.utc)
            if expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=timezone.utc)
            if expires_at < now_utc:

                async def _delete_expired(conn):
                    await conn.execute(
                        """
                        DELETE FROM hpo_advisor_cache
                        WHERE study_name = $1
                          AND dataset_fingerprint = $2
                          AND direction = $3
                          AND advisor_version = $4
                          AND last_trial = $5
                          AND search_space_hash = $6
                          AND objective_schema_hash = $7
                        """,
                        study_name,
                        dataset_fingerprint,
                        direction,
                        advisor_version,
                        int(last_trial),
                        search_space_hash,
                        objective_schema_hash,
                    )

                await self._execute_with_schema(_delete_expired)
                return None

        payload = row["payload"]
        if isinstance(payload, str):
            payload = self._file_manager.json_loads(payload)
        return payload if isinstance(payload, dict) else None
