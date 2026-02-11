"""PostgreSQL-backed storage for HPO trial artifacts and state."""

from __future__ import annotations

from typing import Any

import asyncpg

from pff.infrastructure.persistence.db.repositories.base import PostgresRepository
from pff.shared.core.file_manager import FileManager
from pff.shared.core.logging import logger


class HpoPostgresStore(PostgresRepository):
    """Persist HPO artifacts (trials, checkpoints, best params) in Postgres."""

    def __init__(self, pool: Any | None = None, file_manager: FileManager | None = None) -> None:
        super().__init__(pool=pool, file_manager=file_manager)

    async def _create_schema(self, conn: asyncpg.Connection) -> None:
        """Create HPO storage tables and indexes."""
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS hpo_trial_results (
                id BIGSERIAL PRIMARY KEY,
                study_name TEXT NOT NULL,
                trial_number INTEGER NOT NULL,
                payload JSONB NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
            """
        )
        await conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS hpo_trial_results_unique
            ON hpo_trial_results (study_name, trial_number)
            """
        )
        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS hpo_trial_results_study
            ON hpo_trial_results (study_name)
            """
        )
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS hpo_checkpoints (
                checkpoint_key TEXT PRIMARY KEY,
                payload JSONB NOT NULL,
                updated_at TIMESTAMPTZ DEFAULT NOW()
            )
            """
        )
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS hpo_best_params (
                study_name TEXT PRIMARY KEY,
                best_value DOUBLE PRECISION,
                best_params JSONB,
                updated_at TIMESTAMPTZ DEFAULT NOW()
            )
            """
        )
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS hpo_memory_entries (
                study_name TEXT PRIMARY KEY,
                entries JSONB NOT NULL,
                updated_at TIMESTAMPTZ DEFAULT NOW()
            )
            """
        )

    async def _execute_with_schema(self, operation):
        await self._ensure_pool()
        async with self.pool.acquire() as conn:
            return await operation(conn)

    async def upsert_trial_result(
        self,
        study_name: str,
        trial_number: int,
        payload: dict[str, Any],
    ) -> None:
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
            if "duration" not in metrics_payload and data.get("elapsed_time") is not None:
                try:
                    metrics_payload["duration"] = float(data["elapsed_time"])
                except Exception as exc:
                    logger.debug(f"Failed to coerce elapsed_time into duration: {exc}")
            metrics.append(metrics_payload)
        return metrics

    async def load_all_results(self, study_name: str) -> list[dict[str, Any]]:
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

    async def upsert_checkpoint(self, checkpoint_key: str, payload: dict[str, Any]) -> None:
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
            "best_value": (float(row["best_value"]) if row["best_value"] is not None else None),
            "best_params": payload or {},
        }

    async def upsert_memory_entries(self, study_name: str, entries: list[dict[str, Any]]) -> None:
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
        await self._ensure_pool()

        async def _op(conn):
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

        await self._execute_with_schema(_op)
