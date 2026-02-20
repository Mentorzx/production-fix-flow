"""
AuditSemanticsRepository - Repository Pattern for calibration + EVT artifacts.

This repository stores compact JSONB payloads that are reused across runs:
    - calibration models (global + per-relation)
    - EVT/POT parameters (global + per-relation)

Storing these in PostgreSQL enables warm-start and avoids unnecessary file I/O.
"""

from __future__ import annotations

from typing import Any

import asyncpg

from pff.infrastructure.persistence.db.repositories.base import PostgresRepository
from pff.shared.core.logging import logger


class AuditSemanticsRepository(PostgresRepository):
    """Repository for calibration and EVT artifacts keyed by baseline_id."""

    async def _create_schema(self, conn: asyncpg.Connection) -> None:
        """Create audit_calibration_models and audit_evt_params tables."""
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS audit_calibration_models (
                baseline_id TEXT NOT NULL,
                relation TEXT NOT NULL,
                model JSONB NOT NULL,
                metrics JSONB,
                created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (baseline_id, relation)
            )
            """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS audit_evt_params (
                baseline_id TEXT NOT NULL,
                relation TEXT NOT NULL,
                params JSONB NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (baseline_id, relation)
            )
            """)
        logger.debug("audit_semantics tables verified/created automatically")

    async def save_calibration_models(
        self,
        *,
        baseline_id: str,
        models_by_relation: dict[str, dict[str, Any]],
    ) -> int:
        """Persist calibration models keyed by baseline_id and relation."""

        if not models_by_relation:
            return 0

        rows = []
        for relation, payload in models_by_relation.items():
            model = payload.get("model")
            metrics = payload.get("metrics")
            if not isinstance(model, dict):
                continue
            rows.append(
                (
                    baseline_id,
                    str(relation),
                    self._file_manager.json_dumps(model),
                    (self._file_manager.json_dumps(metrics) if isinstance(metrics, dict) else None),
                )
            )

        async def _op(conn: asyncpg.Connection) -> int:
            """Execute op.



            Args:

                conn: Input value used by this callable.



            Returns:

                Return value produced by the callable.

            """

            inserted = 0
            async with conn.transaction():
                await conn.execute("""
                    CREATE TEMP TABLE IF NOT EXISTS tmp_audit_calibration (
                        baseline_id TEXT,
                        relation TEXT,
                        model JSONB,
                        metrics JSONB
                    ) ON COMMIT DROP
                    """)
                await conn.copy_records_to_table(
                    table_name="tmp_audit_calibration",
                    columns=("baseline_id", "relation", "model", "metrics"),
                    records=rows,
                )
                inserted = await conn.fetchval("""
                    WITH ins AS (
                        INSERT INTO audit_calibration_models (baseline_id, relation, model, metrics)
                        SELECT baseline_id, relation, model, metrics
                        FROM tmp_audit_calibration
                        ON CONFLICT (baseline_id, relation) DO UPDATE SET
                            model = EXCLUDED.model,
                            metrics = EXCLUDED.metrics
                        RETURNING 1
                    )
                    SELECT COUNT(*) FROM ins
                    """)
            return int(inserted or 0)

        return int(await self._execute_with_schema(_op))

    async def load_calibration_models(self, *, baseline_id: str) -> dict[str, dict[str, Any]]:
        """Execute load calibration models.



        Args:

            baseline_id: Input value used by this callable.



        Returns:

            Return value produced by the callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        async def _op(conn: asyncpg.Connection):
            return await conn.fetch(
                """
                SELECT relation, model, metrics
                FROM audit_calibration_models
                WHERE baseline_id = $1
                """,
                baseline_id,
            )

        rows = await self._execute_with_schema(_op)
        models: dict[str, dict[str, Any]] = {}
        for row in rows:
            model = row["model"]
            metrics = row["metrics"]
            if isinstance(model, (str, bytes)):
                model = self._file_manager.json_loads(model)
            if isinstance(metrics, (str, bytes)):
                metrics = self._file_manager.json_loads(metrics)
            if isinstance(model, dict):
                models[str(row["relation"])] = {
                    "model": model,
                    "metrics": metrics if isinstance(metrics, dict) else None,
                }
        return models

    async def save_evt_params(
        self,
        *,
        baseline_id: str,
        params_by_relation: dict[str, dict[str, Any]],
    ) -> int:
        """Persist EVT parameters keyed by baseline_id and relation."""

        if not params_by_relation:
            return 0

        rows = []
        for relation, params in params_by_relation.items():
            rows.append((baseline_id, str(relation), self._file_manager.json_dumps(params)))

        async def _op(conn: asyncpg.Connection) -> int:
            """Execute op.



            Args:

                conn: Input value used by this callable.



            Returns:

                Return value produced by the callable.

            """

            inserted = 0
            async with conn.transaction():
                await conn.execute("""
                    CREATE TEMP TABLE IF NOT EXISTS tmp_audit_evt (
                        baseline_id TEXT,
                        relation TEXT,
                        params JSONB
                    ) ON COMMIT DROP
                    """)
                await conn.copy_records_to_table(
                    table_name="tmp_audit_evt",
                    columns=("baseline_id", "relation", "params"),
                    records=rows,
                )
                inserted = await conn.fetchval("""
                    WITH ins AS (
                        INSERT INTO audit_evt_params (baseline_id, relation, params)
                        SELECT baseline_id, relation, params
                        FROM tmp_audit_evt
                        ON CONFLICT (baseline_id, relation) DO UPDATE SET
                            params = EXCLUDED.params
                        RETURNING 1
                    )
                    SELECT COUNT(*) FROM ins
                    """)
            return int(inserted or 0)

        return int(await self._execute_with_schema(_op))

    async def load_evt_params(self, *, baseline_id: str) -> dict[str, dict[str, Any]]:
        """Execute load evt params.



        Args:

            baseline_id: Input value used by this callable.



        Returns:

            Return value produced by the callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        async def _op(conn: asyncpg.Connection):
            return await conn.fetch(
                """
                SELECT relation, params
                FROM audit_evt_params
                WHERE baseline_id = $1
                """,
                baseline_id,
            )

        rows = await self._execute_with_schema(_op)
        params_by_relation: dict[str, dict[str, Any]] = {}
        for row in rows:
            params = row["params"]
            if isinstance(params, (str, bytes)):
                params = self._file_manager.json_loads(params)
            if isinstance(params, dict):
                params_by_relation[str(row["relation"])] = params
        return params_by_relation
