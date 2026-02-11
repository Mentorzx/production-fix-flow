"""
AuditAnalysisRepository - Repository Pattern for audit analysis outputs.

This repository persists:
    - JSON Schema validation reports (input documents)
    - Statistical baseline profiles
    - Drift reports per run

All payloads are stored as JSONB to keep the pipeline PostgreSQL-first and avoid
unnecessary file generation.
"""

from __future__ import annotations

from typing import Any

import asyncpg

from pff.infrastructure.persistence.db.repositories.base import PostgresRepository
from pff.shared.core.logging import logger


class AuditAnalysisRepository(PostgresRepository):
    """Repository for schema/profile/drift artifacts produced by the audit pipeline."""

    async def _create_schema(self, conn: asyncpg.Connection) -> None:
        """Create audit_schema_reports, audit_profile_baselines, audit_run_profiles tables."""
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS audit_schema_reports (
                run_id TEXT PRIMARY KEY REFERENCES audit_runs(run_id) ON DELETE CASCADE,
                schema_id TEXT,
                schema_version TEXT,
                report JSONB NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS audit_profile_baselines (
                baseline_id TEXT PRIMARY KEY,
                profile JSONB NOT NULL,
                digest JSONB NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS audit_run_profiles (
                run_id TEXT PRIMARY KEY REFERENCES audit_runs(run_id) ON DELETE CASCADE,
                profile_current JSONB NOT NULL,
                drift JSONB NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """)
        logger.debug("audit_analysis tables verified/created automatically")

    async def save_schema_report(
        self,
        *,
        run_id: str,
        schema_report: list[dict[str, Any]],
        schema_id: str | None = None,
        schema_version: str | int | None = None,
    ) -> None:
        report_json = self._file_manager.json_dumps(schema_report)

        async def _op(conn: asyncpg.Connection) -> None:
            await conn.execute(
                """
                INSERT INTO audit_schema_reports (run_id, schema_id, schema_version, report)
                VALUES ($1, $2, $3, $4::jsonb)
                ON CONFLICT (run_id)
                DO UPDATE SET
                    schema_id = EXCLUDED.schema_id,
                    schema_version = EXCLUDED.schema_version,
                    report = EXCLUDED.report
                """,
                run_id,
                schema_id,
                str(schema_version) if schema_version is not None else None,
                report_json,
            )

        await self._execute_with_schema(_op)

    async def load_schema_report(self, *, run_id: str) -> list[dict[str, Any]] | None:
        async def _op(conn: asyncpg.Connection):
            return await conn.fetchrow(
                """
                SELECT report
                FROM audit_schema_reports
                WHERE run_id = $1
                """,
                run_id,
            )

        row = await self._execute_with_schema(_op)
        if row is None:
            return None
        report = row["report"]
        if isinstance(report, (str, bytes)):
            report = self._file_manager.json_loads(report)
        if isinstance(report, list):
            return [dict(x) for x in report if isinstance(x, dict)]
        return None

    async def save_baseline_profile(
        self,
        *,
        baseline_id: str,
        profile: dict[str, Any],
        digest: dict[str, Any],
    ) -> None:
        profile_json = self._file_manager.json_dumps(profile)
        digest_json = self._file_manager.json_dumps(digest)

        async def _op(conn: asyncpg.Connection) -> None:
            await conn.execute(
                """
                INSERT INTO audit_profile_baselines (baseline_id, profile, digest)
                VALUES ($1, $2::jsonb, $3::jsonb)
                ON CONFLICT (baseline_id)
                DO UPDATE SET
                    profile = EXCLUDED.profile,
                    digest = EXCLUDED.digest
                """,
                baseline_id,
                profile_json,
                digest_json,
            )

        await self._execute_with_schema(_op)

    async def load_baseline_profile(self, *, baseline_id: str) -> dict[str, Any] | None:
        async def _op(conn: asyncpg.Connection):
            return await conn.fetchrow(
                """
                SELECT profile
                FROM audit_profile_baselines
                WHERE baseline_id = $1
                """,
                baseline_id,
            )

        row = await self._execute_with_schema(_op)
        if row is None:
            return None
        profile = row["profile"]
        if isinstance(profile, (str, bytes)):
            profile = self._file_manager.json_loads(profile)
        return dict(profile) if isinstance(profile, dict) else None

    async def save_run_profile(
        self,
        *,
        run_id: str,
        profile_current: dict[str, Any],
        drift: dict[str, Any],
    ) -> None:
        profile_json = self._file_manager.json_dumps(profile_current)
        drift_json = self._file_manager.json_dumps(drift)

        async def _op(conn: asyncpg.Connection) -> None:
            await conn.execute(
                """
                INSERT INTO audit_run_profiles (run_id, profile_current, drift)
                VALUES ($1, $2::jsonb, $3::jsonb)
                ON CONFLICT (run_id)
                DO UPDATE SET
                    profile_current = EXCLUDED.profile_current,
                    drift = EXCLUDED.drift
                """,
                run_id,
                profile_json,
                drift_json,
            )

        await self._execute_with_schema(_op)

    async def load_run_profile(self, *, run_id: str) -> dict[str, Any] | None:
        async def _op(conn: asyncpg.Connection):
            return await conn.fetchrow(
                """
                SELECT profile_current, drift
                FROM audit_run_profiles
                WHERE run_id = $1
                """,
                run_id,
            )

        row = await self._execute_with_schema(_op)
        if row is None:
            return None
        profile_current = row["profile_current"]
        drift = row["drift"]
        if isinstance(profile_current, (str, bytes)):
            profile_current = self._file_manager.json_loads(profile_current)
        if isinstance(drift, (str, bytes)):
            drift = self._file_manager.json_loads(drift)
        if not isinstance(profile_current, dict) or not isinstance(drift, dict):
            return None
        return {"profile_current": profile_current, "drift": drift}
