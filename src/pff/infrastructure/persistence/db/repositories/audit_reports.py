"""
AuditReportsRepository - Repository Pattern for persisted audit reports.

Stores the final `audit_report.json` payload as JSONB in PostgreSQL so that
consumers do not depend on filesystem artifacts.
"""

from __future__ import annotations

from typing import Any

import asyncpg

from pff.infrastructure.persistence.db.repositories.base import PostgresRepository
from pff.shared.core.logging import logger


class AuditReportsRepository(PostgresRepository):
    """Repository for audit_report.json persistence keyed by run_id."""

    async def _create_schema(self, conn: asyncpg.Connection) -> None:
        """Create audit_reports table."""
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS audit_reports (
                run_id TEXT PRIMARY KEY REFERENCES audit_runs(run_id) ON DELETE CASCADE,
                report JSONB NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        logger.debug("audit_reports table verified/created automatically")

    async def save_report(self, *, run_id: str, report: dict[str, Any]) -> None:
        report_json = self._file_manager.json_dumps(report)

        async def _op(conn: asyncpg.Connection) -> None:
            await conn.execute(
                """
                INSERT INTO audit_reports (run_id, report)
                VALUES ($1, $2::jsonb)
                ON CONFLICT (run_id)
                DO UPDATE SET report = EXCLUDED.report
                """,
                run_id,
                report_json,
            )

        await self._execute_with_schema(_op)

    async def load_report(self, *, run_id: str) -> dict[str, Any] | None:
        async def _op(conn: asyncpg.Connection):
            return await conn.fetchrow(
                """
                SELECT report
                FROM audit_reports
                WHERE run_id = $1
                """,
                run_id,
            )

        row = await self._execute_with_schema(_op)
        if row is None:
            return None
        payload = row["report"]
        if isinstance(payload, (str, bytes)):
            payload = self._file_manager.json_loads(payload)
        return dict(payload) if isinstance(payload, dict) else None
