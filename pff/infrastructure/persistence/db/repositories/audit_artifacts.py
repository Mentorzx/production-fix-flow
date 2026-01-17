"""
AuditArtifactsRepository - Repository Pattern for audit pipeline persistence.

Design Patterns Applied:
    - Repository Pattern: encapsulates PostgreSQL access for audit artifacts.
    - Memento Pattern: stores run snapshots (inputs/derived records) for replay.
    - Unit of Work: transaction boundaries per save operation.

Primary goal:
    Persist canonical records and triples in PostgreSQL to avoid generating
    unnecessary intermediate files while keeping JSON→Graph→JSON provenance.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any
from collections.abc import Sequence

import asyncpg

from pff.config import AUDIT_CONFIG_PATH
from pff.infrastructure.persistence.db.connection import get_connection_pool
from pff.shared import FileManager
from pff.shared.core.logger import logger
from pff.domain.audit.canonicalize import CanonicalRecord, CanonicalTriple


@dataclass(frozen=True)
class AuditStorageConfig:
    """Audit storage configuration loaded via FileManager.

    Args:
        batch_size_records: Batch size for canonical records inserts.
        batch_size_triples: Batch size for triples inserts.
    """

    batch_size_records: int = 5000
    batch_size_triples: int = 5000

    @staticmethod
    def load(file_manager: FileManager | None = None) -> AuditStorageConfig:
        fm = file_manager or FileManager()
        try:
            cfg_obj = fm.read(AUDIT_CONFIG_PATH, return_native=True)
        except FileNotFoundError:
            return AuditStorageConfig()
        if not isinstance(cfg_obj, dict):
            return AuditStorageConfig()
        audit_cfg = cfg_obj.get("audit", cfg_obj)
        if not isinstance(audit_cfg, dict):
            return AuditStorageConfig()
        storage_cfg = audit_cfg.get("storage", {})
        if not isinstance(storage_cfg, dict):
            return AuditStorageConfig()
        return AuditStorageConfig(
            batch_size_records=int(storage_cfg.get("batch_size_records", 5000)),
            batch_size_triples=int(storage_cfg.get("batch_size_triples", 5000)),
        )


class AuditArtifactsRepository:
    """Repository for audit run artifacts (canonical records + triples)."""

    def __init__(self, *, config: AuditStorageConfig | None = None) -> None:
        self.pool: asyncpg.Pool | None = None
        self._file_manager = FileManager()
        self._schema_ready = False
        self._schema_lock = asyncio.Lock()
        self._config = config or AuditStorageConfig.load(self._file_manager)

    async def _ensure_pool(self) -> None:
        if self.pool is None:
            self.pool = await get_connection_pool()
            await self._ensure_schema()

    async def _ensure_schema(self, *, force: bool = False) -> None:
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
                    CREATE TABLE IF NOT EXISTS audit_runs (
                        run_id TEXT PRIMARY KEY,
                        document_id TEXT NOT NULL,
                        baseline_id TEXT NOT NULL,
                        meta JSONB,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )
                await conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS audit_canonical_records (
                        id BIGSERIAL PRIMARY KEY,
                        run_id TEXT NOT NULL REFERENCES audit_runs(run_id) ON DELETE CASCADE,
                        record_hash TEXT NOT NULL,
                        json_pointer TEXT NOT NULL,
                        field_path TEXT NOT NULL,
                        key TEXT,
                        value_type TEXT NOT NULL,
                        normalized_value TEXT NOT NULL,
                        raw_value JSONB,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE (run_id, record_hash)
                    )
                    """
                )
                await conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_audit_records_run
                    ON audit_canonical_records (run_id)
                    """
                )
                await conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_audit_records_field
                    ON audit_canonical_records (run_id, field_path)
                    """
                )
                await conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS audit_triples (
                        id BIGSERIAL PRIMARY KEY,
                        run_id TEXT NOT NULL REFERENCES audit_runs(run_id) ON DELETE CASCADE,
                        triple_hash TEXT NOT NULL,
                        s TEXT NOT NULL,
                        p TEXT NOT NULL,
                        o TEXT NOT NULL,
                        json_pointer TEXT,
                        record_hash TEXT,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE (run_id, triple_hash)
                    )
                    """
                )
                await conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_audit_triples_run
                    ON audit_triples (run_id)
                    """
                )
                await conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_audit_triples_predicate
                    ON audit_triples (run_id, p)
                    """
                )
            logger.debug("audit_artifacts tables verified/created automatically")
            self._schema_ready = True

    async def _execute_with_schema(self, operation):
        await self._ensure_pool()
        assert self.pool is not None
        try:
            async with self.pool.acquire() as conn:
                return await operation(conn)
        except asyncpg.UndefinedTableError:
            logger.warning("audit_artifacts tables missing - recreating automatically.")
            await self._ensure_schema(force=True)
            async with self.pool.acquire() as conn:
                return await operation(conn)

    async def save_run(
        self,
        *,
        run_id: str,
        document_id: str,
        baseline_id: str,
        meta: dict[str, Any] | None = None,
    ) -> None:
        """Upsert an audit run row.

        Args:
            run_id: Unique run identifier.
            document_id: Stable identifier for the input JSON.
            baseline_id: Baseline identifier for drift/calibration artifacts.
            meta: Optional JSONB payload (versions, seeds, timings).
        """

        meta_json = self._file_manager.json_dumps(meta) if meta is not None else None

        async def _op(conn: asyncpg.Connection) -> None:
            await conn.execute(
                """
                INSERT INTO audit_runs (run_id, document_id, baseline_id, meta)
                VALUES ($1, $2, $3, $4::jsonb)
                ON CONFLICT (run_id)
                DO UPDATE SET
                    document_id = EXCLUDED.document_id,
                    baseline_id = EXCLUDED.baseline_id,
                    meta = EXCLUDED.meta
                """,
                run_id,
                document_id,
                baseline_id,
                meta_json,
            )

        await self._execute_with_schema(_op)

    async def save_canonical_records(
        self,
        *,
        run_id: str,
        records: Sequence[CanonicalRecord],
    ) -> int:
        """Persist canonical records for a run.

        Args:
            run_id: Run identifier.
            records: Canonical leaf records.

        Returns:
            Number of inserted rows (best-effort, excludes conflicts).
        """

        if not records:
            return 0

        batch_size = max(1, int(self._config.batch_size_records))

        async def _op(conn: asyncpg.Connection) -> int:
            inserted = 0
            async with conn.transaction():
                await conn.execute(
                    """
                    CREATE TEMP TABLE IF NOT EXISTS tmp_audit_records (
                        record_hash TEXT,
                        json_pointer TEXT,
                        field_path TEXT,
                        key TEXT,
                        value_type TEXT,
                        normalized_value TEXT,
                        raw_value JSONB
                    ) ON COMMIT DROP
                    """
                )
                for offset in range(0, len(records), batch_size):
                    batch = records[offset : offset + batch_size]
                    copy_rows = [
                        (
                            rec.record_hash,
                            rec.json_pointer,
                            rec.field_path,
                            rec.key,
                            rec.value_type,
                            rec.normalized_value,
                            self._file_manager.json_dumps(rec.raw_value),
                        )
                        for rec in batch
                    ]
                    await conn.copy_records_to_table(
                        table_name="tmp_audit_records",
                        columns=(
                            "record_hash",
                            "json_pointer",
                            "field_path",
                            "key",
                            "value_type",
                            "normalized_value",
                            "raw_value",
                        ),
                        records=copy_rows,
                    )
                    inserted += await conn.fetchval(
                        """
                        WITH ins AS (
                            INSERT INTO audit_canonical_records
                                (run_id, record_hash, json_pointer, field_path, key, value_type, normalized_value, raw_value)
                            SELECT $1, record_hash, json_pointer, field_path, key, value_type, normalized_value, raw_value::jsonb
                            FROM tmp_audit_records
                            ON CONFLICT (run_id, record_hash) DO NOTHING
                            RETURNING 1
                        )
                        SELECT COUNT(*) FROM ins
                        """,
                        run_id,
                    )
                    await conn.execute("TRUNCATE tmp_audit_records")
            return inserted

        return int(await self._execute_with_schema(_op))

    async def save_triples(
        self,
        *,
        run_id: str,
        triples: Sequence[CanonicalTriple],
    ) -> int:
        """Persist canonical triples for a run.

        Args:
            run_id: Run identifier.
            triples: Canonical triples with provenance fields.

        Returns:
            Number of inserted rows (best-effort, excludes conflicts).
        """

        if not triples:
            return 0

        batch_size = max(1, int(self._config.batch_size_triples))

        async def _op(conn: asyncpg.Connection) -> int:
            inserted = 0
            async with conn.transaction():
                await conn.execute(
                    """
                    CREATE TEMP TABLE IF NOT EXISTS tmp_audit_triples (
                        triple_hash TEXT,
                        s TEXT,
                        p TEXT,
                        o TEXT,
                        json_pointer TEXT,
                        record_hash TEXT
                    ) ON COMMIT DROP
                    """
                )
                for offset in range(0, len(triples), batch_size):
                    batch = triples[offset : offset + batch_size]
                    copy_rows = [
                        (
                            t.triple_hash,
                            t.s,
                            t.p,
                            t.o,
                            t.json_pointer,
                            t.record_hash,
                        )
                        for t in batch
                    ]
                    await conn.copy_records_to_table(
                        table_name="tmp_audit_triples",
                        columns=(
                            "triple_hash",
                            "s",
                            "p",
                            "o",
                            "json_pointer",
                            "record_hash",
                        ),
                        records=copy_rows,
                    )
                    inserted += await conn.fetchval(
                        """
                        WITH ins AS (
                            INSERT INTO audit_triples
                                (run_id, triple_hash, s, p, o, json_pointer, record_hash)
                            SELECT $1, triple_hash, s, p, o, json_pointer, record_hash
                            FROM tmp_audit_triples
                            ON CONFLICT (run_id, triple_hash) DO NOTHING
                            RETURNING 1
                        )
                        SELECT COUNT(*) FROM ins
                        """,
                        run_id,
                    )
                    await conn.execute("TRUNCATE tmp_audit_triples")
            return inserted

        return int(await self._execute_with_schema(_op))

    async def load_triples(self, *, run_id: str) -> list[dict[str, Any]]:
        """Load triples for a run (ordered by insertion)."""

        async def _op(conn: asyncpg.Connection):
            return await conn.fetch(
                """
                SELECT triple_hash, s, p, o, json_pointer, record_hash
                FROM audit_triples
                WHERE run_id = $1
                ORDER BY id
                """,
                run_id,
            )

        rows = await self._execute_with_schema(_op)
        return [dict(row) for row in rows]

    async def load_records(self, *, run_id: str) -> list[dict[str, Any]]:
        """Load canonical records for a run (ordered by insertion)."""

        async def _op(conn: asyncpg.Connection):
            return await conn.fetch(
                """
                SELECT record_hash, json_pointer, field_path, key, value_type, normalized_value, raw_value
                FROM audit_canonical_records
                WHERE run_id = $1
                ORDER BY id
                """,
                run_id,
            )

        rows = await self._execute_with_schema(_op)
        normalized: list[dict[str, Any]] = []
        for row in rows:
            raw_value = row["raw_value"]
            if isinstance(raw_value, (str, bytes)):
                raw_value = self._file_manager.json_loads(raw_value)
            normalized.append(
                {
                    "record_hash": row["record_hash"],
                    "json_pointer": row["json_pointer"],
                    "field_path": row["field_path"],
                    "key": row["key"],
                    "value_type": row["value_type"],
                    "normalized_value": row["normalized_value"],
                    "raw_value": raw_value,
                }
            )
        return normalized
