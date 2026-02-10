"""
PipelineCheckpointsRepository - Repository Pattern for Pipeline Checkpoints.

Design Patterns Applied:
- Repository Pattern: Encapsulates data access logic
- State Pattern: Manages pipeline state transitions
- Memento Pattern: Saves and restores pipeline state

SOTA Features:
- JSONB for flexible metadata storage
- UPSERT pattern (INSERT ... ON CONFLICT UPDATE)
- Atomic state transitions
- Progress tracking (0.0 to 1.0)
"""

from datetime import datetime
from typing import Any

import asyncpg

from pff.infrastructure.persistence.db.repositories.base import PostgresRepository
from pff.shared.core.logging import logger


class PipelineCheckpointsRepository(PostgresRepository):
    """
    Repository for managing pipeline checkpoints with state persistence.

    Pattern: Repository + State + Memento
    """

    async def _create_schema(self, conn: asyncpg.Connection) -> None:
        """Create pipeline_checkpoints table and indexes."""
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS pipeline_checkpoints (
                id BIGSERIAL PRIMARY KEY,
                pipeline_name VARCHAR(100) NOT NULL,
                step_name VARCHAR(100) NOT NULL,
                status VARCHAR(20) NOT NULL,
                progress DOUBLE PRECISION NOT NULL DEFAULT 0.0,
                metadata JSONB,
                started_at TIMESTAMPTZ,
                completed_at TIMESTAMPTZ,
                created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE (pipeline_name, step_name)
            )
            """
        )
        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_pipeline_checkpoints_lookup
            ON pipeline_checkpoints (pipeline_name)
            """
        )
        logger.debug(" pipeline_checkpoints table verified/created automatically")

    async def save_checkpoint(
        self,
        pipeline_name: str,
        step_name: str,
        status: str,
        progress: float = 0.0,
        metadata: dict | None = None,
        started_at: datetime | None = None,
        completed_at: datetime | None = None,
    ) -> int:
        """
        Save or update pipeline checkpoint.

        Args:
            pipeline_name: Pipeline name ('kg', 'dslfm', 'ensemble')
            step_name: Step name ('preprocess', 'learn_rules', 'ranking', etc.)
            status: Status ('pending', 'running', 'completed', 'failed')
            progress: Progress (0.0 to 1.0)
            metadata: Optional metadata (files generated, errors, etc.)
            started_at: Optional start timestamp
            completed_at: Optional completion timestamp

        Returns:
            Checkpoint ID

        Pattern: UPSERT with ON CONFLICT UPDATE
        """
        logger.debug(
            f"Saving checkpoint: {pipeline_name}/{step_name} ({status}, {progress:.0%})"
        )

        async def _operation(conn):
            return await conn.fetchval(
                """
                INSERT INTO pipeline_checkpoints
                    (pipeline_name, step_name, status, progress, metadata, started_at, completed_at, created_at)
                VALUES ($1, $2, $3, $4, $5::jsonb, $6, $7, NOW())
                ON CONFLICT (pipeline_name, step_name)
                DO UPDATE SET
                    status = EXCLUDED.status,
                    progress = EXCLUDED.progress,
                    metadata = EXCLUDED.metadata,
                    started_at = COALESCE(EXCLUDED.started_at, pipeline_checkpoints.started_at),
                    completed_at = EXCLUDED.completed_at
                RETURNING id
                """,
                pipeline_name,
                step_name,
                status,
                progress,
                self._file_manager.json_dumps(metadata) if metadata else None,
                started_at,
                completed_at,
            )

        checkpoint_id = await self._execute_with_schema(_operation)

        logger.success(f" Checkpoint salvo (ID: {checkpoint_id})")

        return checkpoint_id

    async def get_checkpoint(
        self, pipeline_name: str, step_name: str
    ) -> dict[str, Any] | None:
        """
        Get checkpoint for specific pipeline step.

        Args:
            pipeline_name: Pipeline name
            step_name: Step name

        Returns:
            Checkpoint dict or None

        Pattern: Query with optional result
        """

        async def _operation(conn):
            return await conn.fetchrow(
                """
                SELECT id, pipeline_name, step_name, status, progress, metadata,
                       started_at, completed_at, created_at
                FROM pipeline_checkpoints
                WHERE pipeline_name = $1 AND step_name = $2
                """,
                pipeline_name,
                step_name,
            )

        row = await self._execute_with_schema(_operation)

        if row is None:
            return None

        metadata_value = row["metadata"]
        if isinstance(metadata_value, (str, bytes)):
            metadata_value = self._file_manager.json_loads(metadata_value)

        return {
            "id": row["id"],
            "pipeline_name": row["pipeline_name"],
            "step_name": row["step_name"],
            "status": row["status"],
            "progress": row["progress"],
            "metadata": metadata_value or {},
            "started_at": row["started_at"],
            "completed_at": row["completed_at"],
            "created_at": row["created_at"],
        }

    async def get_pipeline_checkpoints(
        self, pipeline_name: str
    ) -> list[dict[str, Any]]:
        """
        Get all checkpoints for a pipeline.

        Args:
            pipeline_name: Pipeline name

        Returns:
            List of checkpoint dicts

        Pattern: Query all with filter
        """

        async def _operation(conn):
            return await conn.fetch(
                """
                SELECT id, pipeline_name, step_name, status, progress, metadata,
                       started_at, completed_at, created_at
                FROM pipeline_checkpoints
                WHERE pipeline_name = $1
                ORDER BY created_at ASC
                """,
                pipeline_name,
            )

        rows = await self._execute_with_schema(_operation)

        checkpoints = []
        for row in rows:
            checkpoints.append(
                {
                    "id": row["id"],
                    "pipeline_name": row["pipeline_name"],
                    "step_name": row["step_name"],
                    "status": row["status"],
                    "progress": row["progress"],
                    "metadata": (
                        self._file_manager.json_loads(row["metadata"])
                        if isinstance(row["metadata"], (str, bytes))
                        else row["metadata"] or {}
                    ),
                    "started_at": row["started_at"],
                    "completed_at": row["completed_at"],
                    "created_at": row["created_at"],
                }
            )

        return checkpoints

    async def reset_pipeline(self, pipeline_name: str) -> int:
        """
        Reset all checkpoints for a pipeline (mark as pending).

        Args:
            pipeline_name: Pipeline name

        Returns:
            Number of checkpoints reset

        Pattern: Bulk update with filter
        """
        logger.info(f" Resetando pipeline: {pipeline_name}")

        async def _operation(conn):
            return await conn.execute(
                """
                UPDATE pipeline_checkpoints
                SET status = 'pending',
                    progress = 0.0,
                    started_at = NULL,
                    completed_at = NULL
                WHERE pipeline_name = $1
                """,
                pipeline_name,
            )

        result = await self._execute_with_schema(_operation)

        count = int(result.split()[-1])
        logger.success(f" {count} checkpoints resetados")

        return count

    async def delete_pipeline_checkpoints(self, pipeline_name: str) -> int:
        """
        Delete all checkpoints for a pipeline.

        Args:
            pipeline_name: Pipeline name

        Returns:
            Number of checkpoints deleted

        Pattern: Bulk delete with filter
        """
        logger.info(f"  Deletando checkpoints: {pipeline_name}")

        async def _operation(conn):
            return await conn.execute(
                """
                DELETE FROM pipeline_checkpoints
                WHERE pipeline_name = $1
                """,
                pipeline_name,
            )

        result = await self._execute_with_schema(_operation)

        count = int(result.split()[-1])
        logger.success(f" {count} checkpoints deletados")

        return count

    async def checkpoint_exists(self, pipeline_name: str, step_name: str) -> bool:
        """
        Check if checkpoint exists.

        Args:
            pipeline_name: Pipeline name
            step_name: Step name

        Returns:
            True if exists, False otherwise

        Pattern: EXISTS query
        """

        async def _operation(conn):
            return await conn.fetchval(
                """
                SELECT EXISTS(
                    SELECT 1 FROM pipeline_checkpoints
                    WHERE pipeline_name = $1 AND step_name = $2
                )
                """,
                pipeline_name,
                step_name,
            )

        exists = await self._execute_with_schema(_operation)

        return bool(exists)

    async def get_pipeline_progress(self, pipeline_name: str) -> float:
        """
        Get overall pipeline progress (average of step progress).

        Args:
            pipeline_name: Pipeline name

        Returns:
            Overall progress (0.0 to 1.0)

        Pattern: Aggregate query
        """

        async def _operation(conn):
            return await conn.fetchval(
                """
                SELECT COALESCE(AVG(progress), 0.0)
                FROM pipeline_checkpoints
                WHERE pipeline_name = $1
                """,
                pipeline_name,
            )

        avg_progress = await self._execute_with_schema(_operation)

        return float(avg_progress)

    async def delete_all_checkpoints(self) -> int:
        """
        Delete all checkpoints (used in cleanup).

        Returns:
            Number of checkpoints deleted

        Pattern: Truncate equivalent
        """
        logger.warning("  Deleting ALL checkpoints")

        async def _operation(conn):
            count = (
                await conn.fetchval(
                    "SELECT reltuples::bigint FROM pg_class WHERE relname = 'pipeline_checkpoints'"
                )
                or 0
            )
            await conn.execute("TRUNCATE TABLE pipeline_checkpoints RESTART IDENTITY")
            return count

        count = await self._execute_with_schema(_operation)
        logger.success(f" {count} checkpoints deletados")

        return int(count)
