"""
ExecutionLogsRepository - Repository Pattern for Execution Logs.

Design Patterns Applied:
- Repository Pattern: Encapsulates data access logic
- Decorator Pattern: @log_execution decorator for automatic logging
- Builder Pattern: Fluent API for log queries
- Strategy Pattern: Different log levels and operations

SOTA Features:
- JSONB for flexible metadata (PostgreSQL 9.4+)
- GIN index for fast JSONB queries
- Time-series queries with indexes
- Automatic cleanup (TTL pattern)
"""

from datetime import datetime, timedelta
from functools import wraps
import traceback

from pff.shared.core.logger import logger

from pff.infrastructure.persistence.db.connection import get_connection_pool
from pff.shared import FileManager


class ExecutionLogsRepository:
    """
    Repository for managing execution logs with JSONB metadata.

    Pattern: Repository + Time-Series Data
    """

    def __init__(self):
        """Initialize repository with connection pool."""
        self.pool = None
        self._file_manager = FileManager()

    async def _ensure_pool(self):
        """Lazy initialization of connection pool."""
        if self.pool is None:
            self.pool = await get_connection_pool()

    async def create_log(
        self,
        operation: str,
        status: str = "running",
        user_id: str | None = None,
        metadata: dict | None = None,
        duration_seconds: float | None = None,
        error_message: str | None = None,
    ) -> int:
        """
        Create execution log entry.

        Args:
            operation: Operation name ('kg_training', 'dslfm_training', 'inference', etc.)
            status: Status ('running', 'success', 'failed')
            user_id: Optional user UUID
            metadata: Optional JSONB metadata (config, params, etc.)
            duration_seconds: Optional duration
            error_message: Optional error message

        Returns:
            Log ID

        Pattern: Insert with RETURNING for ID
        """
        await self._ensure_pool()

        logger.info(f" Criando log de execução: {operation} ({status})")

        async with self.pool.acquire() as conn:
            log_id = await conn.fetchval(
                """
                INSERT INTO execution_logs
                    (user_id, operation, status, duration_seconds, metadata, error_message, created_at)
                VALUES ($1, $2, $3, $4, $5::jsonb, $6, NOW())
                RETURNING id
                """,
                user_id,
                operation,
                status,
                duration_seconds,
                self._file_manager.json_dumps(metadata) if metadata else None,
                error_message,
            )

        logger.success(f" Log criado (ID: {log_id})")

        return log_id

    async def update_log(
        self,
        log_id: int,
        status: str | None = None,
        duration_seconds: float | None = None,
        metadata: dict | None = None,
        error_message: str | None = None,
    ) -> None:
        """
        Update execution log.

        Args:
            log_id: Log ID to update
            status: New status
            duration_seconds: Duration in seconds
            metadata: Additional metadata (will be merged)
            error_message: Error message

        Pattern: Partial Update (only provided fields)
        """
        await self._ensure_pool()

        # Build dynamic update query
        updates = []
        params = [log_id]
        param_idx = 2

        if status is not None:
            updates.append(f"status = ${param_idx}")
            params.append(status)
            param_idx += 1

        if duration_seconds is not None:
            updates.append(f"duration_seconds = ${param_idx}")
            params.append(duration_seconds)
            param_idx += 1

        if metadata is not None:
            # Merge with existing metadata
            updates.append(
                f"metadata = COALESCE(metadata, '{{}}'::jsonb) || ${param_idx}::jsonb"
            )
            params.append(self._file_manager.json_dumps(metadata))
            param_idx += 1

        if error_message is not None:
            updates.append(f"error_message = ${param_idx}")
            params.append(error_message)
            param_idx += 1

        if not updates:
            return

        query = f"""
            UPDATE execution_logs
            SET {", ".join(updates)}
            WHERE id = $1
        """

        async with self.pool.acquire() as conn:
            await conn.execute(query, *params)

        logger.info(f" Log atualizado (ID: {log_id}, status: {status})")

    async def get_logs(
        self,
        operation: str | None = None,
        status: str | None = None,
        user_id: str | None = None,
        since: datetime | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        """
        Query execution logs with filters.

        Args:
            operation: Filter by operation
            status: Filter by status
            user_id: Filter by user
            since: Filter by date (created_at >= since)
            limit: Max results
            offset: Pagination offset

        Returns:
            List of log dictionaries

        Pattern: Query Builder with filters
        """
        await self._ensure_pool()

        # Build dynamic WHERE clause
        where_clauses = []
        params = []
        param_idx = 1

        if operation:
            where_clauses.append(f"operation = ${param_idx}")
            params.append(operation)
            param_idx += 1

        if status:
            where_clauses.append(f"status = ${param_idx}")
            params.append(status)
            param_idx += 1

        if user_id:
            where_clauses.append(f"user_id = ${param_idx}")
            params.append(user_id)
            param_idx += 1

        if since:
            where_clauses.append(f"created_at >= ${param_idx}")
            params.append(since)
            param_idx += 1

        where_sql = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

        query = f"""
            SELECT
                id, user_id, operation, status,
                duration_seconds, metadata, error_message, created_at
            FROM execution_logs
            {where_sql}
            ORDER BY created_at DESC
            LIMIT ${param_idx}
            OFFSET ${param_idx + 1}
        """

        params.extend([limit, offset])

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *params)

        logs = []
        for row in rows:
            logs.append(
                {
                    "id": row["id"],
                    "user_id": row["user_id"],
                    "operation": row["operation"],
                    "status": row["status"],
                    "duration_seconds": row["duration_seconds"],
                    "metadata": row["metadata"],
                    "error_message": row["error_message"],
                    "created_at": row["created_at"],
                }
            )

        logger.info(f" {len(logs)} logs recuperados")

        return logs

    async def get_log_by_id(self, log_id: int) -> dict | None:
        """
        Get single log by ID.

        Args:
            log_id: Log ID

        Returns:
            Log dictionary or None
        """
        await self._ensure_pool()

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT
                    id, user_id, operation, status,
                    duration_seconds, metadata, error_message, created_at
                FROM execution_logs
                WHERE id = $1
                """,
                log_id,
            )

        if row is None:
            return None

        return {
            "id": row["id"],
            "user_id": row["user_id"],
            "operation": row["operation"],
            "status": row["status"],
            "duration_seconds": row["duration_seconds"],
            "metadata": row["metadata"],
            "error_message": row["error_message"],
            "created_at": row["created_at"],
        }

    async def get_statistics(
        self, operation: str | None = None, since: datetime | None = None
    ) -> dict:
        """
        Get execution statistics.

        Args:
            operation: Filter by operation
            since: Since date

        Returns:
            Statistics dictionary with counts, avg duration, etc.

        Pattern: Aggregation Query
        """
        await self._ensure_pool()

        where_clauses = []
        params = []
        param_idx = 1

        if operation:
            where_clauses.append(f"operation = ${param_idx}")
            params.append(operation)
            param_idx += 1

        if since:
            where_clauses.append(f"created_at >= ${param_idx}")
            params.append(since)
            param_idx += 1

        where_sql = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

        async with self.pool.acquire() as conn:
            # Overall stats
            row = await conn.fetchrow(
                f"""
                SELECT
                    COUNT(*) as total_executions,
                    COUNT(*) FILTER (WHERE status = 'success') as successful,
                    COUNT(*) FILTER (WHERE status = 'failed') as failed,
                    COUNT(*) FILTER (WHERE status = 'running') as running,
                    AVG(duration_seconds) as avg_duration,
                    MIN(duration_seconds) as min_duration,
                    MAX(duration_seconds) as max_duration
                FROM execution_logs
                {where_sql}
                """,
                *params,
            )

            # Per-operation stats
            rows_by_op = await conn.fetch(
                f"""
                SELECT
                    operation,
                    COUNT(*) as count,
                    AVG(duration_seconds) as avg_duration
                FROM execution_logs
                {where_sql}
                GROUP BY operation
                ORDER BY count DESC
                """,
                *params,
            )

        stats = {
            "total_executions": row["total_executions"],
            "successful": row["successful"],
            "failed": row["failed"],
            "running": row["running"],
            "success_rate": (
                row["successful"] / row["total_executions"]
                if row["total_executions"] > 0
                else 0
            ),
            "avg_duration": float(row["avg_duration"]) if row["avg_duration"] else 0,
            "min_duration": float(row["min_duration"]) if row["min_duration"] else 0,
            "max_duration": float(row["max_duration"]) if row["max_duration"] else 0,
            "by_operation": [],
        }

        for op_row in rows_by_op:
            stats["by_operation"].append(
                {
                    "operation": op_row["operation"],
                    "count": op_row["count"],
                    "avg_duration": (
                        float(op_row["avg_duration"]) if op_row["avg_duration"] else 0
                    ),
                }
            )

        return stats

    async def delete_old_logs(self, older_than_days: int = 30) -> int:
        """
        Delete logs older than specified days.

        Args:
            older_than_days: Delete logs older than this many days

        Returns:
            Number of logs deleted

        Pattern: TTL (Time To Live) for log rotation
        """
        await self._ensure_pool()

        cutoff_date = datetime.now() - timedelta(days=older_than_days)

        logger.info(f"  Deletando logs anteriores a {cutoff_date.date()}")

        async with self.pool.acquire() as conn:
            result = await conn.execute(
                """
                DELETE FROM execution_logs
                WHERE created_at < $1
                """,
                cutoff_date,
            )

        deleted = int(result.split()[-1]) if result else 0

        logger.info(f"  {deleted:,} logs deletados")

        return deleted


def log_execution(operation: str):
    """
    Decorator for automatic execution logging.

    Usage:
        @log_execution("kg_training")
        async def train_kg():
            ...

    Pattern: Decorator Pattern for cross-cutting concerns
    """

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            repo = ExecutionLogsRepository()

            # Create log entry
            start_time = datetime.now()
            log_id = await repo.create_log(
                operation=operation,
                status="running",
                metadata={
                    "function": func.__name__,
                    "args_count": len(args),
                    "kwargs_count": len(kwargs),
                },
            )

            try:
                # Execute function
                result = await func(*args, **kwargs)

                # Update log on success
                duration = (datetime.now() - start_time).total_seconds()
                await repo.update_log(
                    log_id=log_id, status="success", duration_seconds=duration
                )

                return result

            except Exception as e:
                # Update log on failure
                duration = (datetime.now() - start_time).total_seconds()
                await repo.update_log(
                    log_id=log_id,
                    status="failed",
                    duration_seconds=duration,
                    error_message=str(e),
                    metadata={"traceback": traceback.format_exc()},
                )

                # Re-raise exception
                raise

        return wrapper

    return decorator
