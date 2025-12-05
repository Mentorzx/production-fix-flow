from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod

from pff.utils.core.logger import logger
from pff.utils.ops.cleanup.config import CLEANUP_CONFIG, _coerce_positive_int

from .base import CleanupCommand


class AbstractDatabaseCleanCommand(CleanupCommand, ABC):
    """Template Method base for database cleanup commands.

    Design Pattern: Template Method. Subclasses implement `get_preview()` and
    `_execute()` while the base class handles execution flow and logging.

    Attributes:
        label: Human-readable description for UI display.
        size_bytes: Estimated size in bytes (populated by preview).
    """

    label: str
    size_bytes: int = 0

    @abstractmethod
    async def get_preview(self) -> dict | None:
        """Fetch a preview of data to be deleted.

        Returns:
            dict | None: Preview dictionary with keys `table_name`, `description`,
                `total_rows`, `size_bytes`, `sample_rows`, or None on failure.
        """
        ...

    @abstractmethod
    async def _execute(self) -> int:
        """Perform the actual deletion.

        Returns:
            int: Number of rows deleted.
        """
        ...

    async def execute_async(self) -> None:
        """Execute the cleanup operation asynchronously.

        Calls `_execute()` and logs the result via `_log_deleted()`.
        """
        deleted = await self._execute()
        self._log_deleted(deleted)

    def execute(self) -> None:
        """Sync wrapper for backward compatibility.

        Runs `execute_async()` in a new event loop.
        """
        asyncio.run(self.execute_async())

    def _log_deleted(self, deleted: int) -> None:
        """Log deletion result at info level (PT-BR).

        Args:
            deleted: Number of rows deleted.
        """
        logger.info(f" {deleted} registros deletados")


class DatabaseCleanCommand(AbstractDatabaseCleanCommand):
    """Cleanup command for execution logs with configurable retention.

    Deletes execution log entries older than a configurable number of days.
    Retention period is read from `CLEANUP_CONFIG["retention"]["execution_logs_days"]`.

    Args:
        retention_days: Override for log retention in days. If None, uses config.

    Attributes:
        label: Display label for UI.
    """

    label = "Limpando logs de execução antigos (PostgreSQL)"

    def __init__(self, retention_days: int | None = None):
        from pff.utils.ops.cleanup import config as cleanup_config

        retention_cfg = (
            cleanup_config.CLEANUP_CONFIG.get("retention")
            if isinstance(cleanup_config.CLEANUP_CONFIG, dict)
            else {}
        )
        default_days = _coerce_positive_int(
            retention_cfg.get("execution_logs_days") if isinstance(retention_cfg, dict) else None,
            30,
        )
        resolved_days = retention_days if retention_days is not None else default_days
        self._retention_days = _coerce_positive_int(resolved_days, default_days)

    async def get_preview(self) -> dict | None:
        """Get preview of data to be deleted."""
        try:
            from pff.db.repositories.execution_logs import ExecutionLogsRepository

            repo = ExecutionLogsRepository()
            await repo._ensure_pool()

            if not repo.pool:
                logger.debug("Connection pool not available for preview")
                return None

            query = f"""
                SELECT id, operation, status, created_at, duration_seconds
                FROM execution_logs
                WHERE created_at < NOW() - INTERVAL '{self._retention_days} days'
                ORDER BY created_at DESC
                LIMIT 3
            """

            conn = await asyncio.wait_for(repo.pool.acquire(), timeout=5.0)
            try:
                rows = await conn.fetch(query)
                count_query = f"""
                    SELECT COUNT(*) as count
                    FROM execution_logs
                    WHERE created_at < NOW() - INTERVAL '{self._retention_days} days'
                """
                count_result = await conn.fetchrow(count_query)
                total = count_result["count"] if count_result else 0

                size_query = "SELECT pg_total_relation_size('execution_logs')"
                total_table_size = await conn.fetchval(size_query)
                avg_row_size = total_table_size / (await conn.fetchval("SELECT COUNT(*) FROM execution_logs") or 1)
                estimated_size = int(avg_row_size * total)

                return {
                    "table_name": "execution_logs",
                    "description": f"Logs de execução (>{self._retention_days} dias)",
                    "total_rows": total,
                    "size_bytes": estimated_size,
                    "sample_rows": [dict(row) for row in rows],
                }
            finally:
                await repo.pool.release(conn)

        except (ImportError, asyncio.TimeoutError, AttributeError):
            return None
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"Error fetching log preview: {exc}")
            return None

    async def _execute(self) -> int:
        """Delete old execution logs."""
        try:
            from pff.db.repositories.execution_logs import ExecutionLogsRepository

            repo = ExecutionLogsRepository()
            deleted = await repo.delete_old_logs(older_than_days=self._retention_days)
            return deleted
        except ImportError:
            logger.debug("ExecutionLogsRepository unavailable")
            return 0
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Error cleaning database logs: {exc}")
            return 0

    def _log_deleted(self, deleted: int) -> None:
        logger.info(f" {deleted} logs de execução deletados (>{self._retention_days} dias)")


class KGDataCleanCommand(AbstractDatabaseCleanCommand):
    """Cleanup for Knowledge Graph data tables.

    Removes all entries from the `kg_splits` table containing train/valid/test
    triples used by KG embedding models.

    Attributes:
        label: Display label for UI.
    """

    label = "Limpando dados do Knowledge Graph (PostgreSQL)"

    async def get_preview(self) -> dict | None:
        try:
            from pff.db.repositories import KGSplitsRepository

            repo = KGSplitsRepository()
            await repo._ensure_pool()

            if not hasattr(repo, "pool") or not repo.pool:
                return None

            query = """
                SELECT split_name, split_type, COUNT(*) as count, source, created_at
                FROM kg_splits
                GROUP BY split_name, split_type, source, created_at
                ORDER BY created_at DESC
                LIMIT 3
            """

            async def fetch_data():
                async with repo.pool.acquire() as conn:
                    rows = await conn.fetch(query)
                    count_query = "SELECT COUNT(*) as count FROM kg_splits"
                    count_result = await conn.fetchrow(count_query)
                    total = count_result["count"] if count_result else 0

                    size_query = "SELECT pg_total_relation_size('kg_splits')"
                    size_bytes = await conn.fetchval(size_query)

                    return rows, total, size_bytes

            rows, total, size_bytes = await asyncio.wait_for(fetch_data(), timeout=5.0)

            return {
                "table_name": "kg_splits",
                "description": "Dados do Knowledge Graph (train/valid/test)",
                "total_rows": total,
                "size_bytes": size_bytes,
                "sample_rows": [dict(row) for row in rows],
            }

        except (ImportError, asyncio.TimeoutError, AttributeError):
            return None
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"Error fetching KG data preview: {exc}")
            return None

    async def _execute(self) -> int:
        try:
            from pff.db.repositories import KGSplitsRepository

            repo = KGSplitsRepository()
            deleted = await repo.delete_all()
            logger.info(f" {deleted} triplas do KG deletadas do PostgreSQL")
            return deleted

        except ImportError:
            logger.debug("KGSplitsRepository unavailable")
            return 0
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Error cleaning KG data: {exc}")
            return 0


class KGRulesCleanCommand(AbstractDatabaseCleanCommand):
    """Cleanup for learned rules table.

    Removes all entries from the `kg_rules` table containing symbolic rules
    learned by AnyBURL or ensemble models.

    Attributes:
        label: Display label for UI.
    """

    label = "Limpando regras aprendidas (PostgreSQL)"

    async def get_preview(self) -> dict | None:
        try:
            from pff.db.repositories.kg_rules import KGRulesRepository

            repo = KGRulesRepository()
            await repo._ensure_pool()

            if not repo.pool:
                return None

            query = """
                SELECT source, iteration, confidence, rule_text
                FROM kg_rules
                ORDER BY confidence DESC NULLS LAST
                LIMIT 3
            """

            async with repo.pool.acquire() as conn:
                rows = await conn.fetch(query)
                count_query = "SELECT COUNT(*) as count FROM kg_rules"
                count_result = await conn.fetchrow(count_query)
                total = count_result["count"] if count_result else 0

                size_query = "SELECT pg_total_relation_size('kg_rules')"
                size_bytes = await conn.fetchval(size_query)

                return {
                    "table_name": "kg_rules",
                    "description": "Regras aprendidas (AnyBURL/Ensemble)",
                    "total_rows": total,
                    "size_bytes": size_bytes,
                    "sample_rows": [dict(row) for row in rows],
                }

        except (ImportError, asyncio.TimeoutError, AttributeError):
            return None
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"Error fetching rules preview: {exc}")
            return None

    async def _execute(self) -> int:
        try:
            from pff.db.repositories.kg_rules import KGRulesRepository

            repo = KGRulesRepository()
            deleted = await repo.delete_all()
            logger.info(f" {deleted} regras deletadas do PostgreSQL")
            return deleted

        except ImportError:
            logger.debug("KGRulesRepository unavailable")
            return 0
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Error cleaning rule tables: {exc}")
            return 0


class PipelineCheckpointsCleanCommand(AbstractDatabaseCleanCommand):
    """Cleanup for pipeline checkpoints.

    Removes all entries from the `pipeline_checkpoints` table used to track
    progress and enable pipeline resumption.

    Attributes:
        label: Display label for UI.
    """

    label = "Limpando checkpoints do pipeline (PostgreSQL)"

    async def get_preview(self) -> dict | None:
        try:
            from pff.db.repositories.pipeline_checkpoints import PipelineCheckpointsRepository

            repo = PipelineCheckpointsRepository()
            await repo._ensure_pool()

            if not repo.pool:
                logger.debug("Connection pool not available for preview")
                return None

            query = """
                SELECT id, pipeline_name, step_name, status, progress, created_at
                FROM pipeline_checkpoints
                ORDER BY created_at DESC
                LIMIT 3
            """

            conn = await asyncio.wait_for(repo.pool.acquire(), timeout=5.0)
            try:
                rows = await conn.fetch(query)
                count_query = "SELECT COUNT(*) as count FROM pipeline_checkpoints"
                count_result = await conn.fetchrow(count_query)
                total = count_result["count"] if count_result else 0

                size_query = "SELECT pg_total_relation_size('pipeline_checkpoints')"
                size_bytes = await conn.fetchval(size_query)

                return {
                    "table_name": "pipeline_checkpoints",
                    "description": "Checkpoints do pipeline",
                    "total_rows": total,
                    "size_bytes": size_bytes,
                    "sample_rows": [dict(row) for row in rows],
                }
            finally:
                await repo.pool.release(conn)

        except (ImportError, asyncio.TimeoutError, AttributeError):
            return None
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"Error fetching checkpoints preview: {exc}")
            return None

    async def _execute(self) -> int:
        try:
            from pff.db.repositories.pipeline_checkpoints import PipelineCheckpointsRepository

            repo = PipelineCheckpointsRepository()
            deleted = await repo.delete_all_checkpoints()
            logger.info(f" {deleted} checkpoints do pipeline deletados")
            return deleted

        except ImportError:
            logger.debug("PipelineCheckpointsRepository unavailable")
            return 0
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Error cleaning pipeline checkpoints: {exc}")
            return 0


__all__ = [
    "AbstractDatabaseCleanCommand",
    "DatabaseCleanCommand",
    "KGDataCleanCommand",
    "KGRulesCleanCommand",
    "PipelineCheckpointsCleanCommand",
]
