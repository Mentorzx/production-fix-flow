"""Provide module-level functionality for the PFF codebase.



Notes:

    File: src/pff/infrastructure/cleanup/commands/database.py

"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import asyncio

import asyncpg

from pff.infrastructure.cleanup.config import _coerce_positive_int
from pff.shared.acceleration.asyncio_runner import run_coroutine_sync
from pff.shared.core.logging import logger

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
    total_rows: int = 0

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
        run_coroutine_sync(self.execute_async())

    def _log_deleted(self, deleted: int) -> None:
        """Log deletion result at debug level.

        Args:
            deleted: Number of rows deleted.
        """
        if deleted > 0:
            logger.debug(f"{deleted} registros deletados")


def _is_missing_relation(exc: Exception) -> bool:
    """Execute is missing relation.



    Args:

        exc: Input value used by this callable.



    Returns:

        Return value produced by the callable.

    """

    undefined_exc = None
    if hasattr(asyncpg, "exceptions"):
        undefined_exc = getattr(asyncpg.exceptions, "UndefinedTableError", None)
    if undefined_exc and isinstance(exc, undefined_exc):
        return True
    if hasattr(asyncpg, "UndefinedTableError") and isinstance(
        exc, asyncpg.UndefinedTableError
    ):
        return True
    return "does not exist" in str(exc).lower()


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
        """Execute init.



        Args:

            retention_days: Optional input value.

        """

        from pff.infrastructure.cleanup import config as cleanup_config

        retention_cfg = cleanup_config.CLEANUP_CONFIG.get("retention", {})
        default_days = _coerce_positive_int(
            (
                retention_cfg.get("execution_logs_days")
                if isinstance(retention_cfg, dict)
                else None
            ),
            30,
        )
        resolved_days = retention_days if retention_days is not None else default_days
        self._retention_days = _coerce_positive_int(resolved_days, default_days)

    async def get_preview(self) -> dict | None:
        """Get preview of data to be deleted."""
        try:
            from pff.infrastructure.persistence.db.repositories.execution_logs import (
                ExecutionLogsRepository,
            )

            repo = ExecutionLogsRepository()
            if not repo.pool:
                logger.debug("Connection pool not available for preview")
                return None

            from pff.infrastructure.cleanup.config import CLEANUP_CONFIG

            db_timeout = CLEANUP_CONFIG.get("database", {}).get(
                "acquire_timeout_s", 5.0
            )

            query = f"""
                SELECT id, operation, status, created_at, duration_seconds
                FROM execution_logs
                WHERE created_at < NOW() - INTERVAL '{self._retention_days} days'
                ORDER BY created_at DESC
                LIMIT 3
            """

            conn = await asyncio.wait_for(repo.pool.acquire(), timeout=db_timeout)
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

                estimation_query = "SELECT reltuples::bigint FROM pg_class WHERE relname = 'execution_logs'"
                estimated_total_rows = await conn.fetchval(estimation_query) or 1

                avg_row_size = total_table_size / max(estimated_total_rows, 1)
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
        except Exception as exc:
            logger.debug(f"Error fetching log preview: {exc}")
            return None

    async def _execute(self) -> int:
        """Delete old execution logs."""
        try:
            from pff.infrastructure.persistence.db.repositories.execution_logs import (
                ExecutionLogsRepository,
            )

            repo = ExecutionLogsRepository()
            deleted = await repo.delete_old_logs(older_than_days=self._retention_days)
            return deleted
        except ImportError:
            logger.debug("ExecutionLogsRepository unavailable")
            return 0
        except Exception as exc:
            if _is_missing_relation(exc):
                logger.debug(f"Database logs table missing: {exc}")
                return 0
            logger.warning(f"Error cleaning database logs: {exc}")
            return 0

    def _log_deleted(self, deleted: int) -> None:
        """Execute log deleted.



        Args:

            deleted: Input value used by this callable.

        """

        if deleted > 0:
            logger.info(
                f" {deleted} logs de execução deletados (>{self._retention_days} dias)"
            )


class KGDataCleanCommand(AbstractDatabaseCleanCommand):
    """Cleanup for Knowledge Graph data tables.

    Removes all entries from the `kg_splits` table containing train/valid/test
    triples used by KG embedding models.

    Attributes:
        label: Display label for UI.
    """

    label = "Limpando dados do Knowledge Graph (LanceDB)"

    async def get_preview(self) -> dict | None:
        """Execute get preview.



        Returns:

            Return value produced by the callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        try:
            from pff.infrastructure.persistence.db.repositories import (
                KGSplitsRepository,
            )

            repo = KGSplitsRepository()

            if hasattr(repo, "get_statistics"):
                stats = await repo.get_statistics()
                if not stats:
                    return None

                total_rows = sum(s["count"] for s in stats.values())

                size_bytes = 0

                sample_rows = []

                description = f"Splits: {', '.join(stats.keys())}"

                return {
                    "table_name": "kg_splits (LanceDB)",
                    "description": description,
                    "total_rows": total_rows,
                    "size_bytes": size_bytes,
                    "sample_rows": sample_rows,
                }

            pool = getattr(repo, "pool", None)
            if not pool:
                return None

            query = """
                SELECT split_name, split_type, COUNT(*) as count, source, created_at
                FROM kg_splits
                GROUP BY split_name, split_type, source, created_at
                ORDER BY created_at DESC
                LIMIT 3
            """

            async def fetch_data():
                """Execute fetch data.



                Returns:

                    Return value produced by the callable.



                Notes:

                    Keep behavior deterministic and free of hidden side effects.

                """

                async with pool.acquire() as conn:
                    rows = await conn.fetch(query)
                    count_query = "SELECT COUNT(*) as count FROM kg_splits"
                    count_result = await conn.fetchrow(count_query)
                    total = count_result["count"] if count_result else 0

                    size_query = "SELECT pg_total_relation_size('kg_splits')"
                    size_bytes = await conn.fetchval(size_query)

                    return rows, total, size_bytes

            from pff.infrastructure.cleanup.config import CLEANUP_CONFIG

            db_timeout = CLEANUP_CONFIG.get("database", {}).get(
                "acquire_timeout_s", 5.0
            )
            rows, total, size_bytes = await asyncio.wait_for(
                fetch_data(), timeout=db_timeout
            )

            return {
                "table_name": "kg_splits",
                "description": "Dados do Knowledge Graph (train/valid/test)",
                "total_rows": total,
                "size_bytes": size_bytes,
                "sample_rows": [dict(row) for row in rows],
            }

        except (ImportError, asyncio.TimeoutError, AttributeError):
            return None
        except Exception as exc:
            logger.debug(f"Error fetching KG data preview: {exc}")
            return None

    async def _execute(self) -> int:
        """Execute execute.



        Returns:

            Return value produced by the callable.

        """

        try:
            from pff.infrastructure.cleanup import config as cleanup_config
            from pff.infrastructure.persistence.db.repositories import (
                KGSplitsRepository,
            )

            repo = KGSplitsRepository()
            deleted = await self._delete_kg_rows(repo)

            if deleted > 0:
                logger.info(f"{deleted} triplas do KG deletadas (LanceDB/Postgres)")
                await self._maybe_vacuum_kg_splits(repo, cleanup_config)

            return deleted

        except ImportError:
            logger.debug("KGSplitsRepository unavailable")
            return 0
        except Exception as exc:
            return self._handle_execute_error(exc)

    @staticmethod
    async def _delete_kg_rows(repo: Any) -> int:
        """Execute delete kg rows.



        Args:

            repo: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        if hasattr(repo, "truncate_all"):
            return await repo.truncate_all()
        return await repo.delete_all()

    async def _maybe_vacuum_kg_splits(self, repo: Any, cleanup_config: Any) -> None:
        """Execute maybe vacuum kg splits.



        Args:

            repo: Input value used by this callable.

            cleanup_config: Input value used by this callable.

        """

        if hasattr(repo, "pool"):
            vacuum_full_enabled = cleanup_config.CLEANUP_CONFIG.get("database", {}).get(
                "vacuum_full_after_truncate"
            )
            if vacuum_full_enabled and hasattr(repo, "vacuum_full"):
                try:
                    await repo.vacuum_full()
                    logger.debug("VACUUM FULL executado para kg_splits")
                except Exception as exc:
                    logger.warning(f"Error running VACUUM FULL for kg_splits: {exc}")
            return
        if hasattr(repo, "vacuum_full"):
            await repo.vacuum_full()

    @staticmethod
    def _handle_execute_error(exc: Exception) -> int:
        """Execute handle execute error.



        Args:

            exc: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        if hasattr(exc, "sqlstate") or "does not exist" in str(exc).lower():
            if _is_missing_relation(exc):
                logger.debug(f"KG data table missing: {exc}")
                return 0
        logger.warning(f"Error cleaning KG data: {exc}")
        return 0


class KGPreprocessedSplitsCleanCommand(AbstractDatabaseCleanCommand):
    """Cleanup for preprocessed KG splits only.

    Removes entries marked as preprocessed from `kg_splits` to force rebuild.
    """

    label = "Limpando splits preprocessados do KG (PostgreSQL)"

    async def get_preview(self) -> dict | None:
        """Execute get preview.



        Returns:

            Return value produced by the callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        try:
            from pff.infrastructure.persistence.db.repositories import (
                KGSplitsRepository,
            )

            repo = KGSplitsRepository()
            pool = getattr(repo, "pool", None)
            if not pool:
                return None

            query = """
                SELECT split_name, split_type, COUNT(*) as count, source, created_at
                FROM kg_splits
                WHERE split_type = 'preprocessed'
                GROUP BY split_name, split_type, source, created_at
                ORDER BY created_at DESC
                LIMIT 3
            """

            async def fetch_data():
                """Execute fetch data.



                Returns:

                    Return value produced by the callable.



                Notes:

                    Keep behavior deterministic and free of hidden side effects.

                """

                async with pool.acquire() as conn:
                    rows = await conn.fetch(query)
                    count_query = "SELECT COUNT(*) as count FROM kg_splits WHERE split_type = 'preprocessed'"
                    count_result = await conn.fetchrow(count_query)
                    total = count_result["count"] if count_result else 0

                    size_query = "SELECT pg_total_relation_size('kg_splits')"
                    size_bytes = await conn.fetchval(size_query)

                    return rows, total, size_bytes

            from pff.infrastructure.cleanup.config import CLEANUP_CONFIG

            db_timeout = CLEANUP_CONFIG.get("database", {}).get(
                "acquire_timeout_s", 5.0
            )
            rows, total, size_bytes = await asyncio.wait_for(
                fetch_data(), timeout=db_timeout
            )

            return {
                "table_name": "kg_splits (preprocessed)",
                "description": "Splits preprocessados do Knowledge Graph",
                "total_rows": total,
                "size_bytes": size_bytes,
                "sample_rows": [dict(row) for row in rows],
            }

        except (ImportError, asyncio.TimeoutError, AttributeError):
            return None
        except Exception as exc:
            logger.debug(f"Error fetching preprocessed KG data preview: {exc}")
            return None

    async def _execute(self) -> int:
        """Execute execute.



        Returns:

            Return value produced by the callable.

        """

        try:
            from pff.infrastructure.persistence.db.repositories import (
                KGSplitsRepository,
            )

            repo = KGSplitsRepository()
            deleted = await repo.delete_preprocessed()
            if deleted > 0:
                logger.info(
                    f" {deleted} triplas preprocessadas do KG deletadas do PostgreSQL"
                )
            return deleted

        except ImportError:
            logger.debug("KGSplitsRepository unavailable")
            return 0
        except Exception as exc:
            if _is_missing_relation(exc):
                logger.debug(f"Preprocessed KG data table missing: {exc}")
                return 0
            logger.warning(f"Error cleaning processed KG data: {exc}")
            return 0


class KGRulesCleanCommand(AbstractDatabaseCleanCommand):
    """Cleanup for learned rules table.

    Removes all entries from the `kg_rules` table containing symbolic rules
    learned by DSLFM-KGC or PC2 models.

    Attributes:
        label: Display label for UI.
    """

    label = "Limpando regras aprendidas (PostgreSQL)"

    async def get_preview(self) -> dict | None:
        """Execute get preview.



        Returns:

            Return value produced by the callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        try:
            from pff.infrastructure.persistence.db.repositories.kg_rules import (
                KGRulesRepository,
            )

            repo = KGRulesRepository()
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
                    "description": "Regras aprendidas",
                    "total_rows": total,
                    "size_bytes": size_bytes,
                    "sample_rows": [dict(row) for row in rows],
                }

        except (ImportError, asyncio.TimeoutError, AttributeError):
            return None
        except Exception as exc:
            logger.debug(f"Error fetching rules preview: {exc}")
            return None

    async def _execute(self) -> int:
        """Execute execute.



        Returns:

            Return value produced by the callable.

        """

        try:
            from pff.infrastructure.cleanup import config as cleanup_config
            from pff.infrastructure.persistence.db.repositories.kg_rules import (
                KGRulesRepository,
            )

            repo = KGRulesRepository()
            if hasattr(repo, "truncate_all"):
                deleted = await repo.truncate_all()
            else:
                deleted = await repo.delete_all()
            if deleted > 0:
                logger.info(f"{deleted} regras deletadas do PostgreSQL")
                vacuum_full_enabled = cleanup_config.CLEANUP_CONFIG.get(
                    "database", {}
                ).get("vacuum_full_after_truncate")
                if vacuum_full_enabled and hasattr(repo, "vacuum_full"):
                    try:
                        await repo.vacuum_full()
                        logger.debug("VACUUM FULL executado para kg_rules")
                    except Exception as exc:
                        logger.warning(f"Error running VACUUM FULL for kg_rules: {exc}")
            return deleted

        except ImportError:
            logger.debug("KGRulesRepository unavailable")
            return 0
        except Exception as exc:
            if _is_missing_relation(exc):
                logger.debug(f"KG rules table missing: {exc}")
                return 0
            logger.warning(f"Error cleaning rule tables: {exc}")
            return 0


class KGMappingsCleanCommand(AbstractDatabaseCleanCommand):
    """Cleanup for KG mappings table."""

    label = "Limpando mappings do Knowledge Graph (PostgreSQL)"

    async def get_preview(self) -> dict | None:
        """Execute get preview.



        Returns:

            Return value produced by the callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        try:
            from pff.infrastructure.persistence.db.repositories.kg_mappings import (
                KGMappingsRepository,
            )

            repo = KGMappingsRepository()
            await repo._ensure_pool()

            if not repo.pool:
                return None

            query = """
                SELECT mapping_type, COUNT(*) as count
                FROM kg_mappings
                GROUP BY mapping_type
                ORDER BY count DESC
                LIMIT 3
            """

            async with repo.pool.acquire() as conn:
                rows = await conn.fetch(query)
                count_result = await conn.fetchrow(
                    "SELECT COUNT(*) as count FROM kg_mappings"
                )
                total = count_result["count"] if count_result else 0
                size_bytes = await conn.fetchval(
                    "SELECT pg_total_relation_size('kg_mappings')"
                )

                return {
                    "table_name": "kg_mappings",
                    "description": "Mappings do Knowledge Graph",
                    "total_rows": total,
                    "size_bytes": size_bytes,
                    "sample_rows": [dict(row) for row in rows],
                }

        except (ImportError, asyncio.TimeoutError, AttributeError):
            return None
        except Exception as exc:
            logger.debug(f"Error fetching mappings preview: {exc}")
            return None

    async def _execute(self) -> int:
        """Execute execute.



        Returns:

            Return value produced by the callable.

        """

        try:
            from pff.infrastructure.persistence.db.repositories.kg_mappings import (
                KGMappingsRepository,
            )

            repo = KGMappingsRepository()
            deleted = await repo.delete_all()
            if deleted > 0:
                logger.info(f" {deleted} mappings deletados do PostgreSQL")
            return deleted

        except ImportError:
            logger.debug("KGMappingsRepository unavailable")
            return 0
        except Exception as exc:
            if _is_missing_relation(exc):
                logger.debug(f"KG mappings table missing: {exc}")
                return 0
            logger.warning(f"Error cleaning KG mappings: {exc}")
            return 0


class KGEmbeddingsCleanCommand(AbstractDatabaseCleanCommand):
    """Cleanup for KG embeddings table."""

    label = "Limpando embeddings do Knowledge Graph (PostgreSQL)"

    async def get_preview(self) -> dict | None:
        """Execute get preview.



        Returns:

            Return value produced by the callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        try:
            from pff.infrastructure.persistence.db.repositories.embeddings import (
                EmbeddingsRepository,
            )

            repo = EmbeddingsRepository(register_listener=False)
            if not repo.pool:
                return None

            query = """
                SELECT entity_type, model_version, COUNT(*) as count
                FROM kg_embeddings
                GROUP BY entity_type, model_version
                ORDER BY count DESC
                LIMIT 3
            """

            async with repo.pool.acquire() as conn:
                rows = await conn.fetch(query)
                count_result = await conn.fetchrow(
                    "SELECT COUNT(*) as count FROM kg_embeddings"
                )
                total = count_result["count"] if count_result else 0
                size_bytes = await conn.fetchval(
                    "SELECT pg_total_relation_size('kg_embeddings')"
                )

                return {
                    "table_name": "kg_embeddings",
                    "description": "Embeddings do Knowledge Graph",
                    "total_rows": total,
                    "size_bytes": size_bytes,
                    "sample_rows": [dict(row) for row in rows],
                }

        except (ImportError, asyncio.TimeoutError, AttributeError):
            return None
        except Exception as exc:
            logger.debug(f"Error fetching embeddings preview: {exc}")
            return None

    async def _execute(self) -> int:
        """Execute execute.



        Returns:

            Return value produced by the callable.

        """

        try:
            from pff.infrastructure.persistence.db.repositories.embeddings import (
                EmbeddingsRepository,
            )

            repo = EmbeddingsRepository(register_listener=False)
            deleted = await repo.delete_embeddings()
            return deleted

        except ImportError:
            logger.debug("EmbeddingsRepository unavailable")
            return 0
        except Exception as exc:
            if _is_missing_relation(exc):
                logger.debug(f"KG embeddings table missing: {exc}")
                return 0
            logger.warning(f"Error cleaning KG embeddings: {exc}")
            return 0


class TrainingMetricsCleanCommand(AbstractDatabaseCleanCommand):
    """Cleanup for training metrics table."""

    label = "Limpando métricas de treino (PostgreSQL)"

    async def get_preview(self) -> dict | None:
        """Execute get preview.



        Returns:

            Return value produced by the callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        try:
            from pff.infrastructure.persistence.db.repositories.training_metrics import (
                TrainingMetricsRepository,
            )

            repo = TrainingMetricsRepository()
            if not repo.pool:
                return None

            query = """
                SELECT model_name, COUNT(*) as count
                FROM training_metrics
                GROUP BY model_name
                ORDER BY count DESC
                LIMIT 3
            """

            async with repo.pool.acquire() as conn:
                rows = await conn.fetch(query)
                count_result = await conn.fetchrow(
                    "SELECT COUNT(*) as count FROM training_metrics"
                )
                total = count_result["count"] if count_result else 0
                size_bytes = await conn.fetchval(
                    "SELECT pg_total_relation_size('training_metrics')"
                )

                return {
                    "table_name": "training_metrics",
                    "description": "Métricas de treinamento",
                    "total_rows": total,
                    "size_bytes": size_bytes,
                    "sample_rows": [dict(row) for row in rows],
                }

        except (ImportError, asyncio.TimeoutError, AttributeError):
            return None
        except Exception as exc:
            logger.debug(f"Error fetching training metrics preview: {exc}")
            return None

    async def _execute(self) -> int:
        """Execute execute.



        Returns:

            Return value produced by the callable.

        """

        try:
            from pff.infrastructure.persistence.db.repositories.training_metrics import (
                TrainingMetricsRepository,
            )

            repo = TrainingMetricsRepository()
            deleted = await repo.delete_metrics()
            return deleted

        except ImportError:
            logger.debug("TrainingMetricsRepository unavailable")
            return 0
        except Exception as exc:
            if _is_missing_relation(exc):
                logger.debug(f"Training metrics table missing: {exc}")
                return 0
            logger.warning(f"Error cleaning training metrics: {exc}")
            return 0


class OptunaTablesCleanCommand(AbstractDatabaseCleanCommand):
    """Cleanup for Optuna RDBStorage tables."""

    label = "Limpando estudos Optuna (PostgreSQL)"

    def __init__(self) -> None:
        """Execute init."""

        self._deleted_studies = 0
        self._deleted_trials = 0

    async def get_preview(self) -> dict | None:
        """Execute get preview.



        Returns:

            Return value produced by the callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        try:
            from pff.infrastructure.persistence.db.connection import get_connection_pool

            pool = await get_connection_pool()

            async def fetch_data():
                """Execute fetch data.



                Returns:

                    Return value produced by the callable.



                Notes:

                    Keep behavior deterministic and free of hidden side effects.

                """

                async with pool.acquire() as conn:
                    exists = await conn.fetchval("SELECT to_regclass('public.studies')")
                    if not exists:
                        return None
                    rows = await conn.fetch("""
                        SELECT study_id, study_name
                        FROM studies
                        ORDER BY study_id DESC
                        LIMIT 3
                        """)
                    total_studies = (
                        await conn.fetchval("SELECT COUNT(*) FROM studies") or 0
                    )
                    total_trials = 0
                    trials_exists = await conn.fetchval(
                        "SELECT to_regclass('public.trials')"
                    )
                    if trials_exists:
                        total_trials = (
                            await conn.fetchval("SELECT COUNT(*) FROM trials") or 0
                        )

                    size_bytes = 0
                    for table in [
                        "studies",
                        "study_user_attributes",
                        "study_system_attributes",
                        "trials",
                        "trial_params",
                        "trial_values",
                        "trial_intermediate_values",
                        "trial_user_attributes",
                        "trial_system_attributes",
                        "trial_heartbeats",
                    ]:
                        reg = await conn.fetchval(
                            "SELECT to_regclass($1)", f"public.{table}"
                        )
                        if reg:
                            size_bytes += (
                                await conn.fetchval(
                                    f"SELECT pg_total_relation_size('{table}')"
                                )
                                or 0
                            )

                    return rows, total_studies, total_trials, size_bytes

            from pff.infrastructure.cleanup.config import CLEANUP_CONFIG

            db_timeout = CLEANUP_CONFIG.get("database", {}).get(
                "acquire_timeout_s", 5.0
            )
            result = await asyncio.wait_for(fetch_data(), timeout=db_timeout)
            if result is None:
                return None
            rows, total_studies, total_trials, size_bytes = result

            description = (
                f"Estudos Optuna (studies={total_studies}, trials={total_trials})"
            )
            return {
                "table_name": "optuna",
                "description": description,
                "total_rows": total_trials,
                "size_bytes": size_bytes,
                "sample_rows": [dict(row) for row in rows],
            }

        except (ImportError, asyncio.TimeoutError, AttributeError):
            return None
        except Exception as exc:
            if _is_missing_relation(exc):
                return None
            logger.debug(f"Error fetching Optuna preview: {exc}")
            return None

    async def _execute(self) -> int:
        """Execute execute.



        Returns:

            Return value produced by the callable.

        """

        try:
            from pff.infrastructure.persistence.db.connection import get_connection_pool

            pool = await get_connection_pool()
            async with pool.acquire() as conn:
                exists = await conn.fetchval("SELECT to_regclass('public.studies')")
                if not exists:
                    return 0
                self._deleted_studies = (
                    await conn.fetchval("SELECT COUNT(*) FROM studies") or 0
                )
                trials_exists = await conn.fetchval(
                    "SELECT to_regclass('public.trials')"
                )
                if trials_exists:
                    self._deleted_trials = (
                        await conn.fetchval("SELECT COUNT(*) FROM trials") or 0
                    )

                tables = [
                    "studies",
                    "trials",
                    "trial_params",
                    "trial_values",
                    "trial_intermediate_values",
                    "trial_user_attributes",
                    "trial_system_attributes",
                    "trial_heartbeats",
                    "study_user_attributes",
                    "study_system_attributes",
                ]

                valid_tables = []
                for t in tables:
                    if await conn.fetchval("SELECT to_regclass($1)", f"public.{t}"):
                        valid_tables.append(t)

                if valid_tables:
                    tables_str = ", ".join(valid_tables)
                    await conn.execute(
                        f"TRUNCATE TABLE {tables_str} RESTART IDENTITY CASCADE"
                    )

                return int(self._deleted_trials)

        except Exception as exc:
            if _is_missing_relation(exc):
                return 0
            logger.warning(f"Error cleaning Optuna tables: {exc}")
            return 0

    def _log_deleted(self, deleted: int) -> None:
        """Execute log deleted.



        Args:

            deleted: Input value used by this callable.

        """

        if deleted > 0 or self._deleted_studies > 0:
            logger.info(
                " Estudos Optuna removidos "
                f"studies={self._deleted_studies} trials={self._deleted_trials}"
            )


class HpoTrialResultsCleanCommand(AbstractDatabaseCleanCommand):
    """Cleanup for HPO trial result table."""

    label = "Limpando resultados HPO (PostgreSQL)"

    def __init__(self) -> None:
        """Execute init."""

        self._deleted_rows = 0

    async def get_preview(self) -> dict | None:
        """Execute get preview.



        Returns:

            Return value produced by the callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        try:
            from pff.infrastructure.persistence.db.connection import get_connection_pool

            pool = await get_connection_pool()

            async def fetch_data():
                """Execute fetch data.



                Returns:

                    Return value produced by the callable.



                Notes:

                    Keep behavior deterministic and free of hidden side effects.

                """

                async with pool.acquire() as conn:
                    exists = await conn.fetchval(
                        "SELECT to_regclass('public.hpo_trial_results')"
                    )
                    if not exists:
                        return None
                    rows = await conn.fetch("""
                        SELECT study_name, trial_number, created_at
                        FROM hpo_trial_results
                        ORDER BY created_at DESC
                        LIMIT 3
                        """)
                    total = (
                        await conn.fetchval("SELECT COUNT(*) FROM hpo_trial_results")
                        or 0
                    )
                    size_bytes = await conn.fetchval(
                        "SELECT pg_total_relation_size('hpo_trial_results')"
                    )
                    return rows, total, size_bytes

            from pff.infrastructure.cleanup.config import CLEANUP_CONFIG

            db_timeout = CLEANUP_CONFIG.get("database", {}).get(
                "acquire_timeout_s", 5.0
            )
            result = await asyncio.wait_for(fetch_data(), timeout=db_timeout)
            if result is None:
                return None
            rows, total, size_bytes = result

            return {
                "table_name": "hpo_trial_results",
                "description": "Resultados de trials HPO",
                "total_rows": total,
                "size_bytes": size_bytes,
                "sample_rows": [dict(row) for row in rows],
            }

        except (ImportError, asyncio.TimeoutError, AttributeError):
            return None
        except Exception as exc:
            if _is_missing_relation(exc):
                return None
            logger.debug(f"Error fetching HPO trial results preview: {exc}")
            return None

    async def _execute(self) -> int:
        """Execute execute.



        Returns:

            Return value produced by the callable.

        """

        try:
            from pff.infrastructure.persistence.db.connection import get_connection_pool

            pool = await get_connection_pool()
            async with pool.acquire() as conn:
                exists = await conn.fetchval(
                    "SELECT to_regclass('public.hpo_trial_results')"
                )
                if not exists:
                    return 0
                self._deleted_rows = (
                    await conn.fetchval("SELECT COUNT(*) FROM hpo_trial_results") or 0
                )
                await conn.execute("TRUNCATE TABLE hpo_trial_results")
                return int(self._deleted_rows)

        except Exception as exc:
            if _is_missing_relation(exc):
                return 0
            logger.warning(f"Error cleaning HPO trial results: {exc}")
            return 0

    def _log_deleted(self, deleted: int) -> None:
        """Execute log deleted.



        Args:

            deleted: Input value used by this callable.

        """

        if deleted > 0:
            logger.info(f" Resultados HPO removidos rows={self._deleted_rows}")


class PipelineCheckpointsCleanCommand(AbstractDatabaseCleanCommand):
    """Cleanup for pipeline checkpoints.

    Removes all entries from the `pipeline_checkpoints` table used to track
    progress and enable pipeline resumption.

    Attributes:
        label: Display label for UI.
    """

    label = "Limpando checkpoints do pipeline (PostgreSQL)"

    async def get_preview(self) -> dict | None:
        """Execute get preview.



        Returns:

            Return value produced by the callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        try:
            from pff.infrastructure.persistence.db.repositories.pipeline_checkpoints import (
                PipelineCheckpointsRepository,
            )

            repo = PipelineCheckpointsRepository()
            if not repo.pool:
                logger.debug("Connection pool not available for preview")
                return None

            query = """
                SELECT id, pipeline_name, step_name, status, progress, created_at
                FROM pipeline_checkpoints
                ORDER BY created_at DESC
                LIMIT 3
            """

            from pff.infrastructure.cleanup.config import CLEANUP_CONFIG

            db_timeout = CLEANUP_CONFIG.get("database", {}).get(
                "acquire_timeout_s", 5.0
            )
            conn = await asyncio.wait_for(repo.pool.acquire(), timeout=db_timeout)
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
        except Exception as exc:
            logger.debug(f"Error fetching checkpoints preview: {exc}")
            return None

    async def _execute(self) -> int:
        """Execute execute.



        Returns:

            Return value produced by the callable.

        """

        try:
            from pff.infrastructure.persistence.db.repositories.pipeline_checkpoints import (
                PipelineCheckpointsRepository,
            )

            repo = PipelineCheckpointsRepository()
            deleted = await repo.delete_all_checkpoints()
            if deleted > 0:
                logger.info(f" {deleted} checkpoints do pipeline deletados")
            return deleted

        except ImportError:
            logger.debug("PipelineCheckpointsRepository unavailable")
            return 0
        except Exception as exc:
            if _is_missing_relation(exc):
                logger.debug(f"Pipeline checkpoints table missing: {exc}")
                return 0
            logger.warning(f"Error cleaning workflow checkpoints: {exc}")
            return 0


class LanceDBOptimizeCommand(AbstractDatabaseCleanCommand):
    """Optimization and cleanup for LanceDB tables.

    Performs vacuum, compaction, and removal of old versions for the 'kg_splits' table.
    """

    label = "Otimizando LanceDB (Vacuum + Compact + Old Versions)"

    async def get_preview(self) -> dict | None:
        """Execute get preview.



        Returns:

            Return value produced by the callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        try:
            import lancedb

            from pff.infrastructure.persistence.db.repositories.kg_splits import (
                LANCE_DB_PATH,
                SPLITS_TABLE,
            )

            db = lancedb.connect(LANCE_DB_PATH)
            if SPLITS_TABLE not in db.list_tables().tables:
                return None

            table = db.open_table(SPLITS_TABLE)

            total_rows = table.count_rows()

            table_path = f"{LANCE_DB_PATH}/{SPLITS_TABLE}.lance"
            try:
                total_size = 0
                import os

                for dirpath, _, filenames in os.walk(table_path):
                    for f in filenames:
                        fp = os.path.join(dirpath, f)
                        if not os.path.islink(fp):
                            total_size += os.path.getsize(fp)
            except Exception:
                total_size = 0

            versions = table.list_versions()
            num_versions = len(versions)

            try:
                import polars as pl
                from typing import Any, cast

                arrow_sample = cast(Any, table).head(3).to_arrow()
                df_sample = cast(pl.DataFrame, pl.from_arrow(arrow_sample))
                sample_rows = df_sample.to_dicts()
            except Exception:
                try:
                    sample_rows = cast(Any, table).head(3).to_arrow().to_pylist()
                except Exception:
                    sample_rows = []

            return {
                "table_name": f"LanceDB ({SPLITS_TABLE})",
                "description": f"Versões: {num_versions} | Vacuum/Compact pendente",
                "total_rows": total_rows,
                "size_bytes": total_size,
                "sample_rows": sample_rows,
            }

        except ImportError:
            logger.debug("LanceDB unavailable")
            return None
        except Exception as exc:
            logger.debug(f"Error fetching LanceDB preview: {exc}")
            return None

    async def _execute(self) -> int:
        """Execute execute.



        Returns:

            Return value produced by the callable.

        """

        try:
            import lancedb

            from pff.infrastructure.persistence.db.repositories.kg_splits import (
                LANCE_DB_PATH,
                SPLITS_TABLE,
            )

            db = lancedb.connect(LANCE_DB_PATH)
            if SPLITS_TABLE not in db.list_tables().tables:
                return 0

            table = db.open_table(SPLITS_TABLE)

            from datetime import timedelta

            table.cleanup_old_versions(older_than=timedelta(days=1))

            _ = table.optimize()

            logger.info(" LanceDB otimizado (Compact + Cleanup < 24h)")
            return 1

        except Exception as exc:
            logger.warning(f"Error optimizing LanceDB: {exc}")
            return 0


class HpoCheckpointsCleanCommand(AbstractDatabaseCleanCommand):
    """Cleanup for HPO checkpoints table."""

    label = "Limpando checkpoints de HPO (PostgreSQL)"

    async def get_preview(self) -> dict | None:
        """Execute get preview.



        Returns:

            Return value produced by the callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        try:
            from pff.infrastructure.persistence.db.connection import get_connection_pool

            pool = await get_connection_pool()

            async def fetch_data():
                """Execute fetch data.



                Returns:

                    Return value produced by the callable.



                Notes:

                    Keep behavior deterministic and free of hidden side effects.

                """

                async with pool.acquire() as conn:
                    exists = await conn.fetchval(
                        "SELECT to_regclass('public.hpo_checkpoints')"
                    )
                    if not exists:
                        return None
                    rows = await conn.fetch("""
                        SELECT checkpoint_key, updated_at
                        FROM hpo_checkpoints
                        ORDER BY updated_at DESC
                        LIMIT 3
                        """)
                    total = (
                        await conn.fetchval("SELECT COUNT(*) FROM hpo_checkpoints") or 0
                    )
                    size_bytes = await conn.fetchval(
                        "SELECT pg_total_relation_size('hpo_checkpoints')"
                    )
                    return rows, total, size_bytes

            from pff.infrastructure.cleanup.config import CLEANUP_CONFIG

            db_timeout = CLEANUP_CONFIG.get("database", {}).get(
                "acquire_timeout_s", 5.0
            )
            result = await asyncio.wait_for(fetch_data(), timeout=db_timeout)
            if result is None:
                return None
            rows, total, size_bytes = result

            return {
                "table_name": "hpo_checkpoints",
                "description": "Checkpoints de otimização HPO",
                "total_rows": total,
                "size_bytes": size_bytes,
                "sample_rows": [dict(row) for row in rows],
            }

        except (ImportError, asyncio.TimeoutError, AttributeError):
            return None
        except Exception as exc:
            if _is_missing_relation(exc):
                return None
            logger.debug(f"Error fetching HPO checkpoints preview: {exc}")
            return None

    async def _execute(self) -> int:
        """Execute execute.



        Returns:

            Return value produced by the callable.

        """

        try:
            from pff.infrastructure.persistence.db.connection import get_connection_pool

            pool = await get_connection_pool()
            async with pool.acquire() as conn:
                exists = await conn.fetchval(
                    "SELECT to_regclass('public.hpo_checkpoints')"
                )
                if not exists:
                    return 0
                total = await conn.fetchval("SELECT COUNT(*) FROM hpo_checkpoints") or 0
                await conn.execute("TRUNCATE TABLE hpo_checkpoints")
                if total > 0:
                    logger.info(f" {total} checkpoints de HPO deletados")
                return int(total)

        except Exception as exc:
            if _is_missing_relation(exc):
                return 0
            logger.warning(f"Error cleaning HPO checkpoints: {exc}")
            return 0


__all__ = [
    "AbstractDatabaseCleanCommand",
    "DatabaseCleanCommand",
    "KGDataCleanCommand",
    "KGPreprocessedSplitsCleanCommand",
    "KGMappingsCleanCommand",
    "KGEmbeddingsCleanCommand",
    "KGRulesCleanCommand",
    "TrainingMetricsCleanCommand",
    "OptunaTablesCleanCommand",
    "HpoTrialResultsCleanCommand",
    "HpoCheckpointsCleanCommand",
    "PipelineCheckpointsCleanCommand",
    "LanceDBOptimizeCommand",
]
