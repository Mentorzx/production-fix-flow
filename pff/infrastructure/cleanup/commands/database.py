from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod

import asyncpg

from pff.shared.core.logger import logger
from pff.infrastructure.cleanup.config import _coerce_positive_int

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
        """Log deletion result at debug level.

        Args:
            deleted: Number of rows deleted.
        """
        if deleted > 0:
            logger.debug(f"{deleted} registros deletados")


def _is_missing_relation(exc: Exception) -> bool:
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
        from pff.infrastructure.cleanup import config as cleanup_config

        retention_cfg = (
            cleanup_config.CLEANUP_CONFIG.get("retention")
            if isinstance(cleanup_config.CLEANUP_CONFIG, dict)
            else {}
        )
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
            await repo._ensure_pool()

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

                # Fast row count estimation using system catalog
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
        except Exception as exc:  # noqa: BLE001
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
        except Exception as exc:  # noqa: BLE001
            if _is_missing_relation(exc):
                logger.debug(f"Database logs table missing: {exc}")
                return 0
            logger.warning(f"Error cleaning database logs: {exc}")
            return 0

    def _log_deleted(self, deleted: int) -> None:
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

    label = "Limpando dados do Knowledge Graph (PostgreSQL)"

    async def get_preview(self) -> dict | None:
        try:
            from pff.infrastructure.persistence.db.repositories import (
                KGSplitsRepository,
            )

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
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"Error fetching KG data preview: {exc}")
            return None

    async def _execute(self) -> int:
        try:
            from pff.infrastructure.persistence.db.repositories import (
                KGSplitsRepository,
            )
            from pff.infrastructure.cleanup import config as cleanup_config

            repo = KGSplitsRepository()
            if hasattr(repo, "truncate_all"):
                deleted = await repo.truncate_all()
            else:
                deleted = await repo.delete_all()
            if deleted > 0:
                logger.info(f"{deleted} triplas do KG deletadas do PostgreSQL")
                vacuum_full_enabled = cleanup_config.CLEANUP_CONFIG.get(
                    "database", {}
                ).get("vacuum_full_after_truncate")
                if vacuum_full_enabled and hasattr(repo, "vacuum_full"):
                    try:
                        await repo.vacuum_full()
                        logger.debug("VACUUM FULL executado para kg_splits")
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            f"Error running VACUUM FULL for kg_splits: {exc}"
                        )
            return deleted

        except ImportError:
            logger.debug("KGSplitsRepository unavailable")
            return 0
        except Exception as exc:  # noqa: BLE001
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
        try:
            from pff.infrastructure.persistence.db.repositories import (
                KGSplitsRepository,
            )

            repo = KGSplitsRepository()
            await repo._ensure_pool()

            if not hasattr(repo, "pool") or not repo.pool:
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
                async with repo.pool.acquire() as conn:
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
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"Error fetching preprocessed KG data preview: {exc}")
            return None

    async def _execute(self) -> int:
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
        except Exception as exc:  # noqa: BLE001
            if _is_missing_relation(exc):
                logger.debug(f"Preprocessed KG data table missing: {exc}")
                return 0
            logger.warning(f"Error cleaning preprocessed KG data: {exc}")
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
        try:
            from pff.infrastructure.persistence.db.repositories.kg_rules import (
                KGRulesRepository,
            )

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
                    "description": "Regras aprendidas",
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
            from pff.infrastructure.persistence.db.repositories.kg_rules import (
                KGRulesRepository,
            )
            from pff.infrastructure.cleanup import config as cleanup_config

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
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(f"Error running VACUUM FULL for kg_rules: {exc}")
            return deleted

        except ImportError:
            logger.debug("KGRulesRepository unavailable")
            return 0
        except Exception as exc:  # noqa: BLE001
            if _is_missing_relation(exc):
                logger.debug(f"KG rules table missing: {exc}")
                return 0
            logger.warning(f"Error cleaning rule tables: {exc}")
            return 0


class KGMappingsCleanCommand(AbstractDatabaseCleanCommand):
    """Cleanup for KG mappings table."""

    label = "Limpando mappings do Knowledge Graph (PostgreSQL)"

    async def get_preview(self) -> dict | None:
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
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"Error fetching mappings preview: {exc}")
            return None

    async def _execute(self) -> int:
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
        except Exception as exc:  # noqa: BLE001
            if _is_missing_relation(exc):
                logger.debug(f"KG mappings table missing: {exc}")
                return 0
            logger.warning(f"Error cleaning KG mappings: {exc}")
            return 0


class KGEmbeddingsCleanCommand(AbstractDatabaseCleanCommand):
    """Cleanup for KG embeddings table."""

    label = "Limpando embeddings do Knowledge Graph (PostgreSQL)"

    async def get_preview(self) -> dict | None:
        try:
            from pff.infrastructure.persistence.db.repositories.embeddings import (
                EmbeddingsRepository,
            )

            repo = EmbeddingsRepository(register_listener=False)
            await repo._ensure_pool()

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
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"Error fetching embeddings preview: {exc}")
            return None

    async def _execute(self) -> int:
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
        except Exception as exc:  # noqa: BLE001
            if _is_missing_relation(exc):
                logger.debug(f"KG embeddings table missing: {exc}")
                return 0
            logger.warning(f"Error cleaning KG embeddings: {exc}")
            return 0


class TrainingMetricsCleanCommand(AbstractDatabaseCleanCommand):
    """Cleanup for training metrics table."""

    label = "Limpando métricas de treino (PostgreSQL)"

    async def get_preview(self) -> dict | None:
        try:
            from pff.infrastructure.persistence.db.repositories.training_metrics import (
                TrainingMetricsRepository,
            )

            repo = TrainingMetricsRepository()
            await repo._ensure_pool()

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
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"Error fetching training metrics preview: {exc}")
            return None

    async def _execute(self) -> int:
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
        except Exception as exc:  # noqa: BLE001
            if _is_missing_relation(exc):
                logger.debug(f"Training metrics table missing: {exc}")
                return 0
            logger.warning(f"Error cleaning training metrics: {exc}")
            return 0


class OptunaTablesCleanCommand(AbstractDatabaseCleanCommand):
    """Cleanup for Optuna RDBStorage tables."""

    label = "Limpando estudos Optuna (PostgreSQL)"

    def __init__(self) -> None:
        self._deleted_studies = 0
        self._deleted_trials = 0

    async def get_preview(self) -> dict | None:
        try:
            from pff.infrastructure.persistence.db.connection import get_connection_pool

            pool = await get_connection_pool()

            async def fetch_data():
                async with pool.acquire() as conn:
                    exists = await conn.fetchval("SELECT to_regclass('public.studies')")
                    if not exists:
                        return None
                    rows = await conn.fetch(
                        """
                        SELECT study_id, study_name
                        FROM studies
                        ORDER BY study_id DESC
                        LIMIT 3
                        """
                    )
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
        except Exception as exc:  # noqa: BLE001
            if _is_missing_relation(exc):
                return None
            logger.debug(f"Error fetching Optuna preview: {exc}")
            return None

    async def _execute(self) -> int:
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
                await conn.execute("TRUNCATE TABLE studies RESTART IDENTITY CASCADE")
                return int(self._deleted_trials)

        except Exception as exc:  # noqa: BLE001
            if _is_missing_relation(exc):
                return 0
            logger.warning(f"Error cleaning Optuna tables: {exc}")
            return 0

    def _log_deleted(self, deleted: int) -> None:
        if deleted > 0 or self._deleted_studies > 0:
            logger.info(
                " Estudos Optuna removidos "
                f"studies={self._deleted_studies} trials={self._deleted_trials}"
            )


class HpoTrialResultsCleanCommand(AbstractDatabaseCleanCommand):
    """Cleanup for HPO trial result table."""

    label = "Limpando resultados HPO (PostgreSQL)"

    def __init__(self) -> None:
        self._deleted_rows = 0

    async def get_preview(self) -> dict | None:
        try:
            from pff.infrastructure.persistence.db.connection import get_connection_pool

            pool = await get_connection_pool()

            async def fetch_data():
                async with pool.acquire() as conn:
                    exists = await conn.fetchval(
                        "SELECT to_regclass('public.hpo_trial_results')"
                    )
                    if not exists:
                        return None
                    rows = await conn.fetch(
                        """
                        SELECT study_name, trial_number, created_at
                        FROM hpo_trial_results
                        ORDER BY created_at DESC
                        LIMIT 3
                        """
                    )
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
        except Exception as exc:  # noqa: BLE001
            if _is_missing_relation(exc):
                return None
            logger.debug(f"Error fetching HPO trial results preview: {exc}")
            return None

    async def _execute(self) -> int:
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

        except Exception as exc:  # noqa: BLE001
            if _is_missing_relation(exc):
                return 0
            logger.warning(f"Error cleaning HPO trial results: {exc}")
            return 0

    def _log_deleted(self, deleted: int) -> None:
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
        try:
            from pff.infrastructure.persistence.db.repositories.pipeline_checkpoints import (
                PipelineCheckpointsRepository,
            )

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
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"Error fetching checkpoints preview: {exc}")
            return None

    async def _execute(self) -> int:
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
        except Exception as exc:  # noqa: BLE001
            if _is_missing_relation(exc):
                logger.debug(f"Pipeline checkpoints table missing: {exc}")
                return 0
            logger.warning(f"Error cleaning pipeline checkpoints: {exc}")
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
    "PipelineCheckpointsCleanCommand",
]
