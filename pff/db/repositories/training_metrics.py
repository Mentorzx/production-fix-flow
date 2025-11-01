"""
TrainingMetricsRepository - Repository Pattern for Training Metrics.

Design Patterns Applied:
- Repository Pattern: Encapsulates data access logic
- Builder Pattern: Fluent API for metric queries
- Strategy Pattern: Different metric types and aggregations
- Time-Series Pattern: Track metrics over epochs

SOTA Features:
- JSONB for flexible metadata (PostgreSQL 9.4+)
- GIN index for fast JSONB queries
- Time-series queries with epoch tracking
- Aggregations (best epoch, average metrics)
"""

from datetime import datetime
from typing import Optional

from loguru import logger

from pff.db.connection import get_connection_pool


class TrainingMetricsRepository:
    """
    Repository for managing training metrics with JSONB metadata.

    Pattern: Repository + Time-Series Data
    """

    def __init__(self):
        """Initialize repository with connection pool."""
        self.pool = None

    async def _ensure_pool(self):
        """Lazy initialization of connection pool."""
        if self.pool is None:
            self.pool = await get_connection_pool()

    async def log_metric(
        self,
        model_name: str,
        metric_name: str,
        metric_value: float,
        epoch: Optional[int] = None,
        split: Optional[str] = None,
        execution_log_id: Optional[int] = None,
        metadata: Optional[dict] = None
    ) -> int:
        """
        Log a single training metric.

        Args:
            model_name: Model name ('transe', 'lightgbm', 'ensemble')
            metric_name: Metric name ('loss', 'mrr', 'hits@1', 'hits@10', 'auc')
            metric_value: Metric value (float)
            epoch: Epoch number (None for final metrics)
            split: Data split ('train', 'valid', 'test')
            execution_log_id: Optional FK to execution_logs
            metadata: Optional JSONB metadata

        Returns:
            Metric ID

        Pattern: Insert with RETURNING for ID
        """
        await self._ensure_pool()

        logger.info(f"📊 Logging metric: {model_name}/{metric_name}={metric_value:.4f} (epoch={epoch}, split={split})")

        async with self.pool.acquire() as conn:
            metric_id = await conn.fetchval(
                """
                INSERT INTO training_metrics
                    (execution_log_id, model_name, epoch, metric_name, metric_value, split, metadata, created_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, NOW())
                RETURNING id
                """,
                execution_log_id,
                model_name,
                epoch,
                metric_name,
                metric_value,
                split,
                None if metadata is None else __import__('json').dumps(metadata)
            )

        logger.debug(f"✅ Metric logged (ID: {metric_id})")

        return metric_id

    async def log_epoch_metrics(
        self,
        model_name: str,
        epoch: int,
        metrics: dict[str, float],
        split: str = "train",
        execution_log_id: Optional[int] = None,
        metadata: Optional[dict] = None
    ) -> list[int]:
        """
        Log multiple metrics for a single epoch.

        Args:
            model_name: Model name
            epoch: Epoch number
            metrics: Dictionary of metric_name -> metric_value
            split: Data split
            execution_log_id: Optional FK to execution_logs
            metadata: Optional JSONB metadata (same for all metrics)

        Returns:
            List of metric IDs

        Pattern: Batch Insert
        """
        await self._ensure_pool()

        logger.info(f"📊 Logging {len(metrics)} metrics for {model_name} epoch {epoch} ({split})")

        async with self.pool.acquire() as conn:
            async with conn.transaction():
                records = [
                    (
                        execution_log_id,
                        model_name,
                        epoch,
                        metric_name,
                        metric_value,
                        split,
                        None if metadata is None else __import__('json').dumps(metadata)
                    )
                    for metric_name, metric_value in metrics.items()
                ]

                result = await conn.executemany(
                    """
                    INSERT INTO training_metrics
                        (execution_log_id, model_name, epoch, metric_name, metric_value, split, metadata, created_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, NOW())
                    """,
                    records
                )

        logger.success(f"✅ {len(metrics)} metrics logged for epoch {epoch}")

        return list(range(len(metrics)))

    async def get_metrics(
        self,
        model_name: Optional[str] = None,
        metric_name: Optional[str] = None,
        epoch: Optional[int] = None,
        split: Optional[str] = None,
        execution_log_id: Optional[int] = None,
        limit: int = 1000,
        offset: int = 0
    ) -> list[dict]:
        """
        Query training metrics with filters.

        Args:
            model_name: Filter by model name
            metric_name: Filter by metric name
            epoch: Filter by epoch
            split: Filter by split
            execution_log_id: Filter by execution log ID
            limit: Max results
            offset: Pagination offset

        Returns:
            List of metric dictionaries

        Pattern: Query Builder with filters
        """
        await self._ensure_pool()

        where_clauses = []
        params = []
        param_idx = 1

        if model_name:
            where_clauses.append(f"model_name = ${param_idx}")
            params.append(model_name)
            param_idx += 1

        if metric_name:
            where_clauses.append(f"metric_name = ${param_idx}")
            params.append(metric_name)
            param_idx += 1

        if epoch is not None:
            where_clauses.append(f"epoch = ${param_idx}")
            params.append(epoch)
            param_idx += 1

        if split:
            where_clauses.append(f"split = ${param_idx}")
            params.append(split)
            param_idx += 1

        if execution_log_id:
            where_clauses.append(f"execution_log_id = ${param_idx}")
            params.append(execution_log_id)
            param_idx += 1

        where_sql = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

        query = f"""
            SELECT
                id, execution_log_id, model_name, epoch,
                metric_name, metric_value, split, metadata, created_at
            FROM training_metrics
            {where_sql}
            ORDER BY created_at DESC, epoch DESC
            LIMIT ${param_idx}
            OFFSET ${param_idx + 1}
        """

        params.extend([limit, offset])

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *params)

        metrics = []
        for row in rows:
            metrics.append({
                'id': row['id'],
                'execution_log_id': row['execution_log_id'],
                'model_name': row['model_name'],
                'epoch': row['epoch'],
                'metric_name': row['metric_name'],
                'metric_value': row['metric_value'],
                'split': row['split'],
                'metadata': row['metadata'],
                'created_at': row['created_at']
            })

        logger.info(f"📊 {len(metrics)} metrics recuperados")

        return metrics

    async def get_epoch_metrics(
        self,
        model_name: str,
        epoch: int,
        split: Optional[str] = None
    ) -> dict[str, float]:
        """
        Get all metrics for a specific epoch.

        Args:
            model_name: Model name
            epoch: Epoch number
            split: Optional split filter

        Returns:
            Dictionary mapping metric_name -> metric_value
        """
        await self._ensure_pool()

        params = [model_name, epoch]
        where_clauses = ["model_name = $1", "epoch = $2"]

        if split:
            where_clauses.append("split = $3")
            params.append(split)

        where_sql = " AND ".join(where_clauses)

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT metric_name, metric_value
                FROM training_metrics
                WHERE {where_sql}
                """,
                *params
            )

        metrics = {row['metric_name']: float(row['metric_value']) for row in rows}

        logger.info(f"📊 {len(metrics)} metrics for {model_name} epoch {epoch}")

        return metrics

    async def get_best_epoch(
        self,
        model_name: str,
        metric_name: str,
        split: str = "valid",
        maximize: bool = True
    ) -> Optional[dict]:
        """
        Find the best epoch based on a specific metric.

        Args:
            model_name: Model name
            metric_name: Metric to optimize (e.g., 'mrr', 'loss')
            split: Data split to consider
            maximize: True to maximize metric (MRR), False to minimize (loss)

        Returns:
            Dictionary with epoch, metric_value, and all metrics for that epoch

        Pattern: Aggregation Query
        """
        await self._ensure_pool()

        order = "DESC" if maximize else "ASC"

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                f"""
                SELECT epoch, metric_value
                FROM training_metrics
                WHERE model_name = $1
                AND metric_name = $2
                AND split = $3
                AND epoch IS NOT NULL
                ORDER BY metric_value {order}
                LIMIT 1
                """,
                model_name, metric_name, split
            )

        if row is None:
            return None

        best_epoch = row['epoch']
        best_value = float(row['metric_value'])

        all_metrics = await self.get_epoch_metrics(model_name, best_epoch, split)

        logger.info(f"🏆 Best epoch for {model_name}: {best_epoch} ({metric_name}={best_value:.4f})")

        return {
            'epoch': best_epoch,
            'metric_name': metric_name,
            'metric_value': best_value,
            'all_metrics': all_metrics
        }

    async def get_metric_history(
        self,
        model_name: str,
        metric_name: str,
        split: Optional[str] = None
    ) -> list[tuple[int, float]]:
        """
        Get metric history across all epochs.

        Args:
            model_name: Model name
            metric_name: Metric name
            split: Optional split filter

        Returns:
            List of (epoch, metric_value) tuples, sorted by epoch

        Pattern: Time-Series Query
        """
        await self._ensure_pool()

        params = [model_name, metric_name]
        where_clauses = ["model_name = $1", "metric_name = $2", "epoch IS NOT NULL"]

        if split:
            where_clauses.append("split = $3")
            params.append(split)

        where_sql = " AND ".join(where_clauses)

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT epoch, metric_value
                FROM training_metrics
                WHERE {where_sql}
                ORDER BY epoch ASC
                """,
                *params
            )

        history = [(row['epoch'], float(row['metric_value'])) for row in rows]

        logger.info(f"📈 {len(history)} epochs in {model_name}/{metric_name} history")

        return history

    async def get_summary_statistics(
        self,
        model_name: Optional[str] = None,
        execution_log_id: Optional[int] = None
    ) -> dict:
        """
        Get summary statistics for training metrics.

        Args:
            model_name: Optional model name filter
            execution_log_id: Optional execution log filter

        Returns:
            Statistics dictionary with counts, averages, etc.

        Pattern: Aggregation Query
        """
        await self._ensure_pool()

        where_clauses = []
        params = []
        param_idx = 1

        if model_name:
            where_clauses.append(f"model_name = ${param_idx}")
            params.append(model_name)
            param_idx += 1

        if execution_log_id:
            where_clauses.append(f"execution_log_id = ${param_idx}")
            params.append(execution_log_id)
            param_idx += 1

        where_sql = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

        async with self.pool.acquire() as conn:
            overall = await conn.fetchrow(
                f"""
                SELECT
                    COUNT(*) as total_metrics,
                    COUNT(DISTINCT model_name) as total_models,
                    COUNT(DISTINCT epoch) as total_epochs,
                    MIN(created_at) as first_metric_date,
                    MAX(created_at) as last_metric_date
                FROM training_metrics
                {where_sql}
                """,
                *params
            )

            by_model = await conn.fetch(
                f"""
                SELECT
                    model_name,
                    COUNT(*) as metric_count,
                    COUNT(DISTINCT epoch) as epoch_count,
                    COUNT(DISTINCT metric_name) as unique_metrics
                FROM training_metrics
                {where_sql}
                GROUP BY model_name
                ORDER BY metric_count DESC
                """,
                *params
            )

        stats = {
            'total_metrics': overall['total_metrics'],
            'total_models': overall['total_models'],
            'total_epochs': overall['total_epochs'],
            'first_metric_date': overall['first_metric_date'],
            'last_metric_date': overall['last_metric_date'],
            'by_model': []
        }

        for row in by_model:
            stats['by_model'].append({
                'model_name': row['model_name'],
                'metric_count': row['metric_count'],
                'epoch_count': row['epoch_count'],
                'unique_metrics': row['unique_metrics']
            })

        return stats

    async def delete_metrics(
        self,
        model_name: Optional[str] = None,
        execution_log_id: Optional[int] = None
    ) -> int:
        """
        Delete training metrics.

        Args:
            model_name: Optional model name filter
            execution_log_id: Optional execution log filter

        Returns:
            Number of metrics deleted

        Pattern: Conditional Delete
        """
        await self._ensure_pool()

        if model_name and execution_log_id:
            query = "DELETE FROM training_metrics WHERE model_name = $1 AND execution_log_id = $2"
            params = [model_name, execution_log_id]
        elif model_name:
            query = "DELETE FROM training_metrics WHERE model_name = $1"
            params = [model_name]
        elif execution_log_id:
            query = "DELETE FROM training_metrics WHERE execution_log_id = $1"
            params = [execution_log_id]
        else:
            query = "DELETE FROM training_metrics"
            params = []

        async with self.pool.acquire() as conn:
            result = await conn.execute(query, *params)

        deleted = int(result.split()[-1]) if result else 0

        logger.info(f"🗑️  {deleted:,} metrics deletados")

        return deleted
