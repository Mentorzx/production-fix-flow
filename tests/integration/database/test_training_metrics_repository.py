"""Provide module-level functionality for the PFF codebase.



Notes:

    File: tests/integration/database/test_training_metrics_repository.py

"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pff.infrastructure.persistence.db.repositories.training_metrics import (
    TrainingMetricsRepository,
)


class AsyncContext:
    """Represent AsyncContext."""

    def __init__(self, result):
        """Execute init.



        Args:

            result: Input value used by this callable.

        """

        self._result = result

    async def __aenter__(self):
        return self._result

    async def __aexit__(self, exc_type, exc, tb):
        return False


@pytest.mark.asyncio
class TestTrainingMetricsRepository:
    """Represent TestTrainingMetricsRepository."""

    @pytest.fixture
    def mock_pool(self):
        """Execute mock pool.



        Returns:

            Return value produced by the callable.

        """

        pool = MagicMock()
        conn = AsyncMock()
        pool.acquire.return_value = AsyncContext(conn)
        conn.transaction = MagicMock(return_value=AsyncContext(None))
        return pool, conn

    async def test_ensure_schema(self, mock_pool):
        """Execute test ensure schema.



        Args:

            mock_pool: Input value used by this callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        pool, conn = mock_pool

        with patch(
            "pff.infrastructure.persistence.db.repositories.base.get_connection_pool",
            return_value=pool,
        ):
            repo = TrainingMetricsRepository()
            repo.pool = pool

            await repo._ensure_schema()

            assert conn.execute.call_count >= 1
            assert (
                "CREATE TABLE IF NOT EXISTS training_metrics"
                in conn.execute.call_args_list[0][0][0]
            )

    async def test_log_metric(self, mock_pool):
        """Execute test log metric.



        Args:

            mock_pool: Input value used by this callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        pool, conn = mock_pool
        conn.fetchval.return_value = 1

        with patch(
            "pff.infrastructure.persistence.db.repositories.base.get_connection_pool",
            return_value=pool,
        ):
            repo = TrainingMetricsRepository()
            repo.pool = pool
            repo._schema_ready = True

            metric_id = await repo.log_metric(
                model_name="transe",
                metric_name="loss",
                metric_value=0.1,
                epoch=1,
                split="train",
            )

            assert metric_id == 1
            conn.fetchval.assert_called_once()

    async def test_log_epoch_metrics_uses_copy(self, mock_pool):
        """Execute test log epoch metrics uses copy.



        Args:

            mock_pool: Input value used by this callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        pool, conn = mock_pool

        with patch(
            "pff.infrastructure.persistence.db.repositories.base.get_connection_pool",
            return_value=pool,
        ):
            repo = TrainingMetricsRepository()
            repo.pool = pool
            repo._schema_ready = True

            metrics = {"mrr": 0.5, "hits@1": 0.3}

            await repo.log_epoch_metrics(
                model_name="transe",
                epoch=2,
                metrics=metrics,
                split="valid",
            )

            assert conn.copy_records_to_table.called
            args, kwargs = conn.copy_records_to_table.call_args
            assert args[0] == "training_metrics"
            assert "columns" in kwargs

    async def test_delete_metrics(self, mock_pool):
        """Execute test delete metrics.



        Args:

            mock_pool: Input value used by this callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        pool, conn = mock_pool
        conn.execute.return_value = "DELETE 5"

        with patch(
            "pff.infrastructure.persistence.db.repositories.base.get_connection_pool",
            return_value=pool,
        ):
            repo = TrainingMetricsRepository()
            repo.pool = pool
            repo._schema_ready = True

            deleted = await repo.delete_metrics(model_name="transe")

            assert deleted == 5
            # Ensure the delete statement was issued (other setup queries may run first)
            assert conn.execute.call_count >= 1
            last_call = conn.execute.call_args_list[-1]
            assert "DELETE FROM training_metrics" in last_call.args[0]
