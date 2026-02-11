"""Integration tests for HPO using a real database connection."""

import pytest

from pff.infrastructure.hpo.trials.postgres_store import HpoPostgresStore


class _ConnectionContextManager:
    def __init__(self, conn):
        self.conn = conn

    async def __aenter__(self):
        return self.conn

    async def __aexit__(self, *args):
        pass


class SingleConnectionPool:
    """Wraps a single asyncpg connection to mimic a pool for HpoPostgresStore."""

    def __init__(self, conn):
        self.conn = conn
        self._loop = None  # To bypass loop check

    def acquire(self):
        return _ConnectionContextManager(self.conn)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_hpo_store_real_db_operations(db_connection):
    """Test HpoPostgresStore with a real database connection."""
    pool_wrapper = SingleConnectionPool(db_connection)
    store = HpoPostgresStore(pool=pool_wrapper)

    # Trigger schema creation
    await store._ensure_schema()

    study_name = "integration_test_study"

    # 1. Test Upsert Trial
    payload = {
        "params": {"lr": 0.01},
        "value": 0.85,
        "metrics": {"mrr": 0.85, "hits@10": 0.9},
    }
    await store.upsert_trial_result(study_name, 1, payload)

    # 2. Test Load Results
    results = await store.load_all_results(study_name)
    assert len(results) == 1
    assert results[0]["params"]["lr"] == 0.01

    # 3. Test Checkpoint
    checkpoint_key = "ckpt_test"
    ckpt_payload = {"status": "running", "step": 10}
    await store.upsert_checkpoint(checkpoint_key, ckpt_payload)

    loaded_ckpt = await store.load_checkpoint(checkpoint_key)
    assert loaded_ckpt is not None
    assert loaded_ckpt["status"] == "running"

    # 4. Test Best Params
    best_params = {"lr": 0.01}
    await store.upsert_best_params(study_name, best_params, 0.85)

    loaded_best = await store.load_best_params(study_name)
    assert loaded_best["best_value"] == 0.85
    assert loaded_best["best_params"] == best_params
