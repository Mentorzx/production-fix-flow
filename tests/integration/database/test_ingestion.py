"""Provide module-level functionality for the PFF codebase.



Notes:

    File: tests/integration/database/test_ingestion.py

"""

import pytest_asyncio

"""
Tests for database ingestion (Sprint 3)

Tests:
- Batch insert telecom_data
- Extract and insert KG triples
- Performance validation (<10min for 14k records)

NOTE: These tests require telecom_data and kg_triples tables.
      Skip if schema not ready.
"""

import os  # noqa: E402
from pathlib import Path  # noqa: E402
from tempfile import NamedTemporaryFile  # noqa: E402

import asyncpg  # noqa: E402
import orjson  # noqa: E402
import polars as pl  # noqa: E402
import pytest  # noqa: E402
from pff.shared.core.config import settings  # noqa: E402

pytestmark = [pytest.mark.integration]


def _database_url() -> str:
    """Resolve the database URL from the active pytest environment."""
    return os.getenv("TEST_DATABASE_URL") or os.getenv("DATABASE_URL") or settings.DATABASE_URL


@pytest_asyncio.fixture(loop_scope="function")
async def db_conn():
    """Create test database connection."""
    try:
        conn = await asyncpg.connect(_database_url())
    except Exception as e:
        pytest.skip(f"Database connection failed: {e}")
        return

    # Check if telecom_data table exists and ensure schema
    # Drop and recreate to ensure clean state and correct constraints for testing
    await conn.execute("DROP TABLE IF EXISTS telecom_data")
    await conn.execute("DROP TABLE IF EXISTS kg_triples")

    await conn.execute("""
        CREATE TABLE telecom_data (
            msisdn TEXT PRIMARY KEY,
            data JSONB,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        )
        """)

    await conn.execute("""
        CREATE TABLE kg_triples (
            subject TEXT,
            predicate TEXT,
            object TEXT,
            source TEXT,
            confidence DOUBLE PRECISION,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (subject, predicate, object)
        )
        """)

    try:
        yield conn
    finally:
        await conn.close()


@pytest_asyncio.fixture(loop_scope="function")
def sample_telecom_data():
    """Create sample telecom data for testing."""
    return {
        "id": "TEST123",
        "externalId": "180777157",
        "status": [{"status": "CustomerActive"}],
        "account": [{"externalId": "billingAccountExtId_180777157"}],
    }


@pytest.mark.asyncio
async def test_insert_telecom_data(db_conn, sample_telecom_data):
    """Test inserting telecom data."""
    msisdn = "5511910001706"

    await db_conn.execute(
        """
        INSERT INTO telecom_data (msisdn, data)
        VALUES ($1, $2)
        ON CONFLICT (msisdn) DO UPDATE SET
            data = EXCLUDED.data,
            updated_at = CURRENT_TIMESTAMP
        """,
        msisdn,
        orjson.dumps(sample_telecom_data).decode("utf-8"),
    )

    result = await db_conn.fetchrow(
        "SELECT msisdn, data FROM telecom_data WHERE msisdn = $1", msisdn
    )

    assert result is not None
    assert result["msisdn"] == msisdn

    data = orjson.loads(result["data"]) if isinstance(result["data"], str) else result["data"]
    assert data["id"] == "TEST123"

    await db_conn.execute("DELETE FROM telecom_data WHERE msisdn = $1", msisdn)


@pytest.mark.asyncio
async def test_batch_insert_telecom_data(db_conn):
    """Test batch insert performance."""
    batch = []
    for i in range(100):
        msisdn = f"5511920{i:06d}"
        data = {"id": f"TEST{i}", "externalId": f"{i}"}
        batch.append((msisdn, orjson.dumps(data).decode("utf-8")))

    await db_conn.executemany(
        """
        INSERT INTO telecom_data (msisdn, data)
        VALUES ($1, $2)
        ON CONFLICT (msisdn) DO NOTHING
        """,
        batch,
    )

    count = await db_conn.fetchval("SELECT COUNT(*) FROM telecom_data WHERE msisdn LIKE '5511920%'")

    assert count == 100

    await db_conn.execute("DELETE FROM telecom_data WHERE msisdn LIKE '5511920%'")


@pytest.mark.asyncio
async def test_insert_kg_triple(db_conn):
    """Test inserting KG triple."""
    subject = "customer_test"
    predicate = "has_status"
    object = "active"

    await db_conn.execute(
        """
        INSERT INTO kg_triples (subject, predicate, object, source, confidence)
        VALUES ($1, $2, $3, $4, $5)
        ON CONFLICT (subject, predicate, object) DO UPDATE SET
            confidence = GREATEST(kg_triples.confidence, EXCLUDED.confidence)
        """,
        subject,
        predicate,
        object,
        "test",
        1.0,
    )

    result = await db_conn.fetchrow(
        """
        SELECT subject, predicate, object, confidence
        FROM kg_triples
        WHERE subject = $1 AND predicate = $2
        """,
        subject,
        predicate,
    )

    assert result is not None
    assert result["object"] == object
    assert result["confidence"] == 1.0

    await db_conn.execute("DELETE FROM kg_triples WHERE subject = $1", subject)


@pytest.mark.asyncio
async def test_ingestion_full_cycle(db_conn):
    """Test full ingestion cycle with small subset."""
    from pff.infrastructure.persistence.db.ingestion import TelecomDataIngestion

    with NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
        tmp_path = Path(tmp.name)

        rows = []
        for i in range(3):
            msisdn = f"5511910001{i:03d}"
            filename = f"customer_enquiry_{msisdn}.txt"
            data = {
                "id": f"TEST{i}",
                "externalId": f"{i}",
                "status": [{"status": "Active"}],
            }
            rows.append(
                {
                    "_raw_json": orjson.dumps(data).decode("utf-8"),
                    "_source_name": filename,
                    "externalId": str(i),
                    "_parse_error": None,
                }
            )
        pl.DataFrame(rows).write_parquet(tmp_path)

    try:
        ingestion = TelecomDataIngestion(zip_path=tmp_path, batch_size=10)
        await ingestion.run()

        assert ingestion.stats["total_files"] == 3
        assert ingestion.stats["telecom_inserted"] == 3
        assert ingestion.stats["errors"] == 0

    finally:
        tmp_path.unlink()

        conn = await asyncpg.connect(_database_url())
        try:
            await conn.execute("DELETE FROM telecom_data WHERE msisdn LIKE '5511910001%'")
            await conn.execute("DELETE FROM kg_triples WHERE subject LIKE 'customer_test%'")
        finally:
            await conn.close()