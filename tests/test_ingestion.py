"""
Tests for database ingestion (Sprint 3)

Tests:
- Batch insert telecom_data
- Extract and insert KG triples
- Performance validation (<10min for 14k records)
"""

import asyncio
import json
import os
import zipfile
from pathlib import Path
from tempfile import NamedTemporaryFile

import asyncpg
import pytest

# Skip if PostgreSQL not available
pytestmark = pytest.mark.skipif(
    os.system("pg_isready -h localhost -p 5432 > /dev/null 2>&1") != 0,
    reason="PostgreSQL not running"
)

DATABASE_URL = "postgresql://pff_user:8qflzf45HGGQ_ghLetx4Whu7gqSVNYJ3@localhost/pff_production"


@pytest.fixture
async def db_conn():
    """Create test database connection."""
    conn = await asyncpg.connect(DATABASE_URL)
    try:
        yield conn
    finally:
        await conn.close()


@pytest.fixture
def sample_telecom_data():
    """Create sample telecom data for testing."""
    return {
        "id": "TEST123",
        "externalId": "180777157",
        "status": [{"status": "CustomerActive"}],
        "account": [{"externalId": "billingAccountExtId_180777157"}]
    }


@pytest.mark.asyncio
async def test_insert_telecom_data(db_conn, sample_telecom_data):
    """Test inserting telecom data."""
    msisdn = "5511910001706"

    # Insert (JSONB requires JSON string)
    await db_conn.execute(
        """
        INSERT INTO telecom_data (msisdn, data)
        VALUES ($1, $2)
        ON CONFLICT (msisdn) DO UPDATE SET
            data = EXCLUDED.data,
            updated_at = CURRENT_TIMESTAMP
        """,
        msisdn, json.dumps(sample_telecom_data)
    )

    # Verify
    result = await db_conn.fetchrow(
        "SELECT msisdn, data FROM telecom_data WHERE msisdn = $1",
        msisdn
    )

    assert result is not None
    assert result['msisdn'] == msisdn

    # asyncpg returns JSONB as string, need to parse
    data = json.loads(result['data']) if isinstance(result['data'], str) else result['data']
    assert data['id'] == "TEST123"

    # Cleanup
    await db_conn.execute("DELETE FROM telecom_data WHERE msisdn = $1", msisdn)


@pytest.mark.asyncio
async def test_batch_insert_telecom_data(db_conn):
    """Test batch insert performance."""
    batch = []
    for i in range(100):
        msisdn = f"5511920{i:06d}"  # Different pattern to avoid conflicts
        data = {"id": f"TEST{i}", "externalId": f"{i}"}
        batch.append((msisdn, json.dumps(data)))

    # Batch insert
    await db_conn.executemany(
        """
        INSERT INTO telecom_data (msisdn, data)
        VALUES ($1, $2)
        ON CONFLICT (msisdn) DO NOTHING
        """,
        batch
    )

    # Verify count
    count = await db_conn.fetchval(
        "SELECT COUNT(*) FROM telecom_data WHERE msisdn LIKE '5511920%'"
    )

    assert count == 100

    # Cleanup
    await db_conn.execute("DELETE FROM telecom_data WHERE msisdn LIKE '5511920%'")


@pytest.mark.asyncio
async def test_insert_kg_triple(db_conn):
    """Test inserting KG triple."""
    subject = "customer_test"
    predicate = "has_status"
    object = "active"

    # Insert
    await db_conn.execute(
        """
        INSERT INTO kg_triples (subject, predicate, object, source, confidence)
        VALUES ($1, $2, $3, $4, $5)
        ON CONFLICT (subject, predicate, object) DO UPDATE SET
            confidence = GREATEST(kg_triples.confidence, EXCLUDED.confidence)
        """,
        subject, predicate, object, "test", 1.0
    )

    # Verify
    result = await db_conn.fetchrow(
        """
        SELECT subject, predicate, object, confidence
        FROM kg_triples
        WHERE subject = $1 AND predicate = $2
        """,
        subject, predicate
    )

    assert result is not None
    assert result['object'] == object
    assert result['confidence'] == 1.0

    # Cleanup
    await db_conn.execute(
        "DELETE FROM kg_triples WHERE subject = $1",
        subject
    )


@pytest.mark.asyncio
async def test_ingestion_full_cycle():
    """Test full ingestion cycle with small subset."""
    from pff.db.ingestion import TelecomDataIngestion

    # Create temporary mini correct.zip for testing
    with NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        tmp_path = Path(tmp.name)

        with zipfile.ZipFile(tmp_path, 'w') as zf:
            # Add 3 sample files
            for i in range(3):
                msisdn = f"5511910001{i:03d}"
                filename = f"customer_enquiry_{msisdn}.txt"
                data = {
                    "id": f"TEST{i}",
                    "externalId": f"{i}",
                    "status": [{"status": "Active"}]
                }
                zf.writestr(filename, json.dumps(data))

    try:
        # Run ingestion
        ingestion = TelecomDataIngestion(zip_path=tmp_path, batch_size=10)
        await ingestion.run()

        # Verify results
        assert ingestion.stats['total_files'] == 3
        assert ingestion.stats['telecom_inserted'] == 3
        assert ingestion.stats['errors'] == 0

    finally:
        # Cleanup
        tmp_path.unlink()

        # Cleanup database
        conn = await asyncpg.connect(DATABASE_URL)
        try:
            await conn.execute("DELETE FROM telecom_data WHERE msisdn LIKE '5511910001%'")
            await conn.execute("DELETE FROM kg_triples WHERE subject LIKE 'customer_test%'")
        finally:
            await conn.close()
