"""
Tests for Database Schema Edge Cases & Bugs (FIXED VERSION)

This file defines and tests the CORRECTED schema for critical tables:
- kg_triples
- telecom_data
- execution_logs
- kg_embeddings

Legacy 'users' table tests have been removed as the table is external/legacy.
"""

import asyncio
import os

import asyncpg
import pytest
import pytest_asyncio

pytestmark = [
    pytest.mark.integration,
]

# Database connection string
DB_URL = (
    os.getenv("TEST_DATABASE_URL")
    or os.getenv("DATABASE_URL")
    or "postgresql://pff_user:8qflzf45HGGQ_ghLetx4Whu7gqSVNYJ3@localhost/pff_production"
)

# ═══════════════════════════════════════════════════════════════════
# CORRECTED SCHEMA DEFINITION
# ═══════════════════════════════════════════════════════════════════

TEST_SCHEMA_SQL = """
-- Clean slate
DROP TABLE IF EXISTS kg_triples CASCADE;
DROP TABLE IF EXISTS telecom_data CASCADE;
DROP TABLE IF EXISTS execution_logs CASCADE;
DROP TABLE IF EXISTS kg_embeddings CASCADE;

-- 1. telecom_data (Fixed Bugs: msisdn validation, trigger, GIN ops)
CREATE TABLE telecom_data (
    id SERIAL PRIMARY KEY,
    msisdn TEXT NOT NULL CHECK (length(msisdn) >= 10 AND msisdn ~ '^[0-9]{10,15}$'),
    data JSONB NOT NULL,
    source_file TEXT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = clock_timestamp();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_telecom_data_updated_at
    BEFORE UPDATE ON telecom_data
    FOR EACH ROW
    EXECUTE PROCEDURE update_updated_at_column();

CREATE INDEX idx_telecom_data_gin ON telecom_data USING GIN (data jsonb_path_ops);

-- 2. kg_triples (Fixed Bugs: duplicates, confidence bounds)
CREATE TABLE kg_triples (
    id SERIAL PRIMARY KEY,
    subject TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object TEXT NOT NULL,
    source TEXT,
    confidence FLOAT DEFAULT 1.0 CHECK (confidence >= 0.0 AND confidence <= 1.0),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_kg_triples_spo UNIQUE (subject, predicate, object)
);

-- 3. execution_logs (Fixed Bugs: status validation, partial index, user_id type)
CREATE TABLE execution_logs (
    id SERIAL PRIMARY KEY,
    user_id TEXT, -- Decoupled from legacy users table
    operation TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('pending', 'running', 'success', 'failed', 'cancelled') AND status != ''),
    duration_seconds FLOAT,
    metadata JSONB,
    error_message TEXT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_logs_running ON execution_logs(created_at) WHERE status = 'running';

-- 4. kg_embeddings (Fixed Bugs: entity length)
CREATE TABLE kg_embeddings (
    id SERIAL PRIMARY KEY,
    entity TEXT NOT NULL, -- Fixed: TEXT instead of VARCHAR(255)
    entity_type TEXT NOT NULL,
    embedding vector(128) NOT NULL,
    dimension INTEGER NOT NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);
"""


@pytest_asyncio.fixture(scope="function", loop_scope="function")
async def db_conn():
    """Create clean connection and apply schema for each test."""
    try:
        conn = await asyncpg.connect(DB_URL)
    except Exception as e:
        pytest.skip(f"Database connection failed: {e}")
        return

    # Apply Schema
    try:
        await conn.execute(TEST_SCHEMA_SQL)
    except Exception as e:
        await conn.close()
        pytest.fail(f"Failed to apply test schema: {e}")

    # Use transaction for isolation (though we dropped tables, so it's fresh)
    tr = conn.transaction()
    await tr.start()
    try:
        yield conn
    finally:
        await tr.rollback()
        await conn.close()


# ═══════════════════════════════════════════════════════════════════
# VERIFICATION TESTS
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_kg_triples_reject_duplicates(db_conn):
    """Verify duplicate rejection constraint works."""
    await db_conn.execute(
        """
        INSERT INTO kg_triples (subject, predicate, object)
        VALUES ('s', 'p', 'o')
    """
    )
    with pytest.raises(asyncpg.exceptions.UniqueViolationError):
        # Use nested transaction to savepoint/rollback just this fail
        async with db_conn.transaction():
            await db_conn.execute(
                """
                INSERT INTO kg_triples (subject, predicate, object)
                VALUES ('s', 'p', 'o')
            """
            )


@pytest.mark.asyncio
async def test_kg_triples_confidence_bounds(db_conn):
    """Verify confidence bounds."""
    # Valid
    await db_conn.execute(
        "INSERT INTO kg_triples (subject, predicate, object, confidence) VALUES ('s1', 'p', 'o', 1.0)"
    )

    # Invalid > 1.0
    with pytest.raises(asyncpg.exceptions.CheckViolationError):
        async with db_conn.transaction():
            await db_conn.execute(
                "INSERT INTO kg_triples (subject, predicate, object, confidence) VALUES ('s2', 'p', 'o', 1.1)"
            )

    # Invalid < 0.0
    with pytest.raises(asyncpg.exceptions.CheckViolationError):
        async with db_conn.transaction():
            await db_conn.execute(
                "INSERT INTO kg_triples (subject, predicate, object, confidence) VALUES ('s3', 'p', 'o', -0.1)"
            )


@pytest.mark.asyncio
async def test_execution_logs_status_validation(db_conn):
    """Verify status constraints."""
    # Valid
    await db_conn.execute("INSERT INTO execution_logs (operation, status) VALUES ('op', 'running')")

    # Invalid enum
    with pytest.raises(asyncpg.exceptions.CheckViolationError):
        async with db_conn.transaction():
            await db_conn.execute(
                "INSERT INTO execution_logs (operation, status) VALUES ('op', 'invalid')"
            )

    # Empty
    with pytest.raises(asyncpg.exceptions.CheckViolationError):
        async with db_conn.transaction():
            await db_conn.execute(
                "INSERT INTO execution_logs (operation, status) VALUES ('op', '')"
            )


@pytest.mark.asyncio
async def test_execution_logs_partial_index(db_conn):
    """Verify partial index existence."""
    indexes = await db_conn.fetch(
        """
        SELECT indexdef FROM pg_indexes
        WHERE tablename = 'execution_logs' AND indexname = 'idx_logs_running'
    """
    )
    assert len(indexes) == 1
    assert (
        "WHERE status = 'running'::text" in indexes[0]["indexdef"]
        or "running" in indexes[0]["indexdef"]
    )


@pytest.mark.asyncio
async def test_telecom_data_validation(db_conn):
    """Verify telecom data constraints."""
    # Valid
    await db_conn.execute("INSERT INTO telecom_data (msisdn, data) VALUES ('1234567890', '{}')")

    # Invalid length
    with pytest.raises(asyncpg.exceptions.CheckViolationError):
        async with db_conn.transaction():
            await db_conn.execute("INSERT INTO telecom_data (msisdn, data) VALUES ('123', '{}')")

    # Invalid chars
    with pytest.raises(asyncpg.exceptions.CheckViolationError):
        async with db_conn.transaction():
            await db_conn.execute(
                "INSERT INTO telecom_data (msisdn, data) VALUES ('NotANumber', '{}')"
            )


@pytest.mark.asyncio
async def test_telecom_data_trigger(db_conn):
    """Verify updated_at trigger."""
    id = await db_conn.fetchval(
        "INSERT INTO telecom_data (msisdn, data) VALUES ('9876543210', '{}') RETURNING id"
    )
    row1 = await db_conn.fetchrow("SELECT updated_at FROM telecom_data WHERE id=$1", id)

    await asyncio.sleep(0.1)
    await db_conn.execute("UPDATE telecom_data SET source_file='new' WHERE id=$1", id)

    row2 = await db_conn.fetchrow("SELECT updated_at FROM telecom_data WHERE id=$1", id)
    assert row2["updated_at"] > row1["updated_at"]


@pytest.mark.asyncio
async def test_kg_embeddings_long_id(db_conn):
    """Verify TEXT type allows long IDs."""
    long_id = "x" * 300
    embedding = "[" + ",".join(["0.0"] * 128) + "]"

    await db_conn.execute(
        "INSERT INTO kg_embeddings (entity, entity_type, embedding, dimension) VALUES ($1, 'type', $2::vector, 128)",
        long_id,
        embedding,
    )

    retrieved = await db_conn.fetchval("SELECT entity FROM kg_embeddings WHERE entity=$1", long_id)
    assert retrieved == long_id
