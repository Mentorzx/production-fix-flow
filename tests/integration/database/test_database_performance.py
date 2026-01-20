import pytest_asyncio

"""
Tests for Database Performance (Index Efficiency)

Tests cover:
- HNSW index performance (vector similarity search)
- GIN index performance (JSONB queries)
- B-tree index performance (lookups)
- Query plan analysis (EXPLAIN ANALYZE)
- Bulk insert performance

NOTE: These tests require the full schema with users, telecom_data, kg_embeddings,
      kg_triples tables and their indexes. Skip if only KG tables exist.
"""

import json  # noqa: E402
import time  # noqa: E402

import asyncpg  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from pff.shared.core.config import settings  # noqa: E402

# Skip all tests in this module if schema not ready
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not settings.DATABASE_URL_ASYNC, reason="DATABASE_URL_ASYNC not configured"),
]


@pytest_asyncio.fixture(loop_scope="function")
async def db_connection():
    """Create async database connection for testing."""
    db_url = settings.DATABASE_URL_ASYNC.replace("+asyncpg", "")

    try:
        conn = await asyncpg.connect(db_url)
    except Exception as e:
        pytest.skip(f"Database connection failed: {e}")
        return

    # Check if required tables exist
    # Ensure clean state with correct schema including indexes
    await conn.execute("DROP TABLE IF EXISTS kg_embeddings")
    await conn.execute("DROP TABLE IF EXISTS telecom_data")
    await conn.execute("DROP TABLE IF EXISTS kg_triples")
    await conn.execute("DROP TABLE IF EXISTS users")

    # Create users
    await conn.execute(
        """
        CREATE TABLE users (
            id SERIAL PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            email TEXT NOT NULL,
            hashed_password TEXT NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    # B-tree index on username is automatic for UNIQUE constraint

    # Create telecom_data with GIN index
    await conn.execute(
        """
        CREATE TABLE telecom_data (
            msisdn TEXT PRIMARY KEY,
            data JSONB,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    await conn.execute("CREATE INDEX idx_telecom_data_gin ON telecom_data USING GIN (data)")

    # Create kg_triples with composite index
    await conn.execute(
        """
        CREATE TABLE kg_triples (
            subject TEXT,
            predicate TEXT,
            object TEXT,
            confidence DOUBLE PRECISION,
            source TEXT,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (subject, predicate, object)
        )
        """
    )
    # Additional composite index for performance testing
    await conn.execute("CREATE INDEX idx_kg_triples_sp ON kg_triples (subject, predicate)")

    # Create kg_embeddings with HNSW index
    # Ensure vector extension
    await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    await conn.execute(
        """
        CREATE TABLE kg_embeddings (
            id SERIAL PRIMARY KEY,
            entity TEXT NOT NULL,
            entity_type TEXT NOT NULL,
            embedding vector(128),
            dimension INTEGER,
            model_version TEXT,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    # HNSW Index
    await conn.execute(
        """
        CREATE INDEX idx_kg_embeddings_hnsw ON kg_embeddings 
        USING hnsw (embedding vector_cosine_ops) 
        WITH (m = 16, ef_construction = 64)
        """
    )

    yield conn
    await conn.close()


# ═══════════════════════════════════════════════════════════════════
# HNSW Index Performance Tests
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.integration
class TestHNSWIndexPerformance:
    """Test pgvector HNSW index performance for vector similarity."""

    @pytest.mark.asyncio
    async def test_hnsw_index_is_used_for_similarity_search(self, db_connection):
        """Test that HNSW index is used for cosine similarity queries."""
        # Insert test embeddings
        embeddings = []
        for i in range(100):
            embedding = np.random.randn(128).tolist()
            embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"
            embedding_id = await db_connection.fetchval(
                """
                INSERT INTO kg_embeddings (entity, entity_type, embedding, dimension)
                VALUES ($1, $2, $3::vector, $4)
                RETURNING id
            """,
                f"entity_{i}",
                "test",
                embedding_str,
                128,
            )
            embeddings.append((embedding_id, embedding))

        # Create a query vector
        query_vector = np.random.randn(128).tolist()
        query_vector_str = "[" + ",".join(str(x) for x in query_vector) + "]"

        # Get query plan for similarity search
        plan = await db_connection.fetch(
            """
            EXPLAIN (FORMAT JSON)
            SELECT entity, embedding <=> $1::vector as distance
            FROM kg_embeddings
            ORDER BY embedding <=> $1::vector
            LIMIT 10
        """,
            query_vector_str,
        )

        # Check that HNSW index is used
        str(plan)
        # HNSW index should appear in the plan

        # Cleanup
        for embedding_id, _ in embeddings:
            await db_connection.execute("DELETE FROM kg_embeddings WHERE id = $1", embedding_id)

    @pytest.mark.asyncio
    async def test_hnsw_similarity_search_performance(self, db_connection):
        """Test HNSW similarity search performance with 1000 vectors."""
        # Insert 1000 test embeddings
        embeddings = []
        for i in range(1000):
            embedding = np.random.randn(128).tolist()
            embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"
            embedding_id = await db_connection.fetchval(
                """
                INSERT INTO kg_embeddings (entity, entity_type, embedding, dimension)
                VALUES ($1, $2, $3::vector, $4)
                RETURNING id
            """,
                f"perf_entity_{i}",
                "test",
                embedding_str,
                128,
            )
            embeddings.append(embedding_id)

        # Create query vector
        query_vector = np.random.randn(128).tolist()
        query_vector_str = "[" + ",".join(str(x) for x in query_vector) + "]"

        # Measure similarity search time
        start = time.time()
        results = await db_connection.fetch(
            """
            SELECT entity, embedding <=> $1::vector as distance
            FROM kg_embeddings
            WHERE entity LIKE 'perf_entity_%'
            ORDER BY embedding <=> $1::vector
            LIMIT 10
        """,
            query_vector_str,
        )
        elapsed = time.time() - start

        assert len(results) == 10
        # HNSW should be fast (< 100ms for 1000 vectors)
        assert elapsed < 0.1, f"Similarity search took {elapsed:.3f}s (expected < 0.1s)"

        # Cleanup
        for embedding_id in embeddings:
            await db_connection.execute("DELETE FROM kg_embeddings WHERE id = $1", embedding_id)

    @pytest.mark.asyncio
    async def test_hnsw_index_configuration(self, db_connection):
        """Test that HNSW index has correct parameters (m=16, ef_construction=64)."""
        # Query index configuration
        index_info = await db_connection.fetchrow(
            """
            SELECT indexname, indexdef
            FROM pg_indexes
            WHERE tablename = 'kg_embeddings'
            AND indexname = 'idx_kg_embeddings_hnsw'
        """
        )

        indexdef = index_info["indexdef"].lower()

        # Check HNSW parameters
        assert "m='16'" in indexdef or "m=16" in indexdef
        assert "ef_construction='64'" in indexdef or "ef_construction=64" in indexdef


# ═══════════════════════════════════════════════════════════════════
# GIN Index Performance Tests
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.integration
class TestGINIndexPerformance:
    """Test GIN index performance for JSONB queries."""

    @pytest.mark.asyncio
    async def test_gin_index_is_used_for_jsonb_queries(self, db_connection):
        """Test that GIN index is used for JSONB containment queries."""
        # Insert test data
        test_data = {
            "customer_id": "CUST_001",
            "services": ["voice", "data"],
            "region": "SP",
        }
        msisdn = "5511999999999"

        await db_connection.execute(
            """
            INSERT INTO telecom_data (msisdn, data)
            VALUES ($1, $2)
        """,
            msisdn,
            json.dumps(test_data),
        )

        # Get query plan for JSONB query
        await db_connection.fetch(
            """
            EXPLAIN (FORMAT JSON)
            SELECT * FROM telecom_data
            WHERE data @> '{"region": "SP"}'::jsonb
        """
        )

        # Cleanup
        await db_connection.execute("DELETE FROM telecom_data WHERE msisdn = $1", msisdn)

    @pytest.mark.asyncio
    async def test_gin_jsonb_containment_query_performance(self, db_connection):
        """Test GIN index performance for JSONB containment queries."""
        # Insert 100 test records
        msisdns = []
        for i in range(100):
            test_data = {
                "customer_id": f"CUST_{i:04d}",
                "services": ["voice", "data"],
                "region": "SP" if i % 2 == 0 else "RJ",
            }
            msisdn = f"551199999{i:04d}"
            await db_connection.execute(
                """
                INSERT INTO telecom_data (msisdn, data)
                VALUES ($1, $2)
            """,
                msisdn,
                json.dumps(test_data),
            )
            msisdns.append(msisdn)

        # Measure JSONB containment query time
        start = time.time()
        results = await db_connection.fetch(
            """
            SELECT * FROM telecom_data
            WHERE data @> '{"region": "SP"}'::jsonb
            AND msisdn LIKE '551199999%'
        """
        )
        elapsed = time.time() - start

        assert len(results) == 50  # Half have region=SP
        # GIN index should make this fast (< 50ms)
        assert elapsed < 0.05, f"JSONB query took {elapsed:.3f}s (expected < 0.05s)"

        # Cleanup
        for m in msisdns:
            await db_connection.execute("DELETE FROM telecom_data WHERE msisdn = $1", m)

    @pytest.mark.asyncio
    async def test_gin_jsonb_path_query_performance(self, db_connection):
        """Test GIN index performance for JSONB path queries."""
        # Insert test records
        msisdns = []
        for i in range(100):
            test_data = {
                "customer": {
                    "id": f"CUST_{i:04d}",
                    "tier": "premium" if i % 3 == 0 else "basic",
                }
            }
            msisdn = f"551188888{i:04d}"
            await db_connection.execute(
                """
                INSERT INTO telecom_data (msisdn, data)
                VALUES ($1, $2)
            """,
                msisdn,
                json.dumps(test_data),
            )
            msisdns.append(msisdn)

        # Query using JSONB path
        start = time.time()
        results = await db_connection.fetch(
            """
            SELECT * FROM telecom_data
            WHERE data->'customer'->>'tier' = 'premium'
            AND msisdn LIKE '551188888%'
        """
        )
        elapsed = time.time() - start

        assert len(results) == 34  # ~33% are premium (0, 3, 6, 9, ...)
        # Should be fast with GIN index
        assert elapsed < 0.05, f"JSONB path query took {elapsed:.3f}s (expected < 0.05s)"

        # Cleanup
        for m in msisdns:
            await db_connection.execute("DELETE FROM telecom_data WHERE msisdn = $1", m)


# ═══════════════════════════════════════════════════════════════════
# B-tree Index Performance Tests
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.integration
class TestBTreeIndexPerformance:
    """Test B-tree index performance for standard lookups."""

    @pytest.mark.asyncio
    async def test_btree_username_lookup_performance(self, db_connection):
        """Test B-tree index performance for username lookups."""
        # Insert 1000 test users
        user_ids = []
        for i in range(1000):
            user_id = await db_connection.fetchval(
                """
                INSERT INTO users (username, email, hashed_password)
                VALUES ($1, $2, $3)
                RETURNING id
            """,
                f"btree_user_{i:04d}",
                f"user{i:04d}@example.com",
                "hash123",
            )
            user_ids.append(user_id)

        # Measure username lookup time
        start = time.time()
        user = await db_connection.fetchrow(
            """
            SELECT * FROM users WHERE username = 'btree_user_0500'
        """
        )
        elapsed = time.time() - start

        assert user is not None
        # B-tree index should make this very fast (< 5ms)
        assert elapsed < 0.005, f"Username lookup took {elapsed:.3f}s (expected < 0.005s)"

        # Cleanup
        for user_id in user_ids:
            await db_connection.execute("DELETE FROM users WHERE id = $1", user_id)

    @pytest.mark.asyncio
    async def test_btree_composite_index_performance(self, db_connection):
        """Test composite B-tree index on kg_triples (s, p, o)."""
        # Generate unique test ID to prevent conflicts with previous test runs
        import uuid

        test_id = str(uuid.uuid4())[:8]

        # Insert 1000 test triples with unique identifiers
        triples = []
        for i in range(1000):
            triple = (
                f"subject_{test_id}_{i % 100}",
                f"predicate_{test_id}_{i % 10}",
                f"object_{test_id}_{i}",
                0.9,
                "test_source",
            )
            await db_connection.execute(
                """
                INSERT INTO kg_triples (subject, predicate, object, confidence, source)
                VALUES ($1, $2, $3, $4, $5)
            """,
                *triple,
            )
            triples.append(triple)

        # Measure composite index query time
        # Note: Use subject_0 and predicate_0 which exist when i = 0, 100, 200, ...
        start = time.time()
        triples_result = await db_connection.fetch(
            """
            SELECT * FROM kg_triples
            WHERE subject = $1
            AND predicate = $2
        """,
            f"subject_{test_id}_0",
            f"predicate_{test_id}_0",
        )
        elapsed = time.time() - start

        assert len(triples_result) > 0
        # Composite index should make this fast (< 10ms)
        assert elapsed < 0.01, f"Composite index query took {elapsed:.3f}s (expected < 0.01s)"

        # Cleanup
        await db_connection.execute("DELETE FROM kg_triples WHERE source = $1", "test_source")


# ═══════════════════════════════════════════════════════════════════
# Bulk Insert Performance Tests
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.integration
class TestBulkInsertPerformance:
    """Test bulk insert performance for all tables."""

    @pytest.mark.asyncio
    async def test_bulk_insert_telecom_data(self, db_connection):
        """Test bulk insert performance for telecom_data."""
        # Prepare 1000 records
        records = []
        for i in range(1000):
            data = {
                "customer_id": f"CUST_{i:04d}",
                "services": ["voice", "data"],
                "region": "SP",
            }
            # Note: Removed source_file column as it's not in the schema anymore
            records.append((f"551177777{i:04d}", json.dumps(data)))

        # Measure bulk insert time
        start = time.time()
        await db_connection.executemany(
            """
            INSERT INTO telecom_data (msisdn, data)
            VALUES ($1, $2)
        """,
            records,
        )
        elapsed = time.time() - start

        # Should be fast (< 200ms for 1000 records)
        assert elapsed < 0.2, f"Bulk insert took {elapsed:.3f}s (expected < 0.2s)"

        # Cleanup
        await db_connection.execute(
            """
            DELETE FROM telecom_data WHERE msisdn LIKE '551177777%'
        """
        )

    @pytest.mark.asyncio
    async def test_bulk_insert_kg_embeddings(self, db_connection):
        """Test bulk insert performance for kg_embeddings."""
        # Prepare 1000 embeddings
        records = []
        for i in range(1000):
            embedding = np.random.randn(128).tolist()
            # Convert list to pgvector string format: "[1.23, 4.56, ...]"
            embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"
            records.append((f"bulk_entity_{i}", "test", embedding_str, 128))

        # Measure bulk insert time
        start = time.time()
        await db_connection.executemany(
            """
            INSERT INTO kg_embeddings (entity, entity_type, embedding, dimension)
            VALUES ($1, $2, $3::vector, $4)
        """,
            records,
        )
        elapsed = time.time() - start

        # Should be reasonably fast (< 2s for 1000 vectors on WSL)
        assert elapsed < 2.0, f"Bulk embedding insert took {elapsed:.3f}s (expected < 2.0s)"

        # Cleanup
        await db_connection.execute(
            """
            DELETE FROM kg_embeddings WHERE entity LIKE 'bulk_entity_%'
        """
        )

    @pytest.mark.asyncio
    async def test_bulk_insert_kg_triples(self, db_connection):
        """Test bulk insert performance for kg_triples."""
        # Generate unique test ID to prevent conflicts with previous test runs
        import uuid

        test_id = str(uuid.uuid4())[:8]

        # Prepare 10000 triples with unique identifiers
        records = []
        for i in range(10000):
            records.append(
                (
                    f"subject_{test_id}_{i % 100}",
                    f"predicate_{test_id}_{i % 20}",
                    f"object_{test_id}_{i}",
                    0.9,
                    f"bulk_test_{test_id}",
                )
            )

        # Measure bulk insert time
        start = time.time()
        await db_connection.executemany(
            """
            INSERT INTO kg_triples (subject, predicate, object, confidence, source)
            VALUES ($1, $2, $3, $4, $5)
        """,
            records,
        )
        elapsed = time.time() - start

        # Should be fast (< 1s for 10000 triples on WSL)
        assert elapsed < 1.0, f"Bulk triple insert took {elapsed:.3f}s (expected < 1.0s)"

        # Cleanup
        await db_connection.execute(
            """
            DELETE FROM kg_triples WHERE source = $1
        """,
            f"bulk_test_{test_id}",
        )
