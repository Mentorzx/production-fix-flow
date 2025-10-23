"""
Tests for Database Schema (PostgreSQL + pgvector + Alembic)

Tests cover:
- Schema validation (tables, columns, constraints)
- Index validation (HNSW, GIN, B-tree)
- CRUD operations on all tables
- Foreign key relationships
- Data type validation (JSONB, vector, timestamps)
- Unique constraints (username, email)
"""

import pytest
import asyncpg
from datetime import datetime
import json
import numpy as np

from pff.config import settings


@pytest.fixture
async def db_connection():
    """Create async database connection for testing."""
    # Parse the async URL to get connection params
    # postgresql+asyncpg://user:pass@host/db -> postgresql://user:pass@host/db
    db_url = settings.DATABASE_URL_ASYNC.replace("+asyncpg", "")

    conn = await asyncpg.connect(db_url)
    yield conn
    await conn.close()


# ═══════════════════════════════════════════════════════════════════
# Schema Validation Tests
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.integration
class TestDatabaseSchema:
    """Test database schema structure and constraints."""

    @pytest.mark.asyncio
    async def test_all_tables_exist(self, db_connection):
        """Test that all required tables exist."""
        tables = await db_connection.fetch("""
            SELECT tablename
            FROM pg_tables
            WHERE schemaname = 'public'
            AND tablename != 'alembic_version'
            ORDER BY tablename
        """)

        table_names = [t['tablename'] for t in tables]

        assert 'users' in table_names
        assert 'telecom_data' in table_names
        assert 'kg_embeddings' in table_names
        assert 'execution_logs' in table_names
        assert 'kg_triples' in table_names

    @pytest.mark.asyncio
    async def test_users_table_structure(self, db_connection):
        """Test users table has correct columns and types."""
        columns = await db_connection.fetch("""
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name = 'users'
            ORDER BY ordinal_position
        """)

        column_dict = {col['column_name']: col for col in columns}

        # Required columns
        assert 'id' in column_dict
        assert 'username' in column_dict
        assert 'email' in column_dict
        assert 'hashed_password' in column_dict
        assert 'is_active' in column_dict
        assert 'is_superuser' in column_dict
        assert 'created_at' in column_dict

        # Check NOT NULL constraints
        assert column_dict['username']['is_nullable'] == 'NO'
        assert column_dict['email']['is_nullable'] == 'NO'
        assert column_dict['hashed_password']['is_nullable'] == 'NO'

    @pytest.mark.asyncio
    async def test_telecom_data_table_structure(self, db_connection):
        """Test telecom_data table has JSONB column."""
        columns = await db_connection.fetch("""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = 'telecom_data'
        """)

        column_dict = {col['column_name']: col['data_type'] for col in columns}

        assert 'id' in column_dict
        assert 'msisdn' in column_dict
        assert 'data' in column_dict
        assert column_dict['data'] == 'jsonb'

    @pytest.mark.asyncio
    async def test_kg_embeddings_table_structure(self, db_connection):
        """Test kg_embeddings table has vector column."""
        # Query the table structure directly
        columns = await db_connection.fetch("""
            SELECT column_name, udt_name
            FROM information_schema.columns
            WHERE table_name = 'kg_embeddings'
        """)

        column_dict = {col['column_name']: col['udt_name'] for col in columns}

        assert 'id' in column_dict
        assert 'entity' in column_dict
        assert 'entity_type' in column_dict
        assert 'embedding' in column_dict
        assert column_dict['embedding'] == 'vector'  # pgvector type

    @pytest.mark.asyncio
    async def test_kg_triples_table_structure(self, db_connection):
        """Test kg_triples table for SPO storage."""
        columns = await db_connection.fetch("""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = 'kg_triples'
        """)

        column_names = [col['column_name'] for col in columns]

        assert 'subject' in column_names
        assert 'predicate' in column_names
        assert 'object' in column_names
        assert 'confidence' in column_names

    @pytest.mark.asyncio
    async def test_execution_logs_table_structure(self, db_connection):
        """Test execution_logs table structure."""
        columns = await db_connection.fetch("""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = 'execution_logs'
        """)

        column_dict = {col['column_name']: col['data_type'] for col in columns}

        assert 'operation' in column_dict
        assert 'status' in column_dict
        assert 'metadata' in column_dict
        assert column_dict['metadata'] == 'jsonb'


# ═══════════════════════════════════════════════════════════════════
# Index Validation Tests
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.integration
class TestDatabaseIndices:
    """Test that all required indices exist and are correct type."""

    @pytest.mark.asyncio
    async def test_users_indices_exist(self, db_connection):
        """Test users table has username and email indices."""
        indices = await db_connection.fetch("""
            SELECT indexname, indexdef
            FROM pg_indexes
            WHERE tablename = 'users'
            AND indexname LIKE 'idx_%'
        """)

        index_names = [idx['indexname'] for idx in indices]

        assert 'idx_users_username' in index_names
        assert 'idx_users_email' in index_names

    @pytest.mark.asyncio
    async def test_telecom_data_gin_index(self, db_connection):
        """Test telecom_data has GIN index on JSONB column."""
        indices = await db_connection.fetch("""
            SELECT indexname, indexdef
            FROM pg_indexes
            WHERE tablename = 'telecom_data'
            AND indexname = 'idx_telecom_data_gin'
        """)

        assert len(indices) == 1
        assert 'gin' in indices[0]['indexdef'].lower()

    @pytest.mark.asyncio
    async def test_kg_embeddings_hnsw_index(self, db_connection):
        """Test kg_embeddings has HNSW index on vector column."""
        indices = await db_connection.fetch("""
            SELECT indexname, indexdef
            FROM pg_indexes
            WHERE tablename = 'kg_embeddings'
            AND indexname = 'idx_kg_embeddings_hnsw'
        """)

        assert len(indices) == 1
        indexdef = indices[0]['indexdef'].lower()
        assert 'hnsw' in indexdef
        assert 'vector_cosine_ops' in indexdef

    @pytest.mark.asyncio
    async def test_kg_triples_composite_index(self, db_connection):
        """Test kg_triples has composite index on (s, p, o)."""
        indices = await db_connection.fetch("""
            SELECT indexname, indexdef
            FROM pg_indexes
            WHERE tablename = 'kg_triples'
            AND indexname = 'idx_kg_triples_spo'
        """)

        assert len(indices) == 1
        indexdef = indices[0]['indexdef']
        assert 'subject' in indexdef
        assert 'predicate' in indexdef
        assert 'object' in indexdef


# ═══════════════════════════════════════════════════════════════════
# CRUD Operation Tests
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.integration
class TestDatabaseOperations:
    """Test basic CRUD operations on all tables."""

    @pytest.mark.asyncio
    async def test_insert_user(self, db_connection):
        """Test inserting a user into users table."""
        # Insert test user
        user_id = await db_connection.fetchval("""
            INSERT INTO users (username, email, hashed_password, is_active, is_superuser)
            VALUES ($1, $2, $3, $4, $5)
            RETURNING id
        """, 'test_user', 'test@example.com', 'hashed_password_123', True, False)

        assert user_id is not None

        # Verify user was inserted
        user = await db_connection.fetchrow("""
            SELECT username, email, is_active FROM users WHERE id = $1
        """, user_id)

        assert user['username'] == 'test_user'
        assert user['email'] == 'test@example.com'
        assert user['is_active'] is True

        # Cleanup
        await db_connection.execute("DELETE FROM users WHERE id = $1", user_id)

    @pytest.mark.asyncio
    async def test_insert_telecom_data_with_jsonb(self, db_connection):
        """Test inserting telecom data with JSONB."""
        test_data = {
            "msisdn": "5511999999999",
            "customer_id": "CUST_001",
            "services": ["voice", "data"],
            "metadata": {"region": "SP", "plan": "premium"}
        }

        record_id = await db_connection.fetchval("""
            INSERT INTO telecom_data (msisdn, data, source_file)
            VALUES ($1, $2, $3)
            RETURNING id
        """, "5511999999999", json.dumps(test_data), "test_file.json")

        assert record_id is not None

        # Verify JSONB data
        record = await db_connection.fetchrow("""
            SELECT data FROM telecom_data WHERE id = $1
        """, record_id)

        # asyncpg returns JSONB as dict automatically
        data = record['data'] if isinstance(record['data'], dict) else json.loads(record['data'])
        assert data['msisdn'] == "5511999999999"
        assert 'voice' in data['services']

        # Cleanup
        await db_connection.execute("DELETE FROM telecom_data WHERE id = $1", record_id)

    @pytest.mark.asyncio
    async def test_insert_kg_embedding_with_vector(self, db_connection):
        """Test inserting embedding with vector type."""
        # Create a 128-dimensional vector
        embedding = np.random.randn(128).tolist()
        # Convert to string format for pgvector: [0.1, 0.2, ...]
        embedding_str = '[' + ','.join(str(x) for x in embedding) + ']'

        embedding_id = await db_connection.fetchval("""
            INSERT INTO kg_embeddings (entity, entity_type, embedding, dimension)
            VALUES ($1, $2, $3::vector, $4)
            RETURNING id
        """, 'user_123', 'user', embedding_str, 128)

        assert embedding_id is not None

        # Verify embedding
        record = await db_connection.fetchrow("""
            SELECT entity, entity_type, dimension FROM kg_embeddings WHERE id = $1
        """, embedding_id)

        assert record['entity'] == 'user_123'
        assert record['entity_type'] == 'user'
        assert record['dimension'] == 128

        # Cleanup
        await db_connection.execute("DELETE FROM kg_embeddings WHERE id = $1", embedding_id)

    @pytest.mark.asyncio
    async def test_insert_kg_triple(self, db_connection):
        """Test inserting knowledge graph triple."""
        triple_id = await db_connection.fetchval("""
            INSERT INTO kg_triples (subject, predicate, object, confidence, source)
            VALUES ($1, $2, $3, $4, $5)
            RETURNING id
        """, 'user_123', 'has_service', 'premium_plan', 0.95, 'inferred')

        assert triple_id is not None

        # Verify triple
        triple = await db_connection.fetchrow("""
            SELECT subject, predicate, object, confidence FROM kg_triples WHERE id = $1
        """, triple_id)

        assert triple['subject'] == 'user_123'
        assert triple['predicate'] == 'has_service'
        assert triple['object'] == 'premium_plan'
        assert triple['confidence'] == 0.95

        # Cleanup
        await db_connection.execute("DELETE FROM kg_triples WHERE id = $1", triple_id)

    @pytest.mark.asyncio
    async def test_insert_execution_log(self, db_connection):
        """Test inserting execution log."""
        log_metadata = {
            "batch_size": 1000,
            "processing_time": 45.2,
            "records_processed": 950
        }

        log_id = await db_connection.fetchval("""
            INSERT INTO execution_logs (operation, status, duration_seconds, metadata)
            VALUES ($1, $2, $3, $4)
            RETURNING id
        """, 'kg_training', 'success', 45.2, json.dumps(log_metadata))

        assert log_id is not None

        # Verify log
        log = await db_connection.fetchrow("""
            SELECT operation, status, metadata FROM execution_logs WHERE id = $1
        """, log_id)

        assert log['operation'] == 'kg_training'
        assert log['status'] == 'success'
        # asyncpg returns JSONB as dict automatically
        metadata = log['metadata'] if isinstance(log['metadata'], dict) else json.loads(log['metadata'])
        assert metadata['batch_size'] == 1000

        # Cleanup
        await db_connection.execute("DELETE FROM execution_logs WHERE id = $1", log_id)


# ═══════════════════════════════════════════════════════════════════
# Constraint Tests
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.integration
class TestDatabaseConstraints:
    """Test unique constraints and foreign keys."""

    @pytest.mark.asyncio
    async def test_users_unique_username(self, db_connection):
        """Test that username must be unique."""
        # Insert first user
        user_id = await db_connection.fetchval("""
            INSERT INTO users (username, email, hashed_password)
            VALUES ($1, $2, $3)
            RETURNING id
        """, 'unique_test', 'test1@example.com', 'hash123')

        # Try to insert duplicate username
        with pytest.raises(asyncpg.UniqueViolationError):
            await db_connection.fetchval("""
                INSERT INTO users (username, email, hashed_password)
                VALUES ($1, $2, $3)
                RETURNING id
            """, 'unique_test', 'test2@example.com', 'hash456')

        # Cleanup
        await db_connection.execute("DELETE FROM users WHERE id = $1", user_id)

    @pytest.mark.asyncio
    async def test_users_unique_email(self, db_connection):
        """Test that email must be unique."""
        # Insert first user
        user_id = await db_connection.fetchval("""
            INSERT INTO users (username, email, hashed_password)
            VALUES ($1, $2, $3)
            RETURNING id
        """, 'user1', 'unique@example.com', 'hash123')

        # Try to insert duplicate email
        with pytest.raises(asyncpg.UniqueViolationError):
            await db_connection.fetchval("""
                INSERT INTO users (username, email, hashed_password)
                VALUES ($1, $2, $3)
                RETURNING id
            """, 'user2', 'unique@example.com', 'hash456')

        # Cleanup
        await db_connection.execute("DELETE FROM users WHERE id = $1", user_id)

    @pytest.mark.asyncio
    async def test_execution_logs_foreign_key(self, db_connection):
        """Test foreign key relationship between execution_logs and users."""
        # Create test user
        user_id = await db_connection.fetchval("""
            INSERT INTO users (username, email, hashed_password)
            VALUES ($1, $2, $3)
            RETURNING id
        """, 'fk_test_user', 'fk@example.com', 'hash123')

        # Insert log with valid user_id
        log_id = await db_connection.fetchval("""
            INSERT INTO execution_logs (user_id, operation, status)
            VALUES ($1, $2, $3)
            RETURNING id
        """, user_id, 'test_operation', 'success')

        assert log_id is not None

        # Verify the log references the correct user
        log = await db_connection.fetchrow("""
            SELECT user_id FROM execution_logs WHERE id = $1
        """, log_id)

        assert log['user_id'] == user_id

        # Cleanup (must delete log first due to FK)
        await db_connection.execute("DELETE FROM execution_logs WHERE id = $1", log_id)
        await db_connection.execute("DELETE FROM users WHERE id = $1", user_id)
