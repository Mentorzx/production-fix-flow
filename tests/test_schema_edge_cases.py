"""
Tests for Database Schema Edge Cases & Bugs

Sprint 2: Schema Validation - Tests que DEVEM FALHAR para expor bugs

Bugs esperados a serem expostos:
1. kg_triples: Permite duplicatas (falta UNIQUE constraint)
2. execution_logs.status: Aceita valores inválidos (falta CHECK)
3. kg_triples.confidence: Aceita valores fora de [0,1] (falta CHECK)
4. telecom_data.msisdn: Aceita formato inválido (falta CHECK)
5. users.updated_at: Não atualiza automaticamente (falta TRIGGER)
6. JSONB index: Não usa jsonb_path_ops (ineficiente)
7. kg_embeddings.entity: VARCHAR(255) trunca IDs longos (deveria ser TEXT)
8. Faltam índices parciais para queries comuns

Status: BASELINE (bugs conhecidos, testes documentam comportamento incorreto)
"""

import asyncio
import pytest
import asyncpg
from datetime import datetime
import time


# Database connection string
DB_URL = "postgresql://pff_user:8qflzf45HGGQ_ghLetx4Whu7gqSVNYJ3@localhost/pff_production"


@pytest.fixture(scope="function")
async def db_conn():
    """Create clean connection for each test."""
    conn = await asyncpg.connect(DB_URL)
    # Use transaction for isolation
    tr = conn.transaction()
    await tr.start()
    try:
        yield conn
    finally:
        await tr.rollback()
        await conn.close()


# ═══════════════════════════════════════════════════════════════════
# BUG 1: kg_triples permite duplicatas
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_kg_triples_reject_duplicates(db_conn):
    """
    BUG BASELINE: Permite inserir triplas duplicadas
    ESPERADO: Deve falhar com UniqueViolationError
    FIX: ADD CONSTRAINT uq_kg_triples_spo UNIQUE (subject, predicate, object)
    """
    # Insert primeira tripla
    await db_conn.execute("""
        INSERT INTO kg_triples (subject, predicate, object, source)
        VALUES ('customer_123', 'has_plan', 'prepaid', 'manual')
    """)

    # Tentar inserir duplicata - DEVE FALHAR com UniqueViolationError (constraint funciona!)
    with pytest.raises(asyncpg.exceptions.UniqueViolationError):
        await db_conn.execute("""
            INSERT INTO kg_triples (subject, predicate, object, source)
            VALUES ('customer_123', 'has_plan', 'prepaid', 'inferred')
        """)

    # Test passes if UniqueViolationError was raised (constraint working!)


# ═══════════════════════════════════════════════════════════════════
# BUG 2: execution_logs.status aceita valores inválidos
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_execution_logs_status_validation(db_conn):
    """
    BUG BASELINE: Aceita status='lol_invalid'
    ESPERADO: Deve falhar com CheckViolationError
    FIX: ADD CONSTRAINT check_status CHECK (status IN ('pending', 'running', 'success', 'failed', 'cancelled'))
    """
    # Tentar inserir status inválido - DEVE FALHAR mas PASSA (BUG)
    with pytest.raises(asyncpg.exceptions.CheckViolationError):
        await db_conn.execute("""
            INSERT INTO execution_logs (operation, status)
            VALUES ('test_op', 'lol_invalid')
        """)


@pytest.mark.asyncio
async def test_execution_logs_status_not_empty(db_conn):
    """
    BUG BASELINE: Aceita status=''
    ESPERADO: Deve falhar com CheckViolationError
    FIX: ADD CHECK (status != '')
    """
    with pytest.raises(asyncpg.exceptions.CheckViolationError):
        await db_conn.execute("""
            INSERT INTO execution_logs (operation, status)
            VALUES ('test', '')
        """)


# ═══════════════════════════════════════════════════════════════════
# BUG 3: kg_triples.confidence aceita valores fora de [0, 1]
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_kg_triples_confidence_max_bound(db_conn):
    """
    BUG BASELINE: Aceita confidence=1.5
    ESPERADO: Deve falhar com CheckViolationError
    FIX: ADD CONSTRAINT check_confidence CHECK (confidence BETWEEN 0.0 AND 1.0)
    """
    with pytest.raises(asyncpg.exceptions.CheckViolationError):
        await db_conn.execute("""
            INSERT INTO kg_triples (subject, predicate, object, confidence)
            VALUES ('s1', 'p1', 'o1', 1.5)
        """)


@pytest.mark.asyncio
async def test_kg_triples_confidence_min_bound(db_conn):
    """
    BUG BASELINE: Aceita confidence=-0.5
    ESPERADO: Deve falhar com CheckViolationError
    """
    with pytest.raises(asyncpg.exceptions.CheckViolationError):
        await db_conn.execute("""
            INSERT INTO kg_triples (subject, predicate, object, confidence)
            VALUES ('s2', 'p2', 'o2', -0.5)
        """)


# ═══════════════════════════════════════════════════════════════════
# BUG 4: telecom_data.msisdn aceita formato inválido
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_telecom_data_msisdn_format_validation(db_conn):
    """
    BUG BASELINE: Aceita msisdn='lol not a phone'
    ESPERADO: Deve falhar com CheckViolationError
    FIX: ADD CHECK (msisdn ~ '^[0-9]{10,15}$')
    """
    with pytest.raises(asyncpg.exceptions.CheckViolationError):
        await db_conn.execute("""
            INSERT INTO telecom_data (msisdn, data)
            VALUES ('lol not a phone', '{}')
        """)


@pytest.mark.asyncio
async def test_telecom_data_msisdn_not_empty(db_conn):
    """
    BUG BASELINE: Aceita msisdn=''
    ESPERADO: Deve falhar com CheckViolationError
    FIX: ADD CHECK (msisdn != '' AND length(msisdn) >= 10)
    """
    with pytest.raises(asyncpg.exceptions.CheckViolationError):
        await db_conn.execute("""
            INSERT INTO telecom_data (msisdn, data)
            VALUES ('', '{}')
        """)


# ═══════════════════════════════════════════════════════════════════
# BUG 5: users.updated_at não atualiza automaticamente
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_users_updated_at_trigger(db_conn):
    """
    BUG BASELINE: updated_at permanece igual após UPDATE
    ESPERADO: updated_at deve ser > created_at após update
    FIX: CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users ...
    """
    # Insert user
    user_id = await db_conn.fetchval("""
        INSERT INTO users (username, email, hashed_password)
        VALUES ('triggertest', 'trigger@test.com', 'hash123')
        RETURNING id
    """)

    # Get initial timestamps
    row1 = await db_conn.fetchrow("""
        SELECT created_at, updated_at FROM users WHERE id = $1
    """, user_id)

    # Wait 100ms
    await asyncio.sleep(0.1)

    # Update user
    await db_conn.execute("""
        UPDATE users SET full_name = 'Updated Name' WHERE id = $1
    """, user_id)

    # Get updated timestamps
    row2 = await db_conn.fetchrow("""
        SELECT created_at, updated_at FROM users WHERE id = $1
    """, user_id)

    # ESPERADO: updated_at mudou
    # ATUAL (BUG): updated_at igual a created_at
    assert row2['updated_at'] > row1['updated_at'], \
        f"updated_at não mudou: {row1['updated_at']} == {row2['updated_at']}"


@pytest.mark.asyncio
async def test_telecom_data_updated_at_trigger(db_conn):
    """
    BUG BASELINE: updated_at não muda após UPDATE em telecom_data
    FIX: CREATE TRIGGER update_telecom_data_updated_at ...
    """
    # Insert
    record_id = await db_conn.fetchval("""
        INSERT INTO telecom_data (msisdn, data)
        VALUES ('5511910001706', '{"plan": "prepaid"}')
        RETURNING id
    """)

    row1 = await db_conn.fetchrow("""
        SELECT created_at, updated_at FROM telecom_data WHERE id = $1
    """, record_id)

    await asyncio.sleep(0.1)

    # Update
    await db_conn.execute("""
        UPDATE telecom_data SET data = '{"plan": "postpaid"}' WHERE id = $1
    """, record_id)

    row2 = await db_conn.fetchrow("""
        SELECT created_at, updated_at FROM telecom_data WHERE id = $1
    """, record_id)

    assert row2['updated_at'] > row1['updated_at'], "updated_at não mudou"


# ═══════════════════════════════════════════════════════════════════
# BUG 6: JSONB GIN index não usa jsonb_path_ops (ineficiente)
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_jsonb_gin_index_uses_path_ops(db_conn):
    """
    PERFORMANCE: GIN index deveria usar jsonb_path_ops para @> queries
    ATUAL: Usa GIN default (menos eficiente)
    FIX: CREATE INDEX CONCURRENTLY idx_telecom_data_gin ON telecom_data USING gin (data jsonb_path_ops)
    """
    # Check index definition
    index_def = await db_conn.fetchval("""
        SELECT indexdef FROM pg_indexes
        WHERE tablename = 'telecom_data' AND indexname = 'idx_telecom_data_gin'
    """)

    assert 'jsonb_path_ops' in index_def, \
        f"GIN index deveria usar jsonb_path_ops para melhor performance. Atual: {index_def}"


# ═══════════════════════════════════════════════════════════════════
# BUG 7: kg_embeddings.entity VARCHAR(255) trunca IDs longos
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_kg_embeddings_entity_long_id(db_conn):
    """
    FIXED: entity agora é TEXT e aceita IDs longos
    ESPERADO: Deve aceitar IDs longos (customer_enquiry_* pode ter >255)
    FIX: ALTER TABLE kg_embeddings ALTER COLUMN entity TYPE TEXT (já aplicado)
    """
    long_entity = "customer_enquiry_" + "x" * 250  # 266 chars

    # Insert (embedding como vector string '[0.1, 0.1, ...]')
    embedding_vector = '[' + ','.join(['0.1'] * 128) + ']'
    await db_conn.execute("""
        INSERT INTO kg_embeddings (entity, entity_type, embedding, dimension)
        VALUES ($1, 'customer', $2::vector, 128)
    """, long_entity, embedding_vector)

    # Retrieve
    retrieved = await db_conn.fetchval("""
        SELECT entity FROM kg_embeddings WHERE entity LIKE 'customer_enquiry_%'
    """)

    # ESPERADO: retrieved == long_entity (266 chars)
    # ATUAL (BUG): retrieved truncado para 255 chars
    assert len(retrieved) == len(long_entity), \
        f"Entity truncado! Original: {len(long_entity)}, Recuperado: {len(retrieved)}"
    assert retrieved == long_entity


# ═══════════════════════════════════════════════════════════════════
# BUG 8: Faltam índices parciais para queries comuns
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_partial_index_execution_logs_running(db_conn):
    """
    PERFORMANCE: Deveria ter índice parcial para status='running'
    FIX: CREATE INDEX idx_logs_running ON execution_logs(created_at) WHERE status = 'running'
    """
    indexes = await db_conn.fetch("""
        SELECT indexname, indexdef FROM pg_indexes
        WHERE tablename = 'execution_logs'
        AND indexdef LIKE '%WHERE%status%running%'
    """)

    assert len(indexes) > 0, "Falta índice parcial para execution_logs WHERE status='running'"


@pytest.mark.asyncio
async def test_partial_index_users_active(db_conn):
    """
    PERFORMANCE: Deveria ter índice parcial para is_active=true
    FIX: CREATE INDEX idx_users_active ON users(username) WHERE is_active = true
    """
    indexes = await db_conn.fetch("""
        SELECT indexname, indexdef FROM pg_indexes
        WHERE tablename = 'users'
        AND indexdef LIKE '%WHERE%is_active%'
    """)

    assert len(indexes) > 0, "Falta índice parcial para users WHERE is_active=true"


# ═══════════════════════════════════════════════════════════════════
# BUG 9: users.id deveria ser UUID (security best practice)
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_users_id_is_uuid(db_conn):
    """
    SECURITY BEST PRACTICE: users.id deveria ser UUID para prevenir enumeration attacks
    ATUAL: INTEGER (sequencial, vulnerável)
    FIX: ALTER TABLE users ALTER COLUMN id TYPE UUID USING gen_random_uuid()
    """
    # Check column type
    col_type = await db_conn.fetchval("""
        SELECT data_type FROM information_schema.columns
        WHERE table_name = 'users' AND column_name = 'id'
    """)

    assert col_type == 'uuid', \
        f"users.id deveria ser UUID para segurança. Atual: {col_type}"


# ═══════════════════════════════════════════════════════════════════
# BUG 10: Falta CASCADE em Foreign Keys
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_foreign_key_cascade_delete(db_conn):
    """
    BUG BASELINE: Deletar user não deleta execution_logs (orphaned records)
    ESPERADO: ON DELETE CASCADE deve limpar logs automaticamente
    FIX: ALTER TABLE execution_logs ADD CONSTRAINT fk_user CASCADE
    """
    # Insert user + log
    user_id = await db_conn.fetchval("""
        INSERT INTO users (username, email, hashed_password)
        VALUES ('cascadetest', 'cascade@test.com', 'hash')
        RETURNING id
    """)

    await db_conn.execute("""
        INSERT INTO execution_logs (user_id, operation, status)
        VALUES ($1, 'test', 'success')
    """, user_id)

    # Delete user - ESPERADO: logs são deletados automaticamente
    await db_conn.execute("DELETE FROM users WHERE id = $1", user_id)

    # Verify logs deleted
    log_count = await db_conn.fetchval("""
        SELECT COUNT(*) FROM execution_logs WHERE user_id = $1
    """, user_id)

    assert log_count == 0, f"Logs órfãos! Esperado: 0, Atual: {log_count}"


# ═══════════════════════════════════════════════════════════════════
# Test Summary: Run with --runxfail to see all failures
# ═══════════════════════════════════════════════════════════════════

"""
Para rodar e ver todos os bugs:

pytest tests/test_schema_edge_cases.py -v --runxfail --tb=short

BASELINE (antes da correção):
- test_kg_triples_reject_duplicates: FAIL (permite duplicatas)
- test_execution_logs_status_validation: FAIL (aceita valores inválidos)
- test_kg_triples_confidence_max_bound: FAIL (aceita >1.0)
- test_kg_triples_confidence_min_bound: FAIL (aceita <0.0)
- test_telecom_data_msisdn_format_validation: FAIL (aceita formato inválido)
- test_users_updated_at_trigger: FAIL (não atualiza timestamp)
- test_kg_embeddings_entity_long_id: FAIL (trunca para 255 chars)
- test_partial_index_execution_logs_running: FAIL (índice ausente)
- test_users_id_is_uuid: FAIL (usa INTEGER)
- test_foreign_key_cascade_delete: FAIL (não deleta em cascata)

Total: 10 bugs conhecidos documentados
"""
