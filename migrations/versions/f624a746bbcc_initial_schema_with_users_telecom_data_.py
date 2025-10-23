"""initial schema with users telecom_data kg_embeddings and execution_logs

Revision ID: f624a746bbcc
Revises: 
Create Date: 2025-10-19 07:03:04.655456

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f624a746bbcc'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # ═══════════════════════════════════════════════════════════════════
    # 1. Users Table (Authentication & Authorization)
    # ═══════════════════════════════════════════════════════════════════
    op.create_table(
        'users',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('username', sa.String(100), nullable=False, unique=True),
        sa.Column('email', sa.String(255), nullable=False, unique=True),
        sa.Column('full_name', sa.String(255), nullable=True),
        sa.Column('hashed_password', sa.String(255), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('is_superuser', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text('CURRENT_TIMESTAMP')),
    )

    # Create index on username for fast lookups
    op.create_index('idx_users_username', 'users', ['username'])
    op.create_index('idx_users_email', 'users', ['email'])

    # ═══════════════════════════════════════════════════════════════════
    # 2. Telecom Data Table (JSONB for flexible schema)
    # ═══════════════════════════════════════════════════════════════════
    op.create_table(
        'telecom_data',
        sa.Column('id', sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column('msisdn', sa.String(20), nullable=False),  # Phone number
        sa.Column('data', sa.dialects.postgresql.JSONB(), nullable=False),  # Flexible JSON data
        sa.Column('source_file', sa.String(255), nullable=True),  # Original file reference
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text('CURRENT_TIMESTAMP')),
    )

    # Create GIN index on JSONB data for fast queries
    op.create_index(
        'idx_telecom_data_gin',
        'telecom_data',
        ['data'],
        postgresql_using='gin'
    )

    # B-tree index on msisdn for lookups
    op.create_index('idx_telecom_msisdn', 'telecom_data', ['msisdn'])

    # ═══════════════════════════════════════════════════════════════════
    # 3. Knowledge Graph Embeddings Table (pgvector HNSW)
    # ═══════════════════════════════════════════════════════════════════
    # First create the table structure without the vector column
    op.execute("""
        CREATE TABLE kg_embeddings (
            id BIGSERIAL PRIMARY KEY,
            entity VARCHAR(255) NOT NULL,
            entity_type VARCHAR(50) NOT NULL,
            embedding vector(128) NOT NULL,
            dimension INTEGER NOT NULL DEFAULT 128,
            model_version VARCHAR(50),
            created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # B-tree index on entity for lookups
    op.create_index('idx_kg_entity', 'kg_embeddings', ['entity'])
    op.create_index('idx_kg_entity_type', 'kg_embeddings', ['entity_type'])

    # HNSW index for vector similarity search
    # Note: This requires pgvector extension to be installed
    op.execute("""
        CREATE INDEX idx_kg_embeddings_hnsw
        ON kg_embeddings
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
    """)

    # ═══════════════════════════════════════════════════════════════════
    # 4. Execution Logs Table (System monitoring & debugging)
    # ═══════════════════════════════════════════════════════════════════
    op.create_table(
        'execution_logs',
        sa.Column('id', sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column('user_id', sa.Integer(), sa.ForeignKey('users.id'), nullable=True),
        sa.Column('operation', sa.String(100), nullable=False),  # e.g., 'kg_training', 'inference'
        sa.Column('status', sa.String(20), nullable=False),  # 'success', 'failed', 'running'
        sa.Column('duration_seconds', sa.Float(), nullable=True),
        sa.Column('metadata', sa.dialects.postgresql.JSONB(), nullable=True),  # Additional info
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text('CURRENT_TIMESTAMP')),
    )

    # B-tree indexes for filtering logs
    op.create_index('idx_logs_user_id', 'execution_logs', ['user_id'])
    op.create_index('idx_logs_operation', 'execution_logs', ['operation'])
    op.create_index('idx_logs_status', 'execution_logs', ['status'])
    op.create_index('idx_logs_created_at', 'execution_logs', ['created_at'])

    # ═══════════════════════════════════════════════════════════════════
    # 5. KG Triples Table (Knowledge Graph storage)
    # ═══════════════════════════════════════════════════════════════════
    op.create_table(
        'kg_triples',
        sa.Column('id', sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column('subject', sa.String(255), nullable=False),
        sa.Column('predicate', sa.String(255), nullable=False),
        sa.Column('object', sa.String(255), nullable=False),
        sa.Column('confidence', sa.Float(), nullable=True),  # Optional confidence score
        sa.Column('source', sa.String(100), nullable=True),  # e.g., 'manual', 'inferred'
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text('CURRENT_TIMESTAMP')),
    )

    # Composite index for efficient triple queries
    op.create_index('idx_kg_triples_spo', 'kg_triples', ['subject', 'predicate', 'object'])
    op.create_index('idx_kg_triples_subject', 'kg_triples', ['subject'])
    op.create_index('idx_kg_triples_predicate', 'kg_triples', ['predicate'])
    op.create_index('idx_kg_triples_object', 'kg_triples', ['object'])


def downgrade() -> None:
    """Downgrade schema."""
    # Drop tables in reverse order (respecting foreign keys)
    op.drop_table('kg_triples')
    op.drop_table('execution_logs')
    op.drop_table('kg_embeddings')
    op.drop_table('telecom_data')
    op.drop_table('users')
