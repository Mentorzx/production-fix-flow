"""create_kg_tables_splits_mappings_rules

Revision ID: 0ba16afe7bdb
Revises: a6cdd74efd31
Create Date: 2025-10-30 10:55:21.783723

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '0ba16afe7bdb'
down_revision: Union[str, Sequence[str], None] = 'a6cdd74efd31'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Create kg_splits table
    op.create_table(
        'kg_splits',
        sa.Column('id', sa.BigInteger(), nullable=False, autoincrement=True),
        sa.Column('split_name', sa.String(length=20), nullable=False),
        sa.Column('split_type', sa.String(length=20), nullable=False),
        sa.Column('subject', sa.String(length=255), nullable=False),
        sa.Column('predicate', sa.String(length=255), nullable=False),
        sa.Column('object', sa.String(length=255), nullable=False),
        sa.Column('sample_id', sa.String(length=100), nullable=True),
        sa.Column('source', sa.String(length=100), nullable=True, server_default='correct.zip'),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('split_name', 'split_type', 'subject', 'predicate', 'object', name='uq_kg_splits_triple')
    )
    op.create_index('idx_kg_splits_lookup', 'kg_splits', ['split_name', 'split_type'])
    op.create_index('idx_kg_splits_sample', 'kg_splits', ['sample_id'])

    # Create kg_mappings table
    op.create_table(
        'kg_mappings',
        sa.Column('id', sa.BigInteger(), nullable=False, autoincrement=True),
        sa.Column('mapping_type', sa.String(length=20), nullable=False),
        sa.Column('key', sa.String(length=255), nullable=False),
        sa.Column('value', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('mapping_type', 'key', name='uq_kg_mappings_type_key')
    )
    op.create_index('idx_kg_mappings_lookup', 'kg_mappings', ['mapping_type'])

    # Create kg_rules table
    op.create_table(
        'kg_rules',
        sa.Column('id', sa.BigInteger(), nullable=False, autoincrement=True),
        sa.Column('rule_text', sa.Text(), nullable=False),
        sa.Column('confidence', sa.Float(), nullable=True),
        sa.Column('source', sa.String(length=50), nullable=True, server_default='anyburl'),
        sa.Column('iteration', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_kg_rules_source', 'kg_rules', ['source'])
    op.create_index('idx_kg_rules_iteration', 'kg_rules', ['iteration'])
    op.execute("CREATE INDEX idx_kg_rules_text_hash ON kg_rules USING hash (md5(rule_text))")


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index('idx_kg_rules_text_hash', table_name='kg_rules')
    op.drop_index('idx_kg_rules_iteration', table_name='kg_rules')
    op.drop_index('idx_kg_rules_source', table_name='kg_rules')
    op.drop_table('kg_rules')

    op.drop_index('idx_kg_mappings_lookup', table_name='kg_mappings')
    op.drop_table('kg_mappings')

    op.drop_index('idx_kg_splits_sample', table_name='kg_splits')
    op.drop_index('idx_kg_splits_lookup', table_name='kg_splits')
    op.drop_table('kg_splits')
