"""create_pipeline_checkpoints_table

Revision ID: e9a759e2fe2e
Revises: 0ca8ec697d0c
Create Date: 2025-10-30 20:09:12.164227

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e9a759e2fe2e'
down_revision: Union[str, Sequence[str], None] = '0ca8ec697d0c'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'pipeline_checkpoints',
        sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column('pipeline_name', sa.String(length=100), nullable=False),
        sa.Column('step_name', sa.String(length=100), nullable=False),
        sa.Column('status', sa.String(length=20), nullable=False),
        sa.Column('progress', sa.Float(), server_default='0.0', nullable=False),
        sa.Column('metadata', sa.dialects.postgresql.JSONB(), nullable=True),
        sa.Column('started_at', sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column('completed_at', sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('pipeline_name', 'step_name', name='uq_pipeline_checkpoints_pipeline_step')
    )

    op.create_index(
        'idx_pipeline_checkpoints_lookup',
        'pipeline_checkpoints',
        ['pipeline_name']
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index('idx_pipeline_checkpoints_lookup', table_name='pipeline_checkpoints')
    op.drop_table('pipeline_checkpoints')
