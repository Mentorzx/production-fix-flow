"""add uuid_id default gen_random_uuid

Revision ID: f540435978bc
Revises: 473699ca0a14
Create Date: 2025-10-19 11:11:41.935595

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f540435978bc'
down_revision: Union[str, Sequence[str], None] = '473699ca0a14'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add DEFAULT gen_random_uuid() to users.uuid_id."""
    # Add server default for uuid_id
    op.alter_column(
        'users',
        'uuid_id',
        server_default=sa.text('gen_random_uuid()'),
        existing_type=sa.dialects.postgresql.UUID(),
        existing_nullable=False
    )


def downgrade() -> None:
    """Remove DEFAULT from users.uuid_id."""
    op.alter_column(
        'users',
        'uuid_id',
        server_default=None,
        existing_type=sa.dialects.postgresql.UUID(),
        existing_nullable=False
    )
