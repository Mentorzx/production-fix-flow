"""add unique constraint on telecom_data msisdn

Revision ID: c9e34517143e
Revises: f540435978bc
Create Date: 2025-10-19 16:15:11.156943

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c9e34517143e'
down_revision: Union[str, Sequence[str], None] = 'f540435978bc'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add UNIQUE constraint on telecom_data.msisdn for ON CONFLICT support."""
    # Drop existing non-unique index
    op.drop_index('idx_telecom_msisdn', table_name='telecom_data')

    # Create unique constraint (automatically creates unique index)
    op.create_unique_constraint(
        'uq_telecom_data_msisdn',
        'telecom_data',
        ['msisdn']
    )


def downgrade() -> None:
    """Revert to non-unique index."""
    # Drop unique constraint
    op.drop_constraint('uq_telecom_data_msisdn', 'telecom_data', type_='unique')

    # Recreate regular B-tree index
    op.create_index('idx_telecom_msisdn', 'telecom_data', ['msisdn'])
