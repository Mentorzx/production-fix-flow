"""fix updated_at triggers use clock_timestamp

Revision ID: a6cdd74efd31
Revises: 38410b07bd88
Create Date: 2025-10-22 09:59:45.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a6cdd74efd31'
down_revision: Union[str, Sequence[str], None] = '38410b07bd88'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """
    Fix updated_at trigger to use clock_timestamp() instead of CURRENT_TIMESTAMP

    CURRENT_TIMESTAMP returns the transaction start time, so all updates in the
    same transaction get the same timestamp.

    clock_timestamp() returns the actual current time, which is what we want for
    tracking when a row was last modified.
    """
    op.execute("""
        CREATE OR REPLACE FUNCTION update_updated_at_column()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = clock_timestamp();
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
    """)


def downgrade() -> None:
    """Revert to CURRENT_TIMESTAMP (original buggy version)."""
    op.execute("""
        CREATE OR REPLACE FUNCTION update_updated_at_column()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = CURRENT_TIMESTAMP;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
    """)
