"""change users id to uuid for security

Revision ID: 38410b07bd88
Revises: c9e34517143e
Create Date: 2025-10-19 16:51:27.467161

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '38410b07bd88'
down_revision: Union[str, Sequence[str], None] = 'c9e34517143e'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """
    Change users.id from INTEGER to UUID for enhanced security.

    This migration:
    1. Drops FK constraint from execution_logs
    2. Maps execution_logs.user_id from INTEGER to UUID (using users.uuid_id)
    3. Drops users.id primary key and column
    4. Renames uuid_id to id
    5. Sets id as new primary key
    6. Recreates FK constraint with UUID

    WARNING: This is a breaking change. Data will be preserved but
    external references using integer IDs will break.
    """
    # Step 1: Drop the FK constraint first (dependency on users_pkey)
    op.drop_constraint('execution_logs_user_id_fkey', 'execution_logs', type_='foreignkey')

    # Step 2: Add temp UUID column to execution_logs
    op.execute("""
        ALTER TABLE execution_logs
        ADD COLUMN user_uuid_temp UUID
    """)

    # Step 3: Copy UUIDs from users table based on integer ID
    # Map: execution_logs.user_id (int) -> users.id (int) -> users.uuid_id (UUID)
    op.execute("""
        UPDATE execution_logs el
        SET user_uuid_temp = u.uuid_id
        FROM users u
        WHERE el.user_id = u.id
    """)

    # Step 4: Drop old integer user_id column from execution_logs
    op.drop_column('execution_logs', 'user_id')

    # Step 5: Rename temp column to user_id
    op.alter_column('execution_logs', 'user_uuid_temp', new_column_name='user_id')

    # Step 6: Drop the old integer primary key from users
    op.drop_constraint('users_pkey', 'users', type_='primary')

    # Step 7: Drop the old id column from users
    op.drop_column('users', 'id')

    # Step 8: Rename uuid_id to id
    op.alter_column('users', 'uuid_id', new_column_name='id')

    # Step 9: Set id as primary key
    op.create_primary_key('users_pkey', 'users', ['id'])

    # Step 10: Recreate FK constraint
    op.create_foreign_key(
        'execution_logs_user_id_fkey',
        'execution_logs', 'users',
        ['user_id'], ['id'],
        ondelete='CASCADE'
    )


def downgrade() -> None:
    """
    Revert users.id from UUID back to INTEGER.

    WARNING: This will lose data. Use with caution.
    """
    # Step 1: Drop FK constraint
    op.drop_constraint('execution_logs_user_id_fkey', 'execution_logs', type_='foreignkey')

    # Step 2: Rename id back to uuid_id
    op.drop_constraint('users_pkey', 'users', type_='primary')
    op.alter_column('users', 'id', new_column_name='uuid_id')

    # Step 3: Add back integer id column (nullable first to allow adding to existing data)
    op.add_column('users', sa.Column('id', sa.Integer(), autoincrement=True, nullable=True))

    # Step 3.5: Populate id column with sequential values for existing rows
    op.execute("""
        WITH numbered AS (
            SELECT uuid_id, ROW_NUMBER() OVER (ORDER BY created_at) as rn
            FROM users
        )
        UPDATE users
        SET id = numbered.rn
        FROM numbered
        WHERE users.uuid_id = numbered.uuid_id
    """)

    # Step 3.6: Now make id NOT NULL
    op.alter_column('users', 'id', nullable=False)

    # Step 4: Set as primary key
    op.create_primary_key('users_pkey', 'users', ['id'])

    # Step 5: Restore execution_logs.user_id to integer (will lose FK relationships!)
    op.execute('ALTER TABLE execution_logs ALTER COLUMN user_id TYPE INTEGER USING NULL')

    # Step 6: Recreate FK
    op.create_foreign_key(
        'execution_logs_user_id_fkey',
        'execution_logs', 'users',
        ['user_id'], ['id']
    )
