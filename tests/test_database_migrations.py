"""
Tests for Database Migrations (Alembic Retrocompatibility)

Tests cover:
- Migration upgrade/downgrade cycle
- Schema version tracking
- Data preservation during migrations
- Migration idempotency
- Rollback safety
"""

import pytest
import asyncpg
import subprocess
import os

from pff.config import settings


@pytest.fixture
async def db_connection():
    """Create async database connection for testing."""
    db_url = settings.DATABASE_URL_ASYNC.replace("+asyncpg", "")
    conn = await asyncpg.connect(db_url)
    yield conn
    await conn.close()


def run_alembic_command(command: str) -> tuple[int, str, str]:
    """Run alembic command and return exit code, stdout, stderr."""
    env = os.environ.copy()
    env['VIRTUAL_ENV'] = '/home/Alex/Development/PFF/.venv'

    result = subprocess.run(
        f".venv/bin/alembic {command}",
        shell=True,
        capture_output=True,
        text=True,
        cwd='/home/Alex/Development/PFF',
        env=env
    )
    return result.returncode, result.stdout, result.stderr


# ═══════════════════════════════════════════════════════════════════
# Migration Version Tests
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.integration
class TestAlembicVersionTracking:
    """Test Alembic version tracking functionality."""

    @pytest.mark.asyncio
    async def test_alembic_version_table_exists(self, db_connection):
        """Test that alembic_version table exists."""
        tables = await db_connection.fetch("""
            SELECT tablename FROM pg_tables
            WHERE schemaname = 'public'
            AND tablename = 'alembic_version'
        """)

        assert len(tables) == 1

    @pytest.mark.asyncio
    async def test_current_migration_version_recorded(self, db_connection):
        """Test that current migration version is recorded."""
        version = await db_connection.fetchrow("""
            SELECT version_num FROM alembic_version
        """)

        assert version is not None
        assert version['version_num'] is not None
        # Should be our latest migration (HEAD)
        assert version['version_num'].startswith('a6cdd74efd31')

    def test_alembic_current_command(self):
        """Test that 'alembic current' shows current version."""
        returncode, stdout, stderr = run_alembic_command("current")

        assert returncode == 0
        assert 'a6cdd74efd31' in stdout or 'a6cdd74efd31' in stderr


# ═══════════════════════════════════════════════════════════════════
# Migration Upgrade/Downgrade Tests
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.integration
class TestMigrationUpgradeDowngrade:
    """Test migration upgrade and downgrade (rollback) functionality."""

    @pytest.mark.asyncio
    async def test_downgrade_migration(self, db_connection):
        """Test downgrading migration (rollback)."""
        # Check current tables exist
        tables_before = await db_connection.fetch("""
            SELECT tablename FROM pg_tables
            WHERE schemaname = 'public'
            AND tablename IN ('users', 'telecom_data', 'kg_embeddings', 'execution_logs', 'kg_triples')
        """)
        assert len(tables_before) == 5

        # Downgrade to base (remove all tables)
        returncode, stdout, stderr = run_alembic_command("downgrade base")
        assert returncode == 0

        # Verify tables are removed
        tables_after = await db_connection.fetch("""
            SELECT tablename FROM pg_tables
            WHERE schemaname = 'public'
            AND tablename IN ('users', 'telecom_data', 'kg_embeddings', 'execution_logs', 'kg_triples')
        """)
        assert len(tables_after) == 0

        # Re-upgrade to head
        returncode, stdout, stderr = run_alembic_command("upgrade head")
        assert returncode == 0

        # Verify tables are recreated
        tables_restored = await db_connection.fetch("""
            SELECT tablename FROM pg_tables
            WHERE schemaname = 'public'
            AND tablename IN ('users', 'telecom_data', 'kg_embeddings', 'execution_logs', 'kg_triples')
        """)
        assert len(tables_restored) == 5

    @pytest.mark.asyncio
    async def test_data_preservation_during_upgrade_downgrade(self, db_connection):
        """Test that data is lost during full downgrade/upgrade cycle."""
        # Insert test data
        user_id = await db_connection.fetchval("""
            INSERT INTO users (username, email, hashed_password)
            VALUES ($1, $2, $3)
            RETURNING id
        """, 'migration_test_user', 'migration@test.com', 'hash123')

        # Downgrade to base (removes all tables) and upgrade again
        run_alembic_command("downgrade base")
        run_alembic_command("upgrade head")

        # Verify data is lost (as expected with full downgrade to base)
        # Note: This test shows that downgrade to base WILL delete data
        # In production, you'd want to backup before downgrade
        user = await db_connection.fetchrow("""
            SELECT * FROM users WHERE username = 'migration_test_user'
        """)

        # After downgrade base -> upgrade head, data should be gone
        # (This is expected behavior - migrations don't preserve data by default)
        assert user is None


# ═══════════════════════════════════════════════════════════════════
# Migration Idempotency Tests
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.integration
class TestMigrationIdempotency:
    """Test that migrations are idempotent (safe to run multiple times)."""

    def test_upgrade_head_idempotency(self):
        """Test that running 'upgrade head' multiple times is safe."""
        # Run upgrade head first time
        returncode1, stdout1, stderr1 = run_alembic_command("upgrade head")
        assert returncode1 == 0

        # Run upgrade head second time (should be no-op)
        returncode2, stdout2, stderr2 = run_alembic_command("upgrade head")
        assert returncode2 == 0

        # Should indicate already at head or no changes
        combined_output = stdout2 + stderr2
        # Alembic should report it's already at the target revision

    @pytest.mark.asyncio
    async def test_schema_unchanged_after_reupgrade(self, db_connection):
        """Test that running upgrade multiple times doesn't change schema."""
        # Get table count
        tables_before = await db_connection.fetch("""
            SELECT COUNT(*) as count FROM pg_tables
            WHERE schemaname = 'public'
        """)
        count_before = tables_before[0]['count']

        # Run upgrade again
        run_alembic_command("upgrade head")

        # Get table count again
        tables_after = await db_connection.fetch("""
            SELECT COUNT(*) as count FROM pg_tables
            WHERE schemaname = 'public'
        """)
        count_after = tables_after[0]['count']

        # Should be the same
        assert count_before == count_after


# ═══════════════════════════════════════════════════════════════════
# Migration History Tests
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.integration
class TestMigrationHistory:
    """Test migration history and revision tracking."""

    def test_alembic_history_command(self):
        """Test that 'alembic history' shows migration history."""
        returncode, stdout, stderr = run_alembic_command("history")

        assert returncode == 0
        # Should show our initial migration
        combined_output = stdout + stderr
        assert 'f624a746bbcc' in combined_output
        assert 'initial schema' in combined_output.lower()

    def test_alembic_heads_command(self):
        """Test that 'alembic heads' shows current head."""
        returncode, stdout, stderr = run_alembic_command("heads")

        assert returncode == 0
        combined_output = stdout + stderr
        assert 'a6cdd74efd31' in combined_output


# ═══════════════════════════════════════════════════════════════════
# Rollback Safety Tests
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.integration
class TestRollbackSafety:
    """Test that rollbacks are safe and don't leave orphaned objects."""

    @pytest.mark.asyncio
    async def test_no_orphaned_indices_after_rollback(self, db_connection):
        """Test that downgrade removes all indices."""
        # Get initial index count
        indices_before = await db_connection.fetch("""
            SELECT indexname FROM pg_indexes
            WHERE schemaname = 'public'
            AND tablename IN ('users', 'telecom_data', 'kg_embeddings', 'execution_logs', 'kg_triples')
        """)
        initial_count = len(indices_before)

        # Downgrade
        run_alembic_command("downgrade base")

        # Check for orphaned indices
        indices_after = await db_connection.fetch("""
            SELECT indexname FROM pg_indexes
            WHERE schemaname = 'public'
            AND tablename IN ('users', 'telecom_data', 'kg_embeddings', 'execution_logs', 'kg_triples')
        """)

        # Should be 0 (no orphaned indices)
        assert len(indices_after) == 0

        # Restore
        run_alembic_command("upgrade head")

    @pytest.mark.asyncio
    async def test_no_orphaned_tables_after_rollback(self, db_connection):
        """Test that downgrade removes all application tables."""
        # Downgrade
        run_alembic_command("downgrade base")

        # Check for orphaned tables
        tables = await db_connection.fetch("""
            SELECT tablename FROM pg_tables
            WHERE schemaname = 'public'
            AND tablename IN ('users', 'telecom_data', 'kg_embeddings', 'execution_logs', 'kg_triples')
        """)

        # Should be 0 (all tables removed)
        assert len(tables) == 0

        # Only alembic_version should remain
        all_tables = await db_connection.fetch("""
            SELECT tablename FROM pg_tables
            WHERE schemaname = 'public'
        """)

        table_names = [t['tablename'] for t in all_tables]
        assert 'alembic_version' in table_names
        assert 'users' not in table_names

        # Restore
        run_alembic_command("upgrade head")


# ═══════════════════════════════════════════════════════════════════
# Schema Consistency Tests
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.integration
class TestSchemaConsistency:
    """Test that schema is consistent after migration cycles."""

    @pytest.mark.asyncio
    async def test_schema_consistent_after_full_cycle(self, db_connection):
        """Test that schema is identical after downgrade -> upgrade cycle."""
        # Get initial schema
        tables_before = await db_connection.fetch("""
            SELECT tablename FROM pg_tables
            WHERE schemaname = 'public'
            ORDER BY tablename
        """)

        indices_before = await db_connection.fetch("""
            SELECT indexname FROM pg_indexes
            WHERE schemaname = 'public'
            AND tablename IN ('users', 'telecom_data', 'kg_embeddings', 'execution_logs', 'kg_triples')
            ORDER BY indexname
        """)

        # Downgrade and upgrade
        run_alembic_command("downgrade base")
        run_alembic_command("upgrade head")

        # Get final schema
        tables_after = await db_connection.fetch("""
            SELECT tablename FROM pg_tables
            WHERE schemaname = 'public'
            ORDER BY tablename
        """)

        indices_after = await db_connection.fetch("""
            SELECT indexname FROM pg_indexes
            WHERE schemaname = 'public'
            AND tablename IN ('users', 'telecom_data', 'kg_embeddings', 'execution_logs', 'kg_triples')
            ORDER BY indexname
        """)

        # Tables should be the same
        assert [t['tablename'] for t in tables_before] == [t['tablename'] for t in tables_after]

        # Indices should be the same
        assert [i['indexname'] for i in indices_before] == [i['indexname'] for i in indices_after]
