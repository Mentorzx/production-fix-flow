"""PostgreSQL cleanup commands with backup support and safe execution."""

import asyncio
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from pff import settings
from pff.config import POSTGRES_CONFIG_PATH
from pff.db.connection import get_connection_pool
from pff.utils import FileManager, logger


def _load_backup_config() -> dict[str, object]:
    fallback = {
        "dir": settings.OUTPUTS_DIR / "backups" / "postgres",
        "keep_last": 5,
    }
    try:
        cfg = FileManager.read(POSTGRES_CONFIG_PATH)
        if not isinstance(cfg, dict):
            return fallback
        backup_cfg = cfg.get("backup", {})
        if not isinstance(backup_cfg, dict):
            return fallback
        return {
            "dir": backup_cfg.get("dir", fallback["dir"]),
            "keep_last": backup_cfg.get("keep_last", fallback["keep_last"]),
        }
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"Using default backup config (reason: {exc})")
        return fallback


class PostgreSQLBackupCommand:
    """
    Command to backup PostgreSQL tables before cleanup.

    Pattern: Command Pattern + Template Method
    """

    def __init__(
        self,
        tables: list[str],
        backup_dir: Optional[Path] = None,
        keep_backups: Optional[int] = None
    ):
        """
        Initialize backup command.

        Args:
            tables: List of table names to backup
            backup_dir: Directory for backups (default: backups/)
            keep_backups: Number of recent backups to keep
        """
        cfg = _load_backup_config()
        default_dir = Path(cfg["dir"])
        resolved_dir = (
            Path(backup_dir)
            if backup_dir is not None
            else (default_dir if default_dir.is_absolute() else settings.ROOT_DIR / default_dir)
        )
        self.tables = tables
        self.backup_dir = resolved_dir
        self.keep_backups = int(keep_backups if keep_backups is not None else cfg.get("keep_last", 5))

    async def execute(self, dry_run: bool = False) -> Optional[Path]:
        """
        Execute backup command.

        Args:
            dry_run: If True, only simulate backup

        Returns:
            Path to backup file or None if dry-run
        """
        if dry_run:
            logger.info(" [DRY-RUN] Backup PostgreSQL seria criado")
            return None

        self.backup_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = self.backup_dir / f"ml_backup_{timestamp}.sql"

        logger.info(f" Criando backup PostgreSQL: {backup_file.name}")

        try:
            await self._create_backup(backup_file)
            self._cleanup_old_backups()
            return backup_file
        except Exception as e:
            logger.error(f"Failed to create PostgreSQL backup: {e}")
            raise

    async def _create_backup(self, backup_file: Path) -> None:
        """Create PostgreSQL backup using pg_dump."""
        # Get DB connection info from settings
        db_url = settings.DATABASE_URL
        if not db_url:
            raise ValueError("DATABASE_URL nao configurado")

        # Parse connection string
        # Format: postgresql://user:pass@host:port/dbname
        import re
        match = re.match(r"postgresql://([^:]+):([^@]+)@([^:]+):(\d+)/(.+)", db_url)
        if not match:
            raise ValueError(f"DATABASE_URL invalido: {db_url}")

        user, password, host, port, dbname = match.groups()

        # Build pg_dump command safely
        cmd = [
            "pg_dump",
            "-h",
            host,
            "-p",
            str(port),
            "-U",
            user,
            "-d",
            dbname,
            "--no-owner",
            "--no-acl",
            "--clean",
            "--if-exists",
            "-f",
            str(backup_file),
        ]

        for table in self.tables:
            if not table.replace("_", "").isalnum():
                raise ValueError(f"Nome de tabela invalido para backup: {table}")
            cmd.extend(["-t", table])

        env = os.environ.copy()
        env["PGPASSWORD"] = password

        logger.debug(
            f"Executando pg_dump (tabelas={len(self.tables)})",
            extra={"tables": self.tables, "output": str(backup_file)},
        )

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )

        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            error_msg = (stderr or b"").decode() or "Unknown error"
            raise RuntimeError(f"pg_dump failed: {error_msg}")

        size_mb = backup_file.stat().st_size / 1024 / 1024
        logger.success(f" Backup criado: {size_mb:.2f} MB")

    def _cleanup_old_backups(self) -> None:
        """Keep only the N most recent backups."""
        backups = sorted(
            self.backup_dir.glob("ml_backup_*.sql"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )

        # Keep only the most recent backups
        for old_backup in backups[self.keep_backups:]:
            old_backup.unlink()
            logger.debug(f" Backup antigo removido: {old_backup.name}")

        if len(backups) > self.keep_backups:
            logger.info(f"Mantidos os ultimos {self.keep_backups} backups")


class PostgreSQLCleanupCommand:
    """
    Command to clean ML data from PostgreSQL tables.

    Pattern: Command Pattern
    """

    ML_TABLES = [
        "kg_splits",
        "kg_mappings",
        "kg_rules",
        "kg_embeddings",
        "ml_models",
        "training_metrics",
        "execution_logs",
        "pipeline_checkpoints"
    ]

    def __init__(
        self,
        tables: Optional[list[str]] = None,
        create_backup: bool = True
    ):
        """
        Initialize cleanup command.

        Args:
            tables: Specific tables to clean (default: all ML tables)
            create_backup: Create backup before deletion
        """
        self.tables = tables or self.ML_TABLES
        self.create_backup = create_backup
        self.pool = None

    async def _ensure_pool(self):
        """Lazy initialization of connection pool."""
        if self.pool is None:
            self.pool = await get_connection_pool()

    async def get_statistics(self) -> dict:
        """
        Get statistics about tables to be cleaned.

        Returns:
            Dictionary with table statistics
        """
        await self._ensure_pool()

        stats = {}
        total_rows = 0
        total_size = 0

        async with self.pool.acquire() as conn:
            for table in self.tables:
                # Check if table exists
                exists = await conn.fetchval(
                    """
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables
                        WHERE table_schema = 'public'
                        AND table_name = $1
                    )
                    """,
                    table
                )

                if not exists:
                    continue

                # Get row count
                count = await conn.fetchval(f"SELECT COUNT(*) FROM {table}")

                # Get table size (using regclass to convert table name to OID)
                size_bytes = await conn.fetchval(
                    """
                    SELECT pg_total_relation_size($1::regclass)
                    """,
                    table
                )

                stats[table] = {
                    'rows': count,
                    'size_bytes': size_bytes,
                    'size_mb': size_bytes / 1024 / 1024 if size_bytes else 0
                }

                total_rows += count
                total_size += size_bytes if size_bytes else 0

        stats['_total'] = {
            'rows': total_rows,
            'size_bytes': total_size,
            'size_mb': total_size / 1024 / 1024
        }

        return stats

    async def execute(self, dry_run: bool = False) -> dict:
        """
        Execute cleanup command.

        Args:
            dry_run: If True, only simulate cleanup

        Returns:
            Statistics about cleaned data
        """
        # Get statistics before cleaning
        stats_before = await self.get_statistics()

        if stats_before['_total']['rows'] == 0:
            logger.info(" Nenhum dado ML encontrado no PostgreSQL")
            return stats_before

        # Create backup if requested
        backup_file = None
        if self.create_backup and not dry_run:
            backup_cmd = PostgreSQLBackupCommand(
                tables=self.tables,
                keep_backups=5
            )
            backup_file = await backup_cmd.execute(dry_run=False)

        if dry_run:
            logger.info(" [DRY-RUN] As seguintes tabelas seriam limpas:")
            for table, info in stats_before.items():
                if table != '_total' and info['rows'] > 0:
                    logger.info(f"  • {table}: {info['rows']:,} linhas ({info['size_mb']:.2f} MB)")
            logger.info(f"  TOTAL: {stats_before['_total']['rows']:,} linhas ({stats_before['_total']['size_mb']:.2f} MB)")
            return stats_before

        # Execute cleanup
        await self._ensure_pool()

        deleted_counts = {}

        async with self.pool.acquire() as conn:
            async with conn.transaction():
                for table in self.tables:
                    if table not in stats_before or stats_before[table]['rows'] == 0:
                        continue

                    result = await conn.execute(f"DELETE FROM {table}")
                    deleted = int(result.split()[-1]) if result else 0
                    deleted_counts[table] = deleted

                    if deleted > 0:
                        logger.info(f" {table}: {deleted:,} linhas deletadas")

        total_deleted = sum(deleted_counts.values())

        if backup_file:
            logger.success(f" {total_deleted:,} linhas deletadas (backup: {backup_file.name})")
        else:
            logger.success(f" {total_deleted:,} linhas deletadas")

        return {
            'before': stats_before,
            'deleted': deleted_counts,
            'backup_file': str(backup_file) if backup_file else None
        }

    async def print_confirmation_prompt(self) -> str:
        """
        Print visual confirmation prompt with table statistics.

        Returns:
            Formatted confirmation message
        """
        stats = await self.get_statistics()

        if stats['_total']['rows'] == 0:
            return " Nenhum dado ML no PostgreSQL"

        lines = [
            "",
            "╔═══════════════════════════════════════════════════════════════╗",
            "║        CONFIRMAÇÃO DE LIMPEZA - PostgreSQL                   ║",
            "╚═══════════════════════════════════════════════════════════════╝",
            "",
            " TABELAS POSTGRESQL:",
            "┌────────────────────────┬──────────┬──────────┐",
            "│ Tabela                 │ Linhas   │ Tamanho  │",
            "├────────────────────────┼──────────┼──────────┤",
        ]

        for table, info in sorted(stats.items()):
            if table == '_total':
                continue

            if info['rows'] > 0:
                lines.append(
                    f"│ {table:<22} │ {info['rows']:>8,} │ {info['size_mb']:>6.1f} MB │"
                )

        lines.extend([
            "└────────────────────────┴──────────┴──────────┘",
            f"Total PostgreSQL: {stats['_total']['size_mb']:.1f} MB, {stats['_total']['rows']:,} linhas",
            "",
            "═══════════════════════════════════════════════════════════════",
            f"TOTAL A SER LIBERADO: {stats['_total']['size_mb']:.1f} MB (PostgreSQL)",
            "═══════════════════════════════════════════════════════════════",
            "",
            "  ATENCAO: Esta operacao e IRREVERSIVEL!",
            "    Um backup sera criado em: backups/ml_backup_YYYYMMDD_HHMMSS.sql",
            "",
        ])

        return "\n".join(lines)


# Compatibility with existing cleanup system
import asyncio


def create_postgresql_cleanup_command(tables: Optional[list[str]] = None) -> PostgreSQLCleanupCommand:
    """Factory function for creating PostgreSQL cleanup commands."""
    return PostgreSQLCleanupCommand(tables=tables, create_backup=True)
