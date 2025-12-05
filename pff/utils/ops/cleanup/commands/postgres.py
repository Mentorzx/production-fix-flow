from __future__ import annotations

import asyncio
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from pff import settings
from pff.config import POSTGRES_CONFIG_PATH
from pff.db.connection import get_connection_pool
from pff.utils.core.file_manager import FileManager
from pff.utils.core.logger import logger
from pff.utils.ops.cleanup.config import CLEANUP_CONFIG, _coerce_positive_int

from .base import CleanupCommand


def _read_yaml_dict(path: Path) -> dict[str, Any]:
    """Read a YAML config into a dict using FileManager."""
    try:
        raw_cfg = FileManager.read(path)
        return raw_cfg if isinstance(raw_cfg, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"Using fallback config for {path}: {exc}")
        return {}


def _load_backup_config() -> dict[str, object]:
    """Load backup configuration merging cleanup and postgres fallbacks."""
    fallback = {
        "dir": settings.OUTPUTS_DIR / "backups" / "postgres",
        "keep_last": 5,
    }
    cleanup_cfg = CLEANUP_CONFIG if isinstance(CLEANUP_CONFIG, dict) else {}
    backup_cfg = cleanup_cfg.get("backup") if isinstance(cleanup_cfg, dict) else {}
    if isinstance(backup_cfg, dict) and backup_cfg:
        return {
            "dir": backup_cfg.get("dir", fallback["dir"]),
            "keep_last": _coerce_positive_int(
                backup_cfg.get("keep_last"), fallback["keep_last"]
            ),
        }

    postgres_cfg = _read_yaml_dict(POSTGRES_CONFIG_PATH)
    backup_cfg = postgres_cfg.get("backup") if postgres_cfg else {}
    if isinstance(backup_cfg, dict) and backup_cfg:
        return {
            "dir": backup_cfg.get("dir", fallback["dir"]),
            "keep_last": _coerce_positive_int(
                backup_cfg.get("keep_last"), fallback["keep_last"]
            ),
        }

    return fallback


class PostgreSQLBackupCommand(CleanupCommand):
    """
    Command to backup PostgreSQL tables before cleanup.

    Pattern: Command Pattern + Template Method
    """

    def __init__(
        self,
        tables: list[str],
        backup_dir: Optional[Path] = None,
        keep_backups: Optional[int] = None,
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
        default_keep_last = _coerce_positive_int(cfg.get("keep_last"), 5)
        self.tables = tables
        self.backup_dir = resolved_dir
        self.keep_backups = _coerce_positive_int(
            keep_backups if keep_backups is not None else default_keep_last,
            default_keep_last,
        )

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

        await self._create_backup(backup_file)
        self._cleanup_old_backups()
        return backup_file

    async def _create_backup(self, backup_file: Path) -> None:
        """Create PostgreSQL backup using pg_dump."""
        db_url = settings.DATABASE_URL
        if not db_url:
            raise ValueError("DATABASE_URL nao configurado")

        import re

        match = re.match(r"postgresql://([^:]+):([^@]+)@([^:]+):(\d+)/(.+)", db_url)
        if not match:
            raise ValueError(f"DATABASE_URL invalido: {db_url}")

        user, password, host, port, dbname = match.groups()

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
            reverse=True,
        )

        for old_backup in backups[self.keep_backups:]:
            old_backup.unlink()
            logger.debug(f"Old backup removed: {old_backup.name}")

        if len(backups) > self.keep_backups:
            logger.info(f"Mantidos os ultimos {self.keep_backups} backups")


class PostgreSQLCleanupCommand(CleanupCommand):
    """Clean ML tables from PostgreSQL with optional backup."""

    ML_TABLES = [
        "kg_splits",
        "kg_mappings",
        "kg_rules",
        "kg_embeddings",
        "ml_models",
        "training_metrics",
        "execution_logs",
        "pipeline_checkpoints",
    ]

    def __init__(self, tables: Optional[list[str]] = None, create_backup: bool = True):
        self.tables = tables or self.ML_TABLES
        self.create_backup = create_backup
        self.pool = None

    async def _ensure_pool(self):
        if self.pool is None:
            self.pool = await get_connection_pool()

    async def get_statistics(self) -> dict:
        await self._ensure_pool()
        stats = {}
        total_rows = 0
        total_size_mb = 0.0

        for table in self.tables:
            async with self.pool.acquire() as conn:
                row_count = await conn.fetchval(f"SELECT COUNT(*) FROM {table}")
                size_mb = await conn.fetchval(
                    f"SELECT pg_total_relation_size('{table}') / 1024 / 1024"
                )
                stats[table] = {"rows": row_count, "size_mb": float(size_mb or 0)}
                total_rows += row_count or 0
                total_size_mb += float(size_mb or 0)

        stats["_total"] = {"rows": total_rows, "size_mb": total_size_mb}
        return stats

    async def print_confirmation_prompt(self) -> str:
        stats = await self.get_statistics()

        if stats["_total"]["rows"] == 0:
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
            if table == "_total":
                continue

            if info["rows"] > 0:
                lines.append(
                    f"│ {table:<22} │ {info['rows']:>8,} │ {info['size_mb']:>6.1f} MB │"
                )

        lines.extend(
            [
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
            ]
        )

        return "\n".join(lines)


__all__ = ["PostgreSQLBackupCommand", "PostgreSQLCleanupCommand"]
