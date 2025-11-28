"""Centralized PostgreSQL configuration utilities.

This module consolidates environment-driven settings (via ``pff.config``)
with YAML configuration defaults to expose a single source of truth for
database connection parameters, pooling behaviour, retry policy, and SSL
options.  Consumers such as ``pff.db.connection`` and ingestion pipelines
use these helpers to guarantee consistent behaviour across the codebase.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from pff.config import settings
from pff.config import POSTGRES_CONFIG_PATH
from pff.utils import FileManager


_DEFAULT_CONFIG_FILE = POSTGRES_CONFIG_PATH


@dataclass(frozen=True)
class PostgresRetryConfig:
    attempts: int = 3
    backoff_seconds: float = 1.5


@dataclass(frozen=True)
class PostgresSSLConfig:
    enabled: bool = False
    sslmode: str = "verify-full"
    ca_file: Optional[Path] = None
    cert_file: Optional[Path] = None
    key_file: Optional[Path] = None

    def ssl_context(self):
        if not self.enabled:
            return None

        import ssl

        ctx = ssl.create_default_context(cafile=str(self.ca_file) if self.ca_file else None)
        ctx.check_hostname = self.sslmode != "allow"
        ctx.verify_mode = ssl.CERT_REQUIRED if self.sslmode != "allow" else ssl.CERT_NONE

        if self.cert_file and self.key_file:
            ctx.load_cert_chain(certfile=str(self.cert_file), keyfile=str(self.key_file))

        return ctx


@dataclass(frozen=True)
class PostgresPoolConfig:
    min_size: int = 2
    max_size: int = 10
    command_timeout: float = 60.0
    max_queries: int = 50_000
    max_inactive_connection_lifetime: float = 300.0
    statement_timeout: int = 0  # milliseconds – 0 disables

    def to_asyncpg_kwargs(self) -> dict[str, Any]:
        return {
            "min_size": self.min_size,
            "max_size": self.max_size,
            "command_timeout": self.command_timeout,
            "max_queries": self.max_queries,
            "max_inactive_connection_lifetime": self.max_inactive_connection_lifetime,
        }


@dataclass(frozen=True)
class PostgresConfig:
    dsn_asyncpg: str
    dsn_sqlalchemy: str
    pool: PostgresPoolConfig
    retry: PostgresRetryConfig
    ssl: PostgresSSLConfig

    def apply_statement_timeout_sql(self) -> Optional[str]:
        if self.pool.statement_timeout > 0:
            return f"SET statement_timeout = {self.pool.statement_timeout};"
        return None


def _load_yaml_config(config_path: Path) -> Mapping[str, Any]:
    if not config_path.exists():
        return {}
    fm = FileManager()
    return fm.read(config_path)


def _merge_pool_config(cfg: Mapping[str, Any]) -> PostgresPoolConfig:
    pool_cfg = cfg.get("pool", {}) if isinstance(cfg, Mapping) else {}
    return PostgresPoolConfig(
        min_size=int(pool_cfg.get("min_size", 2)),
        max_size=int(pool_cfg.get("max_size", 10)),
        command_timeout=float(pool_cfg.get("command_timeout", 60)),
        max_queries=int(pool_cfg.get("max_queries", 50_000)),
        max_inactive_connection_lifetime=float(
            pool_cfg.get("max_inactive_connection_lifetime", 300)
        ),
        statement_timeout=int(pool_cfg.get("statement_timeout", 0)),
    )


def _merge_retry_config(cfg: Mapping[str, Any]) -> PostgresRetryConfig:
    retry_cfg = cfg.get("retry", {}) if isinstance(cfg, Mapping) else {}
    return PostgresRetryConfig(
        attempts=int(retry_cfg.get("attempts", 3)),
        backoff_seconds=float(retry_cfg.get("backoff_seconds", 1.5)),
    )


def _merge_ssl_config(cfg: Mapping[str, Any]) -> PostgresSSLConfig:
    ssl_cfg = cfg.get("ssl", {}) if isinstance(cfg, Mapping) else {}
    enabled = bool(ssl_cfg.get("enabled", False))
    ca_file = ssl_cfg.get("ca_file")
    cert_file = ssl_cfg.get("cert_file")
    key_file = ssl_cfg.get("key_file")
    return PostgresSSLConfig(
        enabled=enabled,
        sslmode=str(ssl_cfg.get("sslmode", "verify-full")),
        ca_file=Path(ca_file).expanduser() if ca_file else None,
        cert_file=Path(cert_file).expanduser() if cert_file else None,
        key_file=Path(key_file).expanduser() if key_file else None,
    )


def _build_asyncpg_dsn() -> str:
    """Ensure DSN is compatible with asyncpg (postgresql://...)."""
    return settings.DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")


def _build_sqlalchemy_dsn() -> str:
    return settings.DATABASE_URL_ASYNC


_CACHED_CONFIG: Optional[PostgresConfig] = None


def get_postgres_config(force_reload: bool = False) -> PostgresConfig:
    """Return cached Postgres configuration merging YAML + settings."""

    global _CACHED_CONFIG
    if _CACHED_CONFIG is not None and not force_reload:
        return _CACHED_CONFIG

    raw_cfg = _load_yaml_config(_DEFAULT_CONFIG_FILE)

    pool_cfg = _merge_pool_config(raw_cfg)
    retry_cfg = _merge_retry_config(raw_cfg)
    ssl_cfg = _merge_ssl_config(raw_cfg)

    config = PostgresConfig(
        dsn_asyncpg=_build_asyncpg_dsn(),
        dsn_sqlalchemy=_build_sqlalchemy_dsn(),
        pool=pool_cfg,
        retry=retry_cfg,
        ssl=ssl_cfg,
    )

    _CACHED_CONFIG = config
    return config
