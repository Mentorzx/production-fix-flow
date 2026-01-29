"""Optuna storage factory for HPO."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from pff.infrastructure.hpo.config_loader import load_storage_settings
from pff.shared import logger
from pff.shared.core.config import settings
from pff.shared.core.file_manager import FileManager


def _build_postgres_url() -> str:
    return (
        "postgresql+psycopg2://"
        f"{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}"
        f"@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
    )


def _redact_url(url: str) -> str:
    """Redact credentials from a URL before logging."""
    try:
        parts = urlsplit(url)
    except Exception:
        return url
    if not parts.netloc:
        return url
    if parts.username is None and parts.password is None:
        return url
    host = parts.hostname or ""
    if parts.port:
        host = f"{host}:{parts.port}"
    user_prefix = f"{parts.username}:***@" if parts.username else "***@"
    redacted_netloc = f"{user_prefix}{host}"
    return urlunsplit(
        (parts.scheme, redacted_netloc, parts.path, parts.query, parts.fragment)
    )


def create_optuna_storage(
    *,
    storage_path: Path,
    file_manager: FileManager | None = None,
) -> tuple[Any | None, str]:
    """Create Optuna storage based on config.

    Supported backends:
    - sqlite: Local SQLite file (default)
    - postgres/rdb: PostgreSQL RDBStorage
    - grpc: gRPC proxy to central storage
    - journal: JournalStorage for parallel trials (low overhead)
    """
    fm = file_manager or FileManager()
    storage_cfg = load_storage_settings(fm)
    backend = str(storage_cfg.get("backend", "sqlite")).lower()

    storage: Any = None

    if backend in {"journal", "journal_storage"}:
        journal_path = storage_cfg.get("journal_path") or str(storage_path).replace(
            ".db", ".log"
        )
        try:
            import optuna
            from optuna.storages import JournalStorage
            from optuna.storages.journal import JournalFileBackend

            logger.info(f"hpo_storage backend=journal caminho={journal_path}")
            storage = JournalStorage(JournalFileBackend(journal_path))
            return storage, f"journal://{journal_path}"
        except (ImportError, Exception) as exc:
            raise RuntimeError(
                f"JournalStorage failed: {exc}. Fix the configured journal backend."
            ) from exc

    if backend in {"grpc", "grpc_proxy"}:
        grpc_cfg = (
            storage_cfg.get("grpc_proxy", {})
            if isinstance(storage_cfg.get("grpc_proxy"), dict)
            else {}
        )
        host = str(grpc_cfg.get("host", "localhost"))
        port = int(grpc_cfg.get("port", 13000))
        try:
            import optuna as optuna_grpc
        except Exception as exc:
            raise ImportError("Optuna is required for HPO storage") from exc
        logger.info(f"hpo_storage backend=grpc host={host} porta={port}")
        storage = optuna_grpc.storages.GrpcStorageProxy(host=host, port=port)  # type: ignore[assignment]
        return storage, f"grpc://{host}:{port}"
    if backend in {"postgres", "postgresql", "rdb", "rdbstorage"}:
        url = storage_cfg.get("url") or _build_postgres_url()
        engine_kwargs = (
            storage_cfg.get("engine", {})
            if isinstance(storage_cfg.get("engine"), dict)
            else {}
        )
        try:
            import optuna

            logger.info(f"hpo_storage backend=postgres url={_redact_url(str(url))}")
            storage = optuna.storages.RDBStorage(url=url, engine_kwargs=engine_kwargs)  # type: ignore[assignment]
            return storage, url
        except (ImportError, Exception) as exc:
            raise RuntimeError(
                "Failed to initialize Postgres storage (driver missing or connection error). "
                "Fix the configured Postgres backend."
            ) from exc

    storage_url = f"sqlite:///{storage_path}"
    logger.info(f"hpo_storage backend=sqlite caminho={storage_path}")
    return None, storage_url
