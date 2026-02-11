"""Optuna gRPC storage proxy server helpers."""

from __future__ import annotations

from typing import Any

from pff.infrastructure.hpo.config_loader import load_storage_settings
from pff.infrastructure.hpo.storage import _build_postgres_url
from pff.shared import logger
from pff.shared.core.file_manager import FileManager


def run_optuna_grpc_proxy(
    *,
    host: str | None = None,
    port: int | None = None,
    storage_url: str | None = None,
    file_manager: FileManager | None = None,
) -> None:
    """Start Optuna's gRPC proxy server with an RDBStorage backend."""
    fm = file_manager or FileManager()
    storage_cfg = load_storage_settings(fm)
    backend = str(storage_cfg.get("backend", "sqlite")).lower()
    if backend in {"sqlite"}:
        logger.warning("Storage backend sqlite does not support grpc; use postgres/rdb")
    if storage_url is None:
        storage_url = storage_cfg.get("url") or _build_postgres_url()

    grpc_cfg = (
        storage_cfg.get("grpc_proxy", {})
        if isinstance(storage_cfg.get("grpc_proxy"), dict)
        else {}
    )
    host = host or str(grpc_cfg.get("host", "0.0.0.0"))
    port = int(port or grpc_cfg.get("port", 13000))

    try:
        import optuna
        from optuna.storages import run_grpc_proxy_server
    except Exception as exc:
        raise ImportError("Optuna is required for gRPC proxy support") from exc

    engine_kwargs: dict[str, Any] = (
        storage_cfg.get("engine", {})
        if isinstance(storage_cfg.get("engine"), dict)
        else {}
    )
    storage = optuna.storages.RDBStorage(url=storage_url, engine_kwargs=engine_kwargs)

    logger.info(
        f"grpc_proxy_iniciando host={host} port={port} storage_url={storage_url}"
    )
    run_grpc_proxy_server(storage, host=host, port=port)
