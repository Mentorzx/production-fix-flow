"""Tests for HPO storage config parsing."""

from __future__ import annotations

import types
import sys
from pathlib import Path

import pytest


def test_load_storage_settings_defaults(monkeypatch):
    """Default storage settings should follow optimization.yaml with sane pooling defaults."""
    from pff.infrastructure.hpo import config_loader
    from pff.infrastructure.hpo.config_loader import clear_config_cache

    monkeypatch.delenv("PFF_HPO_GRPC_HOST", raising=False)
    monkeypatch.delenv("PFF_HPO_GRPC_PORT", raising=False)
    monkeypatch.delenv("PFF_HPO_STORAGE_BACKEND", raising=False)
    monkeypatch.delenv("PFF_HPO_STORAGE_URL", raising=False)

    clear_config_cache()
    settings = config_loader.load_storage_settings()

    assert settings["backend"] == "postgres"
    assert settings["url"] is None
    assert settings["engine"]["pool_size"] == 20
    assert settings["engine"]["max_overflow"] == 10
    assert settings["engine"]["pool_pre_ping"] is True
    assert settings["engine"]["connect_args"]["keepalives"] == 1
    assert settings["grpc_proxy"]["host"] == "localhost"
    assert settings["grpc_proxy"]["port"] == 13000


def test_load_parallel_settings_defaults(monkeypatch):
    """Default parallel settings should come from optimization.yaml."""
    from pff.infrastructure.hpo import config_loader
    from pff.infrastructure.hpo.config_loader import clear_config_cache

    monkeypatch.delenv("PFF_HPO_STORAGE_BACKEND", raising=False)
    monkeypatch.delenv("PFF_HPO_STORAGE_URL", raising=False)

    clear_config_cache()
    settings = config_loader.load_parallel_settings()
    clear_config_cache()
    optimization_cfg = config_loader.load_optimization_config()
    parallel_cfg = optimization_cfg.get("parallel", {})
    expected_n_jobs = 1
    if isinstance(parallel_cfg, dict):
        try:
            expected_n_jobs = max(1, int(parallel_cfg.get("n_jobs", 1)))
        except (TypeError, ValueError):
            expected_n_jobs = 1

    assert settings["n_jobs"] == expected_n_jobs
    assert settings["use_journal_for_parallel"] is True
    assert settings["optuna"]["gc_after_trial"] is True
    assert settings["cv"]["disable_when_cuda"] is True
    assert settings["cv"]["disable_when_dataloader_workers"] is True
    assert settings["cv"]["disable_when_auto_workers"] is True


def test_load_parallel_settings_custom(monkeypatch):
    """Custom parallel settings should override defaults."""
    from pff.infrastructure.hpo import config_loader

    def _fake_loader(*_args, **_kwargs):
        return {
            "parallel": {
                "n_jobs": 4,
                "use_journal_for_parallel": False,
                "optuna": {"gc_after_trial": False},
                "cv": {
                    "disable_when_cuda": False,
                    "disable_when_dataloader_workers": False,
                    "disable_when_auto_workers": False,
                },
            }
        }

    monkeypatch.setattr(config_loader, "load_optimization_config", _fake_loader)

    settings = config_loader.load_parallel_settings()

    assert settings["n_jobs"] == 4
    assert settings["use_journal_for_parallel"] is False
    assert settings["optuna"]["gc_after_trial"] is False
    assert settings["cv"]["disable_when_cuda"] is False
    assert settings["cv"]["disable_when_dataloader_workers"] is False
    assert settings["cv"]["disable_when_auto_workers"] is False


def test_load_storage_settings_custom(monkeypatch):
    """Custom storage settings should override defaults."""
    from pff.infrastructure.hpo import config_loader

    def _fake_loader(*_args, **_kwargs):
        return {
            "storage": {
                "backend": "grpc",
                "url": "postgresql+psycopg2://u:p@host:5432/db",
                "engine": {
                    "pool_size": 5,
                    "max_overflow": 2,
                    "pool_pre_ping": False,
                    "connect_args": {"keepalives": 0, "connect_timeout": 10},
                },
                "grpc_proxy": {"host": "grpc-host", "port": 13001},
            }
        }

    monkeypatch.delenv("PFF_HPO_GRPC_HOST", raising=False)
    monkeypatch.delenv("PFF_HPO_GRPC_PORT", raising=False)
    monkeypatch.delenv("PFF_HPO_STORAGE_BACKEND", raising=False)
    monkeypatch.delenv("PFF_HPO_STORAGE_URL", raising=False)
    monkeypatch.setattr(config_loader, "load_optimization_config", _fake_loader)

    settings = config_loader.load_storage_settings()

    assert settings["backend"] == "grpc"
    assert settings["url"] == "postgresql+psycopg2://u:p@host:5432/db"
    assert settings["engine"]["pool_size"] == 5
    assert settings["engine"]["max_overflow"] == 2
    assert settings["engine"]["pool_pre_ping"] is False
    assert settings["engine"]["connect_args"]["keepalives"] == 0
    assert settings["engine"]["connect_args"]["connect_timeout"] == 10
    assert settings["grpc_proxy"]["host"] == "grpc-host"
    assert settings["grpc_proxy"]["port"] == 13001


def test_load_storage_settings_env_overrides(monkeypatch):
    """Environment settings should override YAML for grpc proxy."""
    from pff.infrastructure.hpo import config_loader

    def _fake_loader(*_args, **_kwargs):
        return {"storage": {"grpc_proxy": {"host": "yaml-host", "port": 13001}}}

    monkeypatch.setenv("PFF_HPO_GRPC_HOST", "env-host")
    monkeypatch.setenv("PFF_HPO_GRPC_PORT", "13099")
    monkeypatch.setenv("PFF_HPO_STORAGE_BACKEND", "postgresql")
    monkeypatch.setenv(
        "PFF_HPO_STORAGE_URL", "postgresql+psycopg2://u:p@localhost:5432/db"
    )
    monkeypatch.setattr(config_loader, "load_optimization_config", _fake_loader)

    settings = config_loader.load_storage_settings()

    assert settings["backend"] == "postgresql"
    assert settings["url"] == "postgresql+psycopg2://u:p@localhost:5432/db"
    assert settings["grpc_proxy"]["host"] == "env-host"
    assert settings["grpc_proxy"]["port"] == 13099


def test_create_optuna_storage_rejects_sqlite_backend(monkeypatch, tmp_path: Path):
    """Storage factory should fail fast when SQLite is configured."""
    from pff.infrastructure.hpo import storage

    monkeypatch.setattr(
        storage, "load_storage_settings", lambda _fm: {"backend": "sqlite"}
    )

    with pytest.raises(RuntimeError, match="Unsupported HPO storage backend"):
        storage.create_optuna_storage(storage_path=tmp_path / "optuna_study.db")


def test_create_optuna_storage_bootstraps_local_postgres(monkeypatch, tmp_path: Path):
    """Storage factory should bootstrap local Postgres before creating RDBStorage."""
    from pff.infrastructure.hpo import storage

    called: dict[str, str] = {}

    def _fake_load(_fm):
        return {
            "backend": "postgres",
            "url": "postgresql+psycopg2://u:p@localhost:5432/db",
            "engine": {},
        }

    class _DummyRDBStorage:
        def __init__(self, *, url, engine_kwargs):
            self.url = url
            self.engine_kwargs = engine_kwargs

    monkeypatch.setattr(storage, "load_storage_settings", _fake_load)
    monkeypatch.setattr(
        storage,
        "ensure_local_postgres_ready",
        lambda url: called.setdefault("url", url),
    )
    monkeypatch.setitem(
        sys.modules,
        "optuna",
        types.SimpleNamespace(
            storages=types.SimpleNamespace(RDBStorage=_DummyRDBStorage)
        ),
    )

    storage_obj, storage_url = storage.create_optuna_storage(
        storage_path=tmp_path / "study.db"
    )

    assert called["url"] == "postgresql+psycopg2://u:p@localhost:5432/db"
    assert storage_url == "postgresql+psycopg2://u:p@localhost:5432/db"
    assert isinstance(storage_obj, _DummyRDBStorage)
