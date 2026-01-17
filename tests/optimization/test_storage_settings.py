"""Tests for HPO storage config parsing."""

from __future__ import annotations


def test_load_storage_settings_defaults(monkeypatch):
    """Default storage settings should follow optimization.yaml with sane pooling defaults."""
    from pff.infrastructure.hpo import config_loader

    monkeypatch.delenv("PFF_HPO_GRPC_HOST", raising=False)
    monkeypatch.delenv("PFF_HPO_GRPC_PORT", raising=False)

    settings = config_loader.load_storage_settings()

    assert settings["backend"] == "postgres"
    assert settings["url"] is None
    assert settings["engine"]["pool_size"] == 20
    assert settings["engine"]["max_overflow"] == 10
    assert settings["engine"]["pool_pre_ping"] is True
    assert settings["engine"]["connect_args"]["keepalives"] == 1
    assert settings["grpc_proxy"]["host"] == "localhost"
    assert settings["grpc_proxy"]["port"] == 13000


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
    monkeypatch.setattr(config_loader, "load_optimization_config", _fake_loader)

    settings = config_loader.load_storage_settings()

    assert settings["grpc_proxy"]["host"] == "env-host"
    assert settings["grpc_proxy"]["port"] == 13099
