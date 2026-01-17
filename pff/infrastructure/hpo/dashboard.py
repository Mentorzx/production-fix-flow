"""Optuna dashboard bootstrap helpers."""

from __future__ import annotations

import os
import subprocess

from pff import settings
from pff.shared import logger
from pff.shared.core.file_manager import FileManager
from pff.infrastructure.hpo.config_loader import (
    load_live_plot_settings,
    load_storage_settings,
)


def _trim_output(text: str, limit: int = 240) -> str:
    clean = " ".join((text or "").split())
    if len(clean) <= limit:
        return clean
    return f"{clean[:limit]}..."


def _dashboard_url() -> str:
    return os.getenv("OPTUNA_DASHBOARD_URL", "http://localhost:8080/dashboard")


def _compose_env() -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("POSTGRES_USER", settings.POSTGRES_USER)
    env.setdefault("POSTGRES_PASSWORD", settings.POSTGRES_PASSWORD)
    env.setdefault("POSTGRES_DB", settings.POSTGRES_DB)
    env.setdefault("POSTGRES_PORT_HOST", str(settings.POSTGRES_PORT))
    env.setdefault("POSTGRES_PORT", str(settings.POSTGRES_PORT))
    env.setdefault(
        "OPTUNA_STORAGE_URL",
        f"postgresql+psycopg2://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}"
        f"@postgres:5432/{settings.POSTGRES_DB}",
    )
    return env


def _is_dashboard_running() -> bool:
    try:
        result = subprocess.run(
            [
                "docker",
                "ps",
                "--filter",
                "name=pff-optuna-dashboard",
                "--format",
                "{{.Names}}",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        logger.warning("component=hpo dashboard_optuna stop_reason=docker_not_found")
        return False
    if result.returncode != 0:
        logger.warning(
            "component=hpo dashboard_optuna stop_reason=checagem_falhou "
            f"stderr={_trim_output(result.stderr)}"
        )
        return False
    return "pff-optuna-dashboard" in set(result.stdout.splitlines())


def ensure_optuna_dashboard_running(file_manager: FileManager | None = None) -> bool:
    """Ensure Optuna Dashboard is running when live dashboard is enabled."""
    fm = file_manager or FileManager()
    live_cfg = load_live_plot_settings(fm)
    if not live_cfg.get("enable_optuna_dashboard", False):
        return False

    storage_cfg = load_storage_settings(fm)
    backend = str(storage_cfg.get("backend", "sqlite")).lower()
    if backend not in {"postgres", "postgresql", "rdb", "rdbstorage"}:
        logger.info(f"component=hpo dashboard_optuna status=ignorado backend={backend}")
        return False

    os.environ.setdefault("OPTUNA_DASHBOARD_URL", _dashboard_url())
    compose_file = settings.ROOT_DIR / "docker-compose.yml"
    if not compose_file.exists():
        logger.warning(
            f"component=hpo dashboard_optuna stop_reason=compose_inexistente path={compose_file}"
        )
        return False

    if _is_dashboard_running():
        logger.debug(
            f"component=hpo dashboard_optuna status=ativo url={_dashboard_url()}"
        )
        return True

    env = _compose_env()
    try:
        result = subprocess.run(
            [
                "docker",
                "compose",
                "-f",
                str(compose_file),
                "up",
                "-d",
                "optuna-dashboard",
            ],
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
    except FileNotFoundError:
        logger.warning("component=hpo dashboard_optuna stop_reason=docker_not_found")
        return False

    if result.returncode != 0:
        logger.warning(
            "component=hpo dashboard_optuna stop_reason=falha_subir "
            f"stderr={_trim_output(result.stderr)}"
        )
        return False

    logger.debug(
        f"component=hpo dashboard_optuna status=ativado url={_dashboard_url()}"
    )
    return True
