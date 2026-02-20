"""Runtime initialization helpers (drivers only)."""

from __future__ import annotations

import multiprocessing as mp
import os
import importlib.util

from pff.shared.core.logging import logger
from pff.shared.core.config import settings
from pff.shared.core.file_manager import FileManager


def _is_main_process() -> bool:
    return mp.current_process().name == "MainProcess"


def _configure_spawn_for_lancedb() -> None:
    """Execute configure spawn for lancedb."""

    if importlib.util.find_spec("lancedb") is None:
        return
    try:
        if mp.get_start_method(allow_none=True) != "spawn":
            mp.set_start_method("spawn", force=True)
    except RuntimeError:
        return


def _ensure_cache_environment() -> None:
    """Execute ensure cache environment."""

    joblib_dir = (settings.CACHE_DIR / "joblib").expanduser()
    joblib_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("JOBLIB_TEMP_FOLDER", str(joblib_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(settings.CACHE_DIR))
    os.environ.setdefault("TORCH_HOME", str(settings.CACHE_DIR / "torch"))
    os.environ.setdefault("HF_HOME", str(settings.CACHE_DIR / "huggingface"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(settings.CACHE_DIR / "huggingface" / "datasets"))
    pycache_dir = settings.CACHE_DIR / "pycache"
    pycache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PYTHONPYCACHEPREFIX", str(pycache_dir))
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "0")


def _initialize_dirs_and_cache(*, clean_mode: bool) -> None:
    """Execute initialize dirs and cache.



    Args:

        clean_mode: Input value used by this callable.

    """

    settings.DATA_DIR.mkdir(exist_ok=True)
    if clean_mode:
        return
    settings.OUTPUTS_DIR.mkdir(exist_ok=True)
    settings.LOGS_DIR.mkdir(exist_ok=True)
    _ensure_cache_environment()


def _load_dotenv_if_present() -> None:
    """Execute load dotenv if present."""

    env_path = settings.ROOT_DIR / ".env"
    if not env_path.exists():
        return
    try:
        for raw_line in FileManager.read_text(env_path).splitlines():
            line = raw_line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                if key not in os.environ:
                    os.environ[key] = value
    except Exception:
        return


def _log_runtime_status(*, version: str | None, clean_mode: bool) -> None:
    """Execute log runtime status.



    Args:

        version: Input value used by this callable.

        clean_mode: Input value used by this callable.

    """

    version_label = version or "unknown"
    logger.info(f" PFF Fênix v{version_label} iniciado.")
    logger.info("Ambiente configurado com sucesso")
    if clean_mode:
        return
    logger.debug(f"   - Logs Directory: {settings.LOGS_DIR}")
    logger.debug(f"   - Outputs Directory: {settings.OUTPUTS_DIR}")


def _apply_runtime_configurations() -> None:
    """Execute apply runtime configurations."""

    try:
        from pff.shared.core.config import apply_permanent_configurations
    except ImportError:
        return
    apply_permanent_configurations()


def initialize_runtime(version: str | None = None) -> None:
    """Initialize runtime directories and environment for main process."""
    if not _is_main_process():
        return

    clean_mode = os.environ.get("PFF_CLEAN_MODE") == "1"
    _configure_spawn_for_lancedb()
    _initialize_dirs_and_cache(clean_mode=clean_mode)
    _load_dotenv_if_present()
    _log_runtime_status(version=version, clean_mode=clean_mode)
    _apply_runtime_configurations()
