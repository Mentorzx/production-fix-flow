"""Runtime initialization helpers (drivers only)."""

from __future__ import annotations

import multiprocessing as mp
import os
import importlib.util

from pff.shared.core.logging import logger
from pff.shared.core.config import settings


def initialize_runtime(version: str | None = None) -> None:
    """Initialize runtime directories and environment for main process."""
    if mp.current_process().name != "MainProcess":
        return

    clean_mode = os.environ.get("PFF_CLEAN_MODE") == "1"

    if importlib.util.find_spec("lancedb") is not None:
        try:
            if mp.get_start_method(allow_none=True) != "spawn":
                mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

    settings.DATA_DIR.mkdir(exist_ok=True)
    if not clean_mode:
        settings.OUTPUTS_DIR.mkdir(exist_ok=True)
        settings.LOGS_DIR.mkdir(exist_ok=True)

        joblib_dir = (settings.CACHE_DIR / "joblib").expanduser()
        joblib_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("JOBLIB_TEMP_FOLDER", str(joblib_dir))
        os.environ.setdefault("XDG_CACHE_HOME", str(settings.CACHE_DIR))
        os.environ.setdefault("TORCH_HOME", str(settings.CACHE_DIR / "torch"))
        os.environ.setdefault("HF_HOME", str(settings.CACHE_DIR / "huggingface"))
        os.environ.setdefault(
            "HF_DATASETS_CACHE", str(settings.CACHE_DIR / "huggingface" / "datasets")
        )
        pycache_dir = settings.CACHE_DIR / "pycache"
        pycache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("PYTHONPYCACHEPREFIX", str(pycache_dir))
        os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "0")
    env_path = settings.ROOT_DIR / ".env"
    if env_path.exists():
        try:
            for raw_line in env_path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    if key not in os.environ:
                        os.environ[key] = value
        except Exception:
            pass

    version_label = version or "unknown"
    logger.info(f" PFF Fênix v{version_label} iniciado. Ambiente configurado.")
    if not clean_mode:
        logger.info(f"   - Diretório de Logs: {settings.LOGS_DIR}")
        logger.info(f"   - Diretório de Saída: {settings.OUTPUTS_DIR}")

    try:
        from pff.shared.core.config import apply_permanent_configurations
    except ImportError:
        return

    apply_permanent_configurations()
