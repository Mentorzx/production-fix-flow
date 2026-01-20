"""Runtime initialization helpers (drivers only)."""

from __future__ import annotations

import multiprocessing as mp
import os

from pff.shared import logger
from pff.shared.core.config import settings


def initialize_runtime(version: str | None = None) -> None:
    """Initialize runtime directories and environment for main process."""
    if mp.current_process().name != "MainProcess":
        return

    settings.DATA_DIR.mkdir(exist_ok=True)
    settings.OUTPUTS_DIR.mkdir(exist_ok=True)
    settings.LOGS_DIR.mkdir(exist_ok=True)

    joblib_dir = (settings.CACHE_DIR / "joblib").expanduser()
    joblib_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("JOBLIB_TEMP_FOLDER", str(joblib_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(settings.CACHE_DIR))
    os.environ.setdefault("TORCH_HOME", str(settings.CACHE_DIR / "torch"))
    os.environ.setdefault("HF_HOME", str(settings.CACHE_DIR / "huggingface"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(settings.CACHE_DIR / "huggingface"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(settings.CACHE_DIR / "huggingface" / "datasets"))

    version_label = version or "unknown"
    logger.info(f" PFF Fênix v{version_label} iniciado. Ambiente configurado.")
    logger.info(f"   - Diretório de Logs: {settings.LOGS_DIR}")
    logger.info(f"   - Diretório de Saída: {settings.OUTPUTS_DIR}")

    try:
        from pff.shared.core.config import apply_permanent_configurations
    except ImportError:
        return

    apply_permanent_configurations()
