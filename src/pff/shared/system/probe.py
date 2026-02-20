"""
Low-level hardware probe utilities.

This module provides raw hardware detection capabilities (RAM, CPU, GPU)
without high-level dependencies like configuration or logging.
It is designed to be imported by core modules (like file_manager) without
creating circular dependencies.
"""

import atexit
import threading

from functools import lru_cache

import psutil  # type: ignore[import-untyped]

# NVML Management
_NVML_INIT_LOCK = threading.Lock()
_NVML_INITIALIZED = False


def _ensure_nvml_initialized() -> None:
    """Execute ensure nvml initialized."""

    global _NVML_INITIALIZED
    if _NVML_INITIALIZED:
        return
    with _NVML_INIT_LOCK:
        if _NVML_INITIALIZED:
            return
        try:
            import pynvml  # type: ignore[import-untyped]

            pynvml.nvmlInit()
            _NVML_INITIALIZED = True
        except ImportError:
            pass
        except Exception:
            pass


def _nvml_shutdown() -> None:
    """Shutdown NVML if initialized."""
    global _NVML_INITIALIZED
    if not _NVML_INITIALIZED:
        return
    with _NVML_INIT_LOCK:
        if not _NVML_INITIALIZED:
            return
        try:
            import pynvml  # type: ignore[import-untyped]

            pynvml.nvmlShutdown()
        except Exception:
            pass
        _NVML_INITIALIZED = False


atexit.register(_nvml_shutdown)


@lru_cache(maxsize=16)
def _nvml_get_device_handle(device_index: int):
    """Execute nvml get device handle.



    Args:

        device_index: Input value used by this callable.



    Returns:

        Return value produced by the callable.

    """

    _ensure_nvml_initialized()
    try:
        import pynvml  # type: ignore[import-untyped]

        return pynvml.nvmlDeviceGetHandleByIndex(int(device_index))
    except Exception:
        return None


@lru_cache(maxsize=32)
def get_gpu_total_memory_gb(device_index: int = 0) -> tuple[bool, float | None]:
    """Return (has_gpu, total_memory_gb) using NVML.

    Returns:
        tuple[bool, float | None]: (True if GPU found, VRAM in GB or None)
    """
    try:
        handle = _nvml_get_device_handle(device_index)
        if not handle:
            return False, None

        import pynvml  # type: ignore[import-untyped]

        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpu_memory_gb = float(mem_info.total) / (1024**3)
        return True, gpu_memory_gb
    except Exception:
        return False, None


def get_safe_cpu_count(*, logical: bool) -> int:
    """Get CPU count with fallback sanitation."""
    value = psutil.cpu_count(logical=logical)
    if value is None or value <= 0:
        fallback_logical = psutil.cpu_count(logical=True)
        if fallback_logical is None or fallback_logical <= 0:
            return 1
        if logical:
            return int(fallback_logical)
        return max(int(fallback_logical) // 2, 1)
    return int(value)


def get_system_ram_gb() -> tuple[float, float]:
    """Get system RAM details.

    Returns:
        tuple[float, float]: (total_ram_gb, available_ram_gb)
    """
    try:
        mem = psutil.virtual_memory()
        total_ram_gb = float(mem.total) / (1024**3)
        available_ram_gb = float(mem.available) / (1024**3)
        return total_ram_gb, available_ram_gb
    except Exception:
        return 8.0, 4.0  # Safe fallback
