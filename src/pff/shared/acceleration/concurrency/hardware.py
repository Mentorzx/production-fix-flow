"""Hardware management - GPU detection and telemetry.

Delegates NVML lifecycle to ``pff.shared.system.probe`` (singleton + atexit)
and CPU counting to ``probe.get_safe_cpu_count``, eliminating the per-call
init/shutdown race condition that existed before.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import psutil

from pff.shared.system.probe import (
    _ensure_nvml_initialized,
    _nvml_get_device_handle,
    get_safe_cpu_count,
)


@dataclass
class GPUInfo:
    """GPU information dataclass."""

    id: int
    name: str
    memory_total: int
    compute_capability: tuple[int, int]
    uuid: str


class HardwareManager:
    """Hardware detection and real-time telemetry manager.

    NVML lifecycle is managed by ``probe.py`` (singleton + atexit).
    CPU counting uses ``probe.get_safe_cpu_count``.
    """

    def __init__(self):
        self.physical_cores = get_safe_cpu_count(logical=False)
        self.logical_cores = get_safe_cpu_count(logical=True)
        self.gpus: list[GPUInfo] = []
        self._prev_cpu_total: float | None = None
        self._prev_cpu_idle: float | None = None
        self._detect_gpus()

    def _detect_gpus(self) -> None:
        """Detect GPUs using probe's singleton NVML lifecycle."""
        try:
            import pynvml
        except ImportError:
            from ...core.logging import logger

            logger.debug("pynvml not available; GPU detection disabled")
            return

        _ensure_nvml_initialized()
        try:
            cnt = pynvml.nvmlDeviceGetCount()
            for i in range(cnt):
                handle = _nvml_get_device_handle(i)
                if handle is None:
                    continue
                nm = pynvml.nvmlDeviceGetName(handle)
                name = nm.decode() if isinstance(nm, (bytes, bytearray)) else nm
                mem = int(pynvml.nvmlDeviceGetMemoryInfo(handle).total)
                cc = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
                uid = pynvml.nvmlDeviceGetUUID(handle)
                uuid = uid.decode() if isinstance(uid, (bytes, bytearray)) else uid
                self.gpus.append(GPUInfo(i, name, mem, cc, uuid))
        except Exception as exc:
            from ...core.logging import logger

            logger.debug(f"Failed to initialize GPU metadata via NVML: {exc}")
            self.gpus = []

    def shutdown(self):
        """No-op; NVML lifecycle managed by probe.py via atexit."""

    def __getstate__(self) -> dict[str, Any]:
        return {
            "physical_cores": self.physical_cores,
            "logical_cores": self.logical_cores,
            "gpus": [asdict(g) for g in self.gpus],
        }

    def __setstate__(self, state: dict[str, Any]):
        self.physical_cores = state.get("physical_cores", 1)
        self.logical_cores = state.get("logical_cores", self.physical_cores)
        gpus_raw = state.get("gpus", [])
        self.gpus = [GPUInfo(**g) for g in gpus_raw if isinstance(g, dict)]
        self._prev_cpu_total = None
        self._prev_cpu_idle = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.shutdown()

    def get_handle(self, gpu: GPUInfo):
        """Get NVML handle for a GPU (cached via probe)."""
        return _nvml_get_device_handle(gpu.id)

    def get_telemetry(self) -> dict[str, Any]:
        """Returns real-time hardware telemetry."""
        mem = psutil.virtual_memory()

        cpu_times = psutil.cpu_times(percpu=False)
        cpu_total = float(sum(cpu_times))
        cpu_idle = float(getattr(cpu_times, "idle", 0.0) + getattr(cpu_times, "iowait", 0.0))

        if self._prev_cpu_total is None or self._prev_cpu_idle is None:
            self._prev_cpu_total = cpu_total
            self._prev_cpu_idle = cpu_idle
            cpu_usage = psutil.cpu_percent(interval=0.1, percpu=False)
        else:
            total_delta = cpu_total - self._prev_cpu_total
            idle_delta = cpu_idle - self._prev_cpu_idle
            self._prev_cpu_total = cpu_total
            self._prev_cpu_idle = cpu_idle
            if total_delta <= 1e-9:
                cpu_usage = psutil.cpu_percent(interval=0.1, percpu=False)
            else:
                cpu_usage = (1.0 - (idle_delta / total_delta)) * 100.0
                cpu_usage = float(max(0.0, min(100.0, cpu_usage)))

        mem_total = float(getattr(mem, "total", 0.0) or 0.0)
        mem_free = float(getattr(mem, "free", 0.0) or 0.0)
        mem_used_incl_cache = max(0.0, mem_total - mem_free)
        ram_usage_pct = (mem_used_incl_cache / mem_total * 100.0) if mem_total > 0 else 0.0

        telemetry: dict[str, Any] = {
            "cpu_usage": cpu_usage,
            "ram_usage_pct": ram_usage_pct,
            "ram_total_gb": mem.total / (1024**3),
            "ram_used_gb": mem_used_incl_cache / (1024**3),
            "gpus": [],
        }

        if self.gpus:
            try:
                import pynvml

                _ensure_nvml_initialized()
                for gpu in self.gpus:
                    handle = _nvml_get_device_handle(gpu.id)
                    if handle is None:
                        continue
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    telemetry["gpus"].append(
                        {
                            "id": gpu.id,
                            "name": gpu.name,
                            "utilization": util.gpu,
                            "vram_total": mem_info.total,
                            "vram_used": mem_info.used,
                            "vram_usage_pct": (mem_info.used / mem_info.total * 100),
                        }
                    )
            except Exception:
                pass

        return telemetry
