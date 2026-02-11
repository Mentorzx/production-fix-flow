"""
Torch profiler instrumentation for DSLFM-KGC hotspot analysis.

This module provides profiling utilities to identify:
- Training step hotspots (forward, backward, optimizer)
- Evaluation hotspots (score computation, rank aggregation)
- Kernel launch overhead vs compute time
- Memory bandwidth bottlenecks

Usage:
    from pff.infrastructure.profiling import DSLFMProfiler, ProfileConfig

    config = ProfileConfig(
        warmup_steps=20,
        measure_steps=200,
        output_dir=Path("outputs/benches/profiles"),
    )
    profiler = DSLFMProfiler(config)

    for step, batch in enumerate(loader):
        with profiler.step(step):
            loss = model.compute_loss(...)
            loss.backward()
            optimizer.step()

    profiler.export_report()
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import polars as pl
import torch

from pff.shared.core.config import settings
from pff.shared.core.file_manager import FileManager
from pff.shared.core.logging import logger


@dataclass
class ProfileConfig:
    """Configuration for DSLFM profiling.

    Attributes:
        warmup_steps: Steps to skip before measuring (JIT compilation, caches).
        measure_steps: Steps to measure for the profile.
        output_dir: Directory for profile outputs.
        trace_memory: Enable CUDA memory profiling.
        with_stack: Include Python stack traces (slower but more informative).
        with_flops: Estimate FLOPs (requires torch >= 2.0).
        with_modules: Record module hierarchy in traces.
        record_shapes: Record tensor shapes (helps identify hotspots).
        profile_cuda: Profile CUDA kernels (requires CUDA).
        export_chrome_trace: Export Chrome-compatible trace JSON.
        export_stacks: Export flamegraph-compatible stack traces.
    """

    warmup_steps: int = 20
    measure_steps: int = 200
    output_dir: Path = field(default_factory=lambda: settings.OUTPUTS_DIR / "benches" / "profiles")
    trace_memory: bool = True
    with_stack: bool = False
    with_flops: bool = True
    with_modules: bool = True
    record_shapes: bool = True
    profile_cuda: bool = True
    export_chrome_trace: bool = True
    export_stacks: bool = False


@dataclass
class ProfileMetrics:
    """Collected profile metrics."""

    wall_time_per_step_ms: list[float] = field(default_factory=list)
    cuda_time_per_step_ms: list[float] = field(default_factory=list)
    gpu_mem_peak_mb: float = 0.0
    top_cuda_kernels: list[dict[str, Any]] = field(default_factory=list)
    top_cpu_ops: list[dict[str, Any]] = field(default_factory=list)
    hotspots: dict[str, float] = field(default_factory=dict)


class DSLFMProfiler:
    """Profiler for DSLFM-KGC training and evaluation hotspot analysis.

    Wraps torch.profiler with DSLFM-specific reporting including:
    - Training step breakdown (forward, backward, optimizer)
    - Triton/Rust kernel performance
    - Memory allocation patterns
    - Launch overhead estimation

    Example:
        >>> profiler = DSLFMProfiler(ProfileConfig(warmup_steps=10, measure_steps=50))
        >>> for step, batch in enumerate(loader):
        ...     with profiler.step(step):
        ...         loss = model(batch)
        ...         loss.backward()
        ...         optimizer.step()
        >>> report = profiler.export_report()
        >>> report["hotspots"]
    """

    def __init__(self, config: ProfileConfig | None = None) -> None:
        """Initialize profiler with configuration.

        Args:
            config: Profile configuration. Uses defaults if None.
        """
        self.config = config or ProfileConfig()
        self._profiler: torch.profiler.profile | None = None
        self._step_times: list[float] = []
        self._cuda_times: list[float] = []
        self._current_step = 0
        self._step_start_time: float | None = None
        self._is_active = False
        self._metrics = ProfileMetrics()

        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def _should_profile(self, step: int) -> bool:
        """Check if current step should be profiled."""
        return (
            step >= self.config.warmup_steps
            and step < self.config.warmup_steps + self.config.measure_steps
        )

    def _create_schedule(self) -> Callable[[int], torch.profiler.ProfilerAction]:
        """Create a profiler schedule."""
        return torch.profiler.schedule(
            skip_first=0,
            wait=self.config.warmup_steps,
            warmup=1,
            active=self.config.measure_steps,
            repeat=1,
        )

    def start(self) -> None:
        """Start the profiler session."""
        if self._is_active:
            logger.warning("Profiler already active, skipping start()")
            return

        activities = [torch.profiler.ProfilerActivity.CPU]
        if self.config.profile_cuda and torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        self._profiler = torch.profiler.profile(
            activities=activities,
            schedule=self._create_schedule(),
            on_trace_ready=self._on_trace_ready,
            record_shapes=self.config.record_shapes,
            profile_memory=self.config.trace_memory,
            with_stack=self.config.with_stack,
            with_flops=self.config.with_flops,
            with_modules=self.config.with_modules,
        )
        self._profiler.__enter__()
        self._is_active = True
        self._current_step = 0
        logger.info(
            f"Profiler iniciado: aquecimento={self.config.warmup_steps}, "
            f"medicao={self.config.measure_steps}"
        )

    def stop(self) -> None:
        """Stop the profiler session."""
        if not self._is_active or self._profiler is None:
            return

        try:
            self._profiler.__exit__(None, None, None)
        except Exception as exc:
            logger.warning(f"Error stopping profiler: {exc}")
        finally:
            self._is_active = False
            self._profiler = None
            logger.info("Profiler finalizado")

    @contextmanager
    def step(self, step_idx: int) -> Generator[None, None, None]:
        """Context manager for profiling a single step.

        Args:
            step_idx: Current step index.

        Yields:
            None - use as context manager.
        """
        self._current_step = step_idx

        if self._should_profile(step_idx):
            self._step_start_time = time.perf_counter()
            cuda_start: torch.cuda.Event | None = None
            cuda_end: torch.cuda.Event | None = None
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                cuda_start = torch.cuda.Event(enable_timing=True)
                cuda_end = torch.cuda.Event(enable_timing=True)
                cuda_start.record()
            else:
                cuda_start = cuda_end = None

        try:
            yield
        finally:
            if self._should_profile(step_idx) and self._step_start_time is not None:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    if cuda_start is not None and cuda_end is not None:
                        cuda_end.record()
                        torch.cuda.synchronize()
                        cuda_time_ms = cuda_start.elapsed_time(cuda_end)
                        self._cuda_times.append(cuda_time_ms)

                wall_time_ms = (time.perf_counter() - self._step_start_time) * 1000.0
                self._step_times.append(wall_time_ms)
                self._step_start_time = None

            if self._profiler is not None:
                self._profiler.step()

    def _on_trace_ready(self, prof: torch.profiler.profile) -> None:
        """Callback when trace is ready for export."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        if self.config.export_chrome_trace:
            trace_path = self.config.output_dir / f"trace_{timestamp}.json"
            prof.export_chrome_trace(str(trace_path))
            logger.info(f"Chrome trace exportado: {trace_path}")

        if self.config.export_stacks:
            stacks_path = self.config.output_dir / f"stacks_{timestamp}.txt"
            prof.export_stacks(str(stacks_path), metric="self_cuda_time_total")
            logger.info(f"Stack traces exportados: {stacks_path}")

    def _analyze_key_averages(
        self, prof: torch.profiler.profile
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Extract top CPU ops and CUDA kernels from profiler."""
        key_averages = prof.key_averages()

        cpu_ops = []
        cuda_kernels = []

        for event in key_averages:
            entry = {
                "name": event.key,
                "count": event.count,
                "cpu_time_total_ms": event.cpu_time_total / 1000.0,
                "self_cpu_time_ms": event.self_cpu_time_total / 1000.0,
            }

            if hasattr(event, "cuda_time_total") and event.cuda_time_total > 0:
                entry["cuda_time_total_ms"] = event.cuda_time_total / 1000.0
                entry["self_cuda_time_ms"] = event.self_cuda_time_total / 1000.0
                cuda_kernels.append(entry)
            else:
                cpu_ops.append(entry)

        cpu_ops.sort(key=lambda x: x["self_cpu_time_ms"], reverse=True)
        cuda_kernels.sort(key=lambda x: x.get("self_cuda_time_ms", 0), reverse=True)

        return cpu_ops[:20], cuda_kernels[:20]

    def get_metrics(self) -> ProfileMetrics:
        """Get collected profile metrics.

        Returns:
            ProfileMetrics with timing data and hotspot analysis.
        """
        import numpy as np

        metrics = ProfileMetrics()
        metrics.wall_time_per_step_ms = self._step_times.copy()
        metrics.cuda_time_per_step_ms = self._cuda_times.copy()

        if torch.cuda.is_available():
            metrics.gpu_mem_peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

        if self._step_times:
            arr = np.array(self._step_times)
            metrics.hotspots["wall_time_median_ms"] = float(np.median(arr))
            metrics.hotspots["wall_time_p10_ms"] = float(np.percentile(arr, 10))
            metrics.hotspots["wall_time_p90_ms"] = float(np.percentile(arr, 90))
            metrics.hotspots["wall_time_mean_ms"] = float(np.mean(arr))

        if self._cuda_times:
            arr = np.array(self._cuda_times)
            metrics.hotspots["cuda_time_median_ms"] = float(np.median(arr))
            metrics.hotspots["cuda_time_p10_ms"] = float(np.percentile(arr, 10))
            metrics.hotspots["cuda_time_p90_ms"] = float(np.percentile(arr, 90))
            metrics.hotspots["cuda_time_mean_ms"] = float(np.mean(arr))

        return metrics

    def export_report(self) -> dict[str, Any]:
        """Export comprehensive profile report.

        Returns:
            Dict with timing stats, hotspots, and recommendations.
        """
        metrics = self.get_metrics()

        report: dict[str, Any] = {
            "config": {
                "warmup_steps": self.config.warmup_steps,
                "measure_steps": self.config.measure_steps,
                "steps_measured": len(self._step_times),
            },
            "timings": {
                "wall_time_per_step_ms": metrics.wall_time_per_step_ms,
                "cuda_time_per_step_ms": metrics.cuda_time_per_step_ms,
            },
            "summary": dict(metrics.hotspots),
            "memory": {
                "gpu_mem_peak_mb": metrics.gpu_mem_peak_mb,
            },
            "top_cuda_kernels": metrics.top_cuda_kernels,
            "top_cpu_ops": metrics.top_cpu_ops,
        }

        if metrics.hotspots.get("wall_time_median_ms"):
            median_ms = metrics.hotspots["wall_time_median_ms"]
            report["summary"]["steps_per_second"] = 1000.0 / median_ms

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_path = self.config.output_dir / f"profile_report_{timestamp}.parquet"
        df = pl.DataFrame([report])
        FileManager.save(df, report_path)
        logger.info(f"Profile report salvo: {report_path}")

        return report


@contextmanager
def profile_section(name: str) -> Generator[None, None, None]:
    """Lightweight profiling for a named code section using torch.profiler.record_function.

    Use this inside a profiler.step() context to label specific operations.

    Args:
        name: Label for this section (appears in trace).

    Example:
        >>> with profiler.step(step_idx):
        ...     with profile_section("forward"):
        ...         logits = model(x)
        ...     with profile_section("loss"):
        ...         loss = criterion(logits, y)
        ...     with profile_section("backward"):
        ...         loss.backward()
    """
    with torch.profiler.record_function(name):
        yield


def quick_benchmark(
    fn: Callable[[], Any],
    *,
    warmup: int = 10,
    measure: int = 100,
    sync_cuda: bool = True,
) -> dict[str, float]:
    """Quick benchmark for a function.

    Args:
        fn: Function to benchmark (no arguments).
        warmup: Warmup iterations.
        measure: Measurement iterations.
        sync_cuda: Synchronize CUDA before/after (for accurate GPU timing).

    Returns:
        Dict with median, mean, p10, p90 timings in milliseconds.
    """
    import numpy as np

    for _ in range(warmup):
        fn()
        if sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()

    timings = []
    for _ in range(measure):
        if sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        if sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        timings.append((time.perf_counter() - start) * 1000.0)

    arr = np.array(timings)
    return {
        "median_ms": float(np.median(arr)),
        "mean_ms": float(np.mean(arr)),
        "p10_ms": float(np.percentile(arr, 10)),
        "p90_ms": float(np.percentile(arr, 90)),
        "std_ms": float(np.std(arr)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
    }


def get_cuda_allocator_config() -> str:
    """Get recommended PYTORCH_CUDA_ALLOC_CONF for RTX 3070 Ti (8GB VRAM).

    Returns:
        Environment variable string for CUDA allocator configuration.
    """

    config_parts = [
        "garbage_collection_threshold:0.6",
        "expandable_segments:True",
        "max_split_size_mb:512",
    ]
    return ",".join(config_parts)


def apply_cuda_allocator_config() -> None:
    """Apply recommended CUDA allocator configuration.

    Sets PYTORCH_CUDA_ALLOC_CONF environment variable if not already set.
    Must be called BEFORE importing torch.cuda or creating any CUDA tensors.
    """
    env_key = "PYTORCH_CUDA_ALLOC_CONF"
    if env_key not in os.environ:
        config = get_cuda_allocator_config()
        os.environ[env_key] = config
        logger.info(f"CUDA allocator configurado: {config}")
    else:
        logger.debug(f"CUDA allocator ja configurado: {os.environ[env_key]}")


def autotune_chunk_size(
    score_fn: Callable[[int], None],
    candidates: list[int] | None = None,
    *,
    warmup: int = 2,
    measure: int = 5,
    vram_gb: float = 8.0,
) -> tuple[int, dict[int, dict[str, float]]]:
    """Auto-tune score_all_tails_chunk_size for optimal throughput.

    Tests different chunk sizes and selects the fastest one that fits in VRAM.
    Designed for RTX 3070 Ti (8GB) but adapts to available memory.

    Args:
        score_fn: Function that takes chunk_size and performs scoring.
                  Should raise RuntimeError/OOM if chunk_size too large.
        candidates: List of chunk sizes to test. Default: [10000, 20000, 40000, 80000].
        warmup: Warmup iterations per candidate.
        measure: Measurement iterations per candidate.
        vram_gb: Available VRAM in GB (for safety margin calculation).

    Returns:
        Tuple of (best_chunk_size, timing_results).

    Example:
        >>> def score_fn(chunk_size):
        ...     model.score_all_tails_chunked(heads, rels, batch_size=chunk_size)
        >>> best, timings = autotune_chunk_size(score_fn)
        >>> f"Best chunk size: {best}"
    """
    import numpy as np

    if candidates is None:
        if vram_gb <= 6:
            candidates = [5000, 10000, 15000, 20000]
        elif vram_gb <= 8:
            candidates = [10000, 20000, 40000, 80000]
        else:
            candidates = [20000, 40000, 80000, 160000]

    results: dict[int, dict[str, Any]] = {}
    best_chunk = candidates[0]
    best_time = float("inf")

    for chunk_size in candidates:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            for _ in range(warmup):
                score_fn(chunk_size)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

            timings = []
            for _ in range(measure):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start = time.perf_counter()
                score_fn(chunk_size)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                timings.append((time.perf_counter() - start) * 1000.0)

            arr = np.array(timings)
            peak_mem_mb = 0.0
            if torch.cuda.is_available():
                peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

            results[chunk_size] = {
                "median_ms": float(np.median(arr)),
                "mean_ms": float(np.mean(arr)),
                "std_ms": float(np.std(arr)),
                "peak_mem_mb": peak_mem_mb,
                "status": "ok",
            }

            if results[chunk_size]["median_ms"] < best_time:
                best_time = results[chunk_size]["median_ms"]
                best_chunk = chunk_size

            logger.debug(
                f"Chunk size {chunk_size}: {results[chunk_size]['median_ms']:.1f}ms, "
                f"peak_mem={peak_mem_mb:.0f}MB"
            )

        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            results[chunk_size] = {
                "median_ms": float("inf"),
                "mean_ms": float("inf"),
                "std_ms": 0.0,
                "peak_mem_mb": 0.0,
                "status": f"oom: {e}",
            }
            logger.warning(f"Chunk size {chunk_size}: OOM - skipping")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    logger.info(
        f"Ajuste automatico de chunk: melhor={best_chunk} ({best_time:.1f}ms), "
        f"testados={list(results.keys())}"
    )

    return best_chunk, results


def get_optimal_chunk_size_for_vram(
    num_entities: int,
    embedding_dim: int,
    vram_gb: float = 8.0,
    *,
    dtype_bytes: int = 4,
    safety_margin: float = 0.7,
) -> int:
    """Estimate optimal chunk size based on VRAM constraints.

    Uses a memory model to estimate the maximum chunk size that fits
    in available VRAM with a safety margin.

    Args:
        num_entities: Total number of entities.
        embedding_dim: Embedding dimension.
        vram_gb: Available VRAM in GB.
        dtype_bytes: Bytes per element (4 for float32, 2 for float16).
        safety_margin: Fraction of VRAM to use (0.7 = 70%).

    Returns:
        Recommended chunk size.
    """

    vram_bytes = vram_gb * 1024 * 1024 * 1024
    usable_bytes = vram_bytes * safety_margin

    batch_size = 256

    bytes_per_chunk_element = (2 * embedding_dim * dtype_bytes + batch_size * dtype_bytes) * 2

    max_chunk = int(usable_bytes / bytes_per_chunk_element)

    chunk_size = (max_chunk // 10000) * 10000
    chunk_size = max(10000, min(chunk_size, num_entities))

    return chunk_size


__all__ = [
    "ProfileConfig",
    "ProfileMetrics",
    "DSLFMProfiler",
    "profile_section",
    "quick_benchmark",
    "get_cuda_allocator_config",
    "apply_cuda_allocator_config",
    "autotune_chunk_size",
    "get_optimal_chunk_size_for_vram",
]
