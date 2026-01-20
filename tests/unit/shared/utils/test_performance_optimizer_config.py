from pathlib import Path

from pff import settings
from pff.infrastructure.performance import (
    AdvancedCompilationBackend,
    CompilationProfiler,
    _load_performance_config,
)


def test_performance_config_uses_yaml_paths() -> None:
    cfg = _load_performance_config()
    perf_cfg = cfg["performance"]

    backend_order = perf_cfg["backends"]["order"]
    profiler = CompilationProfiler()
    resolved_dir = perf_cfg["compilation_logs_dir"]
    expected_dir = (
        Path(resolved_dir)
        if Path(resolved_dir).is_absolute()
        else (settings.ROOT_DIR / Path(resolved_dir))
    )

    assert backend_order, "Expected at least one backend order entry"
    assert profiler.output_dir == expected_dir


def test_backend_order_respected() -> None:
    cfg = _load_performance_config()["performance"]
    order = cfg["backends"]["order"]
    backend = AdvancedCompilationBackend()
    assert backend._backend_order == order


def test_file_io_streaming_thresholds_configured() -> None:
    cfg = _load_performance_config()["performance"]
    file_io = cfg.get("file_io", {})
    thresholds = file_io.get("streaming_thresholds", {})
    assert thresholds.get("low_ram_gb") is not None
    assert thresholds.get("mid_ram_gb") is not None
    assert thresholds.get("low_ram_mb") is not None
    assert thresholds.get("mid_ram_mb") is not None
    assert thresholds.get("high_ram_mb") is not None


def test_file_io_parquet_first_configured() -> None:
    cfg = _load_performance_config()["performance"]
    file_io = cfg.get("file_io", {})
    parquet_first = file_io.get("parquet_first", {})
    assert parquet_first.get("raw_chunk_mb") is not None
    assert parquet_first.get("parsed_row_group_size") is not None
    assert parquet_first.get("container_flush_rows") is not None
    assert parquet_first.get("compression") is not None
    assert parquet_first.get("compression_level") is not None
    assert parquet_first.get("cache_dir") is not None
