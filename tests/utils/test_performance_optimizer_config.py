from pathlib import Path

from pff import settings
from pff.utils.performance.performance import (
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
