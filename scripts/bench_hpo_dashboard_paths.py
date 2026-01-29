from __future__ import annotations

import argparse
import os
import platform
import statistics
import time
from datetime import datetime, timezone
from typing import Any

from pff.infrastructure.hpo.dashboard import server as dashboard_server
from pff.shared.core.file_manager import FileManager


def _pin_threads() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


def _collect_paths_once(iterations: int) -> tuple[float, int]:
    total_paths = 0
    start = time.perf_counter()
    for _ in range(iterations):
        total_paths += len(dashboard_server._collect_dashboard_data_paths())
    elapsed = time.perf_counter() - start
    return elapsed, total_paths


def bench_collect_paths(iterations: int, repeats: int) -> dict[str, Any]:
    durations: list[float] = []
    total_paths = 0
    for _ in range(repeats):
        elapsed, total_paths = _collect_paths_once(iterations)
        durations.append(elapsed)
    median = statistics.median(durations) if durations else 0.0
    per_call_ms = (median / max(iterations, 1)) * 1000.0
    calls_per_sec = (max(iterations, 1) / median) if median else 0.0
    return {
        "iterations": iterations,
        "repeats": repeats,
        "durations_s": durations,
        "median_s": median,
        "per_call_ms": per_call_ms,
        "calls_per_sec": calls_per_sec,
        "paths_per_call": int(total_paths / max(repeats, 1) / max(iterations, 1)),
    }


def _build_result(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "benchmarks": payload,
        "runtime_env": {
            "os": "WSL Linux",
            "python": "3.12",
            "cpu": "AMD Ryzen 5 5600X 6C12T",
            "ram_gb": 32,
            "gpu": "NVIDIA RTX 3070 Ti 8GB",
            "cuda": "13.1",
        },
    }


def _format_text(name: str, result: dict[str, Any]) -> str:
    bench = result["benchmarks"][name]
    lines = [
        f"benchmark={name}",
        f"iterations={bench['iterations']}",
        f"repeats={bench['repeats']}",
        f"median_s={bench['median_s']:.6f}",
        f"per_call_ms={bench['per_call_ms']:.6f}",
        f"calls_per_sec={bench['calls_per_sec']:.2f}",
        f"paths_per_call={bench['paths_per_call']}",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="HPO dashboard collect-paths benchmark")
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-text", required=True)
    args = parser.parse_args()

    _pin_threads()

    benchmark = bench_collect_paths(args.iterations, args.repeats)
    result = _build_result({"collect_dashboard_paths": benchmark})

    FileManager.write_text(FileManager.json_dumps(result, sort_keys=True), args.output_json)
    FileManager.write_text(_format_text("collect_dashboard_paths", result), args.output_text)


if __name__ == "__main__":
    main()
