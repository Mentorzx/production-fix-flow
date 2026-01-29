from __future__ import annotations

import argparse
import os
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

from pff.shared.core.file_manager import FileManager
from pff.shared.core.file_manager.utils import read_raw_bytes


def _pin_threads() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


def _prepare_parquet(path: Path, *, chunk_size: int, chunks: int) -> int:
    if path.exists():
        return path.stat().st_size
    payload = b"a" * chunk_size
    data = [payload] * chunks
    df = pl.DataFrame({"chunk_bytes": pl.Series("chunk_bytes", data, dtype=pl.Binary)})
    FileManager.save(df, path)
    return path.stat().st_size


def bench_read_raw_bytes(path: Path, *, iterations: int, repeats: int) -> dict[str, Any]:
    durations: list[float] = []
    total_bytes = 0
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(iterations):
            total_bytes = len(read_raw_bytes(path))
        durations.append(time.perf_counter() - start)
    median = statistics.median(durations) if durations else 0.0
    per_call_ms = (median / max(iterations, 1)) * 1000.0
    mb_per_sec = (total_bytes / median / (1024 * 1024)) if median else 0.0
    return {
        "iterations": iterations,
        "repeats": repeats,
        "durations_s": durations,
        "median_s": median,
        "per_call_ms": per_call_ms,
        "total_bytes": total_bytes,
        "mb_per_sec": mb_per_sec,
    }


def _build_result(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "benchmarks": payload,
    }


def _format_text(name: str, result: dict[str, Any]) -> str:
    bench = result["benchmarks"][name]
    lines = [
        f"benchmark={name}",
        f"iterations={bench['iterations']}",
        f"repeats={bench['repeats']}",
        f"median_s={bench['median_s']:.6f}",
        f"per_call_ms={bench['per_call_ms']:.6f}",
        f"total_bytes={bench['total_bytes']}",
        f"mb_per_sec={bench['mb_per_sec']:.2f}",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="read_raw_bytes benchmark")
    parser.add_argument("--chunk-size", type=int, default=65536)
    parser.add_argument("--chunks", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-text", required=True)
    parser.add_argument("--data-path", default="outputs/benches/raw_bytes.parquet")
    args = parser.parse_args()

    _pin_threads()

    data_path = Path(args.data_path)
    _prepare_parquet(data_path, chunk_size=args.chunk_size, chunks=args.chunks)
    bench = bench_read_raw_bytes(data_path, iterations=args.iterations, repeats=args.repeats)
    result = _build_result({"read_raw_bytes": bench})

    FileManager.write_text(FileManager.json_dumps(result, sort_keys=True), args.output_json)
    FileManager.write_text(_format_text("read_raw_bytes", result), args.output_text)


if __name__ == "__main__":
    main()
