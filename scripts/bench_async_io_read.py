from __future__ import annotations

import argparse
import asyncio
import os
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pff.shared.core.file_manager import FileManager
from pff.shared.core.file_manager.async_io import read_async_content


def _pin_threads() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


def _prepare_file(path: Path, size_bytes: int) -> None:
    if path.exists() and path.stat().st_size == size_bytes:
        return
    payload = b"0" * size_bytes
    FileManager.write_bytes(payload, path)


async def _read_many(path: Path, chunk_size: int | None, iterations: int) -> int:
    total_bytes = 0
    for _ in range(iterations):
        content = await read_async_content(path, chunk_size=chunk_size)
        total_bytes += len(content)
    return total_bytes


def bench_async_read(
    path: Path, *, chunk_size: int | None, iterations: int, repeats: int
) -> dict[str, Any]:
    durations: list[float] = []
    total_bytes = 0
    for _ in range(repeats):
        start = time.perf_counter()
        total_bytes = asyncio.run(_read_many(path, chunk_size, iterations))
        durations.append(time.perf_counter() - start)
    median = statistics.median(durations) if durations else 0.0
    per_call_ms = (median / max(iterations, 1)) * 1000.0
    bytes_per_sec = (total_bytes / median) if median else 0.0
    mb_per_sec = bytes_per_sec / (1024 * 1024)
    return {
        "iterations": iterations,
        "repeats": repeats,
        "durations_s": durations,
        "median_s": median,
        "per_call_ms": per_call_ms,
        "total_bytes": total_bytes,
        "mb_per_sec": mb_per_sec,
        "chunk_size": chunk_size,
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
        f"chunk_size={bench['chunk_size']}",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Async file read benchmark")
    parser.add_argument("--size-mb", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=65536)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-text", required=True)
    parser.add_argument("--data-path", default="outputs/benches/async_io_payload.bin")
    args = parser.parse_args()

    _pin_threads()

    payload_path = Path(args.data_path)
    size_bytes = int(args.size_mb) * 1024 * 1024
    _prepare_file(payload_path, size_bytes)

    bench = bench_async_read(
        payload_path,
        chunk_size=args.chunk_size,
        iterations=args.iterations,
        repeats=args.repeats,
    )
    result = _build_result({"async_read": bench})

    FileManager.write_text(FileManager.json_dumps(result, sort_keys=True), args.output_json)
    FileManager.write_text(_format_text("async_read", result), args.output_text)


if __name__ == "__main__":
    main()
