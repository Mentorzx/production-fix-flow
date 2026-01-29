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

import polars as pl

from pff.shared.core.file_manager.handlers.csv import CSVHandler


def _pin_threads() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


def _build_frame(rows: int) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "a": list(range(rows)),
            "b": ["x"] * rows,
        }
    )


async def _save_many(handler: CSVHandler, df: pl.DataFrame, path: Path, iterations: int) -> None:
    for _ in range(iterations):
        await handler.async_save(df, path)


def bench_async_save(
    *, rows: int, iterations: int, repeats: int, output_path: Path
) -> dict[str, Any]:
    durations: list[float] = []
    handler = CSVHandler()
    df = _build_frame(rows)
    for _ in range(repeats):
        start = time.perf_counter()
        asyncio.run(_save_many(handler, df, output_path, iterations))
        durations.append(time.perf_counter() - start)
    median = statistics.median(durations) if durations else 0.0
    per_call_ms = (median / max(iterations, 1)) * 1000.0
    return {
        "iterations": iterations,
        "repeats": repeats,
        "rows": rows,
        "durations_s": durations,
        "median_s": median,
        "per_call_ms": per_call_ms,
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
        f"rows={bench['rows']}",
        f"median_s={bench['median_s']:.6f}",
        f"per_call_ms={bench['per_call_ms']:.6f}",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="CSV async_save benchmark")
    parser.add_argument("--rows", type=int, default=50000)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-text", required=True)
    parser.add_argument("--data-path", default="outputs/benches/async_save.csv")
    args = parser.parse_args()

    _pin_threads()

    output_path = Path(args.data_path)
    bench = bench_async_save(
        rows=args.rows,
        iterations=args.iterations,
        repeats=args.repeats,
        output_path=output_path,
    )
    result = _build_result({"csv_async_save": bench})

    output_path.unlink(missing_ok=True)
    output_path.with_suffix(".csv.lock").unlink(missing_ok=True)

    from pff.shared.core.file_manager import FileManager

    FileManager.write_text(FileManager.json_dumps(result, sort_keys=True), args.output_json)
    FileManager.write_text(_format_text("csv_async_save", result), args.output_text)


if __name__ == "__main__":
    main()
