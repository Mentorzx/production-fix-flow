from __future__ import annotations

import argparse
import gc
import os
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

from pff.shared.core.file_manager import FileManager, ParquetBundle
from pff.shared.core.file_manager.materializers.implementations import ContainerMaterializer


def _pin_threads() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


def _build_bundle(root: Path, rows: int) -> ParquetBundle:
    parsed_path = root / "container_parsed.parquet"
    raw_path = root / "container_raw.parquet"
    FileManager.ensure_dir(root)

    names = [f"entry_{idx}" for idx in range(rows)]
    texts = [f"payload_{idx}" for idx in range(rows)]

    df = pl.DataFrame(
        {
            "entry_name": names,
            "entry_ext": [".txt"] * rows,
            "payload_kind": ["text"] * rows,
            "payload_msgpack": pl.Series("payload_msgpack", [None] * rows, dtype=pl.Binary),
            "payload_text": texts,
            "payload_bytes": pl.Series("payload_bytes", [None] * rows, dtype=pl.Binary),
            "payload_parquet_path": pl.Series("payload_parquet_path", [None] * rows, dtype=pl.Utf8),
        }
    )
    FileManager.save(df, parsed_path)

    raw_df = pl.DataFrame({"chunk_bytes": [b""]})
    FileManager.save(raw_df, raw_path)

    return ParquetBundle(
        source_path=root / "source.zip",
        ext=".zip",
        file_id="bench",
        raw_parquet_path=raw_path,
        parsed_parquet_path=parsed_path,
        parsed_kind="container",
    )


def bench_materialize(bundle: ParquetBundle, *, iterations: int, repeats: int) -> dict[str, Any]:
    durations: list[float] = []
    total_entries = 0
    materializer = ContainerMaterializer()
    for _ in range(repeats):
        start = time.perf_counter()
        total_entries = 0
        for _ in range(iterations):
            result = materializer.materialize(bundle)
            total_entries += len(result)
        durations.append(time.perf_counter() - start)
        gc.collect()
    median = statistics.median(durations) if durations else 0.0
    per_call_ms = (median / max(iterations, 1)) * 1000.0
    entries_per_sec = (total_entries / median) if median else 0.0
    return {
        "iterations": iterations,
        "repeats": repeats,
        "durations_s": durations,
        "median_s": median,
        "per_call_ms": per_call_ms,
        "total_entries": total_entries,
        "entries_per_sec": entries_per_sec,
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
        f"total_entries={bench['total_entries']}",
        f"entries_per_sec={bench['entries_per_sec']:.2f}",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Container materializer benchmark")
    parser.add_argument("--rows", type=int, default=2000)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-text", required=True)
    parser.add_argument("--work-dir", default="outputs/benches/container_materializer")
    args = parser.parse_args()

    _pin_threads()

    work_dir = Path(args.work_dir)
    bundle = _build_bundle(work_dir, args.rows)
    bench = bench_materialize(bundle, iterations=args.iterations, repeats=args.repeats)
    result = _build_result({"container_materialize": bench})

    FileManager.write_text(FileManager.json_dumps(result, sort_keys=True), args.output_json)
    FileManager.write_text(_format_text("container_materialize", result), args.output_text)


if __name__ == "__main__":
    main()
