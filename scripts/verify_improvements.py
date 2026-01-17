import sys
import types
from pathlib import Path

# Add repo root to path
sys.path.append(str(Path.cwd()))

# Mock broken/missing modules to allow importing FileManager
sys.modules["pff.celery_app"] = types.ModuleType("pff.celery_app")
sys.modules["pff.config"] = types.ModuleType("pff.config")
sys.modules["pff.config"].settings = None  # Mock settings if needed

import time
import json
import subprocess
import statistics
import polars as pl
from pff.shared.core.file_manager import FileManager

# Ensure we are using the optimized handlers
from pff.shared.core.file_manager.handlers.json import JSONHandler
from pff.shared.core.file_manager.handlers.parquet import ParquetHandler


def run_python_bench(path_str, iterations=100):
    print(f"Starting Python Benchmark (New Baseline) - {iterations} iterations...")
    path = Path(path_str)
    file_size_mb = path.stat().st_size / 1024 / 1024

    # Initialize manager (will use new handlers)
    fm = FileManager()

    print(f"Handler for parquet: {type(fm._handlers['.parquet'])}")
    print(f"Handler for json: {type(fm._handlers['.json'])}")

    latencies = []
    # Warmup
    print("Warming up...")
    for _ in range(5):
        fm.read(path)

    print("Running...")
    for i in range(iterations):
        start = time.perf_counter()
        df = fm.read(path)
        # Force materialization if lazy (though read defaults to eager now)
        if isinstance(df, pl.LazyFrame):
            df = df.collect()
        latencies.append(time.perf_counter() - start)

    latencies.sort()
    mean_lat = sum(latencies) / iterations

    return {
        "name": "Python_Optimized_FileManager",
        "iterations": iterations,
        "mean": {"total_s": mean_lat},
        "p50": {"total_s": latencies[iterations // 2]},
        "p95": {"total_s": latencies[int(iterations * 0.95)]},
        "throughput_mbs": file_size_mb / mean_lat,
    }


def main():
    target_file = "data/models/correct.parquet"
    if not Path(target_file).exists():
        print(f"Error: {target_file} not found.")
        return

    # 1. Run Python Bench
    py_results = run_python_bench(target_file)

    # 2. Run Rust Bench
    print("\nRunning Rust benchmarks for comparison...")
    rust_results = []
    try:
        rust_bin = "./scripts/rust_bench/target/release/parquet_bench"
        if not Path(rust_bin).exists():
            print("Rust binary not found, attempting to build...")
            subprocess.run(
                ["cargo", "build", "--release", "--bin", "parquet_bench"],
                cwd="scripts/rust_bench",
                check=True,
            )

        subprocess.run([rust_bin], check=True)

        with open("outputs/benches/rust_parquet_results.json", "r") as f:
            rust_results = json.load(f)
    except Exception as e:
        print(f"Rust benchmark failed: {e}")

    # 3. Combine and Format
    all_results = [py_results] + rust_results

    print("\n" + "=" * 80)
    print(
        f"{'Implementation':<30} | {'Mean (ms)':<10} | {'p50 (ms)':<10} | {'p95 (ms)':<10} | {'MB/s':<10}"
    )
    print("-" * 80)

    for res in all_results:
        name = res["name"]
        mean = res["mean"]["total_s"] * 1000
        p50 = res["p50"]["total_s"] * 1000
        p95 = res["p95"]["total_s"] * 1000
        tp = res["throughput_mbs"]
        print(f"{name:<30} | {mean:<10.2f} | {p50:<10.2f} | {p95:<10.2f} | {tp:<10.2f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
