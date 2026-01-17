import time
import json
import statistics
import subprocess
from pathlib import Path
import polars as pl
from pff.shared.core.file_manager import FileManager


def run_python_bench(path_str, iterations=100):
    print(f"Starting Python Benchmark ({iterations} iterations)...")
    path = Path(path_str)
    file_size_mb = path.stat().st_size / 1024 / 1024
    fm = FileManager()

    latencies = []
    for i in range(iterations):
        start = time.perf_counter()
        df = fm.read(path)
        # Force materialization if lazy
        if isinstance(df, pl.LazyFrame):
            df = df.collect()
        latencies.append(time.perf_counter() - start)

    latencies.sort()
    mean_lat = sum(latencies) / iterations

    return {
        "name": "Python_FileManager_Polars",
        "iterations": iterations,
        "mean": {"load_s": 0.0, "parse_s": 0.0, "total_s": mean_lat},
        "p50": {"load_s": 0.0, "parse_s": 0.0, "total_s": latencies[iterations // 2]},
        "p95": {"load_s": 0.0, "parse_s": 0.0, "total_s": latencies[int(iterations * 0.95)]},
        "p99": {"load_s": 0.0, "parse_s": 0.0, "total_s": latencies[int(iterations * 0.99)]},
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
    print("Building and running Rust benchmarks...")
    try:
        # Run from root
        rust_bin = "./scripts/rust_bench/target/release/parquet_bench"
        subprocess.run([rust_bin], check=True)

        with open("outputs/benches/rust_parquet_results.json", "r") as f:
            rust_results = json.load(f)
    except Exception as e:
        print(f"Rust benchmark failed: {e}")
        rust_results = []

    # 3. Combine and Format
    all_results = [py_results] + rust_results

    report = "# Parquet File Manager Benchmark Report\n\n"
    report += (
        f"**Target:** `{target_file}` ({Path(target_file).stat().st_size / 1024 / 1024:.2f} MB)\n"
    )
    report += "**Iterations:** 100 per manager\n\n"

    report += "| Manager | Mean Total (ms) | p50 (ms) | p95 (ms) | Throughput (MB/s) |\n"
    report += "| :--- | :---: | :---: | :---: | :---: |\n"

    for res in all_results:
        name = res["name"]
        mean = res["mean"]["total_s"] * 1000
        p50 = res["p50"]["total_s"] * 1000
        p95 = res["p95"]["total_s"] * 1000
        tp = res["throughput_mbs"]
        report += f"| {name} | {mean:.2f} | {p50:.2f} | {p95:.2f} | {tp:.2f} |\n"

    print("\n" + report)

    with open("outputs/benches/final_report.md", "w") as f:
        f.write(report)
    print(f"Final report saved to outputs/benches/final_report.md")


if __name__ == "__main__":
    main()
