import subprocess
import time
from pathlib import Path

import polars as pl

from pff.shared import logger
from pff.shared.core.file_manager import FileManager


def run_python_bench(path_str, iterations=100):
    logger.info(f"Iniciando Benchmark Python ({iterations} iterações)...")
    path = Path(path_str)
    file_size_mb = path.stat().st_size / 1024 / 1024
    fm = FileManager()

    latencies = []
    for i in range(iterations):
        start = time.perf_counter()
        df = fm.read(path)

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

    py_results = run_python_bench(target_file)

    logger.info("Compilando e executando benchmarks em Rust...")
    fm = FileManager()
    try:
        rust_bin = "./scripts/rust_bench/target/release/parquet_bench"
        subprocess.run([rust_bin], check=True)

        rust_results = fm.read("outputs/benches/rust_parquet_results.json", return_native=True)
    except Exception as e:
        logger.error(f"Benchmark Rust falhou: {e}")
        rust_results = []

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

    logger.info("\n" + report)

    fm.save(report, "outputs/benches/final_report.md")
    logger.info("Relatório final salvo em outputs/benches/final_report.md")


if __name__ == "__main__":
    main()
