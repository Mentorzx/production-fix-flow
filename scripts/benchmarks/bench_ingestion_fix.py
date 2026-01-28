import time
from pathlib import Path
import polars as pl
from pff.shared.core.file_manager.handlers.parquet import iter_parquet_as_json


def bench_ingestion_iteration():
    print("Benchmarking Ingestion Iteration (After Fix)...")

    n_rows = 100_000
    mock_dir = Path("outputs/benches/mock_ingestion_fix")
    mock_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = mock_dir / "mock_telecom.parquet"

    # Create structured parquet (optimized format)
    df = pl.DataFrame(
        {
            "id": [f"id_{i}" for i in range(n_rows)],
            "externalId": [f"ext_{i}" for i in range(n_rows)],
            "status": ["active"] * n_rows,
            "account": [{"id": f"acc_{i}"} for i in range(n_rows)],
        }
    )
    df.write_parquet(parquet_path)

    # Bench iter_parquet_as_json (Structured path)
    start = time.perf_counter()
    count = 0
    for _ in iter_parquet_as_json(parquet_path, batch_size=10000, prefer_struct=True):
        count += 1
    end = time.perf_counter()

    print(f"Iterate structured ({n_rows} rows): {(end - start):.4f} s")
    print(f"Throughput: {n_rows / (end - start):.2f} rows/sec")

    import shutil

    shutil.rmtree(mock_dir)


if __name__ == "__main__":
    bench_ingestion_iteration()
