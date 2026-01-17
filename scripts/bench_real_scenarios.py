import time
import orjson
import msgspec
import polars as pl
import pyarrow.parquet as pq
import pyarrow.compute as pc
import statistics
import numpy as np
from pathlib import Path

TARGET = "data/models/correct.parquet"


# --- Setup ---
def setup():
    if not Path(TARGET).exists():
        print(f"Target {TARGET} not found")
        return None
    df = pl.read_parquet(TARGET)
    # Ensure columns exist for tests
    if "s" not in df.columns and len(df.columns) > 0:
        df = df.with_columns(pl.col(df.columns[0]).cast(pl.Utf8).alias("s"))
    return df


def time_func(func, arg):
    times = []
    # Warmup
    try:
        func(arg)
    except Exception as e:
        print(f"Skipping {func.__name__}: {e}")
        return 0.001

    for _ in range(20):
        start = time.perf_counter()
        func(arg)
        times.append(time.perf_counter() - start)
    return statistics.median(times)


# --- Scenarios ---


# 1. Builder Cleaning
def bench_builder_clean_loop(df):
    # Current: iter rows + python clean
    return [str(x).replace("\t", " ").strip() for x in df["s"]]


def bench_builder_clean_vectorized(df):
    # Proposed: Polars Expr
    # strip_chars() is Polars 0.19+, strip() is older. Using strip_chars based on recent Polars.
    return df.select(pl.col("s").str.replace("\t", " ").str.strip_chars()).to_series()


def bench_builder_clean_arrow(df):
    # Arrow Compute
    col = df.to_arrow()["s"]
    return pc.utf8_trim(pc.replace_substring(col, "\t", " "))


# 2. Data Loader
def bench_loader_iter_rows(df):
    # Current: iter_rows
    # Simulate selecting 3 cols if possible
    cols = df.columns[:3]
    return [list(row) for row in df.select(cols).iter_rows()]


def bench_loader_numpy(df):
    # Proposed: Numpy
    cols = df.columns[:3]
    return df.select(cols).to_numpy().tolist()


def bench_loader_arrow(df):
    # Proposed: Arrow
    cols = df.columns[:3]
    return df.select(cols).to_arrow().to_pylist()


# 3. Serialization
def bench_serialization_polars_msgspec(path):
    # Current: Polars -> Dicts -> Msgspec
    df = pl.read_parquet(path)
    encoder = msgspec.json.Encoder()
    return [encoder.encode(row) for row in df.to_dicts()]


def bench_serialization_arrow_orjson(path):
    # Proposed: Arrow -> Pylist -> Orjson
    table = pq.read_table(path)
    return [orjson.dumps(row) for row in table.to_pylist()]


# --- Runner ---
def run_suite():
    print(f"--- Benchmark Real (LZ4 + Mimalloc) ---")
    df = setup()
    if df is None:
        return

    # inflate data for better measurement if small
    if len(df) < 10000:
        df = pl.concat([df] * 10)

    print(f"Dataset Size: {len(df)} rows")

    # 1. Builder Cleaning
    print("\n[Builder Cleaning]")
    t_loop = time_func(bench_builder_clean_loop, df)
    t_vec = time_func(bench_builder_clean_vectorized, df)
    t_arr = time_func(bench_builder_clean_arrow, df)

    print(f"Loop (Python):   {t_loop:.5f}s")
    print(f"Polars (Vector): {t_vec:.5f}s  (Speedup: {t_loop / t_vec:.1f}x)")
    print(f"Arrow (Compute): {t_arr:.5f}s  (Speedup: {t_loop / t_arr:.1f}x)")

    # 2. Data Loader
    print("\n[Data Loader List Conversion]")
    t_iter = time_func(bench_loader_iter_rows, df)
    t_npy = time_func(bench_loader_numpy, df)
    t_arrow_list = time_func(bench_loader_arrow, df)

    print(f"iter_rows():     {t_iter:.5f}s")
    print(f"Numpy tolist():  {t_npy:.5f}s  (Speedup: {t_iter / t_npy:.1f}x)")
    print(f"Arrow to_pylist: {t_arrow_list:.5f}s  (Speedup: {t_iter / t_arrow_list:.1f}x)")

    # 3. Serialization
    print("\n[JSON Serialization]")
    t_pol = time_func(bench_serialization_polars_msgspec, TARGET)
    t_arr_ser = time_func(bench_serialization_arrow_orjson, TARGET)

    print(f"Polars+Msgspec:  {t_pol:.5f}s")
    print(f"Arrow+Orjson:    {t_arr_ser:.5f}s  (Speedup: {t_pol / t_arr_ser:.1f}x)")


if __name__ == "__main__":
    run_suite()
