import os
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv

TARGET = "data/models/correct.parquet"


def time_func(func, *args):
    times = []

    try:
        func(*args)
    except Exception:
        return float("inf")

    for _ in range(15):
        start = time.perf_counter()
        func(*args)
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def setup_data():
    if not Path(TARGET).exists():
        print(f"Target {TARGET} not found")
        return None, None, None

    df = pl.read_parquet(TARGET)

    if "s" not in df.columns and len(df.columns) > 0:
        df = df.with_columns(pl.col(df.columns[0]).cast(pl.Utf8).alias("s"))

    target_rows = 500_000
    current_rows = len(df)
    if current_rows < target_rows:
        factor = (target_rows // current_rows) + 1
        df = pl.concat([df] * factor)

    table = df.to_arrow()

    if isinstance(table, pa.Table):
        pass

    np_s = df["s"].to_numpy()

    print(f"Dataset Size: {len(df):,} rows")
    return df, table, np_s


def bench_unique_polars(df):
    return df["s"].unique()


def bench_unique_arrow(table):
    return pc.unique(table["s"])


def bench_unique_numpy(arr):
    return np.unique(arr)


def bench_sort_polars(df):
    return df.sort("s")


def bench_sort_arrow(table):
    indices = pc.sort_indices(table["s"])
    return table.take(indices)


def bench_sort_numpy(arr):
    return np.sort(arr)


def bench_concat_polars(df):
    return pl.concat([df, df])


def bench_concat_arrow(table):
    return pa.concat_tables([table, table])


def bench_concat_numpy(arr):
    return np.concatenate([arr, arr])


def bench_filter_polars(df):
    return df.filter(pl.col("s").str.contains("a", literal=True))


def bench_filter_arrow(table):
    return table.filter(pc.match_substring(table["s"], "a"))


def bench_filter_numpy(arr):
    return arr[np.char.find(arr.astype(str), "a") != -1]


def bench_csv_polars(path):
    return pl.read_csv(path)


def bench_csv_arrow(path):
    return pacsv.read_csv(path)


def bench_csv_numpy(path):
    return np.genfromtxt(path, delimiter=",", dtype=None, encoding="utf-8")


def run_suite():
    print("--- Benchmark Estendido (Polars vs PyArrow vs Numpy) ---")
    df, table, np_arr = setup_data()
    if df is None:
        return

    print("\n[Unique (Strings)]")
    t_pl = time_func(bench_unique_polars, df)
    t_pa = time_func(bench_unique_arrow, table)
    t_np = time_func(bench_unique_numpy, np_arr)
    print(f"Polars:  {t_pl:.5f}s")
    print(f"Arrow:   {t_pa:.5f}s")
    print(f"Numpy:   {t_np:.5f}s")

    print("\n[Sort (Strings)]")
    t_pl = time_func(bench_sort_polars, df)
    t_pa = time_func(bench_sort_arrow, table)
    t_np = time_func(bench_sort_numpy, np_arr)
    print(f"Polars:  {t_pl:.5f}s")
    print(f"Arrow:   {t_pa:.5f}s")
    print(f"Numpy:   {t_np:.5f}s")

    print("\n[Concat (2x Data)]")
    t_pl = time_func(bench_concat_polars, df)
    t_pa = time_func(bench_concat_arrow, table)
    t_np = time_func(bench_concat_numpy, np_arr)
    print(f"Polars:  {t_pl:.5f}s")
    print(f"Arrow:   {t_pa:.5f}s")
    print(f"Numpy:   {t_np:.5f}s")

    print("\n[Filter (String Contains 'a')]")
    t_pl = time_func(bench_filter_polars, df)
    t_pa = time_func(bench_filter_arrow, table)
    t_np = time_func(bench_filter_numpy, np_arr)
    print(f"Polars:  {t_pl:.5f}s")
    print(f"Arrow:   {t_pa:.5f}s")
    print(f"Numpy:   {t_np:.5f}s")

    print("\n[CSV Read]")
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        df.write_csv(tmp.name)
        tmp_path = tmp.name

    try:
        t_pl = time_func(bench_csv_polars, tmp_path)
        t_pa = time_func(bench_csv_arrow, tmp_path)

        t_np = float("inf")

        print(f"Polars:  {t_pl:.5f}s")
        print(f"Arrow:   {t_pa:.5f}s")
        print(f"Numpy:   {t_np:.5f}s (Skipped/Slow)")
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    run_suite()
