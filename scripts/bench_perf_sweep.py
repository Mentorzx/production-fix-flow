from __future__ import annotations

import asyncio
import importlib.util
import json
import time
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import torch

from pff.domain.kg.builder import KGBuilder
from pff.infrastructure.hpo.trials import data_loader
from pff.shared import FileManager
from pff.shared.system.resource_manager import HardwareDetector


def _measure(fn, runs: int = 3, warmup: int = 1) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    times_ms = [t * 1000 for t in times]
    times_ms.sort()
    mid = len(times_ms) // 2
    if len(times_ms) % 2 == 0:
        median = (times_ms[mid - 1] + times_ms[mid]) / 2.0
    else:
        median = times_ms[mid]
    return {
        "median_ms": float(median),
        "min_ms": float(times_ms[0]),
        "max_ms": float(times_ms[-1]),
        "runs": float(runs),
    }


def _as_lazy_frame(payload: object) -> pl.LazyFrame:
    if isinstance(payload, pl.LazyFrame):
        return payload
    if isinstance(payload, pl.DataFrame):
        return payload.lazy()
    from pff.shared.core.file_manager.bundles import ParquetBundle

    if isinstance(payload, ParquetBundle):
        return payload.lazyframe()
    raise ValueError(f"Unsupported payload type: {type(payload)}")


def _as_dataframe(payload: object) -> pl.DataFrame:
    if isinstance(payload, pl.DataFrame):
        return payload
    if isinstance(payload, pl.LazyFrame):
        return payload.collect(engine="streaming")
    from pff.shared.core.file_manager.bundles import ParquetBundle

    if isinstance(payload, ParquetBundle):
        return payload.lazyframe().collect(engine="streaming")
    raise ValueError(f"Unsupported payload type: {type(payload)}")


def _select_triplet_columns(lf: pl.LazyFrame) -> pl.LazyFrame:
    schema = lf.collect_schema()
    simple_types = {
        pl.Utf8,
        pl.Int64,
        pl.Int32,
        pl.UInt64,
        pl.UInt32,
        pl.Float64,
        pl.Float32,
        pl.Boolean,
    }
    simple_cols = [name for name, dtype in schema.items() if dtype in simple_types]
    if all(name in schema for name in ("s", "p", "o")):
        return lf.select(["s", "p", "o"])
    if len(simple_cols) >= 3:
        return lf.select(simple_cols[:3])
    cols = list(schema.keys())
    if len(cols) >= 3:
        return lf.select(cols[:3])
    return lf


def bench_parquet_scan(path: Path, n_rows: int) -> dict[str, float]:
    fm = FileManager()

    def run() -> None:
        payload = fm.read(path)
        lf = _as_lazy_frame(payload)
        lf = _select_triplet_columns(lf)
        if n_rows > 0:
            lf = lf.head(n_rows)
        lf.collect(engine="streaming")

    return _measure(run)


def bench_pyarrow_dataset_scan(path: Path, n_rows: int) -> dict[str, float]:
    dataset = ds.dataset(str(path), format="parquet")
    columns = list(dataset.schema.names)
    preferred = [c for c in ("id", "externalId", "_source_name") if c in columns]
    select_cols = preferred if preferred else columns[:3]

    def run() -> None:
        scanner = dataset.scanner(columns=select_cols)
        if n_rows > 0:
            scanner.head(n_rows)
        else:
            scanner.to_table()

    return _measure(run)


def _select_parquet_columns(column_names: list[str]) -> list[str]:
    if all(name in column_names for name in ("s", "p", "o")):
        return ["s", "p", "o"]
    if all(name in column_names for name in ("head", "relation", "tail")):
        return ["head", "relation", "tail"]
    return column_names[:3] if len(column_names) >= 3 else column_names


def bench_pyarrow_iter_batches(path: Path, n_rows: int) -> dict[str, float]:
    parquet_file = pq.ParquetFile(path)
    columns = _select_parquet_columns(list(parquet_file.schema_arrow.names))
    batch_size = 4096

    def run() -> None:
        total = 0
        for batch in parquet_file.iter_batches(columns=columns, batch_size=batch_size):
            df = pl.from_arrow(batch, rechunk=False)
            if n_rows > 0:
                remaining = n_rows - total
                if remaining <= 0:
                    break
                if len(df) > remaining:
                    df = df.head(remaining)
            total += len(df)

    return _measure(run)


def bench_pyarrow_dataset_batches(path: Path, n_rows: int) -> dict[str, float]:
    dataset = ds.dataset(str(path), format="parquet")
    columns = _select_parquet_columns(list(dataset.schema.names))
    batch_size = 4096

    def run() -> None:
        total = 0
        for batch in dataset.to_batches(columns=columns, batch_size=batch_size):
            df = pl.from_arrow(batch, rechunk=False)
            if n_rows > 0:
                remaining = n_rows - total
                if remaining <= 0:
                    break
                if len(df) > remaining:
                    df = df.head(remaining)
            total += len(df)

    return _measure(run)


def bench_hpo_unique_counts(path: Path, n_rows: int) -> dict[str, float]:
    fm = FileManager()

    payload = fm.read(path)
    lf = _as_lazy_frame(payload)
    lf = _select_triplet_columns(lf)
    if n_rows > 0:
        lf = lf.head(n_rows)
    df = lf.collect(engine="streaming")
    cols = df.columns
    if not cols:
        return {"median_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0, "runs": 0.0}
    if len(cols) >= 3:
        col_s, _, col_o = cols[:3]
    elif len(cols) == 2:
        col_s, col_o = cols
    else:
        col_s = cols[0]
        col_o = cols[0]

    split_at = max(1, int(len(df) * 0.8))
    train_df = df[:split_at]
    valid_df = df[split_at:]

    def run() -> None:
        data_loader._count_unique_arrow(
            train_df[col_s],
            train_df[col_o],
            valid_df[col_s],
            valid_df[col_o],
        )

    return _measure(run)


def bench_arrow_ipc_read(path: Path, n_rows: int) -> dict[str, float]:
    fm = FileManager()
    bench_dir = Path("outputs/benches")
    arrow_path = bench_dir / "tmp.arrow"
    lf = pl.scan_parquet(path)
    if n_rows > 0:
        lf = lf.head(n_rows)
    df = lf.collect(engine="streaming")
    fm.save(df, arrow_path, compression="uncompressed")

    def run() -> None:
        fm.read(arrow_path, return_native=True)

    result = _measure(run)
    FileManager.delete_file(arrow_path, ignore_errors=True)
    return result


def bench_kg_builder_load(path: Path, n_rows: int) -> dict[str, float]:
    async def run_once() -> None:
        builder = KGBuilder(
            source_path=path,
            output_dir="benches/kg_builder",
            max_members=n_rows if n_rows > 0 else None,
            parallel=False,
        )
        await builder._load_parquet_tabular(
            parquet_path=Path(path),
            collector=None,
            persist=False,
        )

    def run() -> None:
        asyncio.run(run_once())

    return _measure(run)


def _load_hpo_mapping_frames() -> tuple[pl.DataFrame, pl.DataFrame]:
    train_path = Path("outputs/kg/train.parquet")
    valid_path = Path("outputs/kg/valid.parquet")
    if train_path.exists() and valid_path.exists():
        return pl.read_parquet(train_path), pl.read_parquet(valid_path)

    train_rows = 20000
    valid_rows = 5000
    train_df = pl.select(
        pl.format("s{}", pl.arange(0, train_rows)).alias("s"),
        pl.format("p{}", pl.arange(0, train_rows) % 128).alias("p"),
        pl.format("o{}", pl.arange(0, train_rows) * 3).alias("o"),
    )
    valid_df = pl.select(
        pl.format("s{}", pl.arange(train_rows, train_rows + valid_rows)).alias("s"),
        pl.format("p{}", pl.arange(0, valid_rows) % 128).alias("p"),
        pl.format("o{}", pl.arange(train_rows, train_rows + valid_rows) * 3).alias("o"),
    )
    return train_df, valid_df


def bench_hpo_mapping() -> dict[str, float]:
    train_df, valid_df = _load_hpo_mapping_frames()

    def df_to_triples(
        df: pl.DataFrame, entity_map: pl.DataFrame, relation_map: pl.DataFrame
    ) -> None:
        mapped = (
            df.select(["s", "p", "o"])
            .join(entity_map, left_on="s", right_on="label", how="left")
            .rename({"id": "s_id"})
            .join(relation_map, left_on="p", right_on="label", how="left")
            .rename({"id": "p_id"})
            .join(entity_map, left_on="o", right_on="label", how="left")
            .rename({"id": "o_id"})
            .select(["s_id", "p_id", "o_id"])
        )
        mapped.to_numpy()

    def run() -> None:
        entity_labels = (
            pl.concat([train_df["s"], train_df["o"], valid_df["s"], valid_df["o"]]).unique().sort()
        )
        relation_labels = pl.concat([train_df["p"], valid_df["p"]]).unique().sort()
        entity_map = pl.DataFrame({"label": entity_labels}).with_row_index("id")
        relation_map = pl.DataFrame({"label": relation_labels}).with_row_index("id")
        df_to_triples(train_df, entity_map, relation_map)
        df_to_triples(valid_df, entity_map, relation_map)

    return _measure(run)


def bench_hpo_mapping_combined() -> dict[str, float]:
    train_df, valid_df = _load_hpo_mapping_frames()

    def run() -> None:
        entity_labels = (
            pl.concat([train_df["s"], train_df["o"], valid_df["s"], valid_df["o"]]).unique().sort()
        )
        relation_labels = pl.concat([train_df["p"], valid_df["p"]]).unique().sort()
        entity_map = pl.DataFrame({"label": entity_labels}).with_row_index("id")
        relation_map = pl.DataFrame({"label": relation_labels}).with_row_index("id")

        combined = pl.concat(
            [
                train_df.with_columns(pl.lit("train").alias("__split")),
                valid_df.with_columns(pl.lit("valid").alias("__split")),
            ],
        )
        mapped = (
            combined.select(["s", "p", "o", "__split"])
            .join(entity_map, left_on="s", right_on="label", how="left", maintain_order="left")
            .rename({"id": "s_id"})
            .join(relation_map, left_on="p", right_on="label", how="left", maintain_order="left")
            .rename({"id": "p_id"})
            .join(entity_map, left_on="o", right_on="label", how="left", maintain_order="left")
            .rename({"id": "o_id"})
            .select(["__split", "s_id", "p_id", "o_id"])
        )
        mapped_train = mapped.filter(pl.col("__split") == "train").select(["s_id", "p_id", "o_id"])
        mapped_valid = mapped.filter(pl.col("__split") == "valid").select(["s_id", "p_id", "o_id"])
        mapped_train.to_numpy()
        mapped_valid.to_numpy()

    return _measure(run)


def bench_numba_negative_samples(batch_size: int, num_entities: int) -> dict[str, float]:
    from pff.shared.acceleration.numba_kernels import generate_negative_samples

    generate_negative_samples(
        num_negatives=batch_size,
        num_entities=num_entities,
        head_idx=1,
        tail_idx=2,
        rel_idx=3,
        seed=1234,
    )

    def run() -> None:
        generate_negative_samples(
            num_negatives=batch_size,
            num_entities=num_entities,
            head_idx=1,
            tail_idx=2,
            rel_idx=3,
            seed=1234,
        )

    return _measure(run)


def bench_triton_subsample(batch_size: int, num_candidates: int, k: int) -> dict[str, float] | None:
    if not torch.cuda.is_available():
        return None
    try:
        from pff.shared.acceleration.triton_kernels import fused_random_subsample_triton
    except Exception:
        return None

    scores = torch.randn((batch_size, num_candidates), device="cuda")
    fused_random_subsample_triton(scores, k=k, seed=1234)
    torch.cuda.synchronize()

    def run() -> None:
        fused_random_subsample_triton(scores, k=k, seed=1234)
        torch.cuda.synchronize()

    return _measure(run)


def bench_lance_take(rows: int, cache_size: int) -> dict[str, float] | None:
    try:
        import lancedb
    except Exception:
        return None

    bench_dir = Path("outputs/benches")
    db_path = bench_dir / "lance_db"
    FileManager.delete_directory(db_path, ignore_errors=True)
    db = lancedb.connect(str(db_path))
    rng = pa.array([[int(i) for i in range(cache_size)] for _ in range(rows)])
    table = pa.Table.from_pydict({"id": list(range(rows)), "negatives": rng})
    lance_table = db.create_table("negatives", table)
    take_row_ids = getattr(lance_table, "take_row_ids", None)
    take_offsets = getattr(lance_table, "take_offsets", None)
    indices = list(range(min(rows, 1024)))

    def run() -> None:
        if take_row_ids is not None:
            take_row_ids(indices)
        elif take_offsets is not None:
            take_offsets(indices)

    result = _measure(run)
    FileManager.delete_directory(db_path, ignore_errors=True)
    return result


def bench_faiss_search(
    dim: int, n_db: int, n_query: int, k: int
) -> dict[str, dict[str, float]] | None:
    try:
        import faiss
        import numpy as np
    except Exception:
        return None

    rng = np.random.default_rng(1234)
    xb = rng.standard_normal((n_db, dim)).astype(np.float32)
    xq = rng.standard_normal((n_query, dim)).astype(np.float32)

    index_cpu = faiss.IndexFlatL2(dim)
    index_cpu.add(xb)

    def run_cpu() -> None:
        index_cpu.search(xq, k)

    results: dict[str, dict[str, float]] = {"cpu": _measure(run_cpu)}

    if hasattr(faiss, "StandardGpuResources") and torch.cuda.is_available():
        res = faiss.StandardGpuResources()
        index_gpu = faiss.index_cpu_to_gpu(res, 0, index_cpu)

        def run_gpu() -> None:
            index_gpu.search(xq, k)

        results["gpu"] = _measure(run_gpu)

    return results


def _bench_negative_sampling(device: torch.device, batch_size: int) -> dict[str, float]:
    from pff.domain.learning.dslfm.neg_sampling import SamplerConfig, UniformSampler

    torch.manual_seed(1234)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(1234)

    all_scores = torch.randn((batch_size, batch_size), device=device)
    tails = torch.arange(batch_size, device=device, dtype=torch.long)
    sampler = UniformSampler(SamplerConfig())

    def run() -> None:
        sampler.get_positive_negative_scores(all_scores, tails)
        if device.type == "cuda":
            torch.cuda.synchronize()

    return _measure(run)


def bench_negative_sampling(batch_size: int) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    cpu_device = torch.device("cpu")
    results["cpu"] = _bench_negative_sampling(cpu_device, batch_size)
    if torch.cuda.is_available():
        cuda_device = torch.device("cuda")
        results["cuda"] = _bench_negative_sampling(cuda_device, batch_size)
    return results


def _lib_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def main() -> None:
    bench_dir = Path("outputs/benches")
    bench_dir.mkdir(parents=True, exist_ok=True)

    correct_path = Path("data/models/correct.parquet")
    fallback_path = Path("tests/support/fixtures/bench_small.parquet")
    target_path = correct_path if correct_path.exists() else fallback_path
    n_rows = 10_000 if target_path == correct_path else 0

    hardware = HardwareDetector.detect()

    results = {
        "target_path": str(target_path),
        "n_rows": n_rows,
        "hardware": {
            "cpu_threads": hardware.cpu_threads,
            "total_ram_gb": hardware.total_ram_gb,
            "gpu": hardware.has_gpu,
            "gpu_memory_gb": hardware.gpu_memory_gb,
            "platform": hardware.platform,
        },
        "parquet_scan": bench_parquet_scan(target_path, n_rows),
        "pyarrow_dataset_scan": bench_pyarrow_dataset_scan(target_path, n_rows),
        "pyarrow_iter_batches": bench_pyarrow_iter_batches(target_path, n_rows),
        "pyarrow_dataset_batches": bench_pyarrow_dataset_batches(target_path, n_rows),
        "hpo_unique_counts": bench_hpo_unique_counts(target_path, n_rows),
        "arrow_ipc_read": bench_arrow_ipc_read(target_path, n_rows),
        "kg_builder_load": bench_kg_builder_load(target_path, n_rows),
        "numba_negative_samples": bench_numba_negative_samples(batch_size=512, num_entities=10000),
        "negative_sampling": bench_negative_sampling(batch_size=512),
    }
    triton_result = bench_triton_subsample(batch_size=256, num_candidates=4096, k=256)
    if triton_result is not None:
        results["triton_subsample"] = triton_result
    lance_result = bench_lance_take(rows=2048, cache_size=256)
    if lance_result is not None:
        results["lance_take"] = lance_result
    faiss_result = bench_faiss_search(dim=64, n_db=50000, n_query=1000, k=10)
    if faiss_result is not None:
        results["faiss_search"] = faiss_result
    results["lib_availability"] = {
        "flash_attn": _lib_available("flash_attn"),
        "bitsandbytes": _lib_available("bitsandbytes"),
        "deepspeed": _lib_available("deepspeed"),
        "kvikio": _lib_available("kvikio"),
        "galore_torch": _lib_available("galore_torch"),
        "faiss": _lib_available("faiss"),
        "torch_geometric": _lib_available("torch_geometric"),
        "lancedb": _lib_available("lancedb"),
        "triton": _lib_available("triton"),
    }
    results["hpo_mapping"] = bench_hpo_mapping()
    results["hpo_mapping_combined"] = bench_hpo_mapping_combined()

    FileManager.write_text(json.dumps(results, indent=2), bench_dir / "baseline.json")
    FileManager.write_text(json.dumps(results, indent=2), bench_dir / "baseline.txt")
    FileManager.write_text(
        "poetry run python scripts/bench_perf_sweep.py\n",
        bench_dir / "commands.txt",
    )
    summary = {
        "kg_builder_load_ms": results.get("kg_builder_load", {}).get("median_ms"),
        "hpo_mapping_ms": results.get("hpo_mapping", {}).get("median_ms"),
        "hpo_mapping_combined_ms": results.get("hpo_mapping_combined", {}).get("median_ms"),
        "pyarrow_iter_batches_ms": results.get("pyarrow_iter_batches", {}).get("median_ms"),
        "pyarrow_dataset_batches_ms": results.get("pyarrow_dataset_batches", {}).get("median_ms"),
        "negative_sampling_cpu_ms": results.get("negative_sampling", {})
        .get("cpu", {})
        .get("median_ms"),
        "negative_sampling_cuda_ms": results.get("negative_sampling", {})
        .get("cuda", {})
        .get("median_ms"),
    }
    print(json.dumps(summary, ensure_ascii=True))
    FileManager.delete_directory(bench_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
