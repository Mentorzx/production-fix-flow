"""
Performance baseline benchmarks for PFF optimization sweep.

Measures critical hotpaths identified in repo scan:
1. Anomaly scoring - vectorized calibration/EVT
2. Negative sampling - Rust kernel performance
3. Cache hashing - JSON serialization overhead
4. Polars materializations - set() from to_list() patterns
5. Parquet I/O - scan vs read throughput
"""

import gc
import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest

BENCH_DIR = Path("outputs/benches")
BENCH_DIR.mkdir(parents=True, exist_ok=True)


def measure(func, warmup: int = 1, runs: int = 3) -> dict[str, float]:
    """Run function with warmup and return timing stats."""
    for _ in range(warmup):
        func()
    gc.collect()

    times = []
    for _ in range(runs):
        start = time.perf_counter()
        func()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        gc.collect()

    return {
        "median_ms": float(np.median(times) * 1000),
        "min_ms": float(np.min(times) * 1000),
        "max_ms": float(np.max(times) * 1000),
        "runs": runs,
    }


class TestAnomalyScoringBaseline:
    """Benchmark anomaly scoring hotpath."""

    def test_score_with_calibration_1m_items(self):
        N = 1_000_000
        scores = np.random.default_rng(42).standard_normal(N)
        relations = np.random.default_rng(42).choice(["r1", "r2", "r3"], N)

        class MockCalibrator:
            def transform(self, x):
                return 1.0 / (1.0 + np.exp(-x))

        import pff.domain.audit.anomaly_scoring as mod

        original_calibrator_from_dict = mod.calibrator_from_dict
        mod.calibrator_from_dict = lambda x: MockCalibrator()

        calibrators = {
            "__global__": {"model": {}},
            "r1": {"model": {}},
            "r2": {"model": {}},
            "r3": {"model": {}},
        }
        evt_params = {
            "__global__": {"shape": 0.1, "scale": 1.0, "u": 0.5},
            "r1": {"shape": 0.1, "scale": 1.0, "u": 0.5},
            "r2": {"shape": 0.1, "scale": 1.0, "u": 0.5},
            "r3": {"shape": 0.1, "scale": 1.0, "u": 0.5},
        }

        def run():
            return mod.score_with_calibration_and_evt(
                scores=scores,
                relations=relations,
                calibrators_by_relation=calibrators,
                evt_params_by_relation=evt_params,
            )

        stats = measure(run, warmup=1, runs=3)
        stats["items"] = N
        stats["items_per_sec"] = N / (stats["median_ms"] / 1000)

        mod.calibrator_from_dict = original_calibrator_from_dict

        print(
            f"\n[anomaly_scoring] {N:,} items: {stats['median_ms']:.2f}ms ({stats['items_per_sec']:,.0f}/s)"
        )
        assert (
            stats["median_ms"] < 5000
        ), f"Anomaly scoring too slow: {stats['median_ms']:.2f}ms"


class TestNegativeSamplingBaseline:
    """Benchmark negative sampling Rust kernel."""

    def test_corrupt_tails_100k_triples(self):
        N = 100_000
        num_entities = 50_000
        num_negatives = 50
        seed = 42

        rng = np.random.default_rng(seed)
        triples = np.column_stack(
            [
                rng.integers(0, num_entities, N),
                rng.integers(0, 1000, N),
                rng.integers(0, num_entities, N),
            ]
        ).astype(np.int64)

        from pff.domain.audit.negative_sampling import corrupt_tails

        def run():
            return corrupt_tails(
                triples,
                num_entities=num_entities,
                num_negatives=num_negatives,
                seed=seed,
            )

        stats = measure(run, warmup=2, runs=3)
        total_negs = N * num_negatives
        stats["total_negatives"] = total_negs
        stats["negs_per_sec"] = total_negs / (stats["median_ms"] / 1000)

        print(
            f"\n[negative_sampling] {N:,} triples × {num_negatives} neg: {stats['median_ms']:.2f}ms ({stats['negs_per_sec']:,.0f} neg/s)"
        )
        assert (
            stats["median_ms"] < 2000
        ), f"Negative sampling too slow: {stats['median_ms']:.2f}ms"


class TestCacheHashingBaseline:
    """Benchmark cache key generation."""

    def test_hash_function_call_10k_args(self):
        from pff.shared.core.cache import FunctionCallHasher

        hasher = FunctionCallHasher()

        def dummy_fn(x, y, z):
            return x + y + z

        args_list = [
            (
                {"key": f"value_{i}", "nested": {"a": i, "b": i * 2}},
                [1, 2, 3, i],
                f"string_{i}",
            )
            for i in range(10_000)
        ]

        def run():
            return [hasher.hash_function_call(dummy_fn, *args) for args in args_list]

        stats = measure(run, warmup=1, runs=3)
        stats["hashes"] = len(args_list)
        stats["hashes_per_sec"] = len(args_list) / (stats["median_ms"] / 1000)

        print(
            f"\n[cache_hashing] {len(args_list):,} hashes: {stats['median_ms']:.2f}ms ({stats['hashes_per_sec']:,.0f}/s)"
        )
        assert (
            stats["median_ms"] < 2000
        ), f"Cache hashing too slow: {stats['median_ms']:.2f}ms"


class TestPolarsPatternBaseline:
    """Benchmark Polars materialization patterns."""

    def test_set_from_to_list_vs_direct_1m_rows(self):
        N = 1_000_000
        df = pl.DataFrame(
            {
                "s": [f"entity_{i % 10000}" for i in range(N)],
                "p": [f"pred_{i % 100}" for i in range(N)],
                "o": [f"obj_{i % 10000}" for i in range(N)],
            }
        )

        def via_to_list():
            return set(df["s"].unique().to_list())

        def via_direct():
            return set(df["s"].unique())

        stats_list = measure(via_to_list, warmup=1, runs=3)
        stats_direct = measure(via_direct, warmup=1, runs=3)

        speedup = (
            stats_list["median_ms"] / stats_direct["median_ms"]
            if stats_direct["median_ms"] > 0
            else 1.0
        )

        print(
            f"\n[polars_set] to_list(): {stats_list['median_ms']:.2f}ms | direct: {stats_direct['median_ms']:.2f}ms | speedup: {speedup:.2f}x"
        )

    def test_collect_streaming_vs_eager_500k(self):
        N = 500_000
        df = pl.DataFrame(
            {
                "a": np.random.default_rng(42).integers(0, 1000, N),
                "b": np.random.default_rng(42).random(N),
            }
        )

        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.parquet"
            df.write_parquet(path)

            def eager_collect():
                return pl.scan_parquet(path).filter(pl.col("a") > 500).collect()

            def streaming_collect():
                return (
                    pl.scan_parquet(path)
                    .filter(pl.col("a") > 500)
                    .collect(engine="streaming")
                )

            stats_eager = measure(eager_collect, warmup=1, runs=3)
            stats_streaming = measure(streaming_collect, warmup=1, runs=3)

            speedup = (
                stats_eager["median_ms"] / stats_streaming["median_ms"]
                if stats_streaming["median_ms"] > 0
                else 1.0
            )

            print(
                f"\n[polars_collect] eager: {stats_eager['median_ms']:.2f}ms | streaming: {stats_streaming['median_ms']:.2f}ms | speedup: {speedup:.2f}x"
            )


class TestRustKernelsBaseline:
    """Benchmark Rust-compiled kernels."""

    def test_batch_generate_negative_samples_50k(self):
        from pff_rust import batch_generate_negative_samples

        N = 50_000
        num_negatives = 50
        num_entities = 100_000

        rng = np.random.default_rng(42)
        heads = rng.integers(0, num_entities, N, dtype=np.int64)
        rels = rng.integers(0, 1000, N, dtype=np.int64)
        tails = rng.integers(0, num_entities, N, dtype=np.int64)

        batch_generate_negative_samples(
            heads[:100], rels[:100], tails[:100], 5, num_entities, 42
        )

        def run():
            return batch_generate_negative_samples(
                heads, rels, tails, num_negatives, num_entities, 42
            )

        stats = measure(run, warmup=2, runs=3)
        total_negs = N * num_negatives
        stats["total_negatives"] = total_negs
        stats["negs_per_sec"] = total_negs / (stats["median_ms"] / 1000)

        print(
            f"\n[rust_neg_sampling] {N:,}×{num_negatives}: {stats['median_ms']:.2f}ms ({stats['negs_per_sec']:,.0f} neg/s)"
        )
        assert (
            stats["median_ms"] < 500
        ), f"Rust neg sampling too slow: {stats['median_ms']:.2f}ms"


class TestParquetIOBaseline:
    """Benchmark Parquet I/O patterns."""

    def test_parquet_scan_with_projection_vs_full(self):
        N = 500_000
        df = pl.DataFrame(
            {
                "a": np.random.default_rng(42).integers(0, 1000, N),
                "b": np.random.default_rng(42).random(N),
                "c": [f"str_{i}" for i in range(N)],
                "d": np.random.default_rng(42).random(N),
                "e": np.random.default_rng(42).integers(0, 100, N),
            }
        )

        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.parquet"
            df.write_parquet(path, compression="lz4", statistics=True)

            def full_scan():
                return pl.scan_parquet(path).collect()

            def projected_scan():
                return pl.scan_parquet(path).select(["a", "b"]).collect()

            stats_full = measure(full_scan, warmup=1, runs=3)
            stats_proj = measure(projected_scan, warmup=1, runs=3)

            speedup = (
                stats_full["median_ms"] / stats_proj["median_ms"]
                if stats_proj["median_ms"] > 0
                else 1.0
            )

            print(
                f"\n[parquet_projection] full: {stats_full['median_ms']:.2f}ms | projected: {stats_proj['median_ms']:.2f}ms | speedup: {speedup:.2f}x"
            )

    def test_parquet_predicate_pushdown(self):
        N = 1_000_000
        df = pl.DataFrame(
            {
                "id": np.arange(N),
                "value": np.random.default_rng(42).random(N),
                "category": np.random.default_rng(42).choice(["A", "B", "C", "D"], N),
            }
        )

        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.parquet"
            df.write_parquet(
                path, compression="lz4", statistics=True, row_group_size=100_000
            )

            def with_pushdown():
                return pl.scan_parquet(path).filter(pl.col("category") == "A").collect()

            def read_then_filter():
                return pl.read_parquet(path).filter(pl.col("category") == "A")

            stats_push = measure(with_pushdown, warmup=1, runs=3)
            stats_read = measure(read_then_filter, warmup=1, runs=3)

            speedup = (
                stats_read["median_ms"] / stats_push["median_ms"]
                if stats_push["median_ms"] > 0
                else 1.0
            )

            print(
                f"\n[parquet_pushdown] read+filter: {stats_read['median_ms']:.2f}ms | pushdown: {stats_push['median_ms']:.2f}ms | speedup: {speedup:.2f}x"
            )


def collect_all_baselines() -> dict[str, Any]:
    """Run all benchmarks and collect results."""
    results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "benchmarks": {},
    }
    return results


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
