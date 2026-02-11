"""Performance benchmarks for Complex wins optimization sweep.

Measures critical hotpaths:
1. CacheSerializer - path cache for repeated objects
2. DiskCache - parallel file deletions
3. CacheJanitor - parallel purging
"""

import gc
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from pff.shared.core.cache import (
    CacheSerializer,
    DiskCache,
)

BENCH_DIR = Path("outputs/benches")
BENCH_DIR.mkdir(parents=True, exist_ok=True)


def measure(
    func: Callable[[], Any], warmup: int = 1, runs: int = 20
) -> dict[str, float]:
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


class TestCacheSerializerPathCacheBaseline:
    """Benchmark CacheSerializer path cache optimization."""

    def test_serialize_repeated_objects_1k(self, tmp_path: Path):
        """Baseline: serialize 1K repeated objects (tests path cache)."""
        serializer = CacheSerializer()
        cache_root = tmp_path / "cache"
        cache_root.mkdir(parents=True, exist_ok=True)

        objects = [
            {"data": f"value_{i % 100}", "nested": {"a": i % 10}} for i in range(1000)
        ]

        def run():
            return [
                serializer.serialize(obj, cache_root=cache_root, cache_key=f"obj_{i}")
                for i, obj in enumerate(objects)
            ]

        stats = measure(run, warmup=1, runs=20)
        stats["objects"] = len(objects)
        stats["objects_per_sec"] = len(objects) / (stats["median_ms"] / 1000)

        print(
            f"\n[serializer_path_cache] {stats['objects']:,} objects: {stats['median_ms']:.2f}ms ({stats['objects_per_sec']:,.0f}/s)"
        )

        bench_file = BENCH_DIR / "serializer_path_cache_baseline_1k.json"
        bench_file.write_text(str(stats))

        assert stats["objects"] == 1000


class TestDiskCacheParallelDeleteBaseline:
    """Benchmark DiskCache parallel delete optimization."""

    def test_purge_1k_files(self, tmp_path: Path):
        """Baseline: purge 1K cache files (tests parallel deletions)."""
        cache = DiskCache(root=tmp_path / "cache")

        for i in range(1000):
            key = f"key_{i}"
            cache._storage.write(cache.root / f"{key}.pkl", f"data_{i}".encode())

        def run():
            return cache.purge("*.pkl*")

        stats = measure(run, warmup=1, runs=20)
        stats["files"] = 1000
        stats["files_per_sec"] = 1000 / (stats["median_ms"] / 1000)

        print(
            f"\n[disk_cache_purge] {stats['files']:,} files: {stats['median_ms']:.2f}ms ({stats['files_per_sec']:,.0f}/s)"
        )

        bench_file = BENCH_DIR / "disk_cache_purge_baseline_1k.json"
        bench_file.write_text(str(stats))

        assert stats["files"] == 1000

    def test_purge_10k_files(self, tmp_path: Path):
        """Baseline: purge 10K cache files (tests parallel deletions)."""
        cache = DiskCache(root=tmp_path / "cache")

        for i in range(10_000):
            key = f"key_{i}"
            cache._storage.write(cache.root / f"{key}.pkl", f"data_{i}".encode())

        def run():
            return cache.purge("*.pkl*")

        stats = measure(run, warmup=1, runs=20)
        stats["files"] = 10_000
        stats["files_per_sec"] = 10_000 / (stats["median_ms"] / 1000)

        print(
            f"\n[disk_cache_purge_10k] {stats['files']:,} files: {stats['median_ms']:.2f}ms ({stats['files_per_sec']:,.0f}/s)"
        )

        bench_file = BENCH_DIR / "disk_cache_purge_baseline_10k.json"
        bench_file.write_text(str(stats))

        assert stats["files"] == 10000


class TestCacheJanitorParallelPurgeBaseline:
    """Benchmark CacheJanitor parallel purge optimization."""

    def test_purge_stale_1k_entries(self, tmp_path: Path):
        """Baseline: purge 1K stale cache entries (tests parallel purging)."""
        cache = DiskCache(root=tmp_path / "cache")

        for i in range(1000):
            key = f"key_{i}"
            cache._storage.write(cache.root / f"{key}.pkl", f"data_{i}".encode())

        def run():
            return cache.purge("*.pkl*")

        stats = measure(run, warmup=1, runs=20)
        stats["entries"] = 1000
        stats["entries_per_sec"] = 1000 / (stats["median_ms"] / 1000)

        print(
            f"\n[cache_janitor_purge] {stats['entries']:,} entries: {stats['median_ms']:.2f}ms ({stats['entries_per_sec']:,.0f}/s)"
        )

        bench_file = BENCH_DIR / "cache_janitor_purge_baseline_1k.json"
        bench_file.write_text(str(stats))

        assert stats["entries"] == 1000
