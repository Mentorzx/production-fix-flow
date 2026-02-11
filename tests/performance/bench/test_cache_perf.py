"""Performance benchmarks for cache.py optimization sweep.

Measures critical hotpaths:
1. FunctionCallHasher - JSON serialization overhead
2. CacheSerializer - serialize/deserialize throughput
3. CacheManager - LRU operations with locks
"""

import gc
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from pff.shared.core.cache import (
    CacheManager,
    CacheSerializer,
    FunctionCallHasher,
    JsonSafeEncoder,
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


class TestFunctionCallHasherBaseline:
    """Benchmark cache key generation."""

    def test_hash_function_call_1k_simple(self, tmp_path: Path):
        """Baseline: 1K simple function calls with primitive args."""
        hasher = FunctionCallHasher()

        def dummy_fn(x: int, y: int, z: int) -> int:
            return x + y + z

        args_list = [(i, i * 2, i * 3) for i in range(1000)]

        def run():
            return [hasher.hash_function_call(dummy_fn, *args) for args in args_list]

        stats = measure(run, warmup=1, runs=3)
        stats["hashes"] = len(args_list)
        stats["hashes_per_sec"] = len(args_list) / (stats["median_ms"] / 1000)

        print(
            f"\n[hasher_simple] {len(args_list):,} hashes: {stats['median_ms']:.2f}ms ({stats['hashes_per_sec']:,.0f}/s)"
        )

        bench_file = BENCH_DIR / "hasher_baseline_1k_simple.json"
        bench_file.write_text(str(stats))

        assert stats["hashes"] == 1000

    def test_hash_function_call_10k_complex(self, tmp_path: Path):
        """Baseline: 10K function calls with nested dicts/lists."""
        hasher = FunctionCallHasher()

        def dummy_fn(x: dict[str, Any], y: list[int], z: str) -> None:
            pass

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
            f"\n[hasher_complex] {len(args_list):,} hashes: {stats['median_ms']:.2f}ms ({stats['hashes_per_sec']:,.0f}/s)"
        )

        bench_file = BENCH_DIR / "hasher_baseline_10k_complex.json"
        bench_file.write_text(str(stats))

        assert stats["hashes"] == 10000

    def test_determinism_consistency(self, tmp_path: Path):
        """Verify baseline determinism: same args = same hash."""
        hasher = FunctionCallHasher()

        def dummy_fn(x: int, y: dict[str, int]) -> int:
            return x + y.get("value", 0)

        args = (42, {"value": 100, "nested": {"a": 1}})

        hash1 = hasher.hash_function_call(dummy_fn, *args)
        hash2 = hasher.hash_function_call(dummy_fn, *args)
        hash3 = hasher.hash_function_call(dummy_fn, *args)

        assert hash1 == hash2 == hash3, "Hashes must be deterministic"


class TestCacheSerializerBaseline:
    """Benchmark cache serialization/deserialization."""

    def test_serialize_deserialize_1k_dicts(self, tmp_path: Path):
        """Baseline: round-trip 1K dict objects."""
        serializer = CacheSerializer()
        objects = [
            {"key": f"value_{i}", "nested": {"a": i, "b": i * 2}} for i in range(1000)
        ]

        def run():
            serialized = [serializer.serialize(obj) for obj in objects]
            deserialized = [serializer.deserialize(data) for data in serialized]
            return deserialized

        stats = measure(run, warmup=1, runs=3)
        stats["objects"] = len(objects)
        stats["objects_per_sec"] = len(objects) / (stats["median_ms"] / 1000)

        print(
            f"\n[serializer_dicts] {len(objects):,} objects: {stats['median_ms']:.2f}ms ({stats['objects_per_sec']:,.0f}/s)"
        )

        bench_file = BENCH_DIR / "serializer_baseline_1k_dicts.json"
        bench_file.write_text(str(stats))

        assert stats["objects"] == 1000

    def test_serialize_deserialize_100_lazyframes(self, tmp_path: Path):
        """Baseline: round-trip 100 LazyFrame objects (parquet-first)."""
        serializer = CacheSerializer()
        cache_root = tmp_path / "cache"
        cache_root.mkdir(parents=True, exist_ok=True)

        objects = []
        for i in range(100):
            df = pl.DataFrame({"a": range(100), "b": range(100, 200)})
            lf = df.lazy()
            objects.append(lf)

        def run():
            results = []
            for i, obj in enumerate(objects):
                cache_key = f"lf_{i}"
                serialized = serializer.serialize(
                    obj, cache_root=cache_root, cache_key=cache_key
                )
                deserialized = serializer.deserialize(serialized, cache_root=cache_root)
                results.append(deserialized)
            return results

        stats = measure(run, warmup=1, runs=3)
        stats["objects"] = len(objects)
        stats["objects_per_sec"] = len(objects) / (stats["median_ms"] / 1000)

        print(
            f"\n[serializer_lazyframes] {len(objects):,} objects: {stats['median_ms']:.2f}ms ({stats['objects_per_sec']:,.0f}/s)"
        )

        bench_file = BENCH_DIR / "serializer_baseline_100_lazyframes.json"
        bench_file.write_text(str(stats))

        assert stats["objects"] == 100

    def test_serialize_deserialize_1k_primitives(self, tmp_path: Path):
        """Baseline: round-trip 1K primitive values."""
        serializer = CacheSerializer()
        objects = [42, "hello", 3.14, True, None, [1, 2, 3], {"a": 1}] * 142

        def run():
            serialized = [serializer.serialize(obj) for obj in objects]
            deserialized = [serializer.deserialize(data) for data in serialized]
            return deserialized

        stats = measure(run, warmup=1, runs=3)
        stats["objects"] = len(objects)
        stats["objects_per_sec"] = len(objects) / (stats["median_ms"] / 1000)

        print(
            f"\n[serializer_primitives] {len(objects):,} objects: {stats['median_ms']:.2f}ms ({stats['objects_per_sec']:,.0f}/s)"
        )

        bench_file = BENCH_DIR / "serializer_baseline_1k_primitives.json"
        bench_file.write_text(str(stats))

        assert stats["objects"] == 994


class TestCacheManagerBaseline:
    """Benchmark CacheManager LRU operations."""

    def test_get_set_10k_ops(self, tmp_path: Path):
        """Baseline: 10K get/set operations with LRU eviction."""
        manager = CacheManager(cache_dir=tmp_path / "cache", max_memory_items=100)

        def run():
            for i in range(10_000):
                key = f"key_{i % 100}"
                value = {"data": f"value_{i}"}
                manager.set(key, value)
                result = manager.get(key)
                assert result is not None

        stats = measure(run, warmup=1, runs=3)
        stats["operations"] = 10_000
        stats["ops_per_sec"] = 10_000 / (stats["median_ms"] / 1000)

        print(
            f"\n[cache_manager_lru] {stats['operations']:,} ops: {stats['median_ms']:.2f}ms ({stats['ops_per_sec']:,.0f}/s)"
        )

        bench_file = BENCH_DIR / "cache_manager_baseline_10k_ops.json"
        bench_file.write_text(str(stats))

        stats_report = manager.get_stats()
        assert stats_report["hits"] > 0

    def test_get_set_10k_ops_with_ttl(self, tmp_path: Path):
        """Baseline: 10K get/set operations with TTL expiration."""
        manager = CacheManager(cache_dir=tmp_path / "cache", max_memory_items=100)

        def run():
            for i in range(10_000):
                key = f"key_{i % 100}"
                value = {"data": f"value_{i}"}
                ttl = 60 if i % 10 == 0 else None
                manager.set(key, value, ttl=ttl)
                result = manager.get(key)
                assert result is not None

        stats = measure(run, warmup=1, runs=3)
        stats["operations"] = 10_000
        stats["ops_per_sec"] = 10_000 / (stats["median_ms"] / 1000)

        print(
            f"\n[cache_manager_ttl] {stats['operations']:,} ops: {stats['median_ms']:.2f}ms ({stats['ops_per_sec']:,.0f}/s)"
        )

        bench_file = BENCH_DIR / "cache_manager_baseline_10k_ttl.json"
        bench_file.write_text(str(stats))


class TestJsonSafeEncoderBaseline:
    """Benchmark JsonSafeEncoder.make_json_safe."""

    def test_make_json_safe_10k_calls(self, tmp_path: Path):
        """Baseline: 10K calls to make_json_safe with mixed types."""
        encoder = JsonSafeEncoder()

        class CustomClass:
            def __repr__(self) -> str:
                return "<Custom>"

        objects = [
            42,
            "string",
            3.14,
            True,
            None,
            [1, 2, 3],
            {"a": 1},
            b"bytes",
            {1, 2, 3},
            CustomClass(),
        ]

        def run():
            return [encoder.make_json_safe(obj) for obj in objects for _ in range(1000)]

        stats = measure(run, warmup=1, runs=3)
        stats["calls"] = len(objects) * 1000
        stats["calls_per_sec"] = stats["calls"] / (stats["median_ms"] / 1000)

        print(
            f"\n[json_safe_encoder] {stats['calls']:,} calls: {stats['median_ms']:.2f}ms ({stats['calls_per_sec']:,.0f}/s)"
        )

        bench_file = BENCH_DIR / "json_safe_encoder_baseline_10k_calls.json"
        bench_file.write_text(str(stats))

        assert stats["calls"] == 10000


class TestCacheManagerLruOptBaseline:
    """Benchmark CacheManager LRU operations for optimization."""

    def test_get_consecutive_hits_10k(self, tmp_path: Path):
        """Baseline: 10K consecutive get hits (tests move_to_end overhead)."""
        manager = CacheManager(cache_dir=tmp_path / "cache", max_memory_items=1000)
        manager.set("key1", "value1")

        def run():
            for _ in range(10_000):
                manager.get("key1")

        stats = measure(run, warmup=1, runs=20)
        stats["operations"] = 10_000
        stats["ops_per_sec"] = 10_000 / (stats["median_ms"] / 1000)

        print(
            f"\n[cache_lru_consecutive] {stats['operations']:,} ops: {stats['median_ms']:.2f}ms ({stats['ops_per_sec']:,.0f}/s)"
        )

        bench_file = BENCH_DIR / "cache_lru_baseline_consecutive_10k.json"
        bench_file.write_text(str(stats))

    def test_get_different_keys_10k(self, tmp_path: Path):
        """Baseline: 10K gets from 100 different keys (tests LRU rotation)."""
        manager = CacheManager(cache_dir=tmp_path / "cache", max_memory_items=100)
        for i in range(100):
            manager.set(f"key_{i}", f"value_{i}")

        def run():
            for i in range(10_000):
                manager.get(f"key_{i % 100}")

        stats = measure(run, warmup=1, runs=20)
        stats["operations"] = 10_000
        stats["ops_per_sec"] = 10_000 / (stats["median_ms"] / 1000)

        print(
            f"\n[cache_lru_rotation] {stats['operations']:,} ops: {stats['median_ms']:.2f}ms ({stats['ops_per_sec']:,.0f}/s)"
        )

        bench_file = BENCH_DIR / "cache_lru_baseline_rotation_10k.json"
        bench_file.write_text(str(stats))
