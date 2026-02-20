"""Performance benchmarks for logger.py optimization sweep.

Measures critical hotpaths:
1. LogReorderer._extract - log line parsing
2. LogReorderer.reorder - log file reordering with I/O
"""

import gc
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import orjson

from pff.shared.core import LogReorderer

BENCH_DIR = Path("outputs/benches")
BENCH_DIR.mkdir(parents=True, exist_ok=True)


def measure(func: Callable[[], Any], warmup: int = 1, runs: int = 20) -> dict[str, float]:
    """Execute measure.



    Args:

        func: Input value used by this callable.

        warmup: Optional input value.

        runs: Optional input value.



    Returns:

        Return value produced by the callable.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

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


def create_json_log_line(thread: str, msisdn: str | None, message: str) -> str:
    """Execute create json log line.



    Args:

        thread: Input value used by this callable.

        msisdn: Input value used by this callable.

        message: Input value used by this callable.



    Returns:

        Return value produced by the callable.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    record = {
        "record": {
            "thread": {"name": thread},
            "extra": {"task_id": msisdn, "msisdn": msisdn},
        },
        "text": message,
    }
    return orjson.dumps(record).decode("utf-8")


def create_text_log_line(thread: str, msisdn: str | None, message: str) -> str:
    """Execute create text log line.



    Args:

        thread: Input value used by this callable.

        msisdn: Input value used by this callable.

        message: Input value used by this callable.



    Returns:

        Return value produced by the callable.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    task_part = f"[{msisdn}]" if msisdn else "[N/A]"
    return f"2025-01-17 10:00:00.123 | INFO     | module:function:42 | {task_part:^11} - {message}"


class TestLogReordererExtractBaseline:
    """Benchmark LogReorderer._extract parsing."""

    def test_extract_100k_json_lines(self, tmp_path: Path):
        """Baseline: extract from 100K JSON-formatted log lines."""
        lines = []
        for i in range(100_000):
            thread = f"Thread-{i % 10}"
            msisdn = f"55{19998887766 + (i % 10000)}" if i % 3 == 0 else None
            message = f"Processing request {i}"
            lines.append(create_json_log_line(thread, msisdn, message))

        def run():
            """Execute run.



            Returns:

                Return value produced by the callable.

            """

            results = [LogReorderer._extract(line) for line in lines]
            return results

        stats = measure(run, warmup=1, runs=3)
        stats["lines"] = len(lines)
        stats["lines_per_sec"] = len(lines) / (stats["median_ms"] / 1000)

        print(
            f"\n[log_extract_json] {len(lines):,} lines: {stats['median_ms']:.2f}ms ({stats['lines_per_sec']:,.0f}/s)"
        )

        bench_file = BENCH_DIR / "log_extract_baseline_100k_json.json"
        bench_file.write_text(str(stats))

        assert len(lines) == 100000

    def test_extract_100k_text_lines(self, tmp_path: Path):
        """Baseline: extract from 100K text-formatted log lines."""
        lines = []
        for i in range(100_000):
            thread = f"Thread-{i % 10}"
            msisdn = f"55{19998887766 + (i % 10000)}" if i % 3 == 0 else None
            message = f"Processing request {i}"
            lines.append(create_text_log_line(thread, msisdn, message))

        def run():
            """Execute run.



            Returns:

                Return value produced by the callable.

            """

            results = [LogReorderer._extract(line) for line in lines]
            return results

        stats = measure(run, warmup=1, runs=3)
        stats["lines"] = len(lines)
        stats["lines_per_sec"] = len(lines) / (stats["median_ms"] / 1000)

        print(
            f"\n[log_extract_text] {len(lines):,} lines: {stats['median_ms']:.2f}ms ({stats['lines_per_sec']:,.0f}/s)"
        )

        bench_file = BENCH_DIR / "log_extract_baseline_100k_text.json"
        bench_file.write_text(str(stats))

        assert len(lines) == 100000

    def test_extract_100k_mixed_lines(self, tmp_path: Path):
        """Baseline: extract from 100K mixed JSON/text log lines."""
        lines = []
        for i in range(100_000):
            thread = f"Thread-{i % 10}"
            msisdn = f"55{19998887766 + (i % 10000)}" if i % 3 == 0 else None
            message = f"Processing request {i}"
            if i % 2 == 0:
                lines.append(create_json_log_line(thread, msisdn, message))
            else:
                lines.append(create_text_log_line(thread, msisdn, message))

        def run():
            """Execute run.



            Returns:

                Return value produced by the callable.

            """

            results = [LogReorderer._extract(line) for line in lines]
            return results

        stats = measure(run, warmup=1, runs=3)
        stats["lines"] = len(lines)
        stats["lines_per_sec"] = len(lines) / (stats["median_ms"] / 1000)

        print(
            f"\n[log_extract_mixed] {len(lines):,} lines: {stats['median_ms']:.2f}ms ({stats['lines_per_sec']:,.0f}/s)"
        )

        bench_file = BENCH_DIR / "log_extract_baseline_100k_mixed.json"
        bench_file.write_text(str(stats))

        assert len(lines) == 100000


class TestLogReordererReorderBaseline:
    """Benchmark LogReorderer.reorder with I/O."""

    def test_reorder_10k_lines_file(self, tmp_path: Path):
        """Baseline: reorder 10K lines from file."""
        log_file = tmp_path / "test_10k.log"

        with open(log_file, "w", encoding="utf-8") as f:
            f.write("===== HEADER =====\n")
            for i in range(10_000):
                thread = f"Thread-{i % 5}"
                msisdn = f"55{19998887766 + (i % 100)}" if i % 3 == 0 else None
                message = f"Processing request {i}"
                f.write(create_json_log_line(thread, msisdn, message) + "\n")

        def run():
            """Execute run.



            Returns:

                Return value produced by the callable.



            Notes:

                Keep behavior deterministic and free of hidden side effects.

            """

            result = LogReorderer.reorder(log_file)
            return result

        stats = measure(run, warmup=1, runs=3)
        stats["lines"] = 10001
        stats["lines_per_sec"] = 10001 / (stats["median_ms"] / 1000)

        print(
            f"\n[log_reorder_10k] {stats['lines']:,} lines: {stats['median_ms']:.2f}ms ({stats['lines_per_sec']:,.0f}/s)"
        )

        bench_file = BENCH_DIR / "log_reorder_baseline_10k_lines.json"
        bench_file.write_text(str(stats))

        assert log_file.exists()

    def test_reorder_50k_lines_file(self, tmp_path: Path):
        """Baseline: reorder 50K lines from file."""
        log_file = tmp_path / "test_50k.log"

        with open(log_file, "w", encoding="utf-8") as f:
            f.write("===== HEADER =====\n")
            for i in range(50_000):
                thread = f"Thread-{i % 8}"
                msisdn = f"55{19998887766 + (i % 200)}" if i % 3 == 0 else None
                message = f"Processing request {i}"
                f.write(create_json_log_line(thread, msisdn, message) + "\n")

        def run():
            """Execute run.



            Returns:

                Return value produced by the callable.



            Notes:

                Keep behavior deterministic and free of hidden side effects.

            """

            result = LogReorderer.reorder(log_file)
            return result

        stats = measure(run, warmup=1, runs=3)
        stats["lines"] = 50001
        stats["lines_per_sec"] = 50001 / (stats["median_ms"] / 1000)

        print(
            f"\n[log_reorder_50k] {stats['lines']:,} lines: {stats['median_ms']:.2f}ms ({stats['lines_per_sec']:,.0f}/s)"
        )

        bench_file = BENCH_DIR / "log_reorder_baseline_50k_lines.json"
        bench_file.write_text(str(stats))

        assert log_file.exists()


class TestLogReordererIooptBaseline:
    """Benchmark LogReorderer.reorder for I/O optimization."""

    def test_reorder_100k_lines_with_buffers(self, tmp_path: Path):
        """Baseline: reorder 100K lines (tests I/O buffer impact)."""
        log_file = tmp_path / "test_100k.log"

        with open(log_file, "w", encoding="utf-8", buffering=8192) as f:
            f.write("===== HEADER =====\n")
            for i in range(100_000):
                thread = f"Thread-{i % 10}"
                msisdn = f"55{19998887766 + (i % 500)}" if i % 3 == 0 else None
                message = f"Processing request {i}"
                f.write(create_json_log_line(thread, msisdn, message) + "\n")

        def run():
            """Execute run.



            Returns:

                Return value produced by the callable.



            Notes:

                Keep behavior deterministic and free of hidden side effects.

            """

            result = LogReorderer.reorder(log_file)
            return result

        stats = measure(run, warmup=1, runs=20)
        stats["lines"] = 100001
        stats["lines_per_sec"] = 100001 / (stats["median_ms"] / 1000)

        print(
            f"\n[log_reorder_buffers] {stats['lines']:,} lines: {stats['median_ms']:.2f}ms ({stats['lines_per_sec']:,.0f}/s)"
        )

        bench_file = BENCH_DIR / "log_reorder_baseline_100k_buffers.json"
        bench_file.write_text(str(stats))

        assert log_file.exists()


class TestHttpTemplateCacheFlushBaseline:
    """Benchmark HttpTemplateCache index flush optimization."""

    def test_template_cache_set_get_1k_ops(self, tmp_path: Path):
        """Baseline: 1K set/get operations on HttpTemplateCache (tests flush overhead)."""
        from pff.shared.core.cache import CacheManager, HttpTemplateCache

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_manager = CacheManager(cache_dir=cache_dir)
        template_cache = HttpTemplateCache(cache_manager, namespace="bench")

        def run():
            """Execute run.



            Notes:

                Keep behavior deterministic and free of hidden side effects.

            """

            for i in range(1_000):
                url = f"https://api.example.com/subscriber/55{19998887766 + (i % 1000)}"
                template_cache.set(url, endpoint_type="subscriber", method="GET", ttl_days=7)
                template_cache.get(url, "subscriber", "GET")

        stats = measure(run, warmup=1, runs=20)
        stats["operations"] = 2_000
        stats["ops_per_sec"] = 2_000 / (stats["median_ms"] / 1000)

        print(
            f"\n[template_cache_flush] {stats['operations']:,} ops: {stats['median_ms']:.2f}ms ({stats['ops_per_sec']:,.0f}/s)"
        )

        bench_file = BENCH_DIR / "template_cache_baseline_1k_ops.json"
        bench_file.write_text(str(stats))
