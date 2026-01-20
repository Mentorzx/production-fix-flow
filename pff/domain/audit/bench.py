"""Audit micro-benchmarks.

Sprint 0 requires recording a baseline for the audit contract path. This module
benchmarks building + validating an `audit_report.json` payload and writes a
JSON artifact under `outputs/benchmarks/`.
"""

from __future__ import annotations

import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pff.shared import FileManager, logger

from .report import AuditReportBuilder
from .schema import AuditReportSchemaValidator


@dataclass(frozen=True)
class BenchmarkStats:
    """Summary statistics for a benchmark series."""

    n: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    q = min(max(q, 0.0), 1.0)
    idx = int(round((len(sorted_values) - 1) * q))
    return sorted_values[idx]


def _summarize(samples_ms: list[float]) -> BenchmarkStats:
    ordered = sorted(samples_ms)
    n = len(ordered)
    if n == 0:
        return BenchmarkStats(n=0, mean_ms=0.0, p50_ms=0.0, p95_ms=0.0, min_ms=0.0, max_ms=0.0)
    return BenchmarkStats(
        n=n,
        mean_ms=sum(ordered) / n,
        p50_ms=_percentile(ordered, 0.50),
        p95_ms=_percentile(ordered, 0.95),
        min_ms=ordered[0],
        max_ms=ordered[-1],
    )


def run_audit_report_contract_benchmark(
    *,
    iterations: int = 200,
    outputs_dir: Path | None = None,
    file_manager: FileManager | None = None,
) -> Path:
    """Benchmark building+validating a schema-valid audit report.

    Args:
        iterations: Number of repetitions.
        outputs_dir: Root outputs dir (defaults to ./outputs).
        file_manager: Optional FileManager.

    Returns:
        Path to the benchmark JSON artifact under outputs/benchmarks/.
    """
    fm = file_manager or FileManager()
    root_outputs = outputs_dir or Path("outputs")
    fm.ensure_dir(root_outputs)
    bench_dir = root_outputs / "benchmarks"
    fm.ensure_dir(bench_dir)

    validator = AuditReportSchemaValidator(file_manager=fm)
    builder = AuditReportBuilder(
        outputs_dir=root_outputs, schema_validator=validator, file_manager=fm
    )

    document: dict[str, Any] = {"id": 1, "payload": {"x": 1, "y": "abc"}}
    baseline_key: dict[str, Any] = {"name": "benchmark", "window": "synthetic"}
    schema_version: str = "v1"

    samples_ms: list[float] = []
    for _ in range(max(1, int(iterations))):
        t0 = time.perf_counter()
        builder.build_report(
            document=document,
            baseline_key=baseline_key,
            schema_version=schema_version,
            findings=[],
            meta_overrides={"source_system": "bench"},
        )
        t1 = time.perf_counter()
        samples_ms.append((t1 - t0) * 1000.0)

    stats = _summarize(samples_ms)
    payload = {
        "benchmark": "audit_report_contract_build_validate",
        "stats": {
            "n": stats.n,
            "mean_ms": stats.mean_ms,
            "p50_ms": stats.p50_ms,
            "p95_ms": stats.p95_ms,
            "min_ms": stats.min_ms,
            "max_ms": stats.max_ms,
        },
        "env": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    }

    out_path = bench_dir / "audit_report_contract_baseline.json"
    fm.save(payload, out_path)
    logger.info(
        "benchmark_contrato_laudo "
        f"n={stats.n} mean_ms={stats.mean_ms:.3f} "
        f"p50_ms={stats.p50_ms:.3f} p95_ms={stats.p95_ms:.3f} "
        f"path={out_path}"
    )
    return out_path


if __name__ == "__main__":
    run_audit_report_contract_benchmark()
