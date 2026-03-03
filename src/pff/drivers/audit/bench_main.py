"""CLI entrypoint for audit benchmark routines."""

from __future__ import annotations

import argparse
from pathlib import Path

from pff.domain.audit.bench import run_audit_report_contract_benchmark
from pff.shared.core.file_manager import FileManager
from pff.shared.core.logging import logger


def main() -> None:
    """Parse CLI args and run audit benchmark."""
    parser = argparse.ArgumentParser(description="PFF Audit benchmark runner")
    parser.add_argument(
        "--iterations", type=int, default=200, help="Número de iterações"
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=None,
        help="Diretório raiz de saída (padrão: ./outputs)",
    )
    args = parser.parse_args()
    file_manager = FileManager()

    artifact = run_audit_report_contract_benchmark(
        iterations=args.iterations,
        outputs_dir=args.outputs_dir,
        schema_reader=file_manager,
    )
    file_manager.save(artifact.payload, artifact.out_path)
    logger.info(
        "benchmark_contrato_laudo "
        f"n={artifact.stats.n} mean_ms={artifact.stats.mean_ms:.3f} "
        f"p50_ms={artifact.stats.p50_ms:.3f} p95_ms={artifact.stats.p95_ms:.3f} "
        f"path={artifact.out_path}"
    )


if __name__ == "__main__":
    main()
