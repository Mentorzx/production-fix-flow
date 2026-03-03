"""CLI entrypoint for Knowledge Graph Builder.

Composition root for KG builder inbound adapter.
"""

from __future__ import annotations

import argparse

from pff.domain.kg.builder import KGBuilder
from pff.shared.acceleration.asyncio_runner import run_coroutine_sync


def main() -> None:
    """Parse CLI args and execute KG builder."""
    parser = argparse.ArgumentParser(description="PFF Knowledge Graph Builder")
    parser.add_argument(
        "source", nargs="?", help="Caminho para o arquivo ou diretório fonte"
    )
    parser.add_argument("--output", "-o", help="Diretório de saída")
    parser.add_argument(
        "--max-members", "-n", type=int, help="Limite de membros a processar"
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Desativa processamento paralelo",
    )
    parser.add_argument("--workers", "-w", type=int, help="Número de workers")
    parser.add_argument(
        "--disk-cache", action="store_true", help="Ativa cache em disco"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed para reprodutibilidade"
    )

    args = parser.parse_args()

    builder = KGBuilder(
        source_path=args.source,
        output_dir=args.output,
        max_members=args.max_members,
        parallel=not args.no_parallel,
        workers=args.workers,
        disk_cache=args.disk_cache,
        seed=args.seed,
    )
    run_coroutine_sync(builder.run())


if __name__ == "__main__":
    main()
