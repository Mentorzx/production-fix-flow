import argparse
import asyncio
import sys

from pff.utils import logger

from .config import KGConfig
from .pipeline import KGPipeline
from pff.config import settings

"""
Main entry point for the Knowledge Graph Completion (KGC) pipeline.
"""

# Base directory for relative path resolution
BASE_DIRECTORY = settings.ROOT_DIR


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure command line argument parser."""
    parser = argparse.ArgumentParser(description="Pipeline KGC com PyClause e Ray")
    parser.add_argument(
        "--config",
        default=BASE_DIRECTORY / "config" / "kg.yaml",
        help="Caminho para arquivo de configuração YAML",
    )

    parser.add_argument(
        "command",
        choices=["all", "build", "learn", "rank"],
        default="all",
        nargs="?",  # Torna o comando opcional, com 'all' como padrão
        help="Comando a ser executado: 'build' (builder+preprocess), 'learn', 'rank', ou 'all' (executa tudo).",
    )
    return parser


async def main() -> None:
    """
    Main entry point that runs pipeline stages in isolated or sequential manner.
    """
    parser = create_argument_parser()
    args = parser.parse_args()

    try:
        config = KGConfig(args.config)
        pipeline = KGPipeline(config)

        if args.command in ["all", "build"]:
            logger.info("Executando comando 'build' (Builder + Preprocess)...")
            await pipeline.run_build_and_preprocess()

        if args.command in ["all", "learn"]:
            logger.info("Executando comando 'learn' (AnyBURL)...")
            # Executar a aprendizagem de regras
            await pipeline.run_learn_rules()

        if args.command in ["all", "rank"]:
            logger.info("Executando comando 'rank' (PyClause + Ray)...")
            # Executar o ranking
            await pipeline.run_ranking()

    except KeyboardInterrupt:
        logger.info("Pipeline interrompida pelo usuário.")
        sys.exit(0)
    except Exception as e:
        logger.exception(f"Falha crítica na execução da pipeline: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
