#!/usr/bin/env python3
"""KG Data Preprocessing Script.

This script preprocesses Knowledge Graph data to improve DSLFM training quality.
It addresses critical data quality issues found in PFF Telecom KG:

Issues addressed:
- 62% duplicate triples
- 11.7% self-loops
- No inverse relations

Expected improvements:
- MRR: 0.486 -> 0.55-0.65 (matching WN18RR benchmark)
- Hits@10: 71.2% -> 80%+

Usage:
    poetry run python scripts/preprocess_kg.py [--no-inverses] [--no-backup]

Pattern: Command Pattern - encapsulates preprocessing as executable command.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pff.domain.kg.kg.config import KGConfig  # noqa: E402

from pff.domain.kg.data_optimizer import (  # noqa: E402
    OptimizationConfig,
    TelecomDataOptimizer,
)
from pff.shared import logger  # noqa: E402
from pff.shared.core.config import KG_PIPELINE_CONFIG_PATH  # noqa: E402


def main() -> int:
    """Run KG preprocessing pipeline.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    parser = argparse.ArgumentParser(description="Preprocess KG data for improved DSLFM training")
    parser.add_argument(
        "--no-duplicates",
        action="store_true",
        help="Skip duplicate removal (not recommended)",
    )
    parser.add_argument(
        "--no-self-loops",
        action="store_true",
        help="Skip self-loop removal (not recommended)",
    )
    parser.add_argument("--no-inverses", action="store_true", help="Skip adding inverse relations")
    parser.add_argument(
        "--no-backup", action="store_true", help="Skip creating backup of original data"
    )
    parser.add_argument(
        "--min-degree",
        type=int,
        default=2,
        help="Minimum entity degree to keep (default: 2)",
    )
    parser.add_argument(
        "--min-support",
        type=int,
        default=20,
        help="Minimum relation support to keep (default: 20)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=KG_PIPELINE_CONFIG_PATH,
        help="Path to KG config file",
    )

    args = parser.parse_args()

    logger.info("PRE-PROCESSAMENTO DE DADOS KG PARA DSLFM")

    kg_config = KGConfig(args.config)
    train_path = kg_config.get_split_path("train")

    if not train_path.exists():
        logger.error(f"Training data not found at: {train_path}")
        return 1

    logger.info(f"Arquivo de treino: {train_path}")

    config = OptimizationConfig(
        remove_duplicates=not args.no_duplicates,
        remove_self_loops=not args.no_self_loops,
        add_inverse_relations=not args.no_inverses,
        preserve_original=not args.no_backup,
        min_entity_degree=args.min_degree,
        min_relation_support=args.min_support,
        log_statistics=True,
    )

    logger.info("Configuracao:")
    logger.info(f"  - Remover duplicatas: {config.remove_duplicates}")
    logger.info(f"  - Remover self-loops: {config.remove_self_loops}")
    logger.info(f"  - Adicionar inversas: {config.add_inverse_relations}")
    logger.info(f"  - Grau minimo: {config.min_entity_degree}")
    logger.info(f"  - Suporte minimo: {config.min_relation_support}")

    optimizer = TelecomDataOptimizer(config, args.config)

    try:
        optimized_df, stats = optimizer.optimize_telecom_data(train_path)

        logger.success("PRE-PROCESSAMENTO CONCLUIDO COM SUCESSO!")

        original = stats["original_stats"]["num_triples"]
        final = stats["final_stats"]["num_triples"]
        ratio = stats["improvements"]["size_ratio"]

        logger.info(f"Triplas originais: {original:,}")
        logger.info(f"Triplas finais: {final:,}")
        logger.info(f"Razao: {ratio:.1%}")
        logger.info(f"Arquivo otimizado: {stats['paths']['optimized']}")

        logger.info("")
        logger.info("IMPACTO ESPERADO NO DSLFM:")
        logger.info("  MRR atual: ~0.486")
        logger.info("  MRR esperado: 0.55-0.65")
        logger.info("  Hits@10 atual: ~71.2%")
        logger.info("  Hits@10 esperado: 80%+")
        logger.info("")
        logger.info("Proximo passo: Re-treinar o DSLFM com os dados otimizados")
        logger.info("  poetry run pff learn --use-optimized")

        return 0

    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
