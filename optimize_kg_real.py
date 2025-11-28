#!/usr/bin/env python3
"""
Otimização de Hiperparâmetros com DADOS REAIS do PFF Knowledge Graph

Este arquivo usa os dados reais do PFF KG para otimização.
Dados carregados de: /data/models/kg/*.parquet

Execute com:
    python optimize_kg_real.py
"""

import asyncio
import atexit
import faulthandler
import gc
import os
import signal
import sys
import argparse
from pathlib import Path

# Enable faulthandler to get better traceback on segfaults
faulthandler.enable()

# Disable Numba threading for cleaner shutdown
os.environ.setdefault('NUMBA_NUM_THREADS', '1')

# Adicionar scripts ao path
sys.path.insert(0, str(Path(__file__).parent / "scripts"))

from scripts.optimization import optimize_kg_hyperparameters
from pff.utils.core.logger import logger
from pff.db.connection import close_connection_pool
from pff.utils.core.cache import shutdown_all_cache_janitors


def main():
    """
    Otimização usando dados reais do PFF Knowledge Graph
    """
    parser = argparse.ArgumentParser(description="Otimização de Hiperparâmetros PFF KG")
    parser.add_argument("--model", type=str, default="rotate", choices=["rotate"],
                        help="Modelo KGE a ser utilizado (rotate é o único suportado)")
    parser.add_argument("--trials", type=int, default=50,
                        help="Número de trials para otimização")
    parser.add_argument("--study-name", type=str, default=None,
                        help="Nome do estudo Optuna")
    
    args = parser.parse_args()
    
    study_name = args.study_name or f"pff_kg_real_{args.model}"
    
    logger.info("=" * 70)
    logger.info(f" Otimização com DADOS REAIS do PFF Knowledge Graph ({args.model.upper()})")
    logger.info("=" * 70)

    logger.info(" Carregando dados reais...")
    logger.info("   Fonte: /data/models/kg/*.parquet")
    logger.info("   Formato: (subject, predicate, object) triplets")

    logger.info(f" Iniciando otimização com {args.trials} trials...")

    result = optimize_kg_hyperparameters(
        n_trials=args.trials,
        strategy="optuna",
        study_name=study_name,
        enable_mlflow=True,
        kge_model=args.model,
    )

    # Exibir resultados
    logger.info("=" * 70)
    logger.success(" OTIMIZAÇÃO COM DADOS REAIS CONCLUÍDA!")
    logger.info("=" * 70)

    logger.info(" Estatísticas dos dados reais:")
    if 'real_data_info' in result:
        info = result['real_data_info']
        logger.info(f"   • Triplets de treinamento: {info.get('n_train', 'N/A')}")
        logger.info(f"   • Triplets de validação: {info.get('n_valid', 'N/A')}")
        logger.info(f"   • Entidades únicas: {info.get('n_entities', 'N/A')}")
        logger.info(f"   • Predicados: {info.get('n_predicates', 'N/A')}")

    logger.info(" Resultados da otimização:")
    if result.get('best_value') is not None:
        logger.info(f"   • Melhor score: {result['best_value']:.4f}")
    else:
        logger.warning("   • Best score: N/A (Optimization failed or no solution found)")
        
    logger.info(f"   • Trials executados: {result.get('n_trials', 0)}")
    logger.info(f"   • Tempo total: {result.get('optimization_time', 0):.2f}s")
    logger.info(f"   • Framework: {result.get('framework', 'unknown')}")

    logger.info(" Melhores hiperparâmetros:")
    for param, value in result['best_params'].items():
        logger.info(f"   • {param}: {value}")

    logger.info(" Arquivos salvos:")
    if 'best_params_file' in result:
        logger.info(f"   • Parâmetros: {result['best_params_file']}")
    if 'output_dir' in result:
        logger.info(f"   • Gráficos: {result['output_dir']}")

    # Show saved models
    if 'best_models_dir' in result:
        logger.info(f" Melhores modelos salvos em: {result['best_models_dir']}")
        if 'best_model_files' in result:
            for model_name, model_path in result['best_model_files'].items():
                logger.info(f"   • {model_name.upper()}: {model_path.name}")

        # Show individual params files
        logger.info(" Hiperparâmetros individuais:")
        best_models_dir = result['best_models_dir']
        for model_name in ['rotate', 'anyburl', 'lightgbm', 'ensemble']:
            param_file = best_models_dir / f"best_params_{model_name}.json"
            if param_file.exists():
                logger.info(f"   • {model_name}: {param_file.name}")

    if 'mlflow_tracking_uri' in result and result['mlflow_tracking_uri']:
        logger.info(" MLflow UI:")
        logger.info(f"   • URL: {result['mlflow_tracking_uri']}")
        logger.info("   • Comando: mlflow ui")

    logger.info("=" * 70)


def cleanup():
    """Cleanup resources to prevent segfault on exit."""
    # Stop all cache janitor threads FIRST (they're the main culprit)
    try:
        shutdown_all_cache_janitors()
    except Exception:
        pass
    
    # Force garbage collection
    gc.collect()
    
    try:
        # Close PostgreSQL connection pool
        loop = asyncio.new_event_loop()
        loop.run_until_complete(close_connection_pool())
        loop.close()
    except Exception:
        pass
    
    # Cleanup Numba threading
    try:
        from numba.core.runtime import nrt
        # Force NRT finalization
        nrt.rtsys.shutdown()
    except Exception:
        pass
    
    # Final GC
    gc.collect()


# Register cleanup at exit
atexit.register(cleanup)


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("Optimization interrupted by user")
        exit_code = 130
    except Exception as e:
        logger.error(f" Erro: {e}")
        import traceback
        logger.error(traceback.format_exc())
        exit_code = 1
    
    # Explicit cleanup before exit
    cleanup()
    
    # Use os._exit to skip Python cleanup that causes segfault
    os._exit(exit_code)
