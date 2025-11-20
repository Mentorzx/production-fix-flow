#!/usr/bin/env python3
"""
Otimização de Hiperparâmetros com DADOS REAIS do PFF Knowledge Graph

Este arquivo usa os dados reais do PFF KG para otimização.
Dados carregados de: /data/models/kg/*.parquet

Execute com:
    python optimize_kg_real.py
"""

import sys
from pathlib import Path

# Adicionar scripts ao path
sys.path.insert(0, str(Path(__file__).parent / "scripts"))

from scripts.optimization import optimize_kg_hyperparameters
from pff.utils.core.logger import logger


def main():
    """
    Otimização usando dados reais do PFF Knowledge Graph
    """
    logger.info("=" * 70)
    logger.info(" Otimização com DADOS REAIS do PFF Knowledge Graph")
    logger.info("=" * 70)

    logger.info(" Carregando dados reais...")
    logger.info("   Fonte: /data/models/kg/*.parquet")
    logger.info("   Formato: (subject, predicate, object) triplets")

    logger.info(" Iniciando otimização...")

    result = optimize_kg_hyperparameters(
        n_trials=50,
        strategy="optuna",
        study_name="pff_kg_real_data_optimization",
        enable_mlflow=True,
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
        logger.warning("   • Melhor score: N/A (Otimização falhou ou não encontrou solução)")
        
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
        for model_name in ['transe', 'anyburl', 'lightgbm', 'ensemble']:
            param_file = best_models_dir / f"best_params_{model_name}.json"
            if param_file.exists():
                logger.info(f"   • {model_name}: {param_file.name}")

    if 'mlflow_tracking_uri' in result and result['mlflow_tracking_uri']:
        logger.info(" MLflow UI:")
        logger.info(f"   • URL: {result['mlflow_tracking_uri']}")
        logger.info("   • Comando: mlflow ui")

    logger.info("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning(" Otimização interrompida pelo usuário")
        sys.exit(130)
    except Exception as e:
        logger.error(f" Erro: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
