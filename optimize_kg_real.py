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


def main():
    """
    Otimização usando dados reais do PFF Knowledge Graph
    """
    print("=" * 70)
    print("🎯 Otimização com DADOS REAIS do PFF Knowledge Graph")
    print("=" * 70)

    print("\n📊 Carregando dados reais...")
    print("   Fonte: /data/models/kg/*.parquet")
    print("   Formato: (subject, predicate, object) triplets")

    print("\n🚀 Iniciando otimização...")

    result = optimize_kg_hyperparameters(
        n_trials=50,
        strategy="optuna",
        study_name="pff_kg_real_data_optimization",
        enable_mlflow=True,
    )

    # Exibir resultados
    print("\n" + "=" * 70)
    print("✅ OTIMIZAÇÃO COM DADOS REAIS CONCLUÍDA!")
    print("=" * 70)

    print(f"\n📊 Estatísticas dos dados reais:")
    if 'real_data_info' in result:
        info = result['real_data_info']
        print(f"   • Triplets de treinamento: {info.get('n_train', 'N/A')}")
        print(f"   • Triplets de validação: {info.get('n_valid', 'N/A')}")
        print(f"   • Entidades únicas: {info.get('n_entities', 'N/A')}")
        print(f"   • Predicados: {info.get('n_predicates', 'N/A')}")

    print(f"\n🎯 Resultados da otimização:")
    print(f"   • Melhor score: {result['best_value']:.4f}")
    print(f"   • Trials executados: {result['n_trials']}")
    print(f"   • Tempo total: {result['optimization_time']:.2f}s")
    print(f"   • Framework: {result['framework']}")

    print(f"\n⚙️ Melhores hiperparâmetros:")
    for param, value in result['best_params'].items():
        print(f"   • {param}: {value}")

    print(f"\n📁 Arquivos salvos:")
    if 'best_params_file' in result:
        print(f"   • Parâmetros: {result['best_params_file']}")
    if 'output_dir' in result:
        print(f"   • Gráficos: {result['output_dir']}")

    # Show saved models
    if 'best_models_dir' in result:
        print(f"\n📦 Melhores modelos salvos em: {result['best_models_dir']}")
        if 'best_model_files' in result:
            for model_name, model_path in result['best_model_files'].items():
                print(f"   • {model_name.upper()}: {model_path.name}")

        # Show individual params files
        print(f"\n📄 Hiperparâmetros individuais:")
        best_models_dir = result['best_models_dir']
        for model_name in ['transe', 'anyburl', 'lightgbm', 'ensemble']:
            param_file = best_models_dir / f"best_params_{model_name}.json"
            if param_file.exists():
                print(f"   • {model_name}: {param_file.name}")

    if 'mlflow_tracking_uri' in result and result['mlflow_tracking_uri']:
        print(f"\n🌐 MLflow UI:")
        print(f"   • URL: {result['mlflow_tracking_uri']}")
        print(f"   • Comando: mlflow ui")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ Otimização interrompida pelo usuário")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Erro: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
