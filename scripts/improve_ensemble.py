#!/usr/bin/env python3
"""
Script para melhorar performance do ensemble através de:
1. Feature selection (reduzir de 484 para ~300 features)
2. Class balancing (melhorar recall classe 0)
3. Threshold optimization (além de 0.530)

Baseado na análise:
- Gap tuning vs real: 10.7%
- Recall classe 0 baixo: 54.8%
- Features esparsas: 1.4%
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
import polars as pl
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from loguru import logger

from pff import settings
from pff.utils.core.file_manager import FileManager

class EnsembleImprover:
    """Melhora performance do ensemble através de feature engineering."""
    
    def __init__(self):
        self.file_manager = FileManager()
        
    def analyze_feature_importance(self):
        """Analisa importância de features do modelo atual."""
        logger.info("📊 Analisando feature importance do modelo atual...")
        
        # Carregar metadata do modelo
        metadata_path = settings.OUTPUTS_DIR / "ensemble" / "model_metadata.json"
        if not metadata_path.exists():
            logger.warning("⚠️ model_metadata.json não encontrado")
            return None
            
        with open(metadata_path) as f:
            metadata = json.load(f)
            
        logger.info(f"✅ Modelo carregado:")
        logger.info(f"   Features: {metadata.get('n_features', 'N/A')}")
        logger.info(f"   Hybrid contribution: {metadata.get('hybrid_contribution', 'N/A')}")
        logger.info(f"   Symbolic contribution: {metadata.get('symbolic_contribution', 'N/A')}")
        
        return metadata
        
    def suggest_improvements(self, metadata):
        """Sugere melhorias baseadas na análise."""
        logger.info("\n🎯 Sugestões de Melhoria:")
        
        # 1. Feature selection
        n_features = metadata.get('n_features', 484)
        if n_features > 300:
            target = int(n_features * 0.6)  # Reduzir para 60%
            logger.info(f"\n1️⃣ FEATURE SELECTION:")
            logger.info(f"   Atual: {n_features} features")
            logger.info(f"   Target: ~{target} features (top 60%)")
            logger.info(f"   Método: SelectKBest ou feature_importances_")
            
        # 2. Class balancing
        hybrid_contrib = metadata.get('hybrid_contribution', 0.5)
        symbolic_contrib = metadata.get('symbolic_contribution', 0.5)
        
        if abs(hybrid_contrib - 0.5) > 0.15:  # Desbalanceado
            logger.info(f"\n2️⃣ CLASS BALANCING:")
            logger.info(f"   Hybrid: {hybrid_contrib:.1%}")
            logger.info(f"   Symbolic: {symbolic_contrib:.1%}")
            logger.info(f"   Ajustar: class_weight='balanced' no XGBoost")
            
        # 3. Threshold optimization
        logger.info(f"\n3️⃣ THRESHOLD OPTIMIZATION:")
        logger.info(f"   Atual: 0.530 (F1=0.7025)")
        logger.info(f"   Testar: 0.48-0.55 (range expandido)")
        logger.info(f"   Objetivo: Maximizar F1 + balancear recall classe 0")
        
        # 4. Sparsity reduction
        logger.info(f"\n4️⃣ SPARSITY REDUCTION:")
        logger.info(f"   Atual: 1.4% features simbólicas ativas")
        logger.info(f"   Aumentar threshold AnyBURL: 0.0317 → 0.05")
        logger.info(f"   Reduz ruído, mantém regras de alta confiança")

def main():
    """Executa análise e sugestões de melhoria."""
    logger.info("=" * 70)
    logger.info("🚀 ENSEMBLE IMPROVEMENT ANALYZER")
    logger.info("=" * 70)
    
    improver = EnsembleImprover()
    
    # Análise
    metadata = improver.analyze_feature_importance()
    
    if metadata:
        # Sugestões
        improver.suggest_improvements(metadata)
    
    logger.info("\n" + "=" * 70)
    logger.info("📝 PRÓXIMOS PASSOS:")
    logger.info("=" * 70)
    logger.info("1. Implementar feature selection no advanced_trainer.py")
    logger.info("2. Adicionar class_weight='balanced' no XGBoost")
    logger.info("3. Expandir grid search de threshold: 0.45-0.60")
    logger.info("4. Aumentar AnyBURL threshold em config/kg.yaml")
    logger.info("=" * 70)

if __name__ == "__main__":
    main()
