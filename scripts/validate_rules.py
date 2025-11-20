#!/usr/bin/env python3
"""
Validate AnyBURL Rules
======================

This script loads the filtered rules (TSV) and calculates coverage and density metrics
on the training dataset to verify if the symbolic pipeline is viable.

Usage:
    python scripts/validate_rules.py
"""

import sys
import numpy as np
from pathlib import Path
from pff.utils import FileManager, logger
from pff.validators.ensembles.ensemble_wrappers.transformers import SymbolicFeatureExtractor
from pff.validators.ensembles.data_loader import EnsembleDataLoader

def validate_rules():
    logger.info("Iniciando validação de regras AnyBURL...")
    
    file_manager = FileManager()
    possible_paths = [
        Path("outputs/anyburl/rules_filtered.tsv"),
        Path("outputs/optimization_results/kg_ensemble/trials/trial_0000/anyburl/rules_filtered.tsv"),
        Path("outputs/pyclause/rules_anyburl.tsv")
    ]
    
    rules_path = None
    for p in possible_paths:
        if file_manager.exists(p):
            rules_path = p
            break
    
    if not rules_path:
        logger.error(f"No rules file found in: {[str(p) for p in possible_paths]}")
        sys.exit(1)
        
    logger.info(f"Carregando regras de: {rules_path}")
    
    # Load data
    loader = EnsembleDataLoader()
    try:
        X_train, y_train, _, _ = loader.load_ensemble_data()
        X_train = list(X_train)
        y_train = np.asarray(y_train, dtype=int)
        logger.info(f"Dados carregados: {len(X_train)} amostras")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        sys.exit(1)

    # Initialize extractor
    extractor = SymbolicFeatureExtractor(
        rules_path=str(rules_path),
        min_confidence_threshold=0.01,
        min_activation_ratio=0.001, # Permissive for validation
        min_coverage_threshold=0.001, # Permissive for validation
        activation_sample_size=2000
    )
    
    try:
        extractor.fit(X_train, y_train)
        logger.success(f"Extrator ajustado com sucesso. Regras retidas: {len(extractor.rules_)}")
        
        # Calculate metrics
        features = extractor.transform(X_train[:2000])
        features = np.asarray(features)
        
        if features.size == 0:
             logger.error("Feature matrix is empty!")
             sys.exit(1)

        activated = features > 0
        rule_coverage = activated.mean(axis=0)
        global_coverage = activated.any(axis=1).mean()
        avg_density = activated.sum() / activated.size
        
        logger.info("=" * 50)
        logger.info("MÉTRICAS DE VALIDAÇÃO")
        logger.info("=" * 50)
        logger.info(f"Regras Totais: {len(extractor.rules_)}")
        logger.info(f"Cobertura Global (amostras com >=1 regra): {global_coverage:.2%}")
        logger.info(f"Densidade Média da Matriz: {avg_density:.4%}")
        logger.info(f"Regras com Ativação > 0: {np.count_nonzero(rule_coverage > 0)}")
        logger.info(f"Regras com Ativação > 1%: {np.count_nonzero(rule_coverage > 0.01)}")
        
        if global_coverage < 0.01:
            logger.error("CRITICAL: Global coverage below 1%. Optimization will likely fail.")
            sys.exit(1)
        else:
            logger.success("Validação concluída. Cobertura aceitável.")
            
    except Exception as e:
        logger.error(f"Error during validation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    validate_rules()
