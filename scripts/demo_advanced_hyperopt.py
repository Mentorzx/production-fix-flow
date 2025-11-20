#!/usr/bin/env python3
"""
Demo script for Advanced Hyperparameter Optimization Extensions.

Shows all 6 advanced features in action.

Usage:
    python scripts/demo_advanced_hyperopt.py [--feature FEATURE]

Features:
    multi-objective  - Multi-objective optimization with Pareto front
    nas              - Neural Architecture Search
    distributed      - Distributed optimization with Ray Tune
    dashboard        - Optuna Dashboard
    importance       - Hyperparameter importance analysis
    transfer         - Transfer learning from previous runs
    all              - Run all features (demo mode)
"""

import argparse
import sys
from typing import Dict, Any

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score

# Add project root to path
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pff.utils import logger

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print(" Optuna not available. Install with: pip install optuna")
    sys.exit(1)

from scripts.advanced_hyperopt_extensions import (
    AdvancedHyperparameterOptimizer,
    MultiObjectiveOptimizer,
    NeuralArchitectureSearch,
    DistributedOptimizer,
    OptunaReporting,
    ImportanceAnalyzer,
    TransferLearningOptimizer,
)


# ═══════════════════════════════════════════════════════════════════════════
# Demo objective functions
# ═══════════════════════════════════════════════════════════════════════════

def generate_demo_data(n_samples=1000, n_features=20):
    """Generate synthetic classification data."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=15,
        n_redundant=5,
        random_state=42,
    )
    return X, y


def single_objective_function(trial: optuna.Trial) -> float:
    """Single-objective optimization example."""
    # Generate data
    X, y = generate_demo_data()
    
    # Suggest hyperparameters
    max_depth = trial.suggest_int('max_depth', 2, 10)
    n_estimators = trial.suggest_int('n_estimators', 10, 100, step=10)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    
    # Train model
    model = RandomForestClassifier(
        max_depth=max_depth,
        n_estimators=n_estimators,
        min_samples_split=min_samples_split,
        random_state=42,
    )
    
    # Evaluate with CV
    scores = cross_val_score(model, X, y, cv=3, scoring='f1')
    
    return scores.mean()


def multi_objective_function(trial: optuna.Trial) -> tuple:
    """Multi-objective optimization example (returns tuple of objectives)."""
    X, y = generate_demo_data()
    
    max_depth = trial.suggest_int('max_depth', 2, 10)
    n_estimators = trial.suggest_int('n_estimators', 10, 100, step=10)
    
    model = RandomForestClassifier(
        max_depth=max_depth,
        n_estimators=n_estimators,
        random_state=42,
    )
    
    model.fit(X[:800], y[:800])
    y_pred = model.predict(X[800:])
    y_proba = model.predict_proba(X[800:])[:, 1]
    
    f1 = f1_score(y[800:], y_pred)
    auc = roc_auc_score(y[800:], y_proba)
    
    # Return multiple objectives
    return f1, auc


# ═══════════════════════════════════════════════════════════════════════════
# Feature Demos
# ═══════════════════════════════════════════════════════════════════════════

def demo_multi_objective():
    """Demo 1: Multi-objective optimization with Pareto front."""
    print("\n" + "="*70)
    print(" DEMO 1: Multi-Objective Optimization (Pareto Front)")
    print("="*70)
    
    optimizer = MultiObjectiveOptimizer()
    study = optimizer.create_study("demo_multi_obj")
    
    print("Running optimization with 2 objectives: F1-Score + ROC-AUC")
    study.optimize(multi_objective_function, n_trials=30, show_progress_bar=True)
    
    # Extract Pareto front
    pareto_trials, pareto_df = optimizer.extract_pareto_front(study)
    
    print(f"\n Found {len(pareto_trials)} Pareto-optimal solutions")
    print("\nTop 3 Pareto solutions:")
    print(pareto_df.head(3))
    
    # Save Pareto front
    output_file = optimizer.save_pareto_front(study)
    print(f"\n Saved Pareto front to: {output_file}")


def demo_nas():
    """Demo 2: Neural Architecture Search."""
    print("\n" + "="*70)
    print(" DEMO 2: Neural Architecture Search (NAS)")
    print("="*70)
    
    nas = NeuralArchitectureSearch()
    
    # Demo architecture suggestion
    study = optuna.create_study(direction='maximize')
    
    def nas_objective(trial):
        architecture = nas.suggest_architecture(trial)
        print(f"\nTrial {trial.number}: Testing architecture with {architecture['n_layers']} layers")
        # Simulate score (in real use, would train and evaluate the model)
        return np.random.random()
    
    print("Searching for optimal neural network architecture...")
    study.optimize(nas_objective, n_trials=10)
    
    print(f"\n Best architecture found:")
    best_arch = nas.suggest_architecture(study.best_trial)
    print(f"  Layers: {best_arch['n_layers']}")
    print(f"  Optimizer: {best_arch['optimizer']}")
    print(f"  Learning rate: {best_arch['learning_rate']:.6f}")


def demo_importance():
    """Demo 5: Hyperparameter importance analysis."""
    print("\n" + "="*70)
    print(" DEMO 5: Hyperparameter Importance Analysis")
    print("="*70)
    
    # Run optimization
    study = optuna.create_study(direction='maximize')
    print("Running optimization to collect data...")
    study.optimize(single_objective_function, n_trials=50, show_progress_bar=True)
    
    # Analyze importance
    analyzer = ImportanceAnalyzer()
    importance_df = analyzer.analyze_importance(study, top_k=5)
    
    print("\n Hyperparameter Importance Analysis:")
    print(importance_df.to_string(index=False))
    
    # Save analysis
    output_file = analyzer.save_importance_analysis(study)
    print(f"\n Saved importance analysis to: {output_file}")


def demo_transfer():
    """Demo 6: Transfer learning from previous optimizations."""
    print("\n" + "="*70)
    print(" DEMO 6: Transfer Learning from Previous Runs")
    print("="*70)
    
    transfer = TransferLearningOptimizer()
    
    # Run first optimization
    print("\n1) Running FIRST optimization (cold start)...")
    study1 = optuna.create_study(direction='maximize')
    study1.optimize(single_objective_function, n_trials=20)
    
    # Save to history
    transfer.save_history(study1)
    print(f" Best score (cold start): {study1.best_value:.4f}")
    
    # Run second optimization with warm-start
    print("\n2) Running SECOND optimization (warm start)...")
    
    search_space = {
        'max_depth': None,
        'n_estimators': None,
        'min_samples_split': None,
    }
    
    warmstart_params = transfer.get_warmstart_params(search_space)
    print(f"Using {len(warmstart_params)} configurations from previous run")
    
    study2 = optuna.create_study(
        direction='maximize',
        sampler=transfer.create_warmstart_sampler(warmstart_params),
    )
    study2.optimize(single_objective_function, n_trials=20)
    
    print(f" Best score (warm start): {study2.best_value:.4f}")
    
    # Compare
    improvement = (study2.best_value - study1.best_value) / study1.best_value * 100
    print(f"\n Improvement: {improvement:+.2f}%")
    
    if improvement > 0:
        print(" Transfer learning helped find better solution!")
    else:
        print("Transfer learning manteve desempenho adequado")


def demo_all_features():
    """Demo all features integrated."""
    print("\n" + "="*70)
    print(" INTEGRATED DEMO: All 6 Features")
    print("="*70)
    
    # Create unified optimizer with all features
    optimizer = AdvancedHyperparameterOptimizer(
        enable_multi_objective=False,  # Keep simple for demo
        enable_nas=False,
        enable_distributed=False,  # Requires Ray
        enable_dashboard=False,  # Requires separate server
        enable_importance=True,
        enable_transfer=True,
    )
    
    # Run optimization
    search_space = {
        'max_depth': (2, 10),
        'n_estimators': (10, 100),
        'min_samples_split': (2, 10),
    }
    
    print("\nRunning integrated optimization with:")
    print("   Transfer learning (warm-start from previous runs)")
    print("   Importance analysis (identify key hyperparameters)")
    print("   Automated reporting (save comprehensive results)")
    
    study = optimizer.optimize(
        objective_fn=single_objective_function,
        search_space=search_space,
        n_trials=30,
        use_transfer=True,
    )
    
    print(f"\n Optimization complete!")
    print(f"Best score: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Demo Advanced Hyperparameter Optimization Features"
    )
    parser.add_argument(
        '--feature',
        choices=['multi-objective', 'nas', 'importance', 'transfer', 'all'],
        default='all',
        help='Feature to demo (default: all)'
    )
    
    args = parser.parse_args()
    
    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║   Advanced Hyperparameter Optimization Extensions v3.0           ║")
    print("║   Demo: 6 Cutting-Edge Features                                  ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")
    
    try:
        if args.feature == 'multi-objective':
            demo_multi_objective()
        elif args.feature == 'nas':
            demo_nas()
        elif args.feature == 'importance':
            demo_importance()
        elif args.feature == 'transfer':
            demo_transfer()
        elif args.feature == 'all':
            # Run quick demos
            print("\n Running quick demo of key features...")
            demo_importance()
            demo_transfer()
            demo_all_features()
            
            print("\n" + "="*70)
            print(" Want to see more?")
            print("="*70)
            print("Run specific demos:")
            print("  python scripts/demo_advanced_hyperopt.py --feature multi-objective")
            print("  python scripts/demo_advanced_hyperopt.py --feature nas")
            print("  python scripts/demo_advanced_hyperopt.py --feature importance")
            print("  python scripts/demo_advanced_hyperopt.py --feature transfer")
        
        print("\n Demo complete!")
        
    except KeyboardInterrupt:
        print("\n  Demo interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.exception(f"Demo error: {e}")
        print(f"\n Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
