#!/usr/bin/env python3
"""
Auto Threshold Tuner
System for automatic threshold tuning based on cross-validation metrics.

Optimizes:
1. min_confidence_threshold for overfitting prevention
2. max_violation_percentage for rule filtering
3. XGBoost hyperparameters for model balance
"""

import sys
import json
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
import yaml
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("⚠️ Optuna not available. Install with: pip install optuna")

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ThresholdConfig:
    """Configuration for threshold optimization."""
    min_confidence_threshold: float = 0.05
    max_violation_percentage: float = 200.0
    target_violation_range: Tuple[float, float] = (50.0, 150.0)
    target_symbolic_ratio: float = 0.75  # Target: 75% symbolic, 25% hybrid
    target_f1_score: float = 0.75
    cv_folds: int = 5
    n_trials: int = 100
    random_state: int = 42

@dataclass
class OptimizationResult:
    """Results of threshold optimization."""
    best_thresholds: Dict[str, float]
    best_score: float
    cv_scores: Dict[str, List[float]]
    optimization_history: List[Dict[str, Any]]
    timestamp: datetime
    target_metric: str

class AutoThresholdTuner:
    """Automatic threshold tuning system."""

    def __init__(self, config: ThresholdConfig = None):
        self.config = config or ThresholdConfig()
        self.optimization_history = []
        self.best_config = None

    def load_sample_data(self, data_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load sample data for optimization."""
        try:
            # Try to load from cache or generated samples
            if data_path.endswith('.parquet'):
                df = pd.read_parquet(data_path)
                X = df.drop(['label'], errors='ignore').values
                y = df['label'].values if 'label' in df.columns else None
            else:
                # Generate synthetic data if no real data available
                logger.info("🔄 Generating synthetic data for optimization...")
                X, y = self._generate_synthetic_data()

            logger.info(f"✅ Loaded data: X.shape={X.shape}, y.shape={y.shape}")
            return X, y

        except Exception as e:
            logger.error(f"❌ Error loading data: {e}")
            # Fallback to synthetic data
            return self._generate_synthetic_data()

    def _generate_synthetic_data(self, n_samples: int = 1000, n_features: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic data for optimization."""
        np.random.seed(self.config.random_state)

        # Generate features with different patterns
        X = np.random.randn(n_samples, n_features)

        # Add some correlated features to simulate symbolic rules
        for i in range(0, n_features, 10):
            X[:, i] = (X[:, i] > 0).astype(float)  # Binary features

        # Generate labels with some complexity
        weights = np.random.randn(n_features)
        linear_part = X @ weights

        # Add non-linear interactions
        for i in range(0, n_features, 20):
            interaction = X[:, i] * X[:, min(i+1, n_features-1)]
            linear_part += 0.5 * interaction

        # Convert to probabilities and binary labels
        probs = 1 / (1 + np.exp(-linear_part))
        y = (probs > 0.5).astype(int)

        # Ensure balance
        if np.mean(y) < 0.3 or np.mean(y) > 0.7:
            # Rebalance if too skewed
            pos_indices = np.where(y == 1)[0]
            neg_indices = np.where(y == 0)[0]
            min_count = min(len(pos_indices), len(neg_indices))

            selected_pos = np.random.choice(pos_indices, min_count, replace=False)
            selected_neg = np.random.choice(neg_indices, min_count, replace=False)

            all_indices = np.concatenate([selected_pos, selected_neg])
            X = X[all_indices]
            y = y[all_indices]

        return X, y

    def evaluate_thresholds(self, X: np.ndarray, y: np.ndarray,
                          thresholds: Dict[str, float]) -> Dict[str, float]:
        """Evaluate threshold configuration with cross-validation."""
        from sklearn.ensemble import RandomForestClassifier

        cv = StratifiedKFold(n_splits=self.config.cv_folds,
                           shuffle=True,
                           random_state=self.config.random_state)

        # Simulate feature extraction with given thresholds
        X_filtered = self._apply_thresholds(X, thresholds)

        # Train model with CV
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=4,
            random_state=self.config.random_state,
            n_jobs=-1
        )

        # Calculate CV scores
        scores = {}
        scoring_metrics = {
            'f1': lambda y_true, y_pred: f1_score(y_true, y_pred, average='binary'),
            'precision': lambda y_true, y_pred: precision_score(y_true, y_pred, average='binary'),
            'recall': lambda y_true, y_pred: recall_score(y_true, y_pred, average='binary'),
            'roc_auc': lambda y_true, y_pred: roc_auc_score(y_true, y_pred)
        }

        for metric_name, metric_func in scoring_metrics.items():
            cv_scores = cross_val_score(model, X_filtered, y,
                                     cv=cv, scoring=metric_func if hasattr(model, 'score') else None)
            scores[metric_name] = cv_scores.mean()

        # Add penalty functions for optimization targets
        penalty = 0.0

        # Penalty for violation percentage outside target range
        if 'violation_percentage' in thresholds:
            violation_pct = thresholds['violation_percentage']
            if violation_pct < self.config.target_violation_range[0]:
                penalty += (self.config.target_violation_range[0] - violation_pct) * 0.01
            elif violation_pct > self.config.target_violation_range[1]:
                penalty += (violation_pct - self.config.target_violation_range[1]) * 0.01

        # Penalty for symbolic ratio too high
        if 'symbolic_ratio' in thresholds:
            symbolic_ratio = thresholds['symbolic_ratio']
            if symbolic_ratio > self.config.target_symbolic_ratio:
                penalty += (symbolic_ratio - self.config.target_symbolic_ratio) * 0.1

        # Combined score (weighted)
        combined_score = (
            0.4 * scores.get('f1', 0.0) +
            0.3 * scores.get('roc_auc', 0.0) +
            0.2 * scores.get('precision', 0.0) +
            0.1 * scores.get('recall', 0.0) -
            penalty
        )

        scores['combined'] = combined_score
        scores['penalty'] = penalty

        return scores

    def _apply_thresholds(self, X: np.ndarray, thresholds: Dict[str, float]) -> np.ndarray:
        """Simulate applying thresholds to features."""
        X_filtered = X.copy()

        # Simulate confidence threshold filtering
        if 'min_confidence_threshold' in thresholds:
            conf_threshold = thresholds['min_confidence_threshold']
            # Remove low-confidence features (simulate by zeroing out some columns)
            n_features_to_keep = int(X.shape[1] * (1.0 - conf_threshold))
            if n_features_to_keep > 0:
                # Keep features with highest variance (simulate confidence)
                feature_variances = np.var(X, axis=0)
                top_features = np.argsort(feature_variances)[-n_features_to_keep:]
                mask = np.zeros(X.shape[1], dtype=bool)
                mask[top_features] = True
                X_filtered = X_filtered[:, mask]

        # Simulate violation percentage filtering
        if 'max_violation_percentage' in thresholds:
            max_violation = thresholds['max_violation_percentage']
            # Simulate by removing samples that would have too many violations
            if max_violation < 100.0:
                n_samples_to_keep = int(X.shape[0] * (max_violation / 100.0))
                if n_samples_to_keep > 0 and n_samples_to_keep < X.shape[0]:
                    X_filtered = X_filtered[:n_samples_to_keep]

        return X_filtered

    def objective_function(self, trial, X: np.ndarray, y: np.ndarray) -> float:
        """Objective function for Optuna optimization."""
        # Suggest thresholds
        thresholds = {
            'min_confidence_threshold': trial.suggest_float('min_confidence_threshold', 0.01, 0.2),
            'max_violation_percentage': trial.suggest_float('max_violation_percentage', 50.0, 300.0),
        }

        # Evaluate thresholds
        scores = self.evaluate_thresholds(X, y, thresholds)

        # Store optimization history
        self.optimization_history.append({
            'trial_number': trial.number,
            'thresholds': thresholds.copy(),
            'scores': scores.copy(),
            'timestamp': datetime.now()
        })

        # Return combined score for optimization
        return scores['combined']

    def optimize_with_optuna(self, X: np.ndarray, y: np.ndarray) -> OptimizationResult:
        """Optimize thresholds using Optuna."""
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required for optimization")

        logger.info(f"🚀 Starting threshold optimization with Optuna...")
        logger.info(f"   - Data shape: {X.shape}")
        logger.info(f"   - CV folds: {self.config.cv_folds}")
        logger.info(f"   - Trials: {self.config.n_trials}")

        # Create study
        study = optuna.create_study(
            direction='maximize',
            study_name='threshold_optimization',
            sampler=optuna.samplers.TPESampler(seed=self.config.random_state)
        )

        # Optimize
        study.optimize(
            lambda trial: self.objective_function(trial, X, y),
            n_trials=self.config.n_trials,
            timeout=600,  # 10 minutes max
            show_progress_bar=True
        )

        # Extract best results
        best_trial = study.best_trial
        best_thresholds = {
            'min_confidence_threshold': best_trial.params['min_confidence_threshold'],
            'max_violation_percentage': best_trial.params['max_violation_percentage'],
        }

        # Calculate final CV scores for best configuration
        final_scores = self.evaluate_thresholds(X, y, best_thresholds)

        result = OptimizationResult(
            best_thresholds=best_thresholds,
            best_score=best_trial.value,
            cv_scores=final_scores,
            optimization_history=self.optimization_history.copy(),
            timestamp=datetime.now(),
            target_metric='combined'
        )

        logger.info(f"✅ Optimization completed!")
        logger.info(f"   - Best score: {best_trial.value:.4f}")
        logger.info(f"   - Best thresholds: {best_thresholds}")
        logger.info(f"   - CV F1: {final_scores.get('f1', 0):.4f}")
        logger.info(f"   - CV ROC-AUC: {final_scores.get('roc_auc', 0):.4f}")

        return result

    def optimize_grid_search(self, X: np.ndarray, y: np.ndarray) -> OptimizationResult:
        """Optimize thresholds using grid search (fallback when Optuna unavailable)."""
        logger.info("🔄 Using grid search for threshold optimization...")

        # Define search space
        confidence_range = np.linspace(0.01, 0.15, 10)
        violation_range = np.linspace(50.0, 250.0, 10)

        best_score = -np.inf
        best_thresholds = {}
        best_scores = {}

        total_combinations = len(confidence_range) * len(violation_range)
        current = 0

        for conf_threshold in confidence_range:
            for max_violation in violation_range:
                current += 1
                if current % 10 == 0:
                    logger.info(f"   - Progress: {current}/{total_combinations}")

                thresholds = {
                    'min_confidence_threshold': conf_threshold,
                    'max_violation_percentage': max_violation,
                }

                scores = self.evaluate_thresholds(X, y, thresholds)

                if scores['combined'] > best_score:
                    best_score = scores['combined']
                    best_thresholds = thresholds.copy()
                    best_scores = scores.copy()

                # Store history
                self.optimization_history.append({
                    'thresholds': thresholds.copy(),
                    'scores': scores.copy(),
                    'timestamp': datetime.now()
                })

        result = OptimizationResult(
            best_thresholds=best_thresholds,
            best_score=best_score,
            cv_scores=best_scores,
            optimization_history=self.optimization_history.copy(),
            timestamp=datetime.now(),
            target_metric='combined'
        )

        logger.info(f"✅ Grid search completed!")
        logger.info(f"   - Best score: {best_score:.4f}")
        logger.info(f"   - Best thresholds: {best_thresholds}")

        return result

    def apply_optimal_thresholds(self, result: OptimizationResult) -> bool:
        """Apply optimal thresholds to configuration files."""
        try:
            # Update ensemble.yaml
            ensemble_config_path = Path("config/ensemble.yaml")
            if ensemble_config_path.exists():
                with open(ensemble_config_path, 'r') as f:
                    config = yaml.safe_load(f)

                # Update symbolic extractor thresholds
                if 'base_models' in config:
                    for model in config['base_models']:
                        if model.get('type') == 'symbolic':
                            if 'params' not in model:
                                model['params'] = {}
                            model['params']['min_confidence_threshold'] = result.best_thresholds['min_confidence_threshold']

                # Update violation threshold in meta learner
                if 'meta_learner' in config and 'params' in config['meta_learner']:
                    config['meta_learner']['params']['max_violation_percentage'] = result.best_thresholds['max_violation_percentage']

                # Save updated config
                with open(ensemble_config_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False)

                logger.info(f"✅ Updated {ensemble_config_path}")

            # Create threshold optimization report
            report_path = Path("outputs/threshold_optimization_report.json")
            report_path.parent.mkdir(exist_ok=True)

            report = {
                'timestamp': result.timestamp.isoformat(),
                'optimization_method': 'optuna' if OPTUNA_AVAILABLE else 'grid_search',
                'best_thresholds': result.best_thresholds,
                'best_score': result.best_score,
                'cv_scores': result.cv_scores,
                'target_metric': result.target_metric,
                'n_trials_evaluated': len(result.optimization_history),
                'config_used': asdict(self.config)
            }

            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)

            logger.info(f"✅ Saved optimization report to {report_path}")

            self.best_config = result
            return True

        except Exception as e:
            logger.error(f"❌ Error applying optimal thresholds: {e}")
            return False

    def run_optimization(self, data_path: str = None) -> Optional[OptimizationResult]:
        """Run complete threshold optimization pipeline."""
        try:
            # Load data
            X, y = self.load_sample_data(data_path or "data/models/kg/train.parquet")

            # Run optimization
            if OPTUNA_AVAILABLE:
                result = self.optimize_with_optuna(X, y)
            else:
                result = self.optimize_grid_search(X, y)

            # Apply optimal thresholds
            if self.apply_optimal_thresholds(result):
                logger.info("🎯 Threshold optimization completed successfully!")
                return result
            else:
                logger.error("❌ Failed to apply optimal thresholds")
                return None

        except Exception as e:
            logger.error(f"❌ Optimization failed: {e}")
            return None

def main():
    """Main execution function."""
    print("🎯 Auto Threshold Tuner")
    print("Optimizing thresholds for better model balance and overfitting prevention")
    print("=" * 70)

    # Configuration
    config = ThresholdConfig(
        min_confidence_threshold=0.05,
        max_violation_percentage=200.0,
        target_violation_range=(50.0, 150.0),
        target_symbolic_ratio=0.75,
        target_f1_score=0.75,
        cv_folds=5,
        n_trials=50 if OPTUNA_AVAILABLE else 100,  # Fewer trials for demo
        random_state=42
    )

    # Create tuner
    tuner = AutoThresholdTuner(config)

    # Run optimization
    try:
        result = tuner.run_optimization()

        if result:
            print("\n" + "="*70)
            print("📊 OPTIMIZATION RESULTS")
            print("="*70)
            print(f"Best Combined Score: {result.best_score:.4f}")
            print(f"Best Thresholds:")
            for key, value in result.best_thresholds.items():
                print(f"  - {key}: {value:.4f}")
            print(f"\nCross-Validation Scores:")
            for metric, score in result.cv_scores.items():
                if metric != 'penalty':
                    print(f"  - {metric}: {score:.4f}")
            print("="*70)
            sys.exit(0)
        else:
            print("\n❌ Optimization failed")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n⚠️ Optimization interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Optimization error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()