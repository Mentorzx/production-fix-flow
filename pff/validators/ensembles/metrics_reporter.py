"""Ensemble Metrics Reporter.

Extracts metrics reporting responsibilities from AdvancedEnsembleTrainer,
following the Single Responsibility Principle (SRP).

Design Patterns Applied:
    - **Strategy Pattern:** Different report formats can be plugged in.
    - **Dependency Injection:** Accepts FileManager for I/O operations.

Author: PFF Team
Date: 2025-11-25
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)

from pff.utils import FileManager, logger


def _convert_numpy_types(obj: Any) -> Any:
    """Recursively convert numpy types to Python natives for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: _convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_numpy_types(v) for v in obj]
    return obj


class EnsembleMetricsReporter:
    """Generates and persists metrics reports for ensemble models.

    Extracts metrics reporting from AdvancedEnsembleTrainer to follow SRP.
    Handles computation, formatting, and persistence of evaluation metrics.

    Design Pattern: Strategy
        - Different report formats (JSON, console, MLflow) can be plugged in.
        - Report generation logic is decoupled from persistence.

    Attributes:
        output_dir: Directory for saving reports.
        file_manager: FileManager for I/O operations.
    """

    def __init__(
        self,
        output_dir: Path | str,
        file_manager: FileManager | None = None,
    ) -> None:
        """Initialize metrics reporter.

        Args:
            output_dir: Directory for reports.
            file_manager: Optional FileManager instance.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.file_manager = file_manager or FileManager()

    def compute_classification_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Compute comprehensive classification metrics.

        Args:
            y_true: Ground truth labels.
            y_pred: Predicted labels.
            y_proba: Optional prediction probabilities.

        Returns:
            Dictionary with all computed metrics.
        """
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        }

        if y_proba is not None:
            try:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
                metrics["pr_auc"] = float(average_precision_score(y_true, y_proba))
            except ValueError as e:
                logger.warning(f"Could not compute AUC metrics: {e}")
                metrics["roc_auc"] = 0.0
                metrics["pr_auc"] = 0.0

        cm = confusion_matrix(y_true, y_pred)
        metrics["confusion_matrix"] = cm.tolist()

        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics["true_negatives"] = int(tn)
            metrics["false_positives"] = int(fp)
            metrics["false_negatives"] = int(fn)
            metrics["true_positives"] = int(tp)

        return metrics

    def compute_feature_balance(
        self,
        feature_importances: np.ndarray,
        feature_names: list[str],
        symbolic_prefix: str = "rule_",
    ) -> dict[str, float]:
        """Compute feature contribution balance (hybrid vs symbolic).

        Args:
            feature_importances: Array of feature importance values.
            feature_names: List of feature names.
            symbolic_prefix: Prefix identifying symbolic features.

        Returns:
            Dictionary with contribution percentages.
        """
        total_importance = np.sum(np.abs(feature_importances))
        if total_importance == 0:
            return {
                "hybrid_contribution": 0.0,
                "symbolic_contribution": 0.0,
                "symbolic_rules_count": 0,
            }

        symbolic_mask = [name.startswith(symbolic_prefix) for name in feature_names]
        symbolic_importance = np.sum(np.abs(feature_importances[symbolic_mask]))

        symbolic_pct = (symbolic_importance / total_importance) * 100
        hybrid_pct = 100.0 - symbolic_pct

        return {
            "hybrid_contribution": round(hybrid_pct, 2),
            "symbolic_contribution": round(symbolic_pct, 2),
            "symbolic_rules_count": sum(symbolic_mask),
        }

    def generate_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray | None = None,
        feature_balance: dict[str, float] | None = None,
        model_params: dict[str, Any] | None = None,
        training_time: float | None = None,
    ) -> dict[str, Any]:
        """Generate a comprehensive metrics report.

        Args:
            y_true: Ground truth labels.
            y_pred: Predicted labels.
            y_proba: Optional prediction probabilities.
            feature_balance: Optional feature balance dict.
            model_params: Optional model parameters.
            training_time: Optional training duration.

        Returns:
            Complete metrics report dictionary.
        """
        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metrics": self.compute_classification_metrics(y_true, y_pred, y_proba),
        }

        if feature_balance:
            report["feature_balance"] = feature_balance

        if model_params:
            report["model_params"] = _convert_numpy_types(model_params)

        if training_time is not None:
            report["training_time_seconds"] = round(training_time, 2)

        report["sample_counts"] = {
            "total": len(y_true),
            "positive": int(np.sum(y_true == 1)),
            "negative": int(np.sum(y_true == 0)),
        }

        return report

    def save_report(
        self,
        report: dict[str, Any],
        filename: str = "metrics_all.json",
    ) -> Path:
        """Save metrics report to JSON file.

        Args:
            report: Report dictionary.
            filename: Output filename.

        Returns:
            Path to saved report.
        """
        path = self.output_dir / filename
        self.file_manager.save(_convert_numpy_types(report), path)
        logger.info(f"Relatório de métricas salvo: {path}")
        return path

    def log_summary(self, report: dict[str, Any]) -> None:
        """Log a summary of the metrics report.

        Args:
            report: Report dictionary to summarize.
        """
        metrics = report.get("metrics", {})
        balance = report.get("feature_balance", {})

        logger.info("=== Resumo de Métricas do Ensemble ===")
        logger.info(f"  Accuracy: {metrics.get('accuracy', 0):.4f}")
        logger.info(f"  Precision: {metrics.get('precision', 0):.4f}")
        logger.info(f"  Recall: {metrics.get('recall', 0):.4f}")
        logger.info(f"  F1-Score: {metrics.get('f1', 0):.4f}")

        if "roc_auc" in metrics:
            logger.info(f"  ROC-AUC: {metrics.get('roc_auc', 0):.4f}")
            logger.info(f"  PR-AUC: {metrics.get('pr_auc', 0):.4f}")

        if balance:
            logger.info(f"  Contribuição Híbrida: {balance.get('hybrid_contribution', 0):.2f}%")
            logger.info(f"  Contribuição Simbólica: {balance.get('symbolic_contribution', 0):.2f}%")

    def find_optimal_threshold(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        target_metric: str = "f1",
    ) -> tuple[float, float]:
        """Find optimal classification threshold.

        Args:
            y_true: Ground truth labels.
            y_proba: Prediction probabilities.
            target_metric: Metric to optimize ('f1', 'precision', 'recall').

        Returns:
            Tuple of (optimal_threshold, best_score).
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)

        if target_metric == "f1":
            with np.errstate(divide='ignore', invalid='ignore'):
                f1_scores = 2 * (precision * recall) / (precision + recall)
                f1_scores = np.nan_to_num(f1_scores, nan=0.0)
            best_idx = np.argmax(f1_scores[:-1])
            return float(thresholds[best_idx]), float(f1_scores[best_idx])
        elif target_metric == "precision":
            best_idx = np.argmax(precision[:-1])
            return float(thresholds[best_idx]), float(precision[best_idx])
        elif target_metric == "recall":
            best_idx = np.argmax(recall[:-1])
            return float(thresholds[best_idx]), float(recall[best_idx])
        else:
            raise ValueError(f"Unknown target metric: {target_metric}")
