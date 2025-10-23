"""
Base Wrapper - Base class for ensemble wrappers

This module contains:
- Global constants and cache manager
- BaseWrapper (abstract base class)
- Helper functions (get_shared_cache, evaluate_and_save_metrics, _coerce_mapping_df)

Part of Sprint 4 refactoring (ensemble_wrappers.py split into 3 files).
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from pff import settings
from pff.utils import CacheManager, ConcurrencyManager, FileManager, logger

warnings.filterwarnings("ignore", category=UserWarning)

# ══════════════════════════════════════════════════════════════════════════
# GLOBAL CONSTANTS
# ══════════════════════════════════════════════════════════════════════════

_SHARED_CACHE_MANAGER: CacheManager | None = None
_INT_DTYPES: set[pl.DataType] = {
    pl.Int8(),
    pl.Int16(),
    pl.Int32(),
    pl.Int64(),
    pl.UInt8(),
    pl.UInt16(),
    pl.UInt32(),
    pl.UInt64(),
}


# ══════════════════════════════════════════════════════════════════════════
# BASE WRAPPER CLASS
# ══════════════════════════════════════════════════════════════════════════


class BaseWrapper(BaseEstimator, ClassifierMixin, ABC):
    """Abstract base class for all ensemble wrappers."""

    def __init__(self):
        self.classes_ = np.array([0, 1])
        self.file_manager = FileManager()
        self.cache_manager = get_shared_cache()
        self.concurrency_manager = ConcurrencyManager()

    @abstractmethod
    def fit(self, X, y=None) -> "BaseWrapper":
        """Fit the model."""
        pass

    @abstractmethod
    def predict_proba(self, X: list[Any]) -> np.ndarray:
        """Predict class probabilities."""
        pass

    def predict(self, X: list[Any]) -> np.ndarray:
        """Predict classes based on probability scores."""
        probabilities = self.predict_proba(X)
        return (probabilities[:, 1] > 0.5).astype(int)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        for key in (
            "concurrency_manager",
            "cache_manager",
            "file_manager",
            "logger",
        ):
            state.pop(key, None)
        return state

    def __setstate__(self, state: dict[str, Any]):
        self.__dict__.update(state)
        self.file_manager = FileManager()
        self.cache_manager = CacheManager()
        self.concurrency_manager = ConcurrencyManager()
        self.logger = logger

    def score_and_save_metrics(
        self, X_test, y_true, model_name: str | None = None
    ) -> dict:
        """
        Evaluate this model on test data, compute metrics, and save them as a JSON file in settings.PATTERNS_DIR.
        Args:
            X_test: List of test samples
            y_true: Ground truth labels
            model_name: Optional model name (defaults to class name)
        Returns:
            Dictionary with computed metrics
        """
        if model_name is None:
            model_name = self.__class__.__name__
        return evaluate_and_save_metrics(self, X_test, y_true, model_name)


# ══════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════


def get_shared_cache() -> CacheManager:
    """Get shared cache manager instance."""
    global _SHARED_CACHE_MANAGER
    if _SHARED_CACHE_MANAGER is None:
        _SHARED_CACHE_MANAGER = CacheManager()
    return _SHARED_CACHE_MANAGER


def evaluate_and_save_metrics(model, X_test, y_true, model_name: str) -> dict:
    """
    Evaluate a model, compute metrics, and save them in a single JSON file (metrics_all.json) in settings.PATTERNS_DIR.
    """
    cache_key = f"metrics_{model_name}_{hash(str(X_test)[:100])}"
    cached = get_shared_cache().get(cache_key)
    if cached:
        logger.debug(f"Métricas obtidas do cache para {model_name}")
        return cached
    logger.info(f"Avaliando modelo {model_name}")
    y_true = np.asarray(y_true).astype(int)
    logger.info(f"Distribuição y_test: {np.bincount(y_true)}")
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]
    unique_preds, counts = np.unique(y_pred, return_counts=True)
    logger.info(f"Predições únicas {model_name}: {dict(zip(unique_preds, counts))}")
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division="warn"),
        "recall": recall_score(y_true, y_pred, zero_division="warn"),
        "f1": f1_score(y_true, y_pred, zero_division="warn"),
        "auc": roc_auc_score(y_true, y_score) if len(set(y_true)) > 1 else None,
    }
    get_shared_cache().set(cache_key, metrics, ttl=3600)
    out_path = Path(settings.PATTERNS_DIR) / "metrics_all.json"
    file_manager = FileManager()
    if out_path.exists():
        try:
            all_metrics = file_manager.read(out_path)
            if not isinstance(all_metrics, dict):
                all_metrics = {}
        except Exception:
            all_metrics = {}
    else:
        all_metrics = {}
    all_metrics[model_name] = metrics
    try:
        file_manager.save(all_metrics, out_path, indent=2)
        logger.success(f"Métricas salvas em {out_path}")
    except Exception as e:
        logger.error(f"Erro ao salvar métricas em {out_path}: {e}")

    return metrics


def _coerce_mapping_df(df: pl.DataFrame, id_col: str | None = None) -> dict[str, int]:
    """
    Coerce a Polars DataFrame into a dictionary mapping IDs to indices.

    Args:
        df: Polars DataFrame with ID and index columns
        id_col: Name of the ID column (auto-detected if None)

    Returns:
        Dictionary mapping ID strings to integer indices
    """
    if id_col is None:
        id_col = next(
            (c for c in df.columns if df[c].dtype == pl.Utf8),
            None,
        )
    if id_col is None:
        raise ValueError("Nenhuma coluna textual (id/relation) encontrada")
    num_col = next(
        (
            c
            for c in df.columns
            if df[c].dtype in _INT_DTYPES or c.lower() in {"idx", "index"}
        ),
        None,
    )
    if num_col is None:
        raise ValueError("Nenhuma coluna de índice inteiro encontrada")

    return {row[id_col]: int(row[num_col]) for row in df.to_dicts()}
