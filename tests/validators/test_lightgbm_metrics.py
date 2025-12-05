from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score

from pff.validators.rotate.lightgbm_trainer import RotatELightGBMTrainer


def test_lightgbm_helper_computes_pr_auc_mcc_and_gap() -> None:
    """Helper deve calcular pr_auc, mcc e generalization_gap (train_auc - val_auc)."""
    y_val = np.array([0, 1, 0, 1], dtype=int)
    y_val_pred = np.array([0.10, 0.85, 0.25, 0.80], dtype=float)
    y_val_pred_binary = (y_val_pred > 0.5).astype(int)

    y_train = np.array([0, 1, 1, 0], dtype=int)
    y_train_pred = np.array([0.05, 0.75, 0.60, 0.35], dtype=float)

    metrics = RotatELightGBMTrainer._compute_evaluation_metrics(
        y_val=y_val,
        y_val_pred=y_val_pred,
        y_val_pred_binary=y_val_pred_binary,
        y_train=y_train,
        y_train_pred=y_train_pred,
    )

    expected_val_auc = roc_auc_score(y_val, y_val_pred)
    expected_pr_auc = average_precision_score(y_val, y_val_pred)
    expected_mcc = matthews_corrcoef(y_val, y_val_pred_binary)
    expected_train_auc = roc_auc_score(y_train, y_train_pred)

    assert metrics["val_auc"] == expected_val_auc
    assert metrics["pr_auc"] == expected_pr_auc
    assert metrics["mcc"] == expected_mcc
    assert metrics["train_auc"] == expected_train_auc
    assert metrics["generalization_gap"] == expected_train_auc - expected_val_auc


class DummyBooster:
    def __init__(self, value: float = 0.8) -> None:
        self.value = value

    def predict(self, X: np.ndarray) -> np.ndarray:  # noqa: N803
        return np.full((len(X),), self.value, dtype=float)


class _DummyRotateManager:
    def __init__(self) -> None:
        self.config = {"model": {"embedding_dim": 4}}
        self.entity_to_idx = {"h1": 0, "t1": 1}
        self.relation_to_idx = {"r": 0}
        # Simple embeddings (2 dims real + 2 dims imag)
        self.node_embeddings = {
            "entity": np.array([[1.0, 0.0, 0.5, -0.5], [0.2, -0.1, 0.3, 0.4]], dtype=np.float32),
            "relation": np.array([[0.1, 0.2, -0.2, 0.3]], dtype=np.float32),
        }


def test_predict_samples_aggregates_triple_probs() -> None:
    """predict_samples deve retornar média das probabilidades por sample."""
    rotate_manager = _DummyRotateManager()
    trainer = RotatELightGBMTrainer(rotate_manager)
    trainer.lightgbm_model = DummyBooster(value=0.75)

    samples = [[("h1", "r", "t1")], [("h1", "r", "t1"), ("h1", "r", "t1")]]
    probs = trainer.predict_samples(samples)

    assert probs.shape == (2,)
    assert np.allclose(probs, 0.75)
