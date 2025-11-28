import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from pff.utils.explainability.shap_explainer import ShapExplainerService, ShapExplainerConfig


def _make_toy_model():
    X, y = make_classification(
        n_samples=40,
        n_features=5,
        n_informative=3,
        n_redundant=0,
        random_state=42,
    )
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model, X


def test_shap_explainer_shape():
    model, X = _make_toy_model()
    service = ShapExplainerService(
        config_data={
            "shap": {
                "enabled": True,
                "max_background": 20,
                "max_samples": 15,
            }
        }
    )

    explanation = service.explain(model, X, save=False)

    assert explanation is not None
    values = np.asarray(explanation.values)
    if values.ndim == 3:
        values = values[:, 0, :]
    assert values.shape[0] <= 15
    assert values.shape[1] == X.shape[1]


def test_shap_explainer_respects_disabled_flag():
    model, X = _make_toy_model()
    service = ShapExplainerService(
        config_data={"shap": {"enabled": False}}
    )

    explanation = service.explain(model, X)

    assert explanation is None
