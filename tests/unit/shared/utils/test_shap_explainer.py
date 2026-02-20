"""Provide module-level functionality for the PFF codebase.



Notes:

    File: tests/unit/shared/utils/test_shap_explainer.py

"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from pff.infrastructure.shap_explainer import ShapExplainerService


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
    """Execute test shap explainer shape.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

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
    """Execute test shap explainer respects disabled flag."""

    model, X = _make_toy_model()
    service = ShapExplainerService(config_data={"shap": {"enabled": False}})

    explanation = service.explain(model, X)

    assert explanation is None


def test_shap_explainer_sampling_is_deterministic():
    """Execute test shap explainer sampling is deterministic.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    X = np.arange(200).reshape(100, 2)
    service = ShapExplainerService(
        config_data={"shap": {"enabled": True, "max_samples": 10, "max_background": 10}}
    )

    sample_a = service._sample_rows(X, 10)
    sample_b = service._sample_rows(X, 10)

    assert np.array_equal(sample_a, sample_b)
