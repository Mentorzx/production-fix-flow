"""Provide module-level functionality for the PFF codebase.



Notes:

    File: tests/unit/shared/utils/test_audit_calibration_evt_negatives.py

"""

from __future__ import annotations

import numpy as np

from pff.domain.audit.calibration import (
    CalibrationConfig,
    calibrator_from_dict,
    fit_per_relation_calibrators,
)
from pff.domain.audit.evt import EVTConfig, evt_p_value, fit_gpd_pot
from pff.domain.audit.negative_sampling import corrupt_tails


def test_audit_calibration_models_are_serializable() -> None:
    """Execute test audit calibration models are serializable.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    scores = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    relations = np.array(["r"] * scores.size)

    cfg = CalibrationConfig(
        method="isotonic", ece_bins=5, min_samples_per_relation=1, clip_eps=1e-9
    )
    models = fit_per_relation_calibrators(
        scores=scores, labels=labels, relations=relations, config=cfg
    )

    assert "__global__" in models
    assert "r" in models

    payload = models["r"]["model"]
    model = calibrator_from_dict(payload)
    probs = model.transform(scores)
    assert np.all((probs >= 0.0) & (probs <= 1.0))


def test_audit_evt_fit_and_p_values() -> None:
    """Execute test audit evt fit and p values.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    scores = np.array([0.1, 0.2, 0.3, 10.0, 12.0, 15.0], dtype=np.float64)
    cfg = EVTConfig(threshold_quantile=0.50, min_exceedances=2, clip_eps=1e-9)
    params = fit_gpd_pot(scores, config=cfg)
    assert params is not None

    p_low = evt_p_value(0.2, params=params, clip_eps=1e-9)
    p_high = evt_p_value(15.0, params=params, clip_eps=1e-9)
    assert p_low == 1.0
    assert 0.0 < p_high < 1.0


def test_audit_negative_sampling_is_deterministic() -> None:
    """Execute test audit negative sampling is deterministic.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    triples = np.array([[0, 0, 1], [1, 0, 2]], dtype=np.int64)
    neg_first = corrupt_tails(triples, num_entities=5, num_negatives=3, seed=42)
    neg_second = corrupt_tails(triples, num_entities=5, num_negatives=3, seed=42)
    assert np.array_equal(neg_first, neg_second)
    assert neg_first.shape == (6, 3)
    assert np.all(neg_first[:, 1] == 0)
