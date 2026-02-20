"""Provide module-level functionality for the PFF codebase.



Notes:

    File: tests/unit/shared/utils/test_audit_anomaly_scoring.py

"""

from __future__ import annotations

import numpy as np

from pff.domain.audit.anomaly_scoring import score_with_calibration_and_evt
from pff.domain.audit.calibration import (
    CalibrationConfig,
    calibrator_from_dict,
    fit_per_relation_calibrators,
)
from pff.domain.audit.evt import EVTConfig, fit_evt_by_relation
from pff.domain.audit.findings import neuro_symbolic_scores_to_findings


def test_audit_anomaly_scoring_produces_evt_p_values() -> None:
    """Execute test audit anomaly scoring produces evt p values.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    scores = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    relations = np.array(["r"] * scores.size)

    cal_cfg = CalibrationConfig(
        method="platt", ece_bins=5, min_samples_per_relation=1, clip_eps=1e-9
    )
    models = fit_per_relation_calibrators(
        scores=scores, labels=labels, relations=relations, config=cal_cfg
    )

    global_model = calibrator_from_dict(models["__global__"]["model"])
    probs = global_model.transform(scores)
    evt_scores = -np.log(np.clip(probs, 1e-9, 1.0))
    evt_cfg = EVTConfig(threshold_quantile=0.5, min_exceedances=2, clip_eps=1e-9)
    evt_params = fit_evt_by_relation(anomaly_scores=evt_scores, relations=relations, config=evt_cfg)
    assert "__global__" in evt_params

    scored = score_with_calibration_and_evt(
        scores=scores,
        relations=relations,
        calibrators_by_relation=models,
        evt_params_by_relation=evt_params,
        calibration_config=cal_cfg,
        evt_config=evt_cfg,
    )
    assert len(scored) == scores.size
    assert all(0.0 < float(item["evt_p_value"]) <= 1.0 for item in scored)

    findings = neuro_symbolic_scores_to_findings(
        scored,
        p_value_warning=1.0,
        p_value_error=0.5,
        max_findings=10,
    )
    assert findings
