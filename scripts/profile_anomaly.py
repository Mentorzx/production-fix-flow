import cProfile
import io
import pstats

import numpy as np

from pff.domain.audit.anomaly_scoring import score_with_calibration_and_evt
from pff.domain.audit.calibration import (
    calibrator_from_dict,
    fit_per_relation_calibrators,
)
from pff.domain.audit.evt import fit_evt_by_relation


def profile_anomaly_scoring():
    print("Profiling Anomaly Scoring...")

    n_samples = 1_000_000
    n_relations = 100

    scores = np.random.rand(n_samples)
    relations = np.random.randint(0, n_relations, n_samples).astype(str)
    labels = np.random.randint(0, 2, n_samples)

    print("Fitting calibrators...")
    calibrators = fit_per_relation_calibrators(scores=scores, labels=labels, relations=relations)

    global_cal = calibrator_from_dict(calibrators["__global__"]["model"])
    probs = global_cal.transform(scores)
    probs = np.clip(probs, 1e-12, 1.0 - 1e-12)
    anomaly_scores = -np.log(probs)

    print("Fitting EVT...")
    evt_params = fit_evt_by_relation(anomaly_scores=anomaly_scores, relations=relations)

    if "__global__" not in evt_params:
        evt_params["__global__"] = {
            "u": 0.5,
            "shape": 0.1,
            "scale": 1.0,
            "evt_version": 1,
            "threshold_quantile": 0.95,
        }

    print("Starting profiling of score_with_calibration_and_evt...")

    pr = cProfile.Profile()
    pr.enable()

    score_with_calibration_and_evt(
        scores=scores,
        relations=relations,
        calibrators_by_relation=calibrators,
        evt_params_by_relation=evt_params,
    )

    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(20)
    print(s.getvalue())


if __name__ == "__main__":
    profile_anomaly_scoring()
