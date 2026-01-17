import time
import numpy as np


# Mock Calibrator
class MockCalibrator:
    def transform(self, x):
        return 1.0 / (1.0 + np.exp(-x))  # Sigmoid

    def to_dict(self):
        return {}


def test_anomaly_scoring_performance():
    N = 1_000_000
    scores = np.random.randn(N)
    relations = np.random.choice(["r1", "r2", "r3"], N)

    calibrators = {
        "__global__": {"model": {}},
        "r1": {"model": {}},
        "r2": {"model": {}},
        "r3": {"model": {}},
    }

    import pff.domain.audit.anomaly_scoring as mod

    mod.calibrator_from_dict = lambda x: MockCalibrator()

    evt_params = {
        "__global__": {"shape": 0.1, "scale": 1.0, "u": 0.5},
        "r1": {"shape": 0.1, "scale": 1.0, "u": 0.5},
        "r2": {"shape": 0.1, "scale": 1.0, "u": 0.5},
        "r3": {"shape": 0.1, "scale": 1.0, "u": 0.5},
    }

    # Warmup (smaller)
    mod.score_with_calibration_and_evt(
        scores=scores[:100],
        relations=relations[:100],
        calibrators_by_relation=calibrators,
        evt_params_by_relation=evt_params,
    )

    start = time.perf_counter()
    results = mod.score_with_calibration_and_evt(
        scores=scores,
        relations=relations,
        calibrators_by_relation=calibrators,
        evt_params_by_relation=evt_params,
    )
    end = time.perf_counter()

    duration = end - start
    print(f"\nAnomaly scoring for {N} items: {duration * 1000:.2f} ms")
    print(f"Rate: {N / duration:.2f} items/s")

    assert len(results) == N
