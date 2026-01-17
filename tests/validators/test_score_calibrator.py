import numpy as np

from tests.support.score_calibrator import ScoreCalibrator


def test_score_calibrator_roundtrip():
    scores = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    labels = np.array([0, 0, 0, 1, 1])
    calibrator = ScoreCalibrator()
    calibrator.fit(scores, labels)

    probs = calibrator.transform(scores)
    assert probs.shape == scores.shape
    assert probs.min() >= 0.0 and probs.max() <= 1.0

    payload = calibrator.to_dict()
    restored = ScoreCalibrator.from_dict(payload)

    restored_probs = restored.transform(scores)
    assert np.allclose(probs, restored_probs)
