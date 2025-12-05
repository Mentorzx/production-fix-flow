import numpy as np

from pff.utils.evaluation.edas import KGEDASEvaluator


def test_edas_scores_improve_above_reference():
    evaluator = KGEDASEvaluator(reference_value=0.5)
    metrics = {"a": 0.8, "b": 0.7}
    result = evaluator.compute_score(metrics)
    assert 0.0 <= result.score <= 1.0
    assert result.score > 0.5


def test_edas_handles_missing_metrics_gracefully():
    evaluator = KGEDASEvaluator(reference_value=0.5)
    metrics = {"a": None, "b": 0.2}
    result = evaluator.compute_score(metrics)
    assert np.isfinite(result.score)
    assert 0.0 <= result.score <= 1.0
