"""Test that ExactEvaluator raises NotImplementedError for models without score_all_tails.

This test verifies that bug E (random evaluation fallback) is fixed.
The evaluator must NOT silently return random scores when score_all_tails is missing.
"""

from __future__ import annotations

import pytest
import numpy as np

from pff.domain.learning.dslfm.evaluation import ExactEvaluator, EvaluatorConfig


class MockModelNoScorer:
    """Model that does not implement score_all_tails."""

    pass


class MockModelWithScorer:
    """Model with score_all_tails that returns deterministic scores."""

    def score_all_tails(
        self,
        heads: np.ndarray,
        relations: np.ndarray,
        all_embeddings: np.ndarray,
    ) -> np.ndarray:
        batch_size = len(heads)
        num_entities = all_embeddings.shape[0]
        # Return scores where entity index = score (deterministic)
        return np.tile(np.arange(num_entities, dtype=np.float32), (batch_size, 1))


def test_no_random_evaluation_fallback() -> None:
    """Evaluator must raise NotImplementedError when model lacks score_all_tails."""
    evaluator = ExactEvaluator(EvaluatorConfig(batch_size=2))
    model = MockModelNoScorer()
    triples = np.array([[0, 0, 1], [2, 1, 3]], dtype=np.int64)
    embeddings = np.random.randn(10, 32).astype(np.float32)

    with pytest.raises(NotImplementedError, match="score_all_tails"):
        evaluator.evaluate(model, triples, embeddings)


def test_evaluator_uses_model_scorer() -> None:
    """Evaluator should use model.score_all_tails when available."""
    evaluator = ExactEvaluator(EvaluatorConfig(batch_size=10))
    model = MockModelWithScorer()
    # tail=5 gets score 5.0; all others get their index as score
    # For tail=5: entities 6,7,8,9 have higher scores -> rank = 5
    triples = np.array([[0, 0, 5]], dtype=np.int64)
    embeddings = np.random.randn(10, 32).astype(np.float32)

    metrics = evaluator.evaluate(model, triples, embeddings)

    assert "mrr" in metrics
    # tail=5 has rank 5 (4 entities have higher scores: 6,7,8,9)
    expected_mrr = 1.0 / 5.0
    assert metrics["mrr"] == pytest.approx(expected_mrr, rel=1e-5)
