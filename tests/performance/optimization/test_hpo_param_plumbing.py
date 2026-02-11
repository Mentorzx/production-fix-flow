"""Verify HPO parameter plumbing for DSLFM/PC trials."""

from __future__ import annotations

import numpy as np
import pytest

from pff.infrastructure.hpo.trials import evaluator


@pytest.fixture
def tiny_triples() -> tuple[np.ndarray, np.ndarray]:
    train = np.array(
        [
            [0, 0, 1],
            [1, 1, 2],
            [2, 0, 3],
            [3, 1, 0],
        ],
        dtype=np.int64,
    )
    valid = np.array(
        [
            [0, 0, 2],
            [1, 1, 3],
        ],
        dtype=np.int64,
    )
    return train, valid


def test_hpo_applies_core_params(
    monkeypatch: pytest.MonkeyPatch, tiny_triples: tuple[np.ndarray, np.ndarray]
) -> None:
    """HPO training should pass lr/min_delta/patience/rerank/lambda_pc into configs."""
    captured: dict[str, object] = {}

    class DummyManager:
        def __init__(
            self, model_config, training_config, relation_names=None, **kwargs
        ) -> None:  # noqa: ANN001
            captured["model_config"] = model_config
            captured["training_config"] = training_config
            self.observers = kwargs.get("observers", [])

        def train(self, *_args, **_kwargs):  # noqa: ANN001
            return {"final_metrics": {}, "best_val_mrr": 0.0}

    # Patch the source module where the function imports from
    monkeypatch.setattr(
        "pff.domain.learning.dslfm.kgc_manager.DSLFMKGCManager", DummyManager
    )

    params = {
        "learning_rate": 7e-5,
        "early_stopping_patience": 5,
        "min_delta": 0.0002,
        "lambda_pc": 0.08,
        "lambda_logic": 0.03,
        "rerank_top_k": 0,
        "dslfm_epochs": 80,
        "batch_size": 256,
        "validate_every": 4,
    }

    train_triples, valid_triples = tiny_triples

    evaluator._train_dslfm_kgc_model(  # pylint: disable=protected-access
        params=params,
        model_dir=evaluator.Path("/tmp"),
        train_triples=train_triples,
        valid_triples=valid_triples,
        num_entities=4,
        num_relations=2,
        relation_names=["r0", "r1"],
        use_bert=False,
        trial=None,
    )

    model_config = captured["model_config"]
    training_config = captured["training_config"]

    assert model_config.lambda_pc == pytest.approx(0.08)
    assert model_config.lambda_logic == pytest.approx(0.03)
    assert training_config.learning_rate == pytest.approx(7e-5)
    assert training_config.early_stopping_patience == 5
    assert training_config.min_delta == pytest.approx(0.0002)
    assert training_config.rerank_top_k == 0
    assert training_config.epochs == 80
    assert training_config.validate_every == 4


def test_hpo_maps_embedding_and_sampler_params(
    monkeypatch: pytest.MonkeyPatch, tiny_triples: tuple[np.ndarray, np.ndarray]
) -> None:
    """HPO should map embedding_dim and adversarial sampler knobs into model config."""
    captured: dict[str, object] = {}

    class DummyManager:
        def __init__(
            self, model_config, training_config, relation_names=None, **kwargs
        ) -> None:  # noqa: ANN001
            captured["model_config"] = model_config
            captured["training_config"] = training_config
            self.observers = kwargs.get("observers", [])

        def train(self, *_args, **_kwargs):  # noqa: ANN001
            return {"final_metrics": {}, "best_val_mrr": 0.0}

    monkeypatch.setattr(
        "pff.domain.learning.dslfm.kgc_manager.DSLFMKGCManager", DummyManager
    )

    params = {
        "embedding_dim": 64,
        "adversarial_temperature": 2.5,
        "self_adversarial": False,
        "dslfm_epochs": 10,
    }

    train_triples, valid_triples = tiny_triples

    evaluator._train_dslfm_kgc_model(  # pylint: disable=protected-access
        params=params,
        model_dir=evaluator.Path("/tmp"),
        train_triples=train_triples,
        valid_triples=valid_triples,
        num_entities=4,
        num_relations=2,
        relation_names=["r0", "r1"],
        use_bert=False,
        trial=None,
    )

    model_config = captured["model_config"]
    assert model_config.entity_dim == 64
    assert model_config.feature_dim == 64
    assert model_config.sampler_type == "degree_based"
    assert model_config.sampler_temperature == pytest.approx(2.5)
