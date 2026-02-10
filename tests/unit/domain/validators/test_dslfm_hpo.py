from __future__ import annotations

import numpy as np
import pytest

from pff.domain.hpo.models import KGE_MODEL_DSLFM
from pff.domain.hpo.search_space import SearchSpaceFactory, TuningConfigBuilder
from pff.infrastructure.hpo.runner import DEFAULT_KGE_MODEL


def test_default_kge_model_is_dslfm() -> None:
    assert KGE_MODEL_DSLFM == "dslfm"
    assert DEFAULT_KGE_MODEL == KGE_MODEL_DSLFM


def test_dslfm_search_space_structure() -> None:
    space = SearchSpaceFactory.create_dslfm_space(TuningConfigBuilder().build())
    assert set(space.keys()) == {
        "embedding_dim",
        "batch_size",
        "negative_sample_size",
        "num_global_negatives",
        "adversarial_temperature",
        "contrastive_temperature",
        "learning_rate",
        "lambda_logic",
        "kl_weight",
        "t_norm",
        "attr_hidden_dim",
        "lambda_pc",
        "pruning_threshold",
        "rebuild_every",
        "max_circuit_depth",
        "lambda_sum_cap",
        "ibp_alpha",
        "max_communities",
    }
    assert min(space["embedding_dim"]) > 0
    batch_low, batch_high = space["batch_size"]
    assert batch_low <= batch_high
    neg_low, neg_high = space["negative_sample_size"]
    assert neg_low <= neg_high


def test_tuning_builder_overrides_defaults() -> None:
    config = TuningConfigBuilder().with_batch_size(64, 256).build()
    space = SearchSpaceFactory.create_dslfm_space(config)
    assert space["batch_size"] == (64, 256)


def test_dslfm_training_helpers_exposed() -> None:
    from pff.infrastructure.hpo.trials import evaluator

    assert hasattr(evaluator, "_train_dslfm_kgc_model")
    assert callable(evaluator._train_dslfm_kgc_model)


def test_dslfm_pc_defaults_loaded(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from pff.infrastructure.hpo.trials import evaluator

    captured: dict[str, object] = {}

    class DummyManager:
        def __init__(
            self,
            model_config,
            training_config,
            persistence_port=None,
            relation_names=None,
            **kwargs,
        ) -> None:  # noqa: ANN001
            captured["model_config"] = model_config
            captured["training_config"] = training_config
            self.observers = kwargs.get("observers", [])

        def train(self, *_args, **_kwargs):  # noqa: ANN001
            return {"final_metrics": {}, "best_val_mrr": 0.0}

    def fake_settings(*_args, **_kwargs):
        return {
            "kgc": {"model": {}, "training": {}},
            "compile": {},
            "logic": {},
            "pc": {
                "lambda_pc": 0.12,
                "pruning_threshold": 0.34,
                "rebuild_every": 7,
                "max_circuit_depth": 5,
            },
        }

    monkeypatch.setattr(
        "pff.domain.learning.dslfm.kgc_manager.DSLFMKGCManager", DummyManager
    )
    monkeypatch.setattr(evaluator, "load_dslfm_kgc_settings", fake_settings)
    monkeypatch.setattr(
        evaluator, "_compute_binary_metrics", lambda *_args, **_kwargs: {}
    )

    train_triples = np.zeros((2, 3), dtype=np.int64)
    valid_triples = np.zeros((1, 3), dtype=np.int64)

    evaluator._train_dslfm_kgc_model(
        params={},
        model_dir=tmp_path,
        train_triples=train_triples,
        valid_triples=valid_triples,
        num_entities=2,
        num_relations=1,
        relation_names=None,
        use_bert=False,
        trial=None,
    )

    model_config = captured["model_config"]
    assert model_config.lambda_pc == pytest.approx(0.12)
    # The config dataclass uses pc_ prefix
    assert model_config.pc_pruning_threshold == pytest.approx(0.34)
    assert model_config.pc_rebuild_every == 7
    assert model_config.pc_max_depth == 5
