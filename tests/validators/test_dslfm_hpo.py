from __future__ import annotations


from pff.infrastructure.hpo.runner import DEFAULT_KGE_MODEL
from pff.domain.hpo.models import KGE_MODEL_DSLFM
from pff.domain.hpo.search_space import SearchSpaceFactory, TuningConfigBuilder


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
