"""Test that DSLFMKGCConfig has no dead config params.

This test verifies that bug A (gamma/epsilon dead params) is fixed.
DSLFMKGCConfig should NOT contain gamma/epsilon (used for RotatE margin scoring,
not applicable to DSLFM-KGC which uses SBM decoder).
"""

from __future__ import annotations

import pytest


def test_no_dead_config_params() -> None:
    """DSLFMKGCConfig should not have gamma/epsilon (dead params)."""
    from pff.domain.learning.dslfm.dslfm_kgc import DSLFMKGCConfig

    config = DSLFMKGCConfig(num_entities=10, num_relations=5)

    # These params are from RotatE and should NOT exist in DSLFM-KGC
    assert not hasattr(config, "gamma"), "gamma should not exist in DSLFMKGCConfig (RotatE param)"
    assert not hasattr(config, "epsilon"), (
        "epsilon should not exist in DSLFMKGCConfig (RotatE param)"
    )


def test_legacy_config_also_cleaned() -> None:
    """Legacy DSLFMConfig should no longer have gamma/epsilon."""
    with pytest.raises(ImportError):
        from pff.domain.learning.dslfm.dslfm_kgc import DSLFMConfig  # noqa: F401


def test_dslfm_kgc_config_has_required_params() -> None:
    """DSLFMKGCConfig should have all required DSLFM-KGC params."""
    from pff.domain.learning.dslfm.dslfm_kgc import DSLFMKGCConfig

    config = DSLFMKGCConfig(num_entities=10, num_relations=5)

    # Essential params for DSLFM-KGC
    required_params = [
        "num_entities",
        "num_relations",
        "entity_dim",
        "feature_dim",
        "max_communities",
        "hidden_dim",
        "ibp_alpha",
        "kl_weight",
        "sparsity_weight",
        "temperature",
        "sampler_type",
        "sampler_temperature",
        "lambda_logic",
        "lambda_pc",
    ]

    for param in required_params:
        assert hasattr(config, param), f"Missing required param: {param}"


def test_dslfm_training_config_has_new_keys() -> None:
    """DSLFM config file should expose expected training knobs."""
    from pff.domain.learning.dslfm.dslfm_kgc import load_dslfm_kgc_settings
    from pff.shared.core.file_manager import FileManager

    settings = load_dslfm_kgc_settings(FileManager())
    training = settings.get("kgc", {}).get("training", {})
    for key in (
        "num_workers",
        "pin_memory",
        "dataloader_prefetch_factor",
        "dataloader_persistent_workers",
        "num_workers_heuristic",
        "cuda_cache_flush_steps",
        "cuda_cache_flush",
        "train_heartbeat_interval_s",
        "score_all_tails_chunk_size",
        "use_faiss_eval",
        "faiss_candidate_k",
        "allow_tf32",
        "matmul_precision",
        "mask_dense_max_entries",
        "use_compile",
        "optimizer_fused",
        "optimizer_foreach",
    ):
        assert key in training, f"Missing training config key: {key}"


def test_resolve_use_bert_setting_uses_config_default() -> None:
    """Explicit flag should override config; None should follow config."""
    from pff.domain.learning.dslfm.kgc_manager import _resolve_use_bert_setting

    model_defaults = {"use_bert_relations": False}

    assert _resolve_use_bert_setting(True, model_defaults) is True
    assert _resolve_use_bert_setting(False, model_defaults) is False
    assert _resolve_use_bert_setting(None, model_defaults) is False


def test_kgc_training_config_builder_applies_overrides() -> None:
    """KGCTrainingConfigBuilder should apply overrides and fluent setters."""
    from pff.domain.learning.dslfm.kgc_manager import (
        KGCTrainingConfig,
        KGCTrainingConfigBuilder,
    )

    base = KGCTrainingConfig(epochs=5, batch_size=16, effective_batch_size=32)
    config = (
        KGCTrainingConfigBuilder(base).with_epochs(10).apply_overrides({"batch_size": 64}).build()
    )

    assert config.epochs == 10
    assert config.batch_size == 64
