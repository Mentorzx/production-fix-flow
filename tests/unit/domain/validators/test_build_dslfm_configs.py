"""Regression tests for build_dslfm_configs (Achado #7).

Ensures that the centralized config factory produces identical configs for both
production training and HPO trials, preventing parameter drift.
"""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture()
def minimal_settings() -> dict[str, Any]:
    """Minimal YAML-like settings dict (simulates load_dslfm_kgc_settings output)."""
    return {
        "kgc": {
            "model": {
                "entity_dim": 128,
                "feature_dim": 128,
                "hidden_dim": 256,
                "sampler_type": "degree_based",
                "nsc_cache_size": 32,
                "nsc_sample_ratio": 0.3,
                "cache_global_negatives": True,
                "global_negatives_refresh_steps": 25,
                "triton_min_entities": 500,
            },
            "training": {
                "epochs": 50,
                "batch_size": 512,
                "warmup_steps": 500,
                "kl_warmup_epochs": 5,
                "temperature": 0.3,
                "temperature_anneal": 0.95,
                "min_temperature": 0.05,
                "triton_min_entities": 500,
                "cuda_cache_flush": {
                    "enabled": True,
                    "free_ratio_low": 0.1,
                    "free_ratio_high": 0.5,
                },
            },
        },
        "logic": {"lambda_logic": 0.1, "t_norm": "godel"},
        "pc": {"lambda_pc": 0.05, "pruning_threshold": 0.02, "rebuild_every": 10},
        "compile": {"mode": "max-autotune", "dynamic": False},
    }


class TestBuildDslfmConfigs:
    """Tests for build_dslfm_configs factory function."""

    def test_num_triples_is_set(self, minimal_settings: dict[str, Any]) -> None:
        """num_triples MUST be propagated — the key bug this achado fixes."""
        from pff.domain.learning.dslfm.kgc_manager import build_dslfm_configs

        model_cfg, _ = build_dslfm_configs(
            num_entities=100,
            num_relations=10,
            num_triples=5000,
            raw_settings=minimal_settings,
            overrides={},
            checkpoint_dir=Path("/tmp/ckpt"),
        )
        assert model_cfg.num_triples == 5000

    def test_num_triples_never_zero_when_given(
        self, minimal_settings: dict[str, Any]
    ) -> None:
        """Even with empty overrides, num_triples should match the argument."""
        from pff.domain.learning.dslfm.kgc_manager import build_dslfm_configs

        model_cfg, _ = build_dslfm_configs(
            num_entities=50,
            num_relations=5,
            num_triples=1234,
            raw_settings={},
            overrides={},
            checkpoint_dir=Path("/tmp/ckpt"),
        )
        assert model_cfg.num_triples == 1234

    def test_yaml_values_propagated(self, minimal_settings: dict[str, Any]) -> None:
        """YAML settings should be reflected in the produced configs."""
        from pff.domain.learning.dslfm.kgc_manager import build_dslfm_configs

        model_cfg, train_cfg = build_dslfm_configs(
            num_entities=100,
            num_relations=10,
            num_triples=500,
            raw_settings=minimal_settings,
            overrides={},
            checkpoint_dir=Path("/tmp/ckpt"),
        )
        assert model_cfg.entity_dim == 128
        assert model_cfg.hidden_dim == 256
        assert model_cfg.nsc_cache_size == 32
        assert model_cfg.nsc_sample_ratio == 0.3
        assert model_cfg.cache_global_negatives is True
        assert model_cfg.global_negatives_refresh_steps == 25
        assert model_cfg.triton_min_entities == 500
        assert model_cfg.lambda_logic == pytest.approx(0.1)
        assert model_cfg.t_norm == "godel"
        assert model_cfg.lambda_pc == pytest.approx(0.05)
        assert model_cfg.pc_rebuild_every == 10

        assert train_cfg.epochs == 50
        assert train_cfg.batch_size == 512
        assert train_cfg.warmup_steps == 500
        assert train_cfg.kl_warmup_epochs == 5
        assert train_cfg.temperature == pytest.approx(0.3)
        assert train_cfg.temperature_anneal == pytest.approx(0.95)
        assert train_cfg.min_temperature == pytest.approx(0.05)
        assert train_cfg.cuda_cache_flush_enabled is True
        assert train_cfg.cuda_cache_flush_free_ratio_low == pytest.approx(0.1)

    def test_overrides_win_over_yaml(self, minimal_settings: dict[str, Any]) -> None:
        """Explicit overrides must take precedence over YAML defaults."""
        from pff.domain.learning.dslfm.kgc_manager import build_dslfm_configs

        model_cfg, train_cfg = build_dslfm_configs(
            num_entities=100,
            num_relations=10,
            num_triples=500,
            raw_settings=minimal_settings,
            overrides={
                "entity_dim": 64,
                "learning_rate": 0.001,
                "batch_size": 1024,
                "lambda_logic": 0.5,
            },
            checkpoint_dir=Path("/tmp/ckpt"),
        )
        assert model_cfg.entity_dim == 64
        assert model_cfg.lambda_logic == pytest.approx(0.5)
        assert train_cfg.learning_rate == pytest.approx(0.001)
        assert train_cfg.batch_size == 1024

    def test_empty_settings_uses_defaults(self) -> None:
        """With empty settings and no overrides, dataclass defaults should apply."""
        from pff.domain.learning.dslfm.kgc_manager import build_dslfm_configs

        model_cfg, train_cfg = build_dslfm_configs(
            num_entities=10,
            num_relations=3,
            num_triples=100,
            raw_settings={},
            overrides={},
            checkpoint_dir=Path("/tmp/ckpt"),
        )
        assert model_cfg.entity_dim == 256
        assert model_cfg.kl_weight == pytest.approx(0.1)
        assert model_cfg.free_bits == pytest.approx(0.125)
        assert train_cfg.epochs == 200
        assert train_cfg.warmup_steps == 1000
        assert train_cfg.adaptive_batch_size is False

    def test_checkpoint_dir_propagated(self, minimal_settings: dict[str, Any]) -> None:
        from pff.domain.learning.dslfm.kgc_manager import build_dslfm_configs

        _, train_cfg = build_dslfm_configs(
            num_entities=10,
            num_relations=3,
            num_triples=100,
            raw_settings=minimal_settings,
            overrides={},
            checkpoint_dir=Path("/custom/path"),
        )
        assert train_cfg.checkpoint_dir == Path("/custom/path")

    def test_all_dslfm_config_fields_covered(self) -> None:
        """Ensure the factory sets every DSLFMKGCConfig field (no drift possible)."""
        from pff.domain.learning.dslfm.dslfm_kgc import DSLFMKGCConfig
        from pff.domain.learning.dslfm.kgc_manager import build_dslfm_configs

        model_cfg, _ = build_dslfm_configs(
            num_entities=10,
            num_relations=3,
            num_triples=100,
            raw_settings={},
            overrides={},
            checkpoint_dir=Path("/tmp/ckpt"),
        )
        for f in fields(DSLFMKGCConfig):
            assert hasattr(
                model_cfg, f.name
            ), f"DSLFMKGCConfig.{f.name} missing from factory"

    def test_all_training_config_fields_covered(self) -> None:
        """Ensure the factory sets every KGCTrainingConfig field."""
        from pff.domain.learning.dslfm.kgc_manager import (
            KGCTrainingConfig,
            build_dslfm_configs,
        )

        _, train_cfg = build_dslfm_configs(
            num_entities=10,
            num_relations=3,
            num_triples=100,
            raw_settings={},
            overrides={},
            checkpoint_dir=Path("/tmp/ckpt"),
        )
        for f in fields(KGCTrainingConfig):
            assert hasattr(
                train_cfg, f.name
            ), f"KGCTrainingConfig.{f.name} missing from factory"

    def test_malformed_settings_handled_gracefully(self) -> None:
        """Factory should not crash on non-dict config sections."""
        from pff.domain.learning.dslfm.kgc_manager import build_dslfm_configs

        model_cfg, train_cfg = build_dslfm_configs(
            num_entities=10,
            num_relations=3,
            num_triples=100,
            raw_settings={"kgc": "not_a_dict", "logic": None, "compile": 42},
            overrides={},
            checkpoint_dir=Path("/tmp/ckpt"),
        )
        assert model_cfg.entity_dim == 256
        assert train_cfg.epochs == 200


class TestBuildHpoOverrides:
    """Tests for _build_hpo_overrides translation function."""

    def test_embedding_dim_expands(self) -> None:
        from pff.infrastructure.hpo.trials.evaluator import _build_hpo_overrides

        result = _build_hpo_overrides({"embedding_dim": 64, "batch_size": 512})
        assert "embedding_dim" not in result
        assert result["entity_dim"] == 64
        assert result["feature_dim"] == 64
        assert result["batch_size"] == 512

    def test_adversarial_temperature_renamed(self) -> None:
        from pff.infrastructure.hpo.trials.evaluator import _build_hpo_overrides

        result = _build_hpo_overrides({"adversarial_temperature": 2.5})
        assert "adversarial_temperature" not in result
        assert result["sampler_temperature"] == 2.5

    def test_dslfm_epochs_renamed(self) -> None:
        from pff.infrastructure.hpo.trials.evaluator import _build_hpo_overrides

        result = _build_hpo_overrides({"dslfm_epochs": 30})
        assert "dslfm_epochs" not in result
        assert result["epochs"] == 30

    def test_passthrough_keys_preserved(self) -> None:
        from pff.infrastructure.hpo.trials.evaluator import _build_hpo_overrides

        result = _build_hpo_overrides({"learning_rate": 0.01, "kl_weight": 0.2})
        assert result["learning_rate"] == 0.01
        assert result["kl_weight"] == 0.2
