"""Regression tests for HPO override normalization in evaluator."""

from __future__ import annotations

import pytest

from pff.infrastructure.hpo.trials.evaluator import _build_hpo_overrides


def test_build_hpo_overrides_maps_kl_weight_to_scheduler_bounds() -> None:
    """Sampled kl_weight must control warmup scheduler ceiling to avoid silent drift."""
    params = {"kl_weight": 0.003, "dslfm_epochs": 20}

    overrides = _build_hpo_overrides(params)

    assert overrides["kl_weight"] == pytest.approx(0.003)
    assert overrides["max_kl_weight"] == pytest.approx(0.003)
    assert overrides["min_kl_weight"] == pytest.approx(0.0)
    assert overrides["epochs"] == 20


def test_build_hpo_overrides_preserves_explicit_kl_bounds() -> None:
    """Explicit max/min bounds from params should not be overridden by kl_weight defaults."""
    params = {
        "kl_weight": 0.003,
        "max_kl_weight": 0.02,
        "min_kl_weight": 0.005,
    }

    overrides = _build_hpo_overrides(params)

    assert overrides["max_kl_weight"] == pytest.approx(0.02)
    assert overrides["min_kl_weight"] == pytest.approx(0.005)
