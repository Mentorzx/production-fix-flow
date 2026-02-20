"""Tests for HPO runner data loading wiring."""

from __future__ import annotations

import polars as pl

from pff.infrastructure.hpo import runner


def test_runner_enables_loader_fallback(monkeypatch):
    """Runner should call loader with allow_fallback enabled."""
    captured: dict[str, object] = {}

    def _fake_load_preprocessed_from_postgres(
        file_manager,
        require_preprocessed=True,
        auto_populate_if_missing=True,
        config_path=None,
        allow_fallback=False,
    ):
        captured["allow_fallback"] = allow_fallback
        captured["require_preprocessed"] = require_preprocessed
        return (
            pl.DataFrame({"s": [0], "p": [0], "o": [1]}),
            pl.DataFrame({"s": [1], "p": [0], "o": [0]}),
            {"source": "test"},
        )

    monkeypatch.setattr(
        "pff.infrastructure.hpo.trials.data_loader.load_preprocessed_from_postgres",
        _fake_load_preprocessed_from_postgres,
    )

    train_df, valid_df, _info = runner._load_kg_data_for_hpo(
        file_manager=object(),
        use_synthetic_if_dslfm=False,
    )

    assert captured["allow_fallback"] is True
    assert captured["require_preprocessed"] is True
    assert len(train_df) == 1
    assert len(valid_df) == 1
