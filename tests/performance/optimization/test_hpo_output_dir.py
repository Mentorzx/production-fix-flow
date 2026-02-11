from __future__ import annotations

from pathlib import Path

import polars as pl

from pff.infrastructure.hpo.runner import optimize_kg_hyperparameters


def test_optimize_accepts_str_output_dir(tmp_path: Path, monkeypatch) -> None:
    train_df = pl.DataFrame({"s": [0], "p": [0], "o": [1]})
    valid_df = pl.DataFrame({"s": [0], "p": [0], "o": [1]})
    data_info = {
        "n_train": 1,
        "n_valid": 1,
        "n_entities": 2,
        "n_predicates": 1,
        "source": "test",
    }

    monkeypatch.setattr(
        "pff.infrastructure.hpo.trials.data_loader.load_preprocessed_from_postgres",
        lambda *_args, **_kwargs: (train_df, valid_df, data_info),
    )
    monkeypatch.setattr(
        "pff.infrastructure.hpo.runner.create_study_and_run",
        lambda **_kwargs: {"study": None, "best_value": None, "best_params": {}},
    )
    monkeypatch.setattr(
        "pff.infrastructure.hpo.runner.select_best_trials",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        "pff.infrastructure.hpo.runner.HpoPostgresStore",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "pff.infrastructure.hpo.runner._write_checkpoint",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "pff.infrastructure.hpo.runner._load_checkpoint",
        lambda *args, **kwargs: None,
    )

    output_dir = tmp_path / "hpo_output"
    result = optimize_kg_hyperparameters(
        n_trials=0,
        output_dir=str(output_dir),
        resume_mode=False,
        reset_state=False,
    )

    assert output_dir.exists()
    assert result["real_data_info"] == data_info
