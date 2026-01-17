from __future__ import annotations

from pathlib import Path

import polars as pl

from pff.infrastructure.hpo.objective import run_dslfm_objective


def test_objective_facade_runs_single_trial(tmp_path: Path) -> None:
    train_df = pl.DataFrame({"s": [0, 1], "p": [0, 0], "o": [1, 0]})
    valid_df = pl.DataFrame({"s": [0], "p": [0], "o": [1]})
    params = {
        "epochs": 1,
        "batch_size": 2,
        "effective_batch_size": 2,
        "learning_rate": 1e-2,
        "entity_dim": 8,
        "feature_dim": 8,
        "max_communities": 4,
        "validate_every": 1,
        "early_stopping_patience": 1,
        "mixed_precision": False,
        "num_workers": 0,
        "pin_memory": False,
        "eval_batch_size": 2,
        "use_bert": False,
        "binary_negatives": 2,
    }

    score = run_dslfm_objective(
        params=params,
        train_df=train_df,
        valid_df=valid_df,
        target_entity_ratio=1.0,
        trial_number=0,
        output_root=tmp_path / "dslfm_trials",
    )

    assert score > 0.0
