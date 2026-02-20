"""Provide module-level functionality for the PFF codebase.



Notes:

    File: tests/performance/optimization/test_hpo_api_shims.py

"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from pff.infrastructure.hpo.objective import run_dslfm_objective


def test_objective_facade_runs_single_trial(tmp_path: Path) -> None:
    # 6 rows to survive 3-fold CV
    """Execute test objective facade runs single trial.



    Args:

        tmp_path: Input value used by this callable.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    train_df = pl.DataFrame(
        {"s": [0, 1, 0, 1, 0, 1], "p": [0, 0, 0, 0, 0, 0], "o": [1, 0, 1, 0, 1, 0]}
    )
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

    from unittest.mock import AsyncMock, MagicMock

    # Mock the store
    mock_store = MagicMock()
    # Mock async methods to return coroutines
    mock_store.upsert_trial_result = AsyncMock()
    mock_store.upsert_checkpoint = AsyncMock()
    mock_store.load_checkpoint = AsyncMock(return_value=None)
    mock_store.ensure_pool = AsyncMock()
    mock_store.list_trial_metrics = AsyncMock(return_value=[])
    mock_store.load_all_results = AsyncMock(return_value=[])

    score = run_dslfm_objective(
        params=params,
        train_df=train_df,
        valid_df=valid_df,
        target_entity_ratio=1.0,
        trial_number=0,
        output_root=tmp_path / "dslfm_trials",
        store=mock_store,
        study_name="test_optimization",
    )

    assert isinstance(score, float)
