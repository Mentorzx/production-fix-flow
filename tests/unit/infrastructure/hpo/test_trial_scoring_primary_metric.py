"""Provide module-level functionality for the PFF codebase.



Notes:

    File: tests/unit/infrastructure/hpo/test_trial_scoring_primary_metric.py

"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import polars as pl
from pytest import approx

from pff.domain.hpo.scoring import build_weights_from_settings, compute_score
from pff.infrastructure.hpo.config_loader import load_scoring_settings
from pff.infrastructure.hpo.trials.pipeline import TrialEvaluationPipeline
from pff.shared.core.file_manager import FileManager


def test_trial_score_prioritizes_ranking_over_mcc() -> None:
    """Execute test trial score prioritizes ranking over mcc.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    train_df = pl.DataFrame({"s": [0, 1], "p": [0, 1], "o": [1, 0]})
    valid_df = pl.DataFrame({"s": [0], "p": [0], "o": [1]})
    artifact_manager = MagicMock(store=MagicMock())
    artifact_manager.list_metrics.return_value = []

    pipeline = TrialEvaluationPipeline(
        params={"dslfm_epochs": 1},
        train_df=train_df,
        valid_df=valid_df,
        target_entity_ratio=1.0,
        trial_number=0,
        trial_output_root=Path("outputs/tests/trial_scoring_primary_metric"),
        artifact_manager=artifact_manager,
    )

    pipeline.kge_metrics = {
        "mrr": 0.72,
        "best_mrr": 0.75,
        "hits1": 0.60,
        "hits3": 0.70,
        "hits10": 0.82,
        "auc": 0.55,
        "pr_auc": 0.50,
        "precision": 0.52,
        "recall": 0.48,
        "mcc": 0.12,
    }
    pipeline.elapsed_time = 120.0

    pipeline._compute_score()

    scoring_settings = load_scoring_settings(FileManager())
    weights = build_weights_from_settings(scoring_settings)
    expected_score, _, _ = compute_score(
        {**pipeline.kge_metrics, "duration": pipeline.elapsed_time}, [], weights=weights
    )

    assert pipeline.base_score == approx(expected_score, rel=1e-6)
    assert pipeline.composite_score == approx(expected_score, rel=1e-6)
    assert pipeline.composite_score > pipeline.kge_metrics["mcc"]
