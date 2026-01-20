from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import polars as pl

from pff.infrastructure.hpo.trials.pipeline import TrialEvaluationPipeline
from pff.shared.core.file_manager import FileManager


def test_pipeline_runs_and_reports_metrics() -> None:
    fm = FileManager()
    output_root = Path("outputs/tests/dslfm_pipeline_small")
    fm.delete_directory(output_root, ignore_errors=True)

    train_df = pl.DataFrame(
        {
            "s": [0, 1, 2, 0],
            "p": [0, 1, 0, 1],
            "o": [1, 2, 0, 2],
        }
    )
    valid_df = pl.DataFrame(
        {
            "s": [0, 1],
            "p": [0, 1],
            "o": [1, 2],
        }
    )

    params = {
        "epochs": 2,
        "batch_size": 2,
        "effective_batch_size": 2,
        "learning_rate": 1e-2,
        "entity_dim": 8,
        "feature_dim": 8,
        "max_communities": 4,
        "validate_every": 1,
        "early_stopping_patience": 2,
        "mixed_precision": False,
        "num_workers": 0,
        "pin_memory": False,
        "eval_batch_size": 2,
        "use_bert": False,
        "binary_negatives": 5,
    }

    pipeline = TrialEvaluationPipeline(
        params=params,
        train_df=train_df,
        valid_df=valid_df,
        target_entity_ratio=1.0,
        trial_number=0,
        trial_output_root=output_root,
        artifact_manager=MagicMock(store=MagicMock(), study_name="test"),
    )

    score = pipeline.run()

    fm.delete_directory(output_root, ignore_errors=True)

    assert score > -1.0
    assert pipeline.kge_metrics.get("mrr", 0.0) > 0.0
    assert pipeline.elapsed_time > 0.0
