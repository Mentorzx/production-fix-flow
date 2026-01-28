from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import polars as pl

from pff.infrastructure.hpo.trials.artifacts import TrialArtifactManager
from pff.infrastructure.hpo.trials.pipeline import TrialEvaluationPipeline
from pff.shared.core.file_manager import FileManager


def test_pipeline_prefers_best_mcc(monkeypatch) -> None:
    fm = FileManager()
    output_root = Path("outputs/tests/pipeline_best_mcc")
    fm.delete_directory(output_root, ignore_errors=True)

    train_df = pl.DataFrame({"s": [0, 1], "p": [0, 1], "o": [1, 0]})
    valid_df = pl.DataFrame({"s": [0], "p": [0], "o": [1]})

    def _fake_train(*_args, **_kwargs):
        stats = {
            "final_metrics": {"mrr": 0.2, "mcc": 0.1},
            "best_val_mrr": 0.2,
            "best_val_mcc": 0.42,
        }
        return stats, Path(output_root) / "checkpoint.pt"

    monkeypatch.setattr(
        "pff.infrastructure.hpo.trials.pipeline._train_dslfm_kgc_model", _fake_train
    )

    pipeline = TrialEvaluationPipeline(
        params={"epochs": 1, "use_bert": False},
        train_df=train_df,
        valid_df=valid_df,
        target_entity_ratio=1.0,
        trial_number=0,
        trial_output_root=output_root,
        artifact_manager=TrialArtifactManager(study_name="test", store=MagicMock()),
        enable_cross_validation=False,
    )

    pipeline._setup_trial()
    pipeline._train_kge()

    fm.delete_directory(output_root, ignore_errors=True)

    assert pipeline.kge_metrics["mcc"] == 0.42
