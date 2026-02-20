"""Provide module-level functionality for the PFF codebase.



Notes:

    File: tests/unit/infrastructure/hpo/test_pipeline_best_mcc.py

"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import polars as pl

from pff.infrastructure.hpo.trials.artifacts import TrialArtifactManager
from pff.infrastructure.hpo.trials.pipeline import TrialEvaluationPipeline
from pff.shared.core.file_manager import FileManager


def test_pipeline_prefers_best_mcc(monkeypatch) -> None:
    """Execute test pipeline prefers best mcc.



    Args:

        monkeypatch: Input value used by this callable.



    Returns:

        Return value produced by the callable.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    fm = FileManager()
    output_root = Path("outputs/tests/pipeline_best_mcc")
    fm.delete_directory(output_root, ignore_errors=True)

    train_df = pl.DataFrame({"s": [0, 1], "p": [0, 1], "o": [1, 0]})
    valid_df = pl.DataFrame({"s": [0], "p": [0], "o": [1]})

    def _fake_train(*_args, **_kwargs):
        """Execute fake train.



        Args:

            *_args: Additional positional arguments.

            **_kwargs: Additional keyword arguments.



        Returns:

            Return value produced by the callable.

        """

        stats = {
            "final_metrics": {"mrr": 0.2, "mcc": 0.1},
            "best_val_mrr": 0.45,
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
    assert pipeline.kge_metrics["mrr"] == 0.45


def test_prepare_kge_triples_uses_semantic_relation_map() -> None:
    """Prefer semantic relation labels from relation map when IDs are contiguous."""
    fm = FileManager()
    output_root = Path("outputs/tests/pipeline_relation_names")
    fm.delete_directory(output_root, ignore_errors=True)
    fm.ensure_dir(output_root)

    relation_map_path = output_root / "relation_map.parquet"
    fm.save(
        pl.DataFrame(
            {
                "relation_id": [0, 1],
                "relation": ["billCycleChangeType", "homeTimeZone"],
            }
        ),
        relation_map_path,
    )

    train_df = pl.DataFrame({"s": [0, 1, 2], "p": [0, 1, 0], "o": [1, 2, 0]})
    valid_df = pl.DataFrame({"s": [2], "p": [1], "o": [0]})

    pipeline = TrialEvaluationPipeline(
        params={
            "epochs": 1,
            "use_bert": True,
            "relation_map_path": str(relation_map_path),
        },
        train_df=train_df,
        valid_df=valid_df,
        target_entity_ratio=1.0,
        trial_number=0,
        trial_output_root=output_root,
        artifact_manager=TrialArtifactManager(study_name="test", store=MagicMock()),
        enable_cross_validation=False,
    )

    _, _, _, _, relation_names = pipeline._prepare_kge_triples()

    fm.delete_directory(output_root, ignore_errors=True)

    assert relation_names == ["billCycleChangeType", "homeTimeZone"]


def test_prepare_kge_triples_preserves_sparse_relation_ids_when_configured() -> None:
    """Sparse integer relation IDs should be preserved under preserve_sparse policy."""
    pipeline = TrialEvaluationPipeline(
        params={
            "epochs": 1,
            "use_bert": False,
            "relation_id_policy": "preserve_sparse",
        },
        train_df=pl.DataFrame({"s": [0, 1, 2], "p": [5, 7, 5], "o": [1, 2, 0]}),
        valid_df=pl.DataFrame({"s": [2], "p": [7], "o": [0]}),
        target_entity_ratio=1.0,
        trial_number=0,
        trial_output_root=Path("outputs/tests/pipeline_sparse_relations"),
        artifact_manager=TrialArtifactManager(study_name="test", store=MagicMock()),
        enable_cross_validation=False,
    )

    train_triples, valid_triples, num_entities, num_relations, _names = (
        pipeline._prepare_kge_triples()
    )

    assert num_entities == 3
    assert num_relations == 8
    assert set(train_triples[:, 1].tolist()) == {5, 7}
    assert set(valid_triples[:, 1].tolist()) == {7}


def test_prepare_kge_triples_remaps_sparse_relation_ids_when_requested() -> None:
    """Sparse integer relation IDs should be remapped when remap_dense policy is selected."""
    pipeline = TrialEvaluationPipeline(
        params={
            "epochs": 1,
            "use_bert": False,
            "relation_id_policy": "remap_dense",
        },
        train_df=pl.DataFrame({"s": [0, 1, 2], "p": [5, 7, 5], "o": [1, 2, 0]}),
        valid_df=pl.DataFrame({"s": [2], "p": [7], "o": [0]}),
        target_entity_ratio=1.0,
        trial_number=0,
        trial_output_root=Path("outputs/tests/pipeline_sparse_relations"),
        artifact_manager=TrialArtifactManager(study_name="test", store=MagicMock()),
        enable_cross_validation=False,
    )

    train_triples, valid_triples, _num_entities, num_relations, _names = (
        pipeline._prepare_kge_triples()
    )

    assert num_relations == 2
    assert set(train_triples[:, 1].tolist()) == {0, 1}
    assert set(valid_triples[:, 1].tolist()) == {1}
