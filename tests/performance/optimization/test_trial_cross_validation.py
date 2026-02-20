"""Provide module-level functionality for the PFF codebase.



Notes:

    File: tests/performance/optimization/test_trial_cross_validation.py

"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import polars as pl

from pff.infrastructure.hpo.trials.pipeline import TrialEvaluationPipeline
from pff.shared.core.file_manager import FileManager


def test_cross_validation_uses_numpy(monkeypatch) -> None:
    """Execute test cross validation uses numpy.



    Args:

        monkeypatch: Input value used by this callable.



    Returns:

        Return value produced by the callable.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    fm = FileManager()
    output_root = Path("outputs/tests/trial_cv_pipeline")
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
        "binary_negatives": 5,
    }

    pipeline = TrialEvaluationPipeline(
        params=params,
        train_df=train_df,
        valid_df=valid_df,
        target_entity_ratio=1.0,
        trial_number=0,
        trial_output_root=output_root,
        enable_cross_validation=True,
        artifact_manager=MagicMock(store=MagicMock(), study_name="test"),
    )
    pipeline.cv_settings = {"cv_folds": 2, "cv_parallel": False, "cv_max_workers": 1}
    pipeline._setup_trial()

    def _fake_run(self) -> float:
        return 0.5

    monkeypatch.setattr(TrialEvaluationPipeline, "run", _fake_run)
    score = pipeline._run_cross_validation()

    fm.delete_directory(output_root, ignore_errors=True)

    assert score == 0.5


def test_build_fold_indices_stratifies_by_relation_deterministically() -> None:
    """Fold construction should be relation-stratified and deterministic by seed."""
    train_df = pl.DataFrame(
        {
            "s": list(range(12)),
            "p": [0] * 6 + [1] * 6,
            "o": list(range(1, 13)),
        }
    )
    pipeline = TrialEvaluationPipeline(
        params={
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
            "binary_negatives": 5,
        },
        train_df=train_df,
        valid_df=train_df.head(2),
        target_entity_ratio=1.0,
        trial_number=0,
        trial_output_root=Path("outputs/tests/trial_cv_pipeline"),
        enable_cross_validation=True,
        artifact_manager=MagicMock(store=MagicMock(), study_name="test"),
    )
    pipeline.trial_seed = 12345

    fold_indices = pipeline._build_fold_indices(cv_folds=3)
    fold_indices_repeat = pipeline._build_fold_indices(cv_folds=3)

    assert [fold.tolist() for fold in fold_indices] == [
        fold.tolist() for fold in fold_indices_repeat
    ]

    for fold in fold_indices:
        fold_df = train_df[fold]
        rel_counts = fold_df.group_by("p").len().sort("p")["len"].to_list()
        assert rel_counts == [2, 2]


def _build_pipeline() -> TrialEvaluationPipeline:
    """Execute build pipeline.



    Returns:

        Return value produced by the callable.

    """

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
        "binary_negatives": 5,
    }
    return TrialEvaluationPipeline(
        params=params,
        train_df=train_df,
        valid_df=valid_df,
        target_entity_ratio=1.0,
        trial_number=0,
        trial_output_root=Path("outputs/tests/trial_cv_pipeline"),
        enable_cross_validation=True,
        artifact_manager=MagicMock(store=MagicMock(), study_name="test"),
    )


def test_cv_parallel_disabled_when_cuda(monkeypatch) -> None:
    """Execute test cv parallel disabled when cuda.



    Args:

        monkeypatch: Input value used by this callable.

    """

    pipeline = _build_pipeline()
    monkeypatch.setattr("pff.infrastructure.hpo.trials.pipeline.is_cuda_available", lambda: True)
    assert pipeline._resolve_cv_parallel(True) is False


def test_cv_parallel_disabled_when_workers_requested(monkeypatch) -> None:
    """Execute test cv parallel disabled when workers requested.



    Args:

        monkeypatch: Input value used by this callable.

    """

    pipeline = _build_pipeline()
    pipeline.params["num_workers"] = 2
    monkeypatch.setattr("pff.infrastructure.hpo.trials.pipeline.is_cuda_available", lambda: False)
    assert pipeline._resolve_cv_parallel(True) is False


def test_cv_parallel_disabled_when_auto_workers_available(monkeypatch) -> None:
    """Execute test cv parallel disabled when auto workers available.



    Args:

        monkeypatch: Input value used by this callable.

    """

    pipeline = _build_pipeline()
    monkeypatch.setattr("pff.infrastructure.hpo.trials.pipeline.is_cuda_available", lambda: False)
    monkeypatch.setattr(
        "pff.infrastructure.hpo.trials.pipeline.get_memory_safe_workers",
        lambda chunk_size: 2,
    )
    assert pipeline._resolve_cv_parallel(True) is False


def test_cv_parallel_enabled_when_safe(monkeypatch) -> None:
    """Execute test cv parallel enabled when safe.



    Args:

        monkeypatch: Input value used by this callable.

    """

    pipeline = _build_pipeline()
    monkeypatch.setattr("pff.infrastructure.hpo.trials.pipeline.is_cuda_available", lambda: False)
    monkeypatch.setattr(
        "pff.infrastructure.hpo.trials.pipeline.get_memory_safe_workers",
        lambda chunk_size: 0,
    )
    assert pipeline._resolve_cv_parallel(True) is True


def test_cv_run_sets_elapsed_time(monkeypatch) -> None:
    """Execute test cv run sets elapsed time.



    Args:

        monkeypatch: Input value used by this callable.



    Returns:

        Return value produced by the callable.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    pipeline = _build_pipeline()
    fm = FileManager()
    output_root = Path("outputs/tests/trial_cv_pipeline")
    fm.delete_directory(output_root, ignore_errors=True)
    pipeline.cv_settings = {"cv_folds": 2, "cv_parallel": False, "cv_max_workers": 1}

    times = iter([100.0, 103.0])

    class MockTime:
        """Represent MockTime."""

        def time(self):
            """Execute time.



            Returns:

                Return value produced by the callable.

            """

            return next(times)

        def perf_counter(self):
            """Execute perf counter.



            Returns:

                Return value produced by the callable.

            """

            return 0.0

    monkeypatch.setattr("pff.infrastructure.hpo.trials.pipeline.time", MockTime())
    monkeypatch.setattr(TrialEvaluationPipeline, "_run_cross_validation", lambda self: 0.5)

    pipeline.run()

    fm.delete_directory(output_root, ignore_errors=True)
    assert pipeline.elapsed_time == 3.0
