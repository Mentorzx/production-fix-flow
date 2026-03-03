"""Tests for LiveTrainingObserver epoch duration handling."""

from __future__ import annotations

import json
from unittest.mock import patch

from pff.infrastructure.hpo.callbacks_internal.visualizers import (
    LiveTrainingObserver,
    _merge_fold_history_entries,
)
from pff.domain.learning.ml.training_observer import TrainingEvent


def test_live_training_observer_uses_epoch_delta_duration(tmp_path) -> None:
    """Execute test live training observer uses epoch delta duration.



    Args:

        tmp_path: Input value used by this callable.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    output_dir = tmp_path / "live"
    output_dir.mkdir(parents=True, exist_ok=True)

    time_values = [
        1000.0,
        1000.0,
        1010.0,
        1010.0,
        1010.0,
        1025.0,
        1025.0,
        1025.0,
    ]

    with patch("time.time", side_effect=time_values):
        observer = LiveTrainingObserver(output_dir=output_dir, trial_number=1)

        observer.on_event(TrainingEvent(event_type="epoch_end", epoch=0, metrics={"loss": 1.0}))
        observer.on_event(TrainingEvent(event_type="epoch_end", epoch=1, metrics={"loss": 0.9}))

    history = observer.epoch_history
    assert len(history) == 2
    assert history[0]["duration"] == 10.0
    assert history[1]["duration"] == 15.0


def test_live_training_observer_normalizes_loss_and_efficiency(tmp_path) -> None:
    """Execute test live training observer normalizes loss and efficiency.



    Args:

        tmp_path: Input value used by this callable.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    output_dir = tmp_path / "live_norm"
    output_dir.mkdir(parents=True, exist_ok=True)

    time_values = [
        2000.0,
        2000.0,
        2010.0,
        2010.0,
        2010.0,
    ]

    with patch("time.time", side_effect=time_values):
        observer = LiveTrainingObserver(output_dir=output_dir, trial_number=2)
        observer.on_event(
            TrainingEvent(
                event_type="epoch_end",
                epoch=0,
                metrics={"binary_loss": 0.6, "score": 0.3},
            )
        )

    assert len(observer.epoch_history) == 1
    row = observer.epoch_history[0]
    assert row["train_loss"] == 0.6
    assert row["val_loss"] is None
    assert row["loss"] == 0.6
    assert row["efficiency"] == 0.03


def test_live_training_observer_uses_binary_loss_as_val_loss_on_validation_epochs(tmp_path) -> None:
    """Execute test live training observer uses binary loss as val loss on validation epochs."""

    output_dir = tmp_path / "live_val_binary"
    output_dir.mkdir(parents=True, exist_ok=True)

    time_values = [
        2500.0,
        2500.0,
        2510.0,
        2510.0,
        2510.0,
    ]

    with patch("time.time", side_effect=time_values):
        observer = LiveTrainingObserver(output_dir=output_dir, trial_number=22)
        observer.on_event(
            TrainingEvent(
                event_type="epoch_end",
                epoch=0,
                metrics={"binary_loss": 0.45, "mrr": 0.31},
            )
        )

    row = observer.epoch_history[0]
    assert row["train_loss"] == 0.45
    assert row["val_loss"] == 0.45


def test_live_training_observer_prefers_explicit_validation_loss(tmp_path) -> None:
    """Execute test live training observer prefers explicit validation loss."""

    output_dir = tmp_path / "live_val"
    output_dir.mkdir(parents=True, exist_ok=True)

    time_values = [
        3000.0,
        3000.0,
        3012.0,
        3012.0,
        3012.0,
    ]

    with patch("time.time", side_effect=time_values):
        observer = LiveTrainingObserver(output_dir=output_dir, trial_number=3)
        observer.on_event(
            TrainingEvent(
                event_type="epoch_end",
                epoch=0,
                metrics={"binary_loss": 0.55, "eval_loss": 0.72, "mrr": 0.25},
            )
        )

    assert len(observer.epoch_history) == 1
    row = observer.epoch_history[0]
    assert row["train_loss"] == 0.55
    assert row["val_loss"] == 0.72
    assert row["loss"] == 0.55


def test_merge_fold_history_entries_keeps_distinct_folds_and_updates_same_fold() -> None:
    """Ensure fold history merge preserves different folds and updates only same trial/fold."""
    existing = [
        {
            "trial_number": 10,
            "cv_fold_id": 0,
            "epoch": 60,
            "timestamp": 1000.0,
            "confusion_matrix": {"vp": 1, "vn": 2, "fp": 3, "fn": 4},
        },
        {
            "trial_number": 10,
            "cv_fold_id": 1,
            "epoch": 60,
            "timestamp": 1010.0,
            "confusion_matrix": {"vp": 5, "vn": 6, "fp": 7, "fn": 8},
        },
    ]
    incoming_new_fold = {
        "trial_number": 10,
        "cv_fold_id": 2,
        "epoch": 60,
        "timestamp": 1020.0,
        "confusion_matrix": {"vp": 9, "vn": 10, "fp": 11, "fn": 12},
    }
    merged = _merge_fold_history_entries(existing, incoming_new_fold, max_entries=10)
    assert len(merged) == 3
    assert [(r["trial_number"], r["cv_fold_id"]) for r in merged] == [(10, 0), (10, 1), (10, 2)]

    incoming_same_fold_newer = {
        "trial_number": 10,
        "cv_fold_id": 1,
        "epoch": 68,
        "timestamp": 1030.0,
        "confusion_matrix": {"vp": 50, "vn": 60, "fp": 70, "fn": 80},
    }
    merged2 = _merge_fold_history_entries(merged, incoming_same_fold_newer, max_entries=10)
    assert len(merged2) == 3
    fold1 = [r for r in merged2 if r["trial_number"] == 10 and r["cv_fold_id"] == 1][0]
    assert fold1["epoch"] == 68
    assert fold1["confusion_matrix"]["vp"] == 50


def test_live_training_observer_writes_fold_history_to_canonical_dashboard_path(tmp_path) -> None:
    """Ensure fold history is persisted to canonical dashboard outputs path."""
    local_output = tmp_path / "local_plots"
    local_output.mkdir(parents=True, exist_ok=True)
    canonical_outputs = tmp_path / "canonical_outputs"

    with patch(
        "pff.infrastructure.hpo.callbacks_internal.visualizers.settings.OUTPUTS_DIR",
        canonical_outputs,
    ):
        observer = LiveTrainingObserver(output_dir=local_output, trial_number=36, cv_fold_id=0)
        observer.current_epoch = 20
        observer.epoch_history = [{"epoch": 20, "vp": 10, "vn": 20, "fp": 3, "fn": 4}]
        observer._save_fold_to_history()

        observer.cv_fold_id = 1
        observer.current_epoch = 40
        observer.epoch_history = [{"epoch": 40, "vp": 11, "vn": 19, "fp": 4, "fn": 5}]
        observer._save_fold_to_history()

    local_fold_history = local_output / "fold_history.json"
    canonical_fold_history = canonical_outputs / "optimization" / "plots" / "fold_history.json"

    assert local_fold_history.exists()
    assert canonical_fold_history.exists()

    with local_fold_history.open("r", encoding="utf-8") as f_local:
        local_rows = json.load(f_local)
    with canonical_fold_history.open("r", encoding="utf-8") as f_canonical:
        canonical_rows = json.load(f_canonical)

    assert [(r["trial_number"], r["cv_fold_id"]) for r in local_rows] == [(36, 0), (36, 1)]
    assert [(r["trial_number"], r["cv_fold_id"]) for r in canonical_rows] == [(36, 0), (36, 1)]


def test_live_training_observer_persists_study_name_in_live_status(tmp_path) -> None:
    """Observer should persist study_name for dashboard study isolation."""
    output_dir = tmp_path / "live_study"
    output_dir.mkdir(parents=True, exist_ok=True)

    LiveTrainingObserver(
        output_dir=output_dir,
        trial_number=7,
        study_name="study_isolation_demo",
    )

    status_path = output_dir / "live_status.json"
    with status_path.open("r", encoding="utf-8") as status_file:
        status = json.load(status_file)

    assert status.get("study_name") == "study_isolation_demo"


def test_live_training_observer_removes_per_trial_snapshot_on_training_end(
    tmp_path,
) -> None:
    """Per-trial live status file should be removed once training ends."""
    output_dir = tmp_path / "live_cleanup"
    output_dir.mkdir(parents=True, exist_ok=True)

    observer = LiveTrainingObserver(output_dir=output_dir, trial_number=9)
    per_trial_path = output_dir / "live_status" / "trial_000009.json"
    assert per_trial_path.exists()

    observer.on_event(
        TrainingEvent(event_type="training_end", epoch=0, metrics={})
    )

    assert not per_trial_path.exists()
