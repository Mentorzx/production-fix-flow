"""Tests for LiveTrainingObserver epoch duration handling."""

from __future__ import annotations

from unittest.mock import patch

from pff.infrastructure.hpo.callbacks_internal.visualizers import LiveTrainingObserver
from pff.domain.learning.ml.training_observer import TrainingEvent


def test_live_training_observer_uses_epoch_delta_duration(tmp_path) -> None:
    output_dir = tmp_path / "live"
    output_dir.mkdir(parents=True, exist_ok=True)

    time_values = [
        1000.0,  # __init__ start_time
        1000.0,  # _write_status on init
        1010.0,  # epoch 1 elapsed
        1010.0,  # epoch 1 timestamp
        1010.0,  # _write_status after epoch 1
        1025.0,  # epoch 2 elapsed
        1025.0,  # epoch 2 timestamp
        1025.0,  # _write_status after epoch 2
    ]

    with patch("time.time", side_effect=time_values):
        observer = LiveTrainingObserver(output_dir=output_dir, trial_number=1)

        observer.on_event(
            TrainingEvent(event_type="epoch_end", epoch=0, metrics={"loss": 1.0})
        )
        observer.on_event(
            TrainingEvent(event_type="epoch_end", epoch=1, metrics={"loss": 0.9})
        )

    history = observer.epoch_history
    assert len(history) == 2
    assert history[0]["duration"] == 10.0
    assert history[1]["duration"] == 15.0
