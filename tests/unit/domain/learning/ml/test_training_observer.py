"""Provide module-level functionality for the PFF codebase.



Notes:

    File: tests/unit/domain/learning/ml/test_training_observer.py

"""

from pff.domain.learning.ml.training_observer import (
    ConsoleObserver,
    MLflowObserver,
    TrainingEvent,
)


class _DummyMLflow:
    def __init__(self) -> None:
        """Execute init."""

        self.metrics: list[tuple[str, float, int | None]] = []
        self.artifacts: list[str] = []

    def log_metric(self, name: str, value: float, step: int | None = None) -> None:
        """Execute log metric.



        Args:

            name: Input value used by this callable.

            value: Input value used by this callable.

            step: Optional input value.

        """

        self.metrics.append((name, float(value), step))

    def log_artifact(self, path: str) -> None:
        """Execute log artifact.



        Args:

            path: Input value used by this callable.

        """

        self.artifacts.append(path)


def test_console_observer_has_eval_metrics() -> None:
    """Execute test console observer has eval metrics.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    assert ConsoleObserver._has_eval_metrics({"mrr": 0.1})
    assert ConsoleObserver._has_eval_metrics({"hits1": 0.1})
    assert ConsoleObserver._has_eval_metrics({"hits@1": 0.1})
    assert ConsoleObserver._has_eval_metrics({"mcc": 0.1})
    assert ConsoleObserver._has_eval_metrics({"ap10": 0.1})
    assert not ConsoleObserver._has_eval_metrics({"loss": 0.1})


def test_mlflow_observer_log_batch_end() -> None:
    """Execute test mlflow observer log batch end.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    dummy = _DummyMLflow()
    event = TrainingEvent(event_type="batch_end", epoch=1, step=5, metrics={})
    MLflowObserver._log_batch_end(event, dummy)
    assert dummy.metrics == []

    event = TrainingEvent(event_type="batch_end", epoch=1, step=5, metrics={"loss": 0.5})
    MLflowObserver._log_batch_end(event, dummy)
    assert dummy.metrics == [("batch_loss", 0.5, 5)]
