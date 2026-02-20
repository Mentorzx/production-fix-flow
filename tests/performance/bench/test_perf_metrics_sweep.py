"""Provide module-level functionality for the PFF codebase.



Notes:

    File: tests/performance/bench/test_perf_metrics_sweep.py

"""

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from pff.infrastructure.hpo.callbacks_internal.collectors import (
    extract_metric_series,
)


@dataclass
class MockTrial:
    """Represent MockTrial."""

    number: int
    value: float
    user_attrs: dict[str, Any]
    state: Any = None
    duration: Any = None


def create_mock_trials(n: int = 500):
    """Execute create mock trials.



    Args:

        n: Optional input value.



    Returns:

        Return value produced by the callable.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    trials = []
    for i in range(n):
        trials.append(
            MockTrial(
                number=i,
                value=np.random.random(),
                user_attrs={
                    "metrics": {"mrr": np.random.random(), "hits1": np.random.random()},
                    "clf_metrics": {
                        "mcc": np.random.random(),
                        "auc": np.random.random(),
                    },
                    "duration": np.random.random() * 100,
                },
            )
        )
    return trials


def bench_metrics_extraction():
    """Execute bench metrics extraction.



    Returns:

        Return value produced by the callable.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    trials = create_mock_trials(1000)
    metrics_to_extract = ["score", "mrr", "hits1", "mcc", "auc", "duration"]

    start = time.perf_counter()
    results = {}
    for m in metrics_to_extract:
        results[m] = extract_metric_series(trials, m)
    end = time.perf_counter()

    elapsed_ms = (end - start) * 1000
    print(f"Metrics Extraction (1000 trials, 6 metrics): {elapsed_ms:.2f}ms")
    return elapsed_ms


if __name__ == "__main__":
    bench_metrics_extraction()
