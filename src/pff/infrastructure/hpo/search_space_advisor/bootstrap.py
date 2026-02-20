"""Bootstrap utilities for Search Space Advisor."""

from __future__ import annotations

import random
from typing import Any, Callable


def bootstrap_action_support(
    *,
    trials: list[Any],
    min_trials: int,
    bootstrap_samples: int,
    seed: int,
    evaluate_sample: Callable[[list[Any]], bool | None],
) -> float | None:
    """Estimate action support by bootstrap resampling over trials."""
    if len(trials) < int(min_trials):
        return None

    rng = random.Random(seed)
    hits = 0
    evaluated = 0
    for _ in range(int(bootstrap_samples)):
        sample = [trials[rng.randint(0, len(trials) - 1)] for _ in range(len(trials))]
        result = evaluate_sample(sample)
        if result is None:
            continue
        evaluated += 1
        hits += int(bool(result))

    if evaluated <= 0:
        return None
    return float(hits) / float(evaluated)


__all__ = ["bootstrap_action_support"]

