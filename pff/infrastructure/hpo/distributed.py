"""Distributed optimization facade for HPO.

This module provides a lightweight, backward-compatible `DistributedOptimizer`
API used by integration tests and older scripts.

Design Patterns:
    - Facade: Presents a stable API independent of the concrete strategy backend.
    - Strategy: Delegates execution to `ConcurrencyManager` (thread/process backends).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from collections.abc import Callable

from pff.shared import ConcurrencyManager, logger
from pff.shared.hash import stable_hash
from pff.shared.ops.global_interrupt_manager import get_interrupt_manager


@dataclass(frozen=True, slots=True)
class _SimpleTrial:
    number: int
    params: dict[str, Any]


def _sample_params(
    search_space: dict[str, Any], trial_number: int, *, seed: int
) -> dict[str, Any]:
    """Deterministically sample parameters from a minimal search-space schema.

    Supported schemas:
    - `key: [v1, v2, ...]` categorical choices.
    - `key: (low, high)` numeric range (float or int).
    - `key: {\"type\": \"float\"|\"int\", \"low\": ..., \"high\": ...}`.

    Args:
        search_space: Search space definition.
        trial_number: Trial index used for deterministic sampling.
        seed: Global seed for the sampler.

    Returns:
        Parameter dictionary.
    """
    rng_seed = stable_hash((seed, trial_number), truncate=16) & (2**32 - 1)
    try:
        import numpy as np
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"NumPy unavailable for distributed sampling: {exc}")
        np = None  # type: ignore

    params: dict[str, Any] = {}
    for key, spec in (search_space or {}).items():
        if (
            isinstance(spec, (list, tuple))
            and spec
            and not (
                len(spec) == 2
                and isinstance(spec[0], (int, float))
                and isinstance(spec[1], (int, float))
            )
        ):
            idx = trial_number % len(spec)
            params[key] = spec[idx]
            continue

        if (
            isinstance(spec, tuple)
            and len(spec) == 2
            and all(isinstance(x, (int, float)) for x in spec)
        ):
            low, high = spec
            if np is None:
                params[key] = low
                continue
            rng = np.random.default_rng(rng_seed)
            if isinstance(low, int) and isinstance(high, int):
                params[key] = int(rng.integers(int(low), int(high) + 1))
            else:
                params[key] = float(rng.uniform(float(low), float(high)))
            continue

        if isinstance(spec, dict):
            low = spec.get("low")
            high = spec.get("high")
            typ = str(spec.get("type", "float")).lower()
            if low is None or high is None:
                continue
            if np is None:
                params[key] = low
                continue
            rng = np.random.default_rng(rng_seed)
            if typ == "int":
                params[key] = int(rng.integers(int(low), int(high) + 1))
            else:
                params[key] = float(rng.uniform(float(low), float(high)))
            continue

    return params


class DistributedOptimizer:
    """Facade for running trial evaluations with cooperative interrupt handling."""

    def __init__(self, *, seed: int = 1337) -> None:
        self._seed = int(seed)
        self._interrupt_manager = get_interrupt_manager()
        self._concurrency = ConcurrencyManager()

    def run_distributed(
        self,
        objective_fn: Callable[[_SimpleTrial], float],
        search_space: dict[str, Any],
        *,
        n_trials: int,
        num_workers: int = 1,
        task_type: str = "thread",
    ) -> dict[str, Any]:
        """Run objective evaluations across multiple workers.

        Args:
            objective_fn: Callable that consumes a trial-like object.
            search_space: Minimal search space schema.
            n_trials: Number of trials to execute.
            num_workers: Maximum parallel workers requested.
            task_type: Concurrency backend for `ConcurrencyManager.execute_sync`.

        Returns:
            Dictionary with best trial summary and interruption flag.
        """
        if self._interrupt_manager.should_stop:
            return {
                "interrupted": True,
                "n_trials": 0,
                "best_value": None,
                "best_params": {},
            }

        n_trials_int = max(0, int(n_trials))
        max_workers = max(1, int(num_workers))

        trials = [
            _SimpleTrial(
                number=i, params=_sample_params(search_space, i, seed=self._seed)
            )
            for i in range(n_trials_int)
        ]

        def _run_one(trial: _SimpleTrial) -> tuple[int, float, dict[str, Any]]:
            value = float(objective_fn(trial))
            return trial.number, value, dict(trial.params)

        results: list[tuple[int, float, dict[str, Any]]] = []
        interrupted = False

        if max_workers <= 1:
            for trial in trials:
                if self._interrupt_manager.should_stop:
                    interrupted = True
                    break
                try:
                    results.append(_run_one(trial))
                except KeyboardInterrupt:
                    interrupted = True
                    break
        else:
            try:
                args_list = [(trial,) for trial in trials]
                results.extend(
                    self._concurrency.execute_sync(
                        _run_one,
                        args_list,
                        task_type=task_type,
                        max_workers=max_workers,
                        desc="distributed_trials",
                    )
                )
            except KeyboardInterrupt:
                interrupted = True

        best_value = None
        best_params: dict[str, Any] = {}
        if results:
            best_number, best_value, best_params = max(
                results, key=lambda item: item[1]
            )
            logger.info(f"melhor_distribuido trial={best_number} valor={best_value}")

        return {
            "interrupted": interrupted,
            "n_trials": len(results),
            "best_value": best_value,
            "best_params": best_params,
        }
