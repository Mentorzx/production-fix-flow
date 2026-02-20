"""Distributed optimization facade for HPO.

This module provides a lightweight, backward-compatible `DistributedOptimizer`
API used by integration tests and older scripts.

Design Patterns:
    - Facade: Presents a stable API independent of the concrete strategy backend.
    - Strategy: Delegates execution to `ConcurrencyManager` (thread/process backends).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pff.shared import ConcurrencyManager, logger
from pff_rust import stable_hash
from pff.shared.ops.global_interrupt_manager import get_interrupt_manager


@dataclass(frozen=True, slots=True)
class _SimpleTrial:
    number: int
    params: dict[str, Any]


def _sample_params(search_space: dict[str, Any], trial_number: int, *, seed: int) -> dict[str, Any]:
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
    except Exception as exc:
        logger.warning(
            f"component_name=hpo_distributed message='NumPy unavailable for distributed sampling: {exc}'"
        )
        np = None  # type: ignore[assignment]

    params: dict[str, Any] = {}
    for key, spec in (search_space or {}).items():
        handled, value = _sample_spec(spec, trial_number, rng_seed, np)
        if handled:
            params[key] = value
    return params


def _sample_spec(
    spec: Any,
    trial_number: int,
    rng_seed: int,
    np: Any,
) -> tuple[bool, Any]:
    """Execute sample spec.



    Args:

        spec: Input value used by this callable.

        trial_number: Input value used by this callable.

        rng_seed: Input value used by this callable.

        np: Input value used by this callable.



    Returns:

        Return value produced by the callable.

    """

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
        return True, spec[idx]

    if (
        isinstance(spec, tuple)
        and len(spec) == 2
        and all(isinstance(x, (int, float)) for x in spec)
    ):
        return True, _sample_numeric_range(spec[0], spec[1], rng_seed, np)

    if isinstance(spec, dict):
        low = spec.get("low")
        high = spec.get("high")
        if low is None or high is None:
            return False, None
        typ = str(spec.get("type", "float")).lower()
        if typ == "int":
            return True, int(_sample_numeric_range(int(low), int(high), rng_seed, np))
        return True, float(_sample_numeric_range(float(low), float(high), rng_seed, np))

    return False, None


def _sample_numeric_range(low: float, high: float, rng_seed: int, np: Any) -> Any:
    """Execute sample numeric range.



    Args:

        low: Input value used by this callable.

        high: Input value used by this callable.

        rng_seed: Input value used by this callable.

        np: Input value used by this callable.



    Returns:

        Return value produced by the callable.

    """

    if np is None:
        return low
    rng = np.random.default_rng(rng_seed)
    if isinstance(low, int) and isinstance(high, int):
        return int(rng.integers(int(low), int(high) + 1))
    return float(rng.uniform(float(low), float(high)))


class DistributedOptimizer:
    """Facade for running trial evaluations with cooperative interrupt handling."""

    def __init__(self, *, seed: int = 1337) -> None:
        """Execute init.



        Args:

            seed: Optional input value.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

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
            _SimpleTrial(number=i, params=_sample_params(search_space, i, seed=self._seed))
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
            best_number, best_value, best_params = max(results, key=lambda item: item[1])
            logger.info(
                f"component_name=hpo_distributed key_parameters={{'trial': {best_number}, 'valor': {best_value}}} message='Melhor trial distribuído encontrado'"
            )

        return {
            "interrupted": interrupted,
            "n_trials": len(results),
            "best_value": best_value,
            "best_params": best_params,
        }
