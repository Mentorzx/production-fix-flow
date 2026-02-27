"""Time Budget Estimator for Trial Pruning.

This module provides a mechanism to estimate the total runtime of a training trial
and prune it early if it is projected to exceed a specified time budget.
"""

import time
from collections import deque
from dataclasses import dataclass
from typing import Any

from pff.shared.core.logging import logger


@dataclass
class TimeBudgetConfig:
    """Configuration for TimeBudgetEstimator."""

    enabled: bool = False
    max_total_time_s: float = 900.0
    tolerance_start_s: float = 840.0
    tolerance_evals: int = 2
    eval_time_window: int = 3

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TimeBudgetConfig":
        """Create config from dictionary."""
        return cls(
            enabled=data.get("enabled", False),
            max_total_time_s=float(data.get("max_total_time_s", 900.0)),
            tolerance_start_s=float(data.get("tolerance_start_s", 840.0)),
            tolerance_evals=int(data.get("tolerance_evals", 2)),
            eval_time_window=int(data.get("eval_time_window", 3)),
        )


class TimeBudgetEstimator:
    """Estimates trial runtime and recommends pruning."""

    def __init__(
        self,
        config: TimeBudgetConfig,
        total_epochs: int,
        validate_every: int,
        clock: Any = time.perf_counter,
    ) -> None:
        """Execute init.



        Args:

            config: Input value used by this callable.

            total_epochs: Input value used by this callable.

            validate_every: Input value used by this callable.

            clock: Optional input value.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        self.config = config
        self.total_epochs = total_epochs
        self.validate_every = validate_every
        self.clock = clock

        self.start_time = self.clock()
        self._last_eval_end_time = self.start_time

        eval_window = max(1, int(self.config.eval_time_window))
        self._phase1_eval_window = eval_window
        self._eval_intervals: deque[float] = deque(maxlen=eval_window)
        self._eval_count = 0
        self.tolerance_counter = 0

    def start_epoch(self) -> None:
        """Execute start epoch."""

        pass

    def end_epoch(self) -> None:
        """Execute end epoch."""

        pass

    def start_eval(self) -> None:
        """Execute start eval."""

        pass

    def end_eval(self) -> None:
        """Execute end eval."""

        pass

    def record_eval_completion(self) -> None:
        """Execute record eval completion."""

        self._last_eval_end_time = self.clock()

    def check_budget(self, current_epoch: int, loss: float | None = None) -> bool:
        """Execute check budget.



        Args:

            current_epoch: Input value used by this callable.

            loss: Optional input value.



        Returns:

            Return value produced by the callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        if not self.config.enabled:
            return False

        now = self.clock()
        interval_duration = now - self._last_eval_end_time
        elapsed = now - self.start_time

        self._eval_intervals.append(interval_duration)
        self._eval_count += 1

        if current_epoch < self.total_epochs // 2:
            if self._eval_count < self._phase1_eval_window:
                return False

            avg_interval = sum(self._eval_intervals) / len(self._eval_intervals)
            remaining_epochs = self.total_epochs - current_epoch
            projected_remaining = (
                remaining_epochs / self.validate_every
            ) * avg_interval
            total_est = elapsed + projected_remaining

            if total_est > self.config.max_total_time_s:
                logger.warning(
                    f"Trial projected to take {total_est / 60:.1f}min (limit {self.config.max_total_time_s / 60:.1f}min). Pruning."
                )
                return True
            return False

        if elapsed > self.config.tolerance_start_s:
            self.tolerance_counter += 1

            avg_interval = (
                sum(self._eval_intervals) / len(self._eval_intervals)
                if self._eval_intervals
                else interval_duration
            )

            if self.tolerance_counter > self.config.tolerance_evals:
                logger.warning(
                    f"Time budget exceeded ({elapsed / 60:.1f}min) and grace period over. Pruning."
                )
                return True

            if elapsed + avg_interval > self.config.max_total_time_s:
                logger.warning(
                    f"Trial projected to exceed limit ({elapsed / 60:.1f}min). Pruning."
                )
                return True

            logger.info(
                f"Time budget within grace ({elapsed / 60:.1f}min). "
                f"Token {self.tolerance_counter}/{self.config.tolerance_evals} used."
            )
            return False

        return False
