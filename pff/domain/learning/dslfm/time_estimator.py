"""Time Budget Estimator for Trial Pruning.

This module provides a mechanism to estimate the total runtime of a training trial
and prune it early if it is projected to exceed a specified time budget.

It implements a two-phase time guard:
1. Early Phase (epoch < max/2): Prunes if projection to reach halfway point exceeds limit.
   Uses a rolling average of evaluation intervals after an initial first-eval check.
2. Grace Phase (epoch >= max/2): If elapsed time > 14min, allows UP TO a fixed number
   of evaluations (tolerance). Before each one, checks if `elapsed + next_eval_time <= 15min`.
   If not, prunes immediately.

Design Patterns:
    - Component: reusable logic for time tracking
    - Strategy: configurable pruning decision
"""

import time
from collections import deque
from dataclasses import dataclass
from typing import Any

from pff.shared.core.logger import logger


@dataclass
class TimeBudgetConfig:
    """Configuration for TimeBudgetEstimator."""

    enabled: bool = False
    max_total_time_s: float = 900.0  # 15 minutes
    tolerance_start_s: float = 840.0  # 14 minutes
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
        self.config = config
        self.total_epochs = total_epochs
        self.validate_every = validate_every
        self.clock = clock

        self.start_time = self.clock()
        self._last_eval_end_time = self.start_time

        eval_window = max(1, int(self.config.eval_time_window))
        self._phase1_eval_window = eval_window
        self._eval_intervals = deque(maxlen=eval_window)
        self._eval_count = 0

        # State for Phase 2
        self.tolerance_counter = 0

    def start_epoch(self) -> None:
        """Mark the start of a training epoch. (No-op in this strategy)"""
        pass

    def end_epoch(self) -> None:
        """Mark the end of a training epoch. (No-op in this strategy)"""
        pass

    def start_eval(self) -> None:
        """Mark the start of an evaluation phase. (No-op in this strategy)"""
        pass

    def end_eval(self) -> None:
        """Mark the end of an evaluation phase and update timing stats."""
        # Main logic is in check_budget
        pass

    def record_eval_completion(self) -> None:
        """Update timestamp of the last completed evaluation.

        Should be called AFTER check_budget to prepare for the next interval.
        """
        self._last_eval_end_time = self.clock()

    def check_budget(self, current_epoch: int, loss: float | None = None) -> bool:
        """Check if the estimated time exceeds the budget.

        Args:
            current_epoch: The current epoch number (0-based).
            loss: Current training loss.

        Returns:
            True if the trial should be pruned, False otherwise.
        """
        if not self.config.enabled:
            return False

        now = self.clock()
        elapsed = now - self.start_time
        halfway_point = self.total_epochs // 2
        completed_epochs = current_epoch + 1

        loss_str = f"loss={loss:.4f}" if loss is not None else ""

        # Calculate interval duration (time since last check)
        # This represents the time taken for 'validate_every' epochs + 1 validation
        interval_duration = now - self._last_eval_end_time
        self._eval_intervals.append(interval_duration)
        self._eval_count += 1

        # Phase 1: Early Cutoff (Before halfway)
        if completed_epochs < halfway_point:
            epochs_to_half = halfway_point - completed_epochs

            if self._eval_count == 1:
                seconds_per_epoch = interval_duration / self.validate_every
                projected_time_to_half = elapsed + (epochs_to_half * seconds_per_epoch)
                # Optimization: Never prune on the very first evaluation.
                # The first epoch often includes compilation overhead (torch.compile)
                # or data loading initialization, making it a poor predictor.
                should_prune = False

                logger.info(
                    f"epoch={completed_epochs} etapa=time_guard {loss_str}\n"
                    f"phase=1 amostras=1/{self._phase1_eval_window} (warmup)\n"
                    f"proj_to_half={projected_time_to_half:.1f}s limit={self.config.max_total_time_s:.1f}s\n"
                )

                if should_prune:
                    logger.warning(
                        f"time_guard_prune phase=1: projected {projected_time_to_half:.1f}s "
                        f"> {self.config.max_total_time_s}s to reach epoch {halfway_point}"
                    )
                    self._last_eval_end_time = now
                    return True

                self._last_eval_end_time = now
                return False

            if self._eval_count < self._phase1_eval_window:
                logger.info(
                    f"epoch={completed_epochs} etapa=time_guard {loss_str}\n"
                    f"phase=1 status=aguardando_amostras\n"
                    f"amostras={self._eval_count}/{self._phase1_eval_window} last_interval={interval_duration:.1f}s\n"
                )
                self._last_eval_end_time = now
                return False

            avg_interval = sum(self._eval_intervals) / len(self._eval_intervals)
            seconds_per_epoch = avg_interval / self.validate_every
            projected_time_to_half = elapsed + (epochs_to_half * seconds_per_epoch)

            should_prune = projected_time_to_half > self.config.max_total_time_s

            logger.info(
                f"epoch={completed_epochs} etapa=time_guard {loss_str}\n"
                f"phase=1 amostras={self._eval_count}/{self._phase1_eval_window}\n"
                f"proj_to_half={projected_time_to_half:.1f}s limit={self.config.max_total_time_s:.1f}s\n"
            )

            if should_prune:
                logger.warning(
                    f"time_guard_prune phase=1: projected {projected_time_to_half:.1f}s "
                    f"> {self.config.max_total_time_s}s to reach epoch {halfway_point}"
                )
                self._last_eval_end_time = now
                return True

        # Phase 2: Grace Phase (After halfway)
        elif elapsed > self.config.tolerance_start_s:
            self.tolerance_counter += 1

            # Check 1: Hard Limit on Evaluations
            if self.tolerance_counter > self.config.tolerance_evals:
                logger.warning(
                    f"time_guard_prune phase=2: grace evals exceeded ({self.tolerance_counter} > {self.config.tolerance_evals}) "
                    f"after {elapsed:.1f}s"
                )
                self._last_eval_end_time = now
                return True

            # Check 2: Conditional Projection for NEXT evaluation
            # Assume next interval takes same time as current interval
            next_eval_projection = elapsed + interval_duration

            logger.info(
                f"epoch={completed_epochs} etapa=time_guard {loss_str}\n"
                f"phase=2 grace_count={self.tolerance_counter}/{self.config.tolerance_evals}\n"
                f"proj_next={next_eval_projection:.1f}s limit={self.config.max_total_time_s:.1f}s\n"
            )

            if next_eval_projection > self.config.max_total_time_s:
                logger.warning(
                    f"time_guard_prune phase=2: next eval projected {next_eval_projection:.1f}s "
                    f"> {self.config.max_total_time_s}s (no time for grace)"
                )
                self._last_eval_end_time = now
                return True

        else:
            logger.info(
                f"epoch={completed_epochs} etapa=time_guard {loss_str}\n"
                f"phase=2 status=ok (under {self.config.tolerance_start_s}s)\n"
            )

        # IMPORTANT: Prepare for next interval
        self._last_eval_end_time = now
        return False
