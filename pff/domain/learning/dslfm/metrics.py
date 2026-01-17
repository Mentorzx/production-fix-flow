"""DSLFM metrics facade.

Provides stable imports for metrics reporting helpers while delegating to the
existing reporter implementation.
"""

from __future__ import annotations

from .metrics_reporter import DSLFMMetricsReporter  # noqa: F401

__all__ = ["DSLFMMetricsReporter"]
