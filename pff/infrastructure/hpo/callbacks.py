#!/usr/bin/env python3
"""Facade for optimization callbacks.

This module keeps the public API stable while delegating implementation
into `callbacks_internal/` modules.
"""

from __future__ import annotations

from .callbacks_internal.configs import (
    _get_callback_config,
    _save_matplotlib_figure_png,
)
from .callbacks_internal.observers import (
    AdaptiveSamplerController,
    BestScoreObserver,
    CallbackManager,
    CompositeObserver,
    LoggingObserver,
    MaxTrialsCallback,
    MLflowTrialObserver,
    OptimizationObserver,
    StagnationDetector,
)
from .callbacks_internal.visualizers import LivePlotCallback, RealTimeVisualizer

__all__ = [
    "OptimizationObserver",
    "CompositeObserver",
    "LoggingObserver",
    "BestScoreObserver",
    "StagnationDetector",
    "AdaptiveSamplerController",
    "RealTimeVisualizer",
    "LivePlotCallback",
    "CallbackManager",
    "MLflowTrialObserver",
    "MaxTrialsCallback",
    "_get_callback_config",
    "_save_matplotlib_figure_png",
]
