"""Thin wrappers for HPO config loaders used by trial modules."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pff.shared.core.file_manager import FileManager
from pff.infrastructure.hpo import config_loader as hpo_config

_CONFIG_CACHE: dict[str, dict[str, Any]] = {}


def clear_config_cache() -> None:
    """Clear the in-memory optimization config cache."""
    _CONFIG_CACHE.clear()


def get_cached_config(
    path: str | Path, file_manager: FileManager | None = None
) -> dict[str, Any]:
    """Load and memoize the optimization config at `path`.

    Args:
        path: Path to the optimization config file.
        file_manager: Optional FileManager dependency.

    Returns:
        Parsed optimization config dictionary.
    """
    key = str(path)
    if key in _CONFIG_CACHE:
        return _CONFIG_CACHE[key]
    fm = file_manager or FileManager()
    cfg = hpo_config.load_optimization_config(str(path), fm)
    _CONFIG_CACHE[key] = cfg
    return cfg


def load_optuna_settings(file_manager: FileManager | None = None) -> dict[str, Any]:
    """Load Optuna sampler/pruner settings from config."""
    return hpo_config.load_optuna_settings(file_manager)


def load_live_plot_settings(file_manager: FileManager | None = None) -> dict[str, Any]:
    """Load live-plot settings for optimization monitoring."""
    return hpo_config.load_live_plot_settings(file_manager)


def load_multi_objective_settings(
    file_manager: FileManager | None = None,
) -> dict[str, Any]:
    """Load multi-objective settings for HPO."""
    return hpo_config.load_multi_objective_settings(file_manager)


def load_trial_constraints(file_manager: FileManager | None = None) -> dict[str, Any]:
    """Load trial constraint gates from config."""
    return hpo_config.load_trial_constraints(file_manager)


def load_scoring_settings(file_manager: FileManager | None = None) -> dict[str, Any]:
    """Load scoring configuration (weights/bounds/time scale) from config."""
    return hpo_config.load_scoring_settings(file_manager)
