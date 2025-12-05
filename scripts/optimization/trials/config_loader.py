from __future__ import annotations

from pathlib import Path
from typing import Any

from pff.utils import logger
from pff.utils.core.file_manager import FileManager

from pff.config import ENSEMBLE_CONFIG_PATH, ENSEMBLE_HPO_CONFIG_PATH

_CONFIG_CACHE: dict[str, Any] = {}


def clear_config_cache() -> None:
    """Clear the config cache. Useful for testing."""
    _CONFIG_CACHE.clear()


def get_cached_config(path: Path, file_manager: FileManager | None = None) -> dict[str, Any]:
    """Load and cache YAML/JSON config via FileManager."""
    key = str(path)
    if key in _CONFIG_CACHE:
        return _CONFIG_CACHE[key]
    fm = file_manager or FileManager()
    cfg = fm.read(path) or {}
    _CONFIG_CACHE[key] = cfg
    return cfg


def load_ensemble_hpo_bounds(file_manager: FileManager | None = None) -> dict[str, Any]:
    """Load ensemble HPO bounds from config/hpo/ensemble_hpo.yaml or legacy ensemble.yaml."""
    fm = file_manager or FileManager()
    default_bounds = {
        "weights": {
            "neural_weight": {"low": 0.2, "high": 0.45},
            "rules_weight": {"low": 0.1, "high": 0.25},
            "lightgbm_weight": {"low": 0.45, "high": 0.7},
        },
        "thresholds": {
            "neural_threshold": {"low": 0.3, "high": 0.7},
            "rules_threshold": {"low": 0.2, "high": 0.7},
            "lightgbm_threshold": {"low": 0.3, "high": 0.7},
        },
        "target_symbolic_ratio": {"low": 0.3, "high": 0.42},
        "feature_selection_threshold": {"low": 0.3, "high": 0.55},
        "kge": {
            "negative_ratio": {"low": 0.4, "high": 0.8},
            "embedding_dim": {"choices": [128]},
            "self_adversarial": {"choices": [False]},
            "batch_size": {"low": 256, "high": 640},
        },
    }
    try:
        ensemble_config = get_cached_config(ENSEMBLE_HPO_CONFIG_PATH, fm)
        return ensemble_config.get("hpo_bounds", default_bounds) or default_bounds
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"Failed to load ensemble hpo_bounds: {exc}")
        try:
            legacy_config = get_cached_config(ENSEMBLE_CONFIG_PATH, fm)
            return legacy_config.get("hpo_bounds", default_bounds) or default_bounds
        except Exception as legacy_exc:  # noqa: BLE001
            logger.debug(f"Legacy ensemble.yaml load failed for hpo_bounds: {legacy_exc}")
        return default_bounds


def load_optuna_settings(file_manager: FileManager | None = None) -> dict[str, Any]:
    """Load Optuna sampler/pruner settings from config."""
    fm = file_manager or FileManager()
    default_settings = {
        "tpe": {
            "multivariate": True,
            "group": True,
            "n_startup_trials": 5,
            "constant_liar": False,
        },
        "hyperband": {
            "min_resource": 5,
            "max_resource": 50,
            "reduction_factor": 3,
        },
    }
    try:
        ensemble_config = get_cached_config(ENSEMBLE_HPO_CONFIG_PATH, fm)
        optuna_config = ensemble_config.get("optuna", {}) if isinstance(ensemble_config, dict) else {}
        if not isinstance(optuna_config, dict):
            return default_settings

        tpe_cfg = optuna_config.get("tpe", {}) or {}
        hyperband_cfg = optuna_config.get("hyperband", {}) or {}

        settings = {
            "tpe": {
                "multivariate": bool(tpe_cfg.get("multivariate", default_settings["tpe"]["multivariate"])),
                "group": bool(tpe_cfg.get("group", default_settings["tpe"]["group"])),
                "n_startup_trials": int(tpe_cfg.get("n_startup_trials", default_settings["tpe"]["n_startup_trials"])),
                "constant_liar": bool(tpe_cfg.get("constant_liar", default_settings["tpe"]["constant_liar"])),
            },
            "hyperband": {
                "min_resource": int(hyperband_cfg.get("min_resource", default_settings["hyperband"]["min_resource"])),
                "max_resource": int(hyperband_cfg.get("max_resource", default_settings["hyperband"]["max_resource"])),
                "reduction_factor": int(
                    hyperband_cfg.get("reduction_factor", default_settings["hyperband"]["reduction_factor"])
                ),
            },
        }
        return settings
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"Failed to load Optuna settings; using defaults: {exc}")
        return default_settings


def load_live_plot_settings(file_manager: FileManager | None = None) -> dict[str, Any]:
    """Load live plot settings from ensemble HPO config."""
    fm = file_manager or FileManager()
    defaults = {
        "enabled": True,
        "max_trials_axis": 50,
        "output_subdir": "optimization/plots/live",
    }
    try:
        ensemble_config = get_cached_config(ENSEMBLE_HPO_CONFIG_PATH, fm)
        if not isinstance(ensemble_config, dict):
            return defaults
        live_cfg = ensemble_config.get("live_plots", {}) or {}
        return {
            "enabled": bool(live_cfg.get("enabled", defaults["enabled"])),
            "max_trials_axis": float(live_cfg.get("max_trials_axis", defaults["max_trials_axis"])),
            "output_subdir": str(live_cfg.get("output_subdir", defaults["output_subdir"])),
        }
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"Failed to load live plot settings; using defaults: {exc}")
        return defaults


def load_trial_constraints(file_manager: FileManager | None = None) -> dict[str, Any]:
    """Load trial-level symbolic/coverage constraints from ensemble HPO config."""
    fm = file_manager or FileManager()
    defaults = {
        "coverage_gate": 0.25,
        "dominance_gate": 0.85,
        "min_symbolic_activation": 0.01,
        "symbolic_max_rules": None,
    }
    try:
        ensemble_config = get_cached_config(ENSEMBLE_HPO_CONFIG_PATH, fm)
        if not isinstance(ensemble_config, dict):
            return defaults
        constraints = ensemble_config.get("constraints", {}) or {}
        return {
            "coverage_gate": float(constraints.get("coverage_gate", defaults["coverage_gate"])),
            "dominance_gate": float(constraints.get("dominance_gate", defaults["dominance_gate"])),
            "min_symbolic_activation": float(
                constraints.get("min_symbolic_activation", defaults["min_symbolic_activation"])
            ),
            "symbolic_max_rules": (
                int(constraints["symbolic_max_rules"]) if constraints.get("symbolic_max_rules") is not None else None
            ),
        }
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"Failed to load trial constraints; using defaults: {exc}")
        return defaults
