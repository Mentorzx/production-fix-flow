"""HPO configuration loader (config-first, infra-backed).

Centralizes YAML config reading for HPO and exposes helpers for sampler/pruner,
scoring settings, and bounds. All filesystem access stays in infrastructure.

Uses CacheManager for config caching to avoid repeated file I/O during HPO runs.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pff.domain.hpo.scoring import ScoreWeights, build_weights_from_settings
from pff.shared.core.cache import CacheManager
from pff.shared.core.config import (
    ENSEMBLE_CONFIG_PATH,
    ENSEMBLE_HPO_CONFIG_PATH,
    OPTIMIZATION_CONFIG_PATH,
)
from pff.shared.core.file_manager import FileManager, ParquetBundle
from pff.shared.core.logging import logger

DEFAULT_OPT_PATH = OPTIMIZATION_CONFIG_PATH

_config_cache: CacheManager | None = None
_CONFIG_TTL = 3600


def _get_config_cache() -> CacheManager:
    """Get or create config cache singleton."""
    global _config_cache
    if _config_cache is None:
        _config_cache = CacheManager(max_memory_items=50)
    return _config_cache


def _read_native(file_manager: FileManager, path: str | Path) -> Any:
    payload = file_manager.read(path)
    if isinstance(payload, ParquetBundle):
        payload = payload.to_native()

    try:
        if hasattr(payload, "to_dicts") and hasattr(payload, "is_empty"):
            if not payload.is_empty():
                return payload.to_dicts()[0]
            return {}

        import polars as pl

        if isinstance(payload, pl.DataFrame):
            if not payload.is_empty():
                return payload.to_dicts()[0]
            return {}
    except Exception:
        pass

    return payload


def _read_config_cached(file_manager: FileManager, path: str | Path) -> Any:
    """Read config with caching using CacheManager."""
    return _read_native(file_manager, path)


def clear_config_cache() -> None:
    """Clear all cached configs (useful after config file changes)."""
    cache = _get_config_cache()
    cache.invalidate_by_tag("hpo_config")


def load_optimization_config(
    path: str | Path | None = None,
    file_manager: FileManager | None = None,
) -> dict[str, Any]:
    """Load HPO optimization configuration via FileManager.

    Args:
        path: Optional override path to YAML.
        file_manager: Optional FileManager instance.

    Returns:
        Parsed configuration dict (empty if missing or invalid).
    """
    fm = file_manager or FileManager()
    cfg_path = path or DEFAULT_OPT_PATH
    try:
        raw = _read_config_cached(fm, cfg_path) or {}
    except Exception as exc:
        logger.warning(f"Failed to load optimization config from {cfg_path}: {exc}")
        return {}
    if not isinstance(raw, dict):
        logger.warning(f"Optimization config at {cfg_path} is not a mapping")
        return {}
    return raw


def _read_hpo_config(file_manager: FileManager | None = None) -> dict[str, Any]:
    """Read HPO config with fallback to ensemble config (cached)."""
    fm = file_manager or FileManager()
    cfg = load_optimization_config(DEFAULT_OPT_PATH, fm)
    if cfg:
        return cfg
    try:
        legacy = (
            _read_config_cached(fm, ENSEMBLE_HPO_CONFIG_PATH)
            or _read_config_cached(fm, ENSEMBLE_CONFIG_PATH)
            or {}
        )
        if isinstance(legacy, dict):
            return legacy
    except Exception as exc:
        logger.debug(f"Fallback ensemble HPO config load failed: {exc}")
    return {}


def load_parallel_settings(file_manager: FileManager | None = None) -> dict[str, Any]:
    """Load parallel execution settings for HPO.

    Returns:
        Dict with keys:
        - n_jobs: Number of parallel workers (default 1)
        - use_journal_for_parallel: Use JournalStorage when n_jobs > 1 (default True)
    """
    cfg = _read_hpo_config(file_manager)
    parallel = cfg.get("parallel", {}) if isinstance(cfg, dict) else {}
    if not isinstance(parallel, dict):
        parallel = {}
    return {
        "n_jobs": int(parallel.get("n_jobs", 1)),
        "use_journal_for_parallel": bool(parallel.get("use_journal_for_parallel", True)),
    }


def load_hpo_defaults(file_manager: FileManager | None = None) -> dict[str, Any]:
    """Load flattened DSLFM HPO defaults from config/hpo/optimization.yaml."""
    cfg = _read_hpo_config(file_manager)

    dslfm_kgc = cfg.get("dslfm_kgc", {}) if isinstance(cfg, dict) else {}
    if not isinstance(dslfm_kgc, dict):
        return {}
    if not dslfm_kgc:
        return cfg.get("dslfm_defaults", {}) if isinstance(cfg, dict) else {}

    flat: dict[str, Any] = {}
    training = dslfm_kgc.get("training", {}) if isinstance(dslfm_kgc, dict) else {}
    flat["learning_rate_low"] = training.get("lr_low", 5e-6)
    flat["learning_rate_high"] = training.get("lr_high", 5e-4)
    batch_choices = training.get("batch_size_choices", [512, 768, 1024])
    if isinstance(batch_choices, (list, tuple)) and batch_choices:
        batch_choices = [int(choice) for choice in batch_choices]
    else:
        batch_choices = [512, 768, 1024]
    flat["batch_size_choices"] = batch_choices
    flat["batch_size_low"] = int(training.get("batch_size_low", min(batch_choices)))
    flat["batch_size_high"] = int(training.get("batch_size_high", max(batch_choices)))
    flat["negative_sample_size_low"] = training.get("negative_sample_size_low", 64)
    flat["negative_sample_size_high"] = training.get("negative_sample_size_high", 512)
    flat["epochs_low"] = training.get("epochs_low", 50)
    flat["epochs_high"] = training.get("epochs_high", 150)

    arch = dslfm_kgc.get("architecture", {}) if isinstance(dslfm_kgc, dict) else {}
    flat["embedding_dim_choices"] = arch.get("feature_dim_choices", [128, 256])
    flat["attr_hidden_dim_choices"] = arch.get("hidden_dim_choices", [256, 512])
    flat["lambda_kl_low"] = arch.get("kl_weight_low", 1e-4)
    flat["lambda_kl_high"] = arch.get("kl_weight_high", 5e-2)
    flat["lambda_sparsity_low"] = arch.get("sparsity_weight_low", 1e-6)
    flat["lambda_sparsity_high"] = arch.get("sparsity_weight_high", 1e-2)
    flat["ibp_alpha_low"] = arch.get("ibp_alpha_low", 0.5)
    flat["ibp_alpha_high"] = arch.get("ibp_alpha_high", 5.0)
    flat["max_communities_choices"] = arch.get("max_communities_choices", [16, 32, 64, 128])

    contr = dslfm_kgc.get("contrastive", {}) if isinstance(dslfm_kgc, dict) else {}
    sampling = dslfm_kgc.get("sampling", {}) if isinstance(dslfm_kgc, dict) else {}
    flat["contrastive_temperature_low"] = contr.get("temperature_low", 0.05)
    flat["contrastive_temperature_high"] = contr.get("temperature_high", 0.2)
    flat["adversarial_temperature_low"] = sampling.get("adv_temperature_low", 0.5)
    flat["adversarial_temperature_high"] = sampling.get("adv_temperature_high", 5.0)
    flat["margin_low"] = contr.get("margin_low", 0.0)
    flat["margin_high"] = contr.get("margin_high", 0.05)
    flat["negative_sample_size_choices"] = contr.get("num_negatives_choices", [32, 64, 128, 256])
    flat["num_global_negatives_low"] = contr.get("num_global_negatives_low", 64)
    flat["num_global_negatives_high"] = contr.get("num_global_negatives_high", 256)
    flat["neg_sampler_choices"] = contr.get("neg_sampler_choices", ["uniform", "self_adversarial"])
    flat["self_adversarial_choices"] = contr.get("self_adversarial_choices", [False])

    logic = dslfm_kgc.get("logic", {}) if isinstance(dslfm_kgc, dict) else {}
    flat["lambda_logic_low"] = logic.get("lambda_logic_low", 0.0)
    flat["lambda_logic_high"] = logic.get("lambda_logic_high", 0.2)
    flat["t_norm_choices"] = logic.get("t_norm_choices", ["product", "lukasiewicz", "godel"])

    pc = dslfm_kgc.get("pc", {}) if isinstance(dslfm_kgc, dict) else {}
    flat["lambda_pc_low"] = pc.get("lambda_pc_low", 0.1)
    flat["lambda_pc_high"] = pc.get("lambda_pc_high", 2.0)
    flat["max_circuit_depth_choices"] = pc.get("depth_choices", [2, 3, 4])

    defaults = dslfm_kgc.get("defaults", {}) if isinstance(dslfm_kgc, dict) else {}
    if isinstance(defaults, dict):
        flat.update(defaults)

    return flat


def load_adaptive_range_factors(
    file_manager: FileManager | None = None,
) -> dict[str, Any]:
    """Load adaptive range factors from config/hpo/optimization.yaml."""
    cfg = _read_hpo_config(file_manager)
    if not isinstance(cfg, dict):
        return {}
    factors = cfg.get("adaptive_range_factors", {})
    return factors if isinstance(factors, dict) else {}


def load_optuna_settings(file_manager: FileManager | None = None) -> dict[str, Any]:
    """Load Optuna sampler/pruner settings from config."""
    defaults: dict[str, Any] = {
        "sampler": {
            "type": "tpe",
            "n_startup_trials": 10,
            "n_ei_candidates": 48,
            "multivariate": True,
            "group": True,
            "constant_liar": True,
            "consider_prior": True,
            "consider_magic_clip": True,
            "consider_endpoints": False,
            "warn_independent_sampling": True,
            "seed": 42,
        },
        "pruner": {
            "type": "hyperband",
            "n_startup_trials": 5,
            "n_warmup_steps": 10,
            "interval_steps": 1,
            "hyperband": {
                "min_resource": 1,
                "max_resource": 100,
                "reduction_factor": 3,
                "burn_in_epochs": 10,
            },
            "patient": {"patience": 3, "min_delta": 0.0},
            "wilcoxon": {"p_threshold": 0.1, "n_startup_steps": 2},
        },
    }
    cfg = _read_hpo_config(file_manager)
    sampler_cfg = cfg.get("sampler", {}) if isinstance(cfg, dict) else {}
    pruner_cfg = cfg.get("pruner", {}) if isinstance(cfg, dict) else {}

    sampler = dict(defaults["sampler"])
    if isinstance(sampler_cfg, dict):
        sampler.update({k: sampler_cfg.get(k, sampler[k]) for k in sampler.keys()})

    pruner = dict(defaults["pruner"])
    if isinstance(pruner_cfg, dict):
        pruner.update(
            {
                k: pruner_cfg.get(k, pruner[k])
                for k in pruner.keys()
                if k not in {"hyperband", "patient", "wilcoxon"}
            }
        )
        hyper_cfg = (
            pruner_cfg.get("hyperband", {})
            if isinstance(pruner_cfg.get("hyperband", {}), dict)
            else {}
        )
        patient_cfg = (
            pruner_cfg.get("patient", {}) if isinstance(pruner_cfg.get("patient", {}), dict) else {}
        )
        wilcoxon_cfg = (
            pruner_cfg.get("wilcoxon", {})
            if isinstance(pruner_cfg.get("wilcoxon", {}), dict)
            else {}
        )
        pruner["hyperband"] = {
            "min_resource": int(
                hyper_cfg.get("min_resource", defaults["pruner"]["hyperband"]["min_resource"])
            ),
            "max_resource": int(
                hyper_cfg.get("max_resource", defaults["pruner"]["hyperband"]["max_resource"])
            ),
            "reduction_factor": int(
                hyper_cfg.get(
                    "reduction_factor",
                    defaults["pruner"]["hyperband"]["reduction_factor"],
                )
            ),
            "burn_in_epochs": int(
                hyper_cfg.get("burn_in_epochs", defaults["pruner"]["hyperband"]["burn_in_epochs"])
            ),
        }
        pruner["patient"] = {
            "patience": int(patient_cfg.get("patience", defaults["pruner"]["patient"]["patience"])),
            "min_delta": float(
                patient_cfg.get("min_delta", defaults["pruner"]["patient"]["min_delta"])
            ),
        }
        pruner["wilcoxon"] = {
            "p_threshold": float(
                wilcoxon_cfg.get("p_threshold", defaults["pruner"]["wilcoxon"]["p_threshold"])
            ),
            "n_startup_steps": int(
                wilcoxon_cfg.get(
                    "n_startup_steps", defaults["pruner"]["wilcoxon"]["n_startup_steps"]
                )
            ),
        }

    sampler_seed = sampler.get("seed")
    if sampler_seed is not None:
        sampler["seed"] = int(sampler_seed)

    return {"sampler": sampler, "pruner": pruner}


def load_storage_settings(file_manager: FileManager | None = None) -> dict[str, Any]:
    """Load Optuna storage configuration from config/hpo/optimization.yaml."""
    defaults: dict[str, Any] = {
        "backend": "sqlite",
        "url": None,
        "engine": {
            "pool_size": 20,
            "max_overflow": 10,
            "pool_pre_ping": True,
            "connect_args": {"keepalives": 1},
        },
        "grpc_proxy": {
            "host": os.getenv("PFF_HPO_GRPC_HOST", "localhost"),
            "port": int(os.getenv("PFF_HPO_GRPC_PORT", "13000")),
        },
    }
    cfg = _read_hpo_config(file_manager)
    storage_cfg = cfg.get("storage", {}) if isinstance(cfg, dict) else {}
    if not isinstance(storage_cfg, dict):
        return defaults
    engine_cfg = storage_cfg.get("engine", {})
    if not isinstance(engine_cfg, dict):
        engine_cfg = {}
    connect_args = engine_cfg.get("connect_args", {})
    if not isinstance(connect_args, dict):
        connect_args = {}
    grpc_cfg = storage_cfg.get("grpc_proxy", {})
    if not isinstance(grpc_cfg, dict):
        grpc_cfg = {}
    env_grpc_host = os.getenv("PFF_HPO_GRPC_HOST")
    env_grpc_port = os.getenv("PFF_HPO_GRPC_PORT")
    grpc_host = (
        env_grpc_host
        if env_grpc_host is not None
        else grpc_cfg.get("host", defaults["grpc_proxy"]["host"])
    )
    grpc_port = (
        env_grpc_port
        if env_grpc_port is not None
        else grpc_cfg.get("port", defaults["grpc_proxy"]["port"])
    )

    return {
        "backend": str(storage_cfg.get("backend", defaults["backend"])),
        "url": storage_cfg.get("url", defaults["url"]),
        "engine": {
            "pool_size": int(engine_cfg.get("pool_size", defaults["engine"]["pool_size"])),
            "max_overflow": int(engine_cfg.get("max_overflow", defaults["engine"]["max_overflow"])),
            "pool_pre_ping": bool(
                engine_cfg.get("pool_pre_ping", defaults["engine"]["pool_pre_ping"])
            ),
            "connect_args": {**defaults["engine"]["connect_args"], **connect_args},
        },
        "grpc_proxy": {
            "host": str(grpc_host),
            "port": int(grpc_port),
        },
    }


def load_live_plot_settings(file_manager: FileManager | None = None) -> dict[str, Any]:
    defaults = {
        "enabled": True,
        "max_trials_axis": 50,
        "output_subdir": "optimization/plots/live",
        "enable_optuna_dashboard": False,
        "dashboard_interval": 5,
        "dashboard_top_n": 12,
    }
    cfg = _read_hpo_config(file_manager).get("live_plots", {})
    if not isinstance(cfg, dict):
        return defaults
    return {
        "enabled": bool(cfg.get("enabled", defaults["enabled"])),
        "max_trials_axis": float(cfg.get("max_trials_axis", defaults["max_trials_axis"])),
        "output_subdir": str(cfg.get("output_subdir", defaults["output_subdir"])),
        "enable_optuna_dashboard": bool(
            cfg.get("enable_optuna_dashboard", defaults["enable_optuna_dashboard"])
        ),
        "dashboard_interval": int(cfg.get("dashboard_interval", defaults["dashboard_interval"])),
        "dashboard_top_n": int(cfg.get("dashboard_top_n", defaults["dashboard_top_n"])),
    }


def load_multi_objective_settings(
    file_manager: FileManager | None = None,
) -> dict[str, Any]:
    """Load multi-objective HPO settings from config."""
    defaults: dict[str, Any] = {
        "enabled": False,
        "sampler": "motpe",
        "directions": ["maximize", "maximize", "minimize"],
        "secondary_metric": "mcc",
        "tertiary_metric": "duration",
        "population_size": 50,
        "mutation_prob": 0.1,
        "crossover_prob": 0.9,
    }
    cfg = _read_hpo_config(file_manager).get("multi_objective", {})
    if not isinstance(cfg, dict):
        return defaults
    directions = cfg.get("directions", defaults["directions"])
    if not isinstance(directions, list) or not directions:
        directions = defaults["directions"]
    normalized: list[str] = []
    for direction in directions:
        if not isinstance(direction, str):
            continue
        value = direction.strip().lower()
        if value in {"maximize", "minimize"}:
            normalized.append(value)
    if len(normalized) < 2:
        normalized = list(defaults["directions"])  # type: ignore[arg-type]
    return {
        "enabled": bool(cfg.get("enabled", defaults["enabled"])),
        "sampler": str(cfg.get("sampler", defaults["sampler"])),
        "directions": normalized,
        "secondary_metric": str(cfg.get("secondary_metric", defaults["secondary_metric"])),
        "tertiary_metric": str(
            cfg.get("tertiary_metric", defaults.get("tertiary_metric", "duration"))
        ),
        "population_size": int(cfg.get("population_size", defaults["population_size"])),
        "mutation_prob": float(cfg.get("mutation_prob", defaults["mutation_prob"])),
        "crossover_prob": float(cfg.get("crossover_prob", defaults["crossover_prob"])),
    }


def load_trial_constraints(file_manager: FileManager | None = None) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "coverage_gate": 0.25,
        "dominance_gate": 0.85,
        "min_symbolic_activation": 0.01,
        "symbolic_max_rules": None,
    }
    cfg = _read_hpo_config(file_manager).get("constraints", {})
    if not isinstance(cfg, dict):
        return defaults
    return {
        "coverage_gate": float(cfg.get("coverage_gate") or defaults["coverage_gate"]),
        "dominance_gate": float(cfg.get("dominance_gate") or defaults["dominance_gate"]),
        "min_symbolic_activation": float(
            cfg.get("min_symbolic_activation", defaults["min_symbolic_activation"])
        ),
        "symbolic_max_rules": (
            int(cfg["symbolic_max_rules"]) if cfg.get("symbolic_max_rules") is not None else None
        ),
    }


def load_scoring_settings(file_manager: FileManager | None = None) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "eps": 0.02,
        "weights": {"rank_block": 0.60, "clf_block": 0.25, "time_block": 0.15},
        "rank_metrics": {
            "mrr": 0.40,
            "best_mrr": 0.30,
            "hits1": 0.15,
            "hits3": 0.10,
            "hits10": 0.05,
        },
        "clf_metrics": {"auc": 0.40, "pr_auc": 0.30, "precision": 0.15, "recall": 0.15},
        "time_metric_weight": 1.0,
        "time_scale": {
            "t_best": 60.0,
            "t_target": 240.0,
            "t_worst": 900.0,
            "score_at_best": 0.88,
            "score_at_target": 0.55,
            "score_at_worst": 0.20,
        },
    }
    cfg = _read_hpo_config(file_manager).get("scoring", {})
    if not isinstance(cfg, dict):
        return defaults
    return {
        "eps": float(cfg.get("eps", defaults["eps"])),
        "weights": {
            "rank_block": float(
                cfg.get("weights", {}).get("rank_block", defaults["weights"]["rank_block"])
            ),
            "clf_block": float(
                cfg.get("weights", {}).get("clf_block", defaults["weights"]["clf_block"])
            ),
            "time_block": float(
                cfg.get("weights", {}).get("time_block", defaults["weights"]["time_block"])
            ),
        },
        "rank_metrics": {
            "mrr": float(cfg.get("rank_metrics", {}).get("mrr", defaults["rank_metrics"]["mrr"])),
            "best_mrr": float(
                cfg.get("rank_metrics", {}).get("best_mrr", defaults["rank_metrics"]["best_mrr"])
            ),
            "hits1": float(
                cfg.get("rank_metrics", {}).get("hits1", defaults["rank_metrics"]["hits1"])
            ),
            "hits3": float(
                cfg.get("rank_metrics", {}).get("hits3", defaults["rank_metrics"]["hits3"])
            ),
            "hits10": float(
                cfg.get("rank_metrics", {}).get("hits10", defaults["rank_metrics"]["hits10"])
            ),
        },
        "clf_metrics": {
            "auc": float(cfg.get("clf_metrics", {}).get("auc", defaults["clf_metrics"]["auc"])),
            "pr_auc": float(
                cfg.get("clf_metrics", {}).get("pr_auc", defaults["clf_metrics"]["pr_auc"])
            ),
            "precision": float(
                cfg.get("clf_metrics", {}).get("precision", defaults["clf_metrics"]["precision"])
            ),
            "recall": float(
                cfg.get("clf_metrics", {}).get("recall", defaults["clf_metrics"]["recall"])
            ),
        },
        "time_metric_weight": float(cfg.get("time_metric_weight", defaults["time_metric_weight"])),
        "time_scale": cfg.get("time_scale", defaults["time_scale"]) or defaults["time_scale"],
    }


def load_scoring_weights(file_manager: FileManager | None = None) -> ScoreWeights:
    """Load scoring weights from config with safe fallbacks."""
    try:
        settings = load_scoring_settings(file_manager)
        return build_weights_from_settings(settings)
    except Exception as exc:
        logger.warning(f"Failed to load scoring weights; using defaults: {exc}")
        return build_weights_from_settings({})


def load_metric_bounds(file_manager: FileManager | None = None) -> dict[str, Any]:
    """Load metric normalization bounds from config/hpo/optimization.yaml (fallback to legacy)."""
    fm = file_manager or FileManager()
    default_bounds = {
        "kge": {"mrr": {"low": 0.15, "high": 0.75}},
        "rules": {
            "confidence": {"low": 0.4, "high": 0.95},
            "recall": {"low": 0.05, "high": 0.5},
            "coverage": {"low": 0.05, "high": 0.5},
            "relation_coverage": {"low": 0.05, "high": 0.60},
            "rules_per_relation": {"low": 1.0, "high": 80.0},
        },
        "learner": {
            "auc": {"low": 0.5, "high": 0.99},
            "pr_auc": {"low": 0.4, "high": 0.99},
            "precision": {"low": 0.3, "high": 0.95},
            "recall": {"low": 0.3, "high": 0.95},
        },
        "ensemble": {
            "ensemble_ece": {"low": 0.0, "high": 0.10, "invert": True},
            "ensemble_entropy": {"low": 0.0, "high": 0.70, "invert": True},
        },
    }
    try:
        cfg = _read_hpo_config(fm)
        if "metrics_bounds" in cfg and isinstance(cfg["metrics_bounds"], dict):
            return cfg["metrics_bounds"]
        return default_bounds
    except Exception:
        return default_bounds


def get_rules_coverage_weight(file_manager: FileManager | None = None) -> float:
    """Load rules coverage weight clamped to [0.15, 0.40] (legacy compatible)."""
    fm = file_manager or FileManager()
    cfg = _read_hpo_config(fm)
    try:
        balancing = cfg.get("balancing", {})
        rules_config = balancing.get("rules", {})
        raw_weight = float(rules_config.get("coverage_weight", 0.2))
        return max(0.15, min(0.40, raw_weight))
    except Exception:
        return 0.2


def get_rule_component_weights(
    file_manager: FileManager | None = None,
) -> tuple[float, float, float]:
    """Load rule component weights (confidence, recall, coverage) with normalization."""
    coverage_weight = get_rules_coverage_weight(file_manager)
    fm = file_manager or FileManager()
    cfg = _read_hpo_config(fm)
    try:
        rules_cfg = cfg.get("balancing", {}).get("rules", {})
        conf_raw = max(0.0, float(rules_cfg.get("confidence_weight", 0.5)))
        recall_raw = max(0.0, float(rules_cfg.get("recall_weight", 0.3)))
    except Exception:
        conf_raw, recall_raw = 0.5, 0.3

    remaining = max(0.0, 1.0 - coverage_weight)
    base_sum = conf_raw + recall_raw
    if base_sum <= 0:
        conf_weight = recall_weight = remaining * 0.5
    else:
        conf_weight = remaining * (conf_raw / base_sum)
        recall_weight = remaining * (recall_raw / base_sum)
    return conf_weight, recall_weight, coverage_weight
