#!/usr/bin/env python3
"""Run a deterministic paired benchmark for SearchSpaceAdvisor policies."""

from __future__ import annotations

import argparse
import copy
import math
import multiprocessing as mp
import random
from pathlib import Path
from statistics import mean, median
from typing import Any

import optuna
import orjson
import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon

from pff.infrastructure.hpo.search_space_advisor import SearchSpaceAdvisor
from pff.infrastructure.hpo.search_space_advisor.patching import generate_search_space_patch
from pff.shared.core.file_manager import FileManager


SEARCH_SPACE: dict[str, dict[str, Any]] = {
    "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
    "weight_decay": {"type": "float", "low": 1e-6, "high": 1e-2, "log": True},
    "dropout": {"type": "float", "low": 0.0, "high": 0.55},
    "embedding_dim": {"type": "int", "low": 64, "high": 512},
    "lambda_logic": {"type": "float", "low": 0.0, "high": 0.18},
    "lambda_pc": {"type": "float", "low": 0.0, "high": 0.12},
    "temperature": {"type": "float", "low": 0.35, "high": 1.8},
}

IMPORTANCES = {
    "learning_rate": 0.26,
    "weight_decay": 0.09,
    "dropout": 0.14,
    "embedding_dim": 0.18,
    "lambda_logic": 0.13,
    "lambda_pc": 0.08,
    "temperature": 0.12,
}

POLICIES: dict[str, dict[str, Any]] = {
    "random": {"advisor": False, "sampler": "random"},
    "tpe_pure": {"advisor": False},
    "gp_bo": {"advisor": False, "sampler": "gp"},
    "advisor_full": {
        "advisor": True,
        "advisor_config": {"adaptive_perf_enabled": False},
        "enable_bootstrap": True,
        "enable_self_audit": True,
    },
    "advisor_gp_portfolio": {
        "advisor": True,
        "sampler": "gp",
        "advisor_config": {"adaptive_perf_enabled": False},
        "enable_bootstrap": True,
        "enable_self_audit": True,
    },
    "advisor_static_gp": {
        "advisor": True,
        "sampler": "gp",
        "advisor_mode": "static_restart",
        "advisor_config": {"adaptive_perf_enabled": False},
        "enable_bootstrap": True,
        "enable_self_audit": True,
    },
    "advisor_static_gp_guarded": {
        "advisor": True,
        "sampler": "gp",
        "advisor_mode": "static_restart",
        "static_min_width_fraction": 0.5,
        "advisor_config": {"adaptive_perf_enabled": False},
        "enable_bootstrap": True,
        "enable_self_audit": True,
    },
    "advisor_trust_region_gp": {
        "advisor": True,
        "sampler": "gp",
        "advisor_mode": "trust_region_restart",
        "trust_region_min_width_fraction": 0.35,
        "trust_region_top_fraction": 0.4,
        "advisor_config": {"adaptive_perf_enabled": False},
        "enable_bootstrap": True,
        "enable_self_audit": True,
    },
    "advisor_no_surrogate": {
        "advisor": True,
        "advisor_config": {
            "adaptive_perf_enabled": False,
            "enable_surrogate": False,
            "enable_interactions": False,
        },
        "enable_bootstrap": True,
        "enable_self_audit": True,
    },
    "advisor_no_bootstrap": {
        "advisor": True,
        "advisor_config": {"adaptive_perf_enabled": False},
        "enable_bootstrap": False,
        "enable_self_audit": True,
    },
    "advisor_no_self_audit": {
        "advisor": True,
        "advisor_config": {"adaptive_perf_enabled": False},
        "enable_bootstrap": True,
        "enable_self_audit": False,
    },
}


SCENARIOS = {
    "smooth_kgc",
    "narrow_ridge",
    "conditional_regularized",
    "edge_capacity",
}


def _objective(
    params: dict[str, float],
    *,
    seed: int,
    trial_number: int,
    scenario: str,
) -> float:
    if scenario == "narrow_ridge":
        return _objective_narrow_ridge(params, seed=seed, trial_number=trial_number)
    if scenario == "conditional_regularized":
        return _objective_conditional_regularized(params, seed=seed, trial_number=trial_number)
    if scenario == "edge_capacity":
        return _objective_edge_capacity(params, seed=seed, trial_number=trial_number)
    return _objective_smooth_kgc(params, seed=seed, trial_number=trial_number)


def _objective_smooth_kgc(params: dict[str, float], *, seed: int, trial_number: int) -> float:
    lr_term = math.exp(-((math.log10(params["learning_rate"]) - math.log10(8e-4)) ** 2) / 0.22)
    wd_term = math.exp(-((math.log10(params["weight_decay"]) - math.log10(4e-5)) ** 2) / 0.55)
    dropout_term = math.exp(-((params["dropout"] - 0.14) ** 2) / 0.035)
    dim_term = math.exp(-((params["embedding_dim"] - 320.0) ** 2) / (2.0 * 120.0**2))
    logic_term = math.exp(-((params["lambda_logic"] - 0.075) ** 2) / 0.0024)
    pc_term = math.exp(-((params["lambda_pc"] - 0.035) ** 2) / 0.0016)
    temp_term = math.exp(-((params["temperature"] - 0.92) ** 2) / 0.09)
    interaction = 0.035 if params["learning_rate"] < 0.0015 and params["dropout"] < 0.24 else -0.015
    deterministic_noise = 0.01 * math.sin((seed + 1) * 0.37 + trial_number * 1.91)
    score = (
        0.25 * lr_term
        + 0.08 * wd_term
        + 0.12 * dropout_term
        + 0.16 * dim_term
        + 0.12 * logic_term
        + 0.07 * pc_term
        + 0.12 * temp_term
        + interaction
        + deterministic_noise
    )
    return max(0.0, min(1.0, score))


def _objective_narrow_ridge(params: dict[str, float], *, seed: int, trial_number: int) -> float:
    lr = math.log10(params["learning_rate"])
    wd = math.log10(params["weight_decay"])
    ridge = math.exp(-((lr - math.log10(6.5e-4)) ** 2) / 0.08)
    ridge *= math.exp(-((params["dropout"] - 0.12) ** 2) / 0.012)
    ridge *= math.exp(-((params["temperature"] - 0.88) ** 2) / 0.025)
    dim_term = math.exp(-((params["embedding_dim"] - 336.0) ** 2) / (2.0 * 70.0**2))
    logic_term = math.exp(-((params["lambda_logic"] - 0.07) ** 2) / 0.0012)
    pc_term = math.exp(-((params["lambda_pc"] - 0.032) ** 2) / 0.0008)
    wd_term = math.exp(-((wd - math.log10(5e-5)) ** 2) / 0.24)
    deterministic_noise = 0.006 * math.sin((seed + 3) * 0.41 + trial_number * 2.13)
    score = 0.34 * ridge + 0.16 * dim_term + 0.14 * logic_term + 0.12 * pc_term + 0.1 * wd_term
    return max(0.0, min(1.0, score + deterministic_noise))


def _objective_conditional_regularized(
    params: dict[str, float],
    *,
    seed: int,
    trial_number: int,
) -> float:
    lr_term = math.exp(-((math.log10(params["learning_rate"]) - math.log10(1.1e-3)) ** 2) / 0.18)
    dropout_center = 0.08 if params["lambda_logic"] > 0.08 else 0.2
    dropout_term = math.exp(-((params["dropout"] - dropout_center) ** 2) / 0.018)
    logic_term = math.exp(-((params["lambda_logic"] - 0.095) ** 2) / 0.0018)
    pc_term = math.exp(-((params["lambda_pc"] - 0.045) ** 2) / 0.0011)
    temp_term = math.exp(-((params["temperature"] - 1.05) ** 2) / 0.08)
    dim_term = math.exp(-((params["embedding_dim"] - 288.0) ** 2) / (2.0 * 110.0**2))
    penalty = 0.08 if params["lambda_logic"] > 0.11 and params["dropout"] > 0.18 else 0.0
    deterministic_noise = 0.008 * math.sin((seed + 5) * 0.29 + trial_number * 1.73)
    score = (
        0.22 * lr_term
        + 0.18 * dropout_term
        + 0.17 * logic_term
        + 0.12 * pc_term
        + 0.11 * temp_term
        + 0.12 * dim_term
        - penalty
        + deterministic_noise
    )
    return max(0.0, min(1.0, score))


def _objective_edge_capacity(params: dict[str, float], *, seed: int, trial_number: int) -> float:
    lr_term = math.exp(-((math.log10(params["learning_rate"]) - math.log10(3.5e-3)) ** 2) / 0.16)
    wd_term = math.exp(-((math.log10(params["weight_decay"]) - math.log10(2e-5)) ** 2) / 0.38)
    dim_term = math.exp(-((params["embedding_dim"] - 500.0) ** 2) / (2.0 * 90.0**2))
    dropout_term = math.exp(-((params["dropout"] - 0.18) ** 2) / 0.03)
    logic_term = math.exp(-((params["lambda_logic"] - 0.055) ** 2) / 0.002)
    pc_term = math.exp(-((params["lambda_pc"] - 0.025) ** 2) / 0.0013)
    temp_term = math.exp(-((params["temperature"] - 0.75) ** 2) / 0.06)
    deterministic_noise = 0.007 * math.sin((seed + 7) * 0.53 + trial_number * 1.37)
    score = (
        0.23 * lr_term
        + 0.08 * wd_term
        + 0.21 * dim_term
        + 0.12 * dropout_term
        + 0.11 * logic_term
        + 0.08 * pc_term
        + 0.1 * temp_term
        + deterministic_noise
    )
    return max(0.0, min(1.0, score))


def _suggest_param(trial: optuna.Trial, name: str, spec: dict[str, Any]) -> float:
    param_type = str(spec.get("type", "float"))
    if param_type == "fixed":
        return float(spec["value"])
    low = float(spec["low"])
    high = float(spec["high"])
    if param_type == "int":
        return float(trial.suggest_int(name, int(round(low)), int(round(high))))
    return float(trial.suggest_float(name, low, high, log=bool(spec.get("log")) and low > 0))


def _sanitize_space(space: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    sanitized: dict[str, dict[str, Any]] = {}
    for name, spec in space.items():
        base = SEARCH_SPACE[name]
        if str(spec.get("type")) == "fixed":
            value = min(max(float(spec["value"]), float(base["low"])), float(base["high"]))
            sanitized[name] = {**base, "low": value, "high": value}
            continue
        low = max(float(spec.get("low", base["low"])), float(base["low"]))
        high = min(float(spec.get("high", base["high"])), float(base["high"]))
        if high <= low:
            center = min(max((low + high) / 2.0, float(base["low"])), float(base["high"]))
            low, high = center, center
        sanitized[name] = {**base, "low": low, "high": high}
    return sanitized


def _relax_numeric_space(
    space: dict[str, dict[str, Any]],
    *,
    min_width_fraction: float,
) -> dict[str, dict[str, Any]]:
    relaxed = copy.deepcopy(space)
    for name, spec in relaxed.items():
        if str(spec.get("type")) not in {"float", "int"}:
            continue
        base = SEARCH_SPACE[name]
        base_low = float(base["low"])
        base_high = float(base["high"])
        low = float(spec.get("low", base_low))
        high = float(spec.get("high", base_high))
        if high <= low:
            continue
        if bool(spec.get("log")) and base_low > 0 and low > 0:
            base_low_t = math.log10(base_low)
            base_high_t = math.log10(base_high)
            low_t = math.log10(low)
            high_t = math.log10(high)
            min_width = (base_high_t - base_low_t) * min_width_fraction
            width = high_t - low_t
            if width >= min_width:
                continue
            center = (low_t + high_t) / 2.0
            next_low = max(base_low_t, center - min_width / 2.0)
            next_high = min(base_high_t, center + min_width / 2.0)
            spec["low"] = 10**next_low
            spec["high"] = 10**next_high
            continue
        min_width = (base_high - base_low) * min_width_fraction
        width = high - low
        if width >= min_width:
            continue
        center = (low + high) / 2.0
        spec["low"] = max(base_low, center - min_width / 2.0)
        spec["high"] = min(base_high, center + min_width / 2.0)
        if str(spec.get("type")) == "int":
            spec["low"] = int(round(float(spec["low"])))
            spec["high"] = int(round(float(spec["high"])))
    return relaxed


def _quantile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("at least one value is required")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = max(0.0, min(1.0, q)) * (len(ordered) - 1)
    low_idx = int(math.floor(pos))
    high_idx = int(math.ceil(pos))
    if low_idx == high_idx:
        return ordered[low_idx]
    weight = pos - low_idx
    return ordered[low_idx] * (1.0 - weight) + ordered[high_idx] * weight


def _trust_region_space(
    trials_data: list[dict[str, Any]],
    *,
    top_fraction: float,
    min_width_fraction: float,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    completed = [
        trial
        for trial in trials_data
        if isinstance(trial.get("value"), (int, float)) and isinstance(trial.get("params"), dict)
    ]
    ranked = sorted(completed, key=lambda trial: float(trial["value"]), reverse=True)
    top_k = ranked[: max(3, int(math.ceil(len(ranked) * max(0.05, top_fraction))))]
    if not top_k:
        return copy.deepcopy(SEARCH_SPACE), {"patch_params": [], "reason": "no_completed_trials"}

    next_space = copy.deepcopy(SEARCH_SPACE)
    patch_params: list[str] = []
    for name, base in SEARCH_SPACE.items():
        values = [
            float(trial["params"][name])
            for trial in top_k
            if name in trial["params"] and trial["params"][name] is not None
        ]
        if not values:
            continue
        base_low = float(base["low"])
        base_high = float(base["high"])
        if bool(base.get("log")) and base_low > 0 and all(value > 0 for value in values):
            transformed = [math.log10(value) for value in values]
            q10 = _quantile(transformed, 0.1)
            q90 = _quantile(transformed, 0.9)
            domain_low = math.log10(base_low)
            domain_high = math.log10(base_high)
        else:
            transformed = values
            q10 = _quantile(transformed, 0.1)
            q90 = _quantile(transformed, 0.9)
            domain_low = base_low
            domain_high = base_high
        center = _quantile(transformed, 0.5)
        min_width = (domain_high - domain_low) * min(1.0, max(0.0, min_width_fraction))
        width = max(q90 - q10, min_width)
        low = max(domain_low, center - width / 2.0)
        high = min(domain_high, center + width / 2.0)
        if low >= high:
            continue
        if bool(base.get("log")) and base_low > 0 and all(value > 0 for value in values):
            low = 10**low
            high = 10**high
        spec = {**base, "low": low, "high": high}
        if str(base.get("type")) == "int":
            spec["low"] = max(int(base["low"]), int(math.floor(low)))
            spec["high"] = min(int(base["high"]), int(math.ceil(high)))
            if int(spec["low"]) >= int(spec["high"]):
                continue
        if float(spec["low"]) > float(base["low"]) or float(spec["high"]) < float(base["high"]):
            next_space[name] = spec
            patch_params.append(name)
    return next_space, {
        "patch_params": sorted(patch_params),
        "top_k": len(top_k),
        "top_fraction": top_fraction,
        "min_width_fraction": min_width_fraction,
    }


def _advisor_worker(queue: Any, payload: dict[str, Any]) -> None:
    try:
        advice = SearchSpaceAdvisor(config_thresholds={"persistent_cache_enabled": False}).advise(
            **payload
        )
    except Exception as exc:  # pragma: no cover - exercised through parent failure path.
        queue.put({"ok": False, "error": repr(exc)})
        return
    queue.put({"ok": True, "advice": advice})


def _run_advisor_isolated(payload: dict[str, Any]) -> dict[str, Any]:
    start_method = "forkserver" if "forkserver" in mp.get_all_start_methods() else "spawn"
    context = mp.get_context(start_method)
    queue = context.Queue(maxsize=1)
    process = context.Process(target=_advisor_worker, args=(queue, payload))
    process.start()
    process.join(timeout=300)
    if process.is_alive():
        process.terminate()
        process.join(timeout=10)
        raise TimeoutError("SearchSpaceAdvisor isolated benchmark worker timed out")
    if process.exitcode != 0:
        raise RuntimeError(f"SearchSpaceAdvisor isolated worker failed: exitcode={process.exitcode}")
    result = queue.get(timeout=10)
    if not result.get("ok"):
        raise RuntimeError(f"SearchSpaceAdvisor isolated worker failed: {result.get('error')}")
    advice = result.get("advice")
    if not isinstance(advice, dict):
        raise TypeError("SearchSpaceAdvisor isolated worker returned a non-dict advice payload")
    return advice


def _run_advisor_in_process(payload: dict[str, Any]) -> dict[str, Any]:
    return SearchSpaceAdvisor(config_thresholds={"persistent_cache_enabled": False}).advise(**payload)


def _apply_advisor(
    *,
    policy_name: str,
    policy: dict[str, Any],
    search_space: dict[str, dict[str, Any]],
    trials: list[dict[str, Any]],
    seed: int,
    isolate_advisor: bool,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    payload = {
        "search_space": search_space,
        "trials_data": trials,
        "importances": IMPORTANCES,
        "direction": "maximize",
        "study_name": f"paired_{policy_name}_{seed}_{len(trials)}",
        "advisor_config": policy.get("advisor_config") or {},
        "force_recompute": True,
        "enable_bootstrap": bool(policy.get("enable_bootstrap", True)),
        "enable_self_audit": bool(policy.get("enable_self_audit", True)),
    }
    py_random_state = random.getstate()
    np_random_state = np.random.get_state()
    try:
        runner = _run_advisor_isolated if isolate_advisor else _run_advisor_in_process
        advice = runner(payload)
    finally:
        random.setstate(py_random_state)
        np.random.set_state(np_random_state)
    recommendations = [
        rec for rec in advice.get("recommendations", []) if isinstance(rec, dict)
    ]
    patch = generate_search_space_patch(recommendations, current_config=search_space)
    candidate = copy.deepcopy(search_space)
    candidate.update({name: value for name, value in patch.items() if name in candidate})
    return _sanitize_space(candidate), {
        "trial": len(trials),
        "n_recommendations": len(recommendations),
        "action_counts": _action_counts(recommendations),
        "patch_params": sorted(patch),
        "metadata": advice.get("metadata", {}),
    }


def _action_counts(recommendations: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for rec in recommendations:
        action = str(rec.get("action", "keep"))
        counts[action] = int(counts.get(action, 0)) + 1
    return dict(sorted(counts.items()))


def _build_sampler(policy: dict[str, Any], *, seed: int) -> tuple[optuna.samplers.BaseSampler, str]:
    if policy.get("sampler") == "random":
        return optuna.samplers.RandomSampler(seed=seed), "RandomSampler"
    if policy.get("sampler") == "gp" and hasattr(optuna.samplers, "GPSampler"):
        return optuna.samplers.GPSampler(seed=seed, n_startup_trials=5), "GPSampler"
    return optuna.samplers.TPESampler(seed=seed, n_startup_trials=5), "TPESampler"


def _distribution_for(name: str, spec: dict[str, Any]) -> optuna.distributions.BaseDistribution:
    param_type = str(spec.get("type", "float"))
    base = SEARCH_SPACE[name]
    low = float(spec.get("low", base["low"]))
    high = float(spec.get("high", base["high"]))
    if param_type == "int":
        return optuna.distributions.IntDistribution(int(round(low)), int(round(high)))
    return optuna.distributions.FloatDistribution(
        low,
        high,
        log=bool(spec.get("log")) and low > 0,
    )


def _trial_fits_space(
    params: dict[str, Any],
    space: dict[str, dict[str, Any]],
) -> bool:
    for name, spec in space.items():
        if name not in params:
            return False
        value = float(params[name])
        low = float(spec.get("low", SEARCH_SPACE[name]["low"]))
        high = float(spec.get("high", SEARCH_SPACE[name]["high"]))
        if value < low or value > high:
            return False
    return True


def _add_warmup_trials(
    study: optuna.Study,
    *,
    trials_data: list[dict[str, Any]],
    space: dict[str, dict[str, Any]],
) -> int:
    distributions = {name: _distribution_for(name, spec) for name, spec in space.items()}
    added = 0
    for trial in trials_data:
        params = trial.get("params", {})
        value = trial.get("value")
        if not isinstance(params, dict) or not isinstance(value, (int, float)):
            continue
        if not _trial_fits_space(params, space):
            continue
        try:
            study.add_trial(
                optuna.trial.create_trial(
                    params={name: params[name] for name in space},
                    distributions=distributions,
                    value=float(value),
                )
            )
        except ValueError:
            continue
        added += 1
    return added


def _run_policy(
    *,
    policy_name: str,
    policy: dict[str, Any],
    seed: int,
    scenario: str,
    n_trials: int,
    advisor_period: int,
    min_advisor_trials: int,
    isolate_advisor: bool,
) -> dict[str, Any]:
    sampler, sampler_name = _build_sampler(policy, seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    search_space = copy.deepcopy(SEARCH_SPACE)
    trials_data: list[dict[str, Any]] = []
    updates: list[dict[str, Any]] = []
    best_so_far: list[float] = []

    for trial_number in range(n_trials):
        if (
            policy.get("advisor")
            and policy.get("advisor_mode") == "static_restart"
            and trial_number == min_advisor_trials
        ):
            previous_search_space = search_space
            search_space, update = _apply_advisor(
                policy_name=policy_name,
                policy=policy,
                search_space=search_space,
                trials=trials_data,
                seed=seed,
                isolate_advisor=isolate_advisor,
            )
            min_width_fraction = float(policy.get("static_min_width_fraction", 0.0) or 0.0)
            if min_width_fraction > 0.0:
                search_space = _relax_numeric_space(
                    search_space,
                    min_width_fraction=min(1.0, min_width_fraction),
                )
                update["static_min_width_fraction"] = min_width_fraction
            updates.append(update)
            if not update["patch_params"]:
                search_space = previous_search_space
                update["static_restart_skipped"] = "empty_patch"
                continue
            sampler, post_sampler_name = _build_sampler(policy, seed=seed + 10_000)
            sampler_name = f"{sampler_name}+{post_sampler_name}"
            study = optuna.create_study(direction="maximize", sampler=sampler)
            update["warmup_trials_reused"] = _add_warmup_trials(
                study,
                trials_data=trials_data,
                space=search_space,
            )

        if (
            policy.get("advisor")
            and policy.get("advisor_mode") == "trust_region_restart"
            and trial_number == min_advisor_trials
        ):
            search_space, update = _trust_region_space(
                trials_data,
                top_fraction=float(policy.get("trust_region_top_fraction", 0.4)),
                min_width_fraction=float(policy.get("trust_region_min_width_fraction", 0.35)),
            )
            update["trial"] = len(trials_data)
            update["mode"] = "trust_region_restart"
            updates.append(update)
            if not update["patch_params"]:
                update["static_restart_skipped"] = "empty_patch"
                continue
            sampler, post_sampler_name = _build_sampler(policy, seed=seed + 20_000)
            sampler_name = f"{sampler_name}+trust_region+{post_sampler_name}"
            study = optuna.create_study(direction="maximize", sampler=sampler)
            update["warmup_trials_reused"] = _add_warmup_trials(
                study,
                trials_data=trials_data,
                space=search_space,
            )

        if (
            policy.get("advisor")
            and policy.get("advisor_mode") not in {"static_restart", "trust_region_restart"}
            and trial_number >= min_advisor_trials
            and trial_number % advisor_period == 0
        ):
            search_space, update = _apply_advisor(
                policy_name=policy_name,
                policy=policy,
                search_space=search_space,
                trials=trials_data,
                seed=seed,
                isolate_advisor=isolate_advisor,
            )
            updates.append(update)

        trial = study.ask()
        params = {
            name: _suggest_param(trial, name, spec)
            for name, spec in search_space.items()
        }
        value = _objective(
            params,
            seed=seed,
            trial_number=trial_number,
            scenario=scenario,
        )
        study.tell(trial, value)
        trials_data.append(
            {
                "id": trial_number,
                "number": trial_number,
                "state": "COMPLETE",
                "value": value,
                "params": params,
            }
        )
        best_so_far.append(max(best_so_far[-1], value) if best_so_far else value)

    return {
        "policy": policy_name,
        "sampler": sampler_name,
        "seed": seed,
        "scenario": scenario,
        "n_trials": n_trials,
        "best_value": round(best_so_far[-1], 6),
        "best_curve": [round(value, 6) for value in best_so_far],
        "advisor_updates": updates,
        "final_search_space": search_space,
    }


def _wilcoxon_pvalue(values: list[float]) -> float | None:
    if len(values) < 2 or all(abs(value) < 1e-12 for value in values):
        return None
    try:
        return float(wilcoxon(values, alternative="greater").pvalue)
    except ValueError:
        return None


def _bootstrap_mean_ci(values: list[float], *, samples: int = 500, seed: int = 90210) -> list[float] | None:
    if not values:
        return None
    if len(values) == 1:
        return [round(values[0], 6), round(values[0], 6)]
    rng = random.Random(seed)
    means = []
    for _ in range(samples):
        draw = [values[rng.randrange(len(values))] for _ in values]
        means.append(mean(draw))
    means.sort()
    low_idx = int(0.025 * (samples - 1))
    high_idx = int(0.975 * (samples - 1))
    return [round(means[low_idx], 6), round(means[high_idx], 6)]


def _friedman_pvalue(rows: list[dict[str, Any]], policy_names: list[str]) -> float | None:
    blocks = sorted({(str(row["scenario"]), int(row["seed"])) for row in rows})
    if len(blocks) < 2 or len(policy_names) < 3:
        return None
    series: list[list[float]] = []
    for policy_name in policy_names:
        values = []
        for scenario, seed in blocks:
            match = next(
                (
                    row
                    for row in rows
                    if row["policy"] == policy_name
                    and row["scenario"] == scenario
                    and int(row["seed"]) == seed
                ),
                None,
            )
            if match is None:
                return None
            values.append(float(match["best_value"]))
        series.append(values)
    try:
        return float(friedmanchisquare(*series).pvalue)
    except ValueError:
        return None


def run_benchmark(
    *,
    seeds: list[int],
    scenarios: list[str],
    n_trials: int,
    advisor_period: int,
    min_advisor_trials: int,
    isolate_advisor: bool = True,
) -> dict[str, Any]:
    rows = [
        _run_policy(
            policy_name=policy_name,
            policy=policy,
            seed=seed,
            scenario=scenario,
            n_trials=n_trials,
            advisor_period=advisor_period,
            min_advisor_trials=min_advisor_trials,
            isolate_advisor=isolate_advisor,
        )
        for scenario in scenarios
        for seed in seeds
        for policy_name, policy in POLICIES.items()
    ]
    baseline_by_seed = {
        (str(row["scenario"]), int(row["seed"])): float(row["best_value"])
        for row in rows
        if row["policy"] == "tpe_pure"
    }
    gp_by_seed = {
        (str(row["scenario"]), int(row["seed"])): float(row["best_value"])
        for row in rows
        if row["policy"] == "gp_bo"
    }
    policies = []
    for policy_name in POLICIES:
        policy_rows = [row for row in rows if row["policy"] == policy_name]
        best_values = [float(row["best_value"]) for row in policy_rows]
        deltas = [
            float(row["best_value"]) - baseline_by_seed[(str(row["scenario"]), int(row["seed"]))]
            for row in policy_rows
        ]
        gp_deltas = [
            float(row["best_value"]) - gp_by_seed[(str(row["scenario"]), int(row["seed"]))]
            for row in policy_rows
            if (str(row["scenario"]), int(row["seed"])) in gp_by_seed
        ]
        policies.append(
            {
                "policy": policy_name,
                "sampler": str(policy_rows[0].get("sampler") if policy_rows else ""),
                "mean_best_value": round(mean(best_values), 6),
                "mean_best_value_ci95": _bootstrap_mean_ci(best_values),
                "median_best_value": round(median(best_values), 6),
                "mean_delta_vs_tpe": round(mean(deltas), 6),
                "mean_delta_vs_tpe_ci95": _bootstrap_mean_ci(deltas),
                "median_delta_vs_tpe": round(median(deltas), 6),
                "mean_delta_vs_gp_bo": round(mean(gp_deltas), 6) if gp_deltas else None,
                "mean_delta_vs_gp_bo_ci95": _bootstrap_mean_ci(gp_deltas) if gp_deltas else None,
                "median_delta_vs_gp_bo": round(median(gp_deltas), 6) if gp_deltas else None,
                "wins_vs_tpe": sum(1 for value in deltas if value > 0),
                "ties_vs_tpe": sum(1 for value in deltas if abs(value) <= 1e-12),
                "losses_vs_tpe": sum(1 for value in deltas if value < 0),
                "wins_vs_gp_bo": sum(1 for value in gp_deltas if value > 0),
                "ties_vs_gp_bo": sum(1 for value in gp_deltas if abs(value) <= 1e-12),
                "losses_vs_gp_bo": sum(1 for value in gp_deltas if value < 0),
                "wilcoxon_greater_pvalue": (
                    None if policy_name == "tpe_pure" else _wilcoxon_pvalue(deltas)
                ),
                "wilcoxon_greater_vs_gp_bo_pvalue": (
                    None if policy_name == "gp_bo" else _wilcoxon_pvalue(gp_deltas)
                ),
            }
        )
    scenario_summaries = []
    for scenario in scenarios:
        scenario_rows = [row for row in rows if row["scenario"] == scenario]
        scenario_policies = []
        for policy_name in POLICIES:
            policy_rows = [row for row in scenario_rows if row["policy"] == policy_name]
            best_values = [float(row["best_value"]) for row in policy_rows]
            gp_deltas = [
                float(row["best_value"]) - gp_by_seed[(scenario, int(row["seed"]))]
                for row in policy_rows
            ]
            scenario_policies.append(
                {
                    "policy": policy_name,
                    "mean_best_value": round(mean(best_values), 6),
                    "median_best_value": round(median(best_values), 6),
                    "mean_delta_vs_gp_bo": round(mean(gp_deltas), 6),
                    "wins_vs_gp_bo": sum(1 for value in gp_deltas if value > 0),
                }
            )
        scenario_summaries.append({"scenario": scenario, "policies": scenario_policies})
    advisor_candidates = [
        item for item in policies if str(item["policy"]).startswith("advisor_")
    ]
    best_advisor = max(
        advisor_candidates,
        key=lambda item: float(item["mean_best_value"]),
    )
    return {
        "benchmark": "synthetic_paired_tpe_advisor",
        "n_trials": n_trials,
        "seeds": seeds,
        "scenarios": scenarios,
        "advisor_period": advisor_period,
        "min_advisor_trials": min_advisor_trials,
        "claim_candidate_policy": best_advisor["policy"],
        "friedman_pvalue": _friedman_pvalue(rows, list(POLICIES)),
        "policies": policies,
        "scenario_summaries": scenario_summaries,
        "runs": rows,
        "universal_superiority_claim_supported": bool(
            best_advisor["wins_vs_tpe"] == len(seeds) * len(scenarios)
            and float(best_advisor["mean_delta_vs_tpe"]) > 0
            and best_advisor["mean_delta_vs_gp_bo"] is not None
            and float(best_advisor["mean_delta_vs_gp_bo"]) > 0
            and (
                best_advisor["wilcoxon_greater_pvalue"] is not None
                and float(best_advisor["wilcoxon_greater_pvalue"]) < 0.05
            )
        ),
    }


def _parse_seeds(raw: str) -> list[int]:
    seeds = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return seeds


def _parse_scenarios(raw: str) -> list[str]:
    scenarios = [part.strip() for part in raw.split(",") if part.strip()]
    invalid = sorted(set(scenarios) - SCENARIOS)
    if invalid:
        raise argparse.ArgumentTypeError(f"unknown scenarios: {', '.join(invalid)}")
    if not scenarios:
        raise argparse.ArgumentTypeError("at least one scenario is required")
    return scenarios


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run paired TPE vs SearchSpaceAdvisor benchmark."
    )
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--seeds", type=_parse_seeds, default=_parse_seeds("11,17,23,29,31"))
    parser.add_argument(
        "--scenarios",
        type=_parse_scenarios,
        default=_parse_scenarios("smooth_kgc"),
    )
    parser.add_argument("--advisor-period", type=int, default=10)
    parser.add_argument("--min-advisor-trials", type=int, default=10)
    parser.add_argument(
        "--no-isolated-advisor",
        action="store_true",
        help="Run Advisor in-process instead of an isolated child process.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/benches/search_space_advisor/paired_benchmark_50.json"),
    )
    return parser.parse_args()


def main() -> int:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    args = parse_args()
    payload = run_benchmark(
        seeds=list(args.seeds),
        scenarios=list(args.scenarios),
        n_trials=max(1, int(args.trials)),
        advisor_period=max(1, int(args.advisor_period)),
        min_advisor_trials=max(1, int(args.min_advisor_trials)),
        isolate_advisor=not bool(args.no_isolated_advisor),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    FileManager.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2), args.output)
    print(orjson.dumps(payload["policies"], option=orjson.OPT_INDENT_2).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
