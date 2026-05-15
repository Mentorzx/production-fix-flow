#!/usr/bin/env python3
"""Run a deterministic paired benchmark for SearchSpaceAdvisor policies."""

from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path
from statistics import mean, median
from typing import Any

import optuna
import orjson
from scipy.stats import wilcoxon

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
    "tpe_pure": {"advisor": False},
    "gp_bo": {"advisor": False, "sampler": "gp"},
    "advisor_full": {
        "advisor": True,
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


def _objective(params: dict[str, float], *, seed: int, trial_number: int) -> float:
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


def _apply_advisor(
    *,
    policy_name: str,
    policy: dict[str, Any],
    search_space: dict[str, dict[str, Any]],
    trials: list[dict[str, Any]],
    seed: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    advice = SearchSpaceAdvisor(config_thresholds={"persistent_cache_enabled": False}).advise(
        search_space=search_space,
        trials_data=trials,
        importances=IMPORTANCES,
        direction="maximize",
        study_name=f"paired_{policy_name}_{seed}_{len(trials)}",
        advisor_config=policy.get("advisor_config") or {},
        force_recompute=True,
        enable_bootstrap=bool(policy.get("enable_bootstrap", True)),
        enable_self_audit=bool(policy.get("enable_self_audit", True)),
    )
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


def _run_policy(
    *,
    policy_name: str,
    policy: dict[str, Any],
    seed: int,
    n_trials: int,
    advisor_period: int,
    min_advisor_trials: int,
) -> dict[str, Any]:
    if policy.get("sampler") == "gp" and hasattr(optuna.samplers, "GPSampler"):
        sampler = optuna.samplers.GPSampler(seed=seed, n_startup_trials=5)
        sampler_name = "GPSampler"
    else:
        sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=5)
        sampler_name = "TPESampler"
    study = optuna.create_study(direction="maximize", sampler=sampler)
    search_space = copy.deepcopy(SEARCH_SPACE)
    trials_data: list[dict[str, Any]] = []
    updates: list[dict[str, Any]] = []
    best_so_far: list[float] = []

    for trial_number in range(n_trials):
        if (
            policy.get("advisor")
            and trial_number >= min_advisor_trials
            and trial_number % advisor_period == 0
        ):
            search_space, update = _apply_advisor(
                policy_name=policy_name,
                policy=policy,
                search_space=search_space,
                trials=trials_data,
                seed=seed,
            )
            updates.append(update)

        trial = study.ask()
        params = {
            name: _suggest_param(trial, name, spec)
            for name, spec in search_space.items()
        }
        value = _objective(params, seed=seed, trial_number=trial_number)
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


def run_benchmark(
    *,
    seeds: list[int],
    n_trials: int,
    advisor_period: int,
    min_advisor_trials: int,
) -> dict[str, Any]:
    rows = [
        _run_policy(
            policy_name=policy_name,
            policy=policy,
            seed=seed,
            n_trials=n_trials,
            advisor_period=advisor_period,
            min_advisor_trials=min_advisor_trials,
        )
        for seed in seeds
        for policy_name, policy in POLICIES.items()
    ]
    baseline_by_seed = {
        int(row["seed"]): float(row["best_value"])
        for row in rows
        if row["policy"] == "tpe_pure"
    }
    gp_by_seed = {
        int(row["seed"]): float(row["best_value"])
        for row in rows
        if row["policy"] == "gp_bo"
    }
    policies = []
    for policy_name in POLICIES:
        policy_rows = [row for row in rows if row["policy"] == policy_name]
        best_values = [float(row["best_value"]) for row in policy_rows]
        deltas = [
            float(row["best_value"]) - baseline_by_seed[int(row["seed"])]
            for row in policy_rows
        ]
        gp_deltas = [
            float(row["best_value"]) - gp_by_seed[int(row["seed"])]
            for row in policy_rows
            if int(row["seed"]) in gp_by_seed
        ]
        policies.append(
            {
                "policy": policy_name,
                "sampler": str(policy_rows[0].get("sampler") if policy_rows else ""),
                "mean_best_value": round(mean(best_values), 6),
                "median_best_value": round(median(best_values), 6),
                "mean_delta_vs_tpe": round(mean(deltas), 6),
                "median_delta_vs_tpe": round(median(deltas), 6),
                "mean_delta_vs_gp_bo": round(mean(gp_deltas), 6) if gp_deltas else None,
                "median_delta_vs_gp_bo": round(median(gp_deltas), 6) if gp_deltas else None,
                "wins_vs_tpe": sum(1 for value in deltas if value > 0),
                "ties_vs_tpe": sum(1 for value in deltas if abs(value) <= 1e-12),
                "losses_vs_tpe": sum(1 for value in deltas if value < 0),
                "wilcoxon_greater_pvalue": (
                    None if policy_name == "tpe_pure" else _wilcoxon_pvalue(deltas)
                ),
            }
        )
    advisor_full = next(item for item in policies if item["policy"] == "advisor_full")
    return {
        "benchmark": "synthetic_paired_tpe_advisor",
        "n_trials": n_trials,
        "seeds": seeds,
        "advisor_period": advisor_period,
        "min_advisor_trials": min_advisor_trials,
        "policies": policies,
        "runs": rows,
        "universal_superiority_claim_supported": bool(
            advisor_full["wins_vs_tpe"] == len(seeds)
            and float(advisor_full["mean_delta_vs_tpe"]) > 0
            and advisor_full["mean_delta_vs_gp_bo"] is not None
            and float(advisor_full["mean_delta_vs_gp_bo"]) > 0
            and (
                advisor_full["wilcoxon_greater_pvalue"] is not None
                and float(advisor_full["wilcoxon_greater_pvalue"]) < 0.05
            )
        ),
    }


def _parse_seeds(raw: str) -> list[int]:
    seeds = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return seeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run paired TPE vs SearchSpaceAdvisor benchmark."
    )
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--seeds", type=_parse_seeds, default=_parse_seeds("11,17,23,29,31"))
    parser.add_argument("--advisor-period", type=int, default=10)
    parser.add_argument("--min-advisor-trials", type=int, default=10)
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
        n_trials=max(1, int(args.trials)),
        advisor_period=max(1, int(args.advisor_period)),
        min_advisor_trials=max(1, int(args.min_advisor_trials)),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    FileManager.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2), args.output)
    print(orjson.dumps(payload["policies"], option=orjson.OPT_INDENT_2).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
