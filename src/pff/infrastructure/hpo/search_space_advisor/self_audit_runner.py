"""Self-audit runner orchestration for Search Space Advisor."""

from __future__ import annotations

import math
from typing import Any, Callable

from .models import TrialSummary


def build_self_audit_summary(
    *,
    search_space: dict[str, Any],
    importances: dict[str, float],
    completed_trials: list[TrialSummary],
    direction: str,
    study_name: str,
    last_trial: int,
    min_prefix: int,
    min_suffix: int,
    max_prefixes: int,
    min_group_total: int,
    wilson_block: float,
    period_trials: int,
    min_match_points: int,
    min_trials_aggressive: int,
    min_trials_any: int,
    top_k_fraction: float,
    top_k_min: int,
    rust_spearman_min_len: int,
    advisor_factory: Callable[[dict[str, Any]], Any],
    audit_prefix_sizes_fn: Callable[..., list[int]],
    is_directional_action_fn: Callable[[str], bool],
    match_directional_suffix_trend_fn: Callable[..., tuple[bool | None, float | None]],
    wilson_lower_bound_fn: Callable[..., float],
) -> dict[str, Any]:
    """Run periodic directional self-audit over historical trial prefixes."""
    _ = importances
    prefix_sizes = audit_prefix_sizes_fn(
        len(completed_trials),
        min_prefix=min_prefix,
        min_suffix=min_suffix,
        period_trials=1,
        max_prefixes=max_prefixes,
    )
    if not prefix_sizes:
        return {
            "enabled": True,
            "ran": False,
            "reason": "insufficient_trials_for_self_audit",
            "period_trials": period_trials,
            "source_last_trial": last_trial,
        }

    audit_advisor = advisor_factory(
        {
            "min_trials_aggressive": min_trials_aggressive,
            "min_trials_any": min_trials_any,
            "top_k_fraction": top_k_fraction,
            "top_k_min": top_k_min,
            "persistent_cache_enabled": False,
            "rust_spearman_min_len": rust_spearman_min_len,
            "self_audit_period_trials": period_trials,
            "self_audit_min_prefix": min_prefix,
            "self_audit_min_suffix": min_suffix,
            "self_audit_max_prefixes": max_prefixes,
            "self_audit_min_group_total": min_group_total,
            "self_audit_wilson_block": wilson_block,
            "self_audit_min_match_points": min_match_points,
        }
    )

    directional_total = 0
    directional_hits = 0
    prefixes_evaluated = 0
    group_stats: dict[str, dict[str, Any]] = {}

    for prefix_size in prefix_sizes:
        prefix = completed_trials[:prefix_size]
        suffix = completed_trials[prefix_size:]
        if len(suffix) < min_suffix:
            continue
        prefixes_evaluated += 1
        prefix_payload = [
            {
                "id": trial.number,
                "number": trial.number,
                "value": trial.raw_value if trial.raw_value is not None else trial.value,
                "params": trial.params,
                "state": trial.state,
            }
            for trial in prefix
        ]
        prefix_advice = audit_advisor.advise(
            search_space=search_space,
            trials_data=prefix_payload,
            importances={},
            direction=direction,
            study_name=f"{study_name}__self_audit__{prefix_size}",
            force_recompute=True,
            enable_bootstrap=False,
            enable_self_audit=False,
            advisor_config={
                "enable_surrogate": False,
                "enable_interactions": False,
                "disable_internal_importances": True,
            },
        )
        recommendations = prefix_advice.get("recommendations", [])
        for recommendation in recommendations:
            if not isinstance(recommendation, dict):
                continue
            action = str(recommendation.get("action", "keep"))
            if not is_directional_action_fn(action):
                continue
            param_name = str(recommendation.get("param_name", ""))
            if not param_name:
                continue
            matched, rho = match_directional_suffix_trend_fn(
                action=action,
                param_name=param_name,
                suffix_trials=suffix,
                direction=direction,
                min_points=min_match_points,
            )
            if matched is None:
                continue
            directional_total += 1
            directional_hits += int(matched)
            key = f"{param_name}|{action}"
            group = group_stats.setdefault(
                key,
                {
                    "param_name": param_name,
                    "action": action,
                    "hits": 0,
                    "total": 0,
                    "rho_sum": 0.0,
                    "rho_count": 0,
                },
            )
            group["hits"] += int(matched)
            group["total"] += 1
            if rho is not None:
                group["rho_sum"] += float(rho)
                group["rho_count"] += 1

    breakdown: list[dict[str, Any]] = []
    effective_min_group_total = max(
        2,
        min(
            min_group_total,
            max(2, int(math.ceil(float(prefixes_evaluated) * 0.5))),
        ),
    )
    for item in group_stats.values():
        total = int(item.get("total", 0))
        if total <= 0:
            continue
        hits = int(item.get("hits", 0))
        rho_count = int(item.get("rho_count", 0))
        hit_rate = float(hits) / float(total)
        entry = {
            "param_name": item["param_name"],
            "action": item["action"],
            "total": total,
            "hit_rate": round(hit_rate, 4),
            "hit_rate_wilson_lb": round(wilson_lower_bound_fn(successes=hits, total=total), 4),
            "mean_suffix_spearman": round(float(item["rho_sum"]) / float(rho_count), 4)
            if rho_count > 0
            else None,
        }
        breakdown.append(entry)
    breakdown.sort(key=lambda rec: (rec["hit_rate_wilson_lb"], rec["hit_rate"], -rec["total"]))

    villains = [
        entry
        for entry in breakdown
        if int(entry["total"]) >= effective_min_group_total
        and float(entry["hit_rate"]) < 0.5
        and float(entry["hit_rate_wilson_lb"]) < wilson_block
    ]

    param_stats: dict[str, dict[str, Any]] = {}
    action_stats: dict[str, dict[str, Any]] = {}
    for item in group_stats.values():
        total = int(item.get("total", 0))
        if total <= 0:
            continue
        hits = int(item.get("hits", 0))
        param_name = str(item.get("param_name", ""))
        action = str(item.get("action", ""))
        if not param_name or not action:
            continue
        p_group = param_stats.setdefault(param_name, {"hits": 0, "total": 0})
        p_group["hits"] += hits
        p_group["total"] += total
        a_group = action_stats.setdefault(action, {"hits": 0, "total": 0})
        a_group["hits"] += hits
        a_group["total"] += total

    param_diagnostics: list[dict[str, Any]] = []
    for param_name, stats in param_stats.items():
        total = int(stats["total"])
        hits = int(stats["hits"])
        if total <= 0:
            continue
        hit_rate = float(hits) / float(total)
        param_diagnostics.append(
            {
                "param_name": param_name,
                "total": total,
                "hit_rate": round(hit_rate, 4),
                "hit_rate_wilson_lb": round(wilson_lower_bound_fn(successes=hits, total=total), 4),
            }
        )
    param_diagnostics.sort(
        key=lambda row: (row["hit_rate_wilson_lb"], row["hit_rate"], -row["total"])
    )

    action_diagnostics: list[dict[str, Any]] = []
    for action, stats in action_stats.items():
        total = int(stats["total"])
        hits = int(stats["hits"])
        if total <= 0:
            continue
        hit_rate = float(hits) / float(total)
        action_diagnostics.append(
            {
                "action": action,
                "total": total,
                "hit_rate": round(hit_rate, 4),
                "hit_rate_wilson_lb": round(wilson_lower_bound_fn(successes=hits, total=total), 4),
            }
        )
    action_diagnostics.sort(
        key=lambda row: (row["hit_rate_wilson_lb"], row["hit_rate"], -row["total"])
    )

    return {
        "enabled": True,
        "ran": True,
        "period_trials": period_trials,
        "prefixes_evaluated": prefixes_evaluated,
        "directional_signals_total": directional_total,
        "directional_hit_rate": round(float(directional_hits) / float(directional_total), 4)
        if directional_total > 0
        else None,
        "directional_hit_rate_wilson_lb": round(
            wilson_lower_bound_fn(successes=directional_hits, total=directional_total),
            4,
        )
        if directional_total > 0
        else None,
        "effective_min_group_total": int(effective_min_group_total),
        "villains_count": len(villains),
        "villains": villains,
        "diagnostics": {
            "params": param_diagnostics,
            "actions": action_diagnostics,
            "worst_params": [item["param_name"] for item in param_diagnostics[:3]],
            "worst_actions": [item["action"] for item in action_diagnostics[:3]],
        },
        "source_last_trial": last_trial,
    }


def resolve_self_audit_summary(
    *,
    search_space: dict[str, Any],
    importances: dict[str, float],
    completed_trials: list[TrialSummary],
    direction: str,
    study_name: str,
    last_trial: int,
    force_recompute: bool,
    min_prefix: int,
    min_suffix: int,
    period_trials: int,
    get_cached_snapshot_fn: Callable[[str], dict[str, Any] | None],
    set_cached_snapshot_fn: Callable[[str, dict[str, Any]], None],
    build_summary_fn: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    """Resolve self-audit from cache or recompute when periodic trigger is due."""
    study_key = study_name or "__default__"
    n_completed = len(completed_trials)
    min_trials_to_audit = min_prefix + min_suffix
    if n_completed < min_trials_to_audit:
        return {
            "enabled": True,
            "ran": False,
            "reason": "insufficient_trials_for_self_audit",
            "period_trials": period_trials,
            "source_last_trial": last_trial,
        }

    cached = get_cached_snapshot_fn(study_key)
    already_current = isinstance(cached, dict) and int(cached.get("source_last_trial", -10)) == int(
        last_trial
    )
    periodic_due = n_completed % period_trials == 0
    should_run = force_recompute or cached is None or (periodic_due and not already_current)

    if not should_run and cached is not None:
        reused = dict(cached)
        reused["ran"] = False
        reused["reused"] = True
        return reused

    audit = build_summary_fn(
        search_space=search_space,
        importances=importances,
        completed_trials=completed_trials,
        direction=direction,
        study_name=study_name or "__default__",
        last_trial=last_trial,
    )
    set_cached_snapshot_fn(study_key, dict(audit))
    return audit


__all__ = [
    "build_self_audit_summary",
    "resolve_self_audit_summary",
]
