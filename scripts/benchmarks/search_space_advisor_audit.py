#!/usr/bin/env python3
"""Audit SearchSpaceAdvisor consistency and reliability on real dashboard trial history."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import math
from statistics import mean
import time
from typing import Any

import orjson
from scipy.stats import spearmanr

from pff.infrastructure.hpo.search_space_advisor import SearchSpaceAdvisor
from pff.shared.core.file_manager import FileManager


@dataclass(frozen=True)
class AuditConfig:
    input_path: Path
    output_path: Path
    min_prefix: int


def _normalize_direction(value: Any) -> str:
    raw = str(value or "maximize").strip().lower()
    if "." in raw:
        raw = raw.split(".")[-1]
    if raw in {"maximize", "max"}:
        return "maximize"
    if raw in {"minimize", "min"}:
        return "minimize"
    return "maximize"


def _apply_direction(value: float, direction: str) -> float:
    return value if _normalize_direction(direction) == "maximize" else -value


def _load_dashboard_payload(path: Path) -> dict[str, Any]:
    payload = FileManager.read(path, return_native=True)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid dashboard payload at {path}")
    return payload


def _completed_trials(payload: dict[str, Any]) -> list[dict[str, Any]]:
    trials = payload.get("trials", [])
    if not isinstance(trials, list):
        return []
    completed = []
    for trial in trials:
        if not isinstance(trial, dict):
            continue
        state = str(trial.get("state", "")).upper()
        if "." in state:
            state = state.split(".")[-1]
        if state != "COMPLETE":
            continue
        if trial.get("value") is None:
            continue
        completed.append(trial)
    return sorted(completed, key=lambda item: int(item.get("number", item.get("id", 0))))


def _detect_inconsistencies(recommendations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for rec in recommendations:
        param = str(rec.get("param_name", "unknown"))
        action = str(rec.get("action", "keep"))
        recommendation = rec.get("recommendation", {}) or {}
        validation = rec.get("validation", {}) or {}
        if isinstance(validation, dict) and validation.get("passed") is False:
            issues.append(
                {
                    "param": param,
                    "kind": "validation_failed",
                    "action": action,
                    "reason": validation.get("blocked_reason"),
                }
            )
        if action == "reduce_categories":
            keep = set(map(str, recommendation.get("keep", [])))
            remove = set(map(str, recommendation.get("remove", [])))
            if keep & remove:
                issues.append(
                    {
                        "param": param,
                        "kind": "category_overlap",
                        "overlap": sorted(keep & remove),
                    }
                )
    return issues


def _action_matches_suffix_trend(
    *,
    action: str,
    param_name: str,
    suffix_trials: list[dict[str, Any]],
    direction: str,
) -> tuple[bool | None, float | None]:
    values: list[float] = []
    scores: list[float] = []
    for trial in suffix_trials:
        params = trial.get("params", {})
        if not isinstance(params, dict):
            continue
        value = params.get(param_name)
        score = trial.get("value")
        if isinstance(value, (int, float)) and isinstance(score, (int, float)):
            values.append(float(value))
            scores.append(_apply_direction(float(score), direction))
    if len(values) < 6:
        return None, None
    if len(set(values)) < 2 or len(set(scores)) < 2:
        return None, None
    rho, _ = spearmanr(values, scores)
    if rho != rho:
        return None, None
    if action == "expand_upper":
        return bool(rho > 0), float(rho)
    if action == "expand_lower":
        return bool(rho < 0), float(rho)
    return None, float(rho)


def _run_prefix_backtest(
    *,
    advisor: SearchSpaceAdvisor,
    search_space: dict[str, Any],
    importances: dict[str, float],
    direction: str,
    completed_trials: list[dict[str, Any]],
    min_prefix: int,
) -> dict[str, Any]:
    def _wilson_lower_bound(successes: int, total: int, z: float = 1.96) -> float:
        if total <= 0:
            return 0.0
        phat = float(successes) / float(total)
        z2 = z * z
        denom = 1.0 + (z2 / float(total))
        center = phat + (z2 / (2.0 * float(total)))
        margin = z * math.sqrt((phat * (1.0 - phat) / float(total)) + (z2 / (4.0 * (total**2))))
        lower = (center - margin) / denom
        return max(0.0, min(1.0, lower))

    total_prefixes = 0
    directional_hits = 0
    directional_total = 0
    confidence_buckets: dict[str, list[int]] = defaultdict(list)
    directional_groups: dict[str, dict[str, Any]] = {}
    rho_values: list[float] = []

    for prefix_size in range(min_prefix, max(min_prefix, len(completed_trials) - 5)):
        prefix_trials = completed_trials[:prefix_size]
        suffix_trials = completed_trials[prefix_size:]
        if len(suffix_trials) < 5:
            continue
        total_prefixes += 1
        advice = advisor.advise(
            search_space=search_space,
            trials_data=prefix_trials,
            importances=importances,
            direction=direction,
            study_name=f"audit_prefix_{prefix_size}",
        )
        recs = advice.get("recommendations", [])
        for rec in recs:
            if not isinstance(rec, dict):
                continue
            action = str(rec.get("action", "keep"))
            if action not in {"expand_upper", "expand_lower"}:
                continue
            param_name = str(rec.get("param_name", ""))
            matched, rho = _action_matches_suffix_trend(
                action=action,
                param_name=param_name,
                suffix_trials=suffix_trials,
                direction=direction,
            )
            if rho is not None:
                rho_values.append(rho)
            if matched is None:
                continue
            directional_total += 1
            directional_hits += int(matched)
            confidence = str(rec.get("confidence", "low"))
            confidence_buckets[confidence].append(int(matched))
            group_key = f"{param_name}|{action}"
            group = directional_groups.setdefault(
                group_key,
                {
                    "param_name": param_name,
                    "action": action,
                    "hits": 0,
                    "total": 0,
                    "rho_values": [],
                    "confidence_histogram": {},
                },
            )
            group["total"] += 1
            group["hits"] += int(matched)
            if rho is not None:
                group["rho_values"].append(float(rho))
            hist = group["confidence_histogram"]
            hist[confidence] = int(hist.get(confidence, 0)) + 1

    confidence_success = {
        bucket: round(sum(values) / len(values), 4)
        for bucket, values in confidence_buckets.items()
        if values
    }
    directional_breakdown = []
    for group in directional_groups.values():
        total = int(group["total"])
        if total <= 0:
            continue
        rhos = group.get("rho_values", [])
        directional_breakdown.append(
            {
                "param_name": group["param_name"],
                "action": group["action"],
                "total": total,
                "hit_rate": round(float(group["hits"]) / float(total), 4),
                "hit_rate_wilson_lb": round(
                    _wilson_lower_bound(int(group["hits"]), int(total)),
                    4,
                ),
                "mean_suffix_spearman": round(sum(rhos) / len(rhos), 4) if rhos else None,
                "confidence_histogram": group["confidence_histogram"],
            }
        )
    directional_breakdown.sort(key=lambda item: (item["hit_rate"], -item["total"]))
    effective_min_group_total = min(5, max(2, total_prefixes))
    villains = [
        item
        for item in directional_breakdown
        if int(item["total"]) >= effective_min_group_total
        and float(item.get("hit_rate", 1.0)) < 0.5
        and float(item.get("hit_rate_wilson_lb", 0.0)) < 0.35
    ]
    return {
        "prefixes_evaluated": total_prefixes,
        "directional_signals_total": directional_total,
        "directional_signals_hit_rate": round(directional_hits / directional_total, 4)
        if directional_total
        else None,
        "confidence_success_rate": confidence_success,
        "mean_suffix_spearman": round(mean(rho_values), 4) if rho_values else None,
        "directional_breakdown": directional_breakdown,
        "effective_min_group_total": effective_min_group_total,
        "villains": villains,
    }


def _action_counts(recommendations: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for rec in recommendations:
        action = str(rec.get("action", "keep"))
        counts[action] = int(counts.get(action, 0)) + 1
    return dict(sorted(counts.items()))


def _mean_confidence_score(recommendations: list[dict[str, Any]]) -> float | None:
    scores = [
        float(rec["confidence_score"])
        for rec in recommendations
        if isinstance(rec.get("confidence_score"), (int, float))
    ]
    if not scores:
        return None
    return round(sum(scores) / len(scores), 4)


def _summarize_advice_variant(
    *,
    name: str,
    elapsed_ms: float,
    advice: dict[str, Any],
) -> dict[str, Any]:
    recommendations = [
        rec for rec in advice.get("recommendations", []) if isinstance(rec, dict)
    ]
    metadata = advice.get("metadata", {}) if isinstance(advice.get("metadata"), dict) else {}
    reliability = metadata.get("reliability_summary", {})
    self_audit = metadata.get("self_audit", {})
    acceleration = metadata.get("acceleration", {})
    return {
        "name": name,
        "elapsed_ms": round(float(elapsed_ms), 2),
        "metadata_compute_time_ms": round(float(metadata.get("compute_time_ms") or 0.0), 2),
        "n_recommendations": len(recommendations),
        "action_counts": _action_counts(recommendations),
        "mean_confidence_score": _mean_confidence_score(recommendations),
        "validation_pass_wilson_lb": reliability.get("validation_pass_wilson_lb"),
        "directional_hit_rate_wilson_lb": self_audit.get("directional_hit_rate_wilson_lb"),
        "blocked_actions_current": self_audit.get("blocked_actions_current"),
        "surrogate_enabled": acceleration.get("surrogate_enabled"),
        "interactions_enabled": acceleration.get("interactions_enabled"),
        "internal_importances_disabled": acceleration.get("internal_importances_disabled"),
    }


def _run_ablation_matrix(
    *,
    search_space: dict[str, Any],
    importances: dict[str, float],
    direction: str,
    completed_trials: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    variants = [
        (
            "full_no_adaptive",
            {"adaptive_perf_enabled": False},
            True,
            True,
        ),
        (
            "no_surrogate",
            {
                "adaptive_perf_enabled": False,
                "enable_surrogate": False,
                "enable_interactions": False,
            },
            True,
            True,
        ),
        (
            "no_interactions",
            {
                "adaptive_perf_enabled": False,
                "enable_interactions": False,
            },
            True,
            True,
        ),
        (
            "no_internal_importances",
            {
                "adaptive_perf_enabled": False,
                "disable_internal_importances": True,
            },
            True,
            True,
        ),
        (
            "no_bootstrap",
            {"adaptive_perf_enabled": False},
            False,
            True,
        ),
        (
            "no_self_audit",
            {"adaptive_perf_enabled": False},
            True,
            False,
        ),
    ]
    rows: list[dict[str, Any]] = []
    for name, advisor_config, enable_bootstrap, enable_self_audit in variants:
        t0 = time.monotonic()
        advice = SearchSpaceAdvisor().advise(
            search_space=search_space,
            trials_data=completed_trials,
            importances=importances,
            direction=direction,
            study_name=f"audit_ablation_{name}",
            advisor_config=advisor_config,
            force_recompute=True,
            enable_bootstrap=enable_bootstrap,
            enable_self_audit=enable_self_audit,
        )
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        rows.append(
            _summarize_advice_variant(
                name=name,
                elapsed_ms=elapsed_ms,
                advice=advice,
            )
        )
    return rows


def run_audit(cfg: AuditConfig) -> dict[str, Any]:
    payload = _load_dashboard_payload(cfg.input_path)
    direction_raw = payload.get("direction", "maximize")
    direction = _normalize_direction(direction_raw)
    search_space = payload.get("searchSpace", {})
    importances = payload.get("importances", {})
    completed = _completed_trials(payload)

    advisor = SearchSpaceAdvisor()
    advice_main = advisor.advise(
        search_space=search_space,
        trials_data=completed,
        importances=importances,
        direction=direction_raw,
        study_name="audit_main_raw_direction",
    )
    advice_lower = advisor.advise(
        search_space=search_space,
        trials_data=completed,
        importances=importances,
        direction=direction,
        study_name="audit_main_normalized_direction",
    )
    recs_main = advice_main.get("recommendations", [])
    recs_lower = advice_lower.get("recommendations", [])
    action_mismatch = sum(
        1
        for left, right in zip(recs_main, recs_lower, strict=False)
        if isinstance(left, dict)
        and isinstance(right, dict)
        and left.get("param_name") == right.get("param_name")
        and left.get("action") != right.get("action")
    )

    inconsistency_report = _detect_inconsistencies([r for r in recs_main if isinstance(r, dict)])
    backtest = _run_prefix_backtest(
        advisor=SearchSpaceAdvisor(),
        search_space=search_space if isinstance(search_space, dict) else {},
        importances=importances if isinstance(importances, dict) else {},
        direction=direction,
        completed_trials=completed,
        min_prefix=cfg.min_prefix,
    )
    ablations = _run_ablation_matrix(
        search_space=search_space if isinstance(search_space, dict) else {},
        importances=importances if isinstance(importances, dict) else {},
        direction=direction,
        completed_trials=completed,
    )
    summary = {
        "n_completed_trials": len(completed),
        "direction_input": str(direction_raw),
        "direction_normalized": direction,
        "direction_case_action_mismatch": action_mismatch,
        "n_recommendations": len(recs_main) if isinstance(recs_main, list) else 0,
        "n_detected_inconsistencies": len(inconsistency_report),
        "inconsistencies": inconsistency_report,
        "backtest": backtest,
        "ablations": ablations,
        "metadata": advice_main.get("metadata", {}),
    }
    return summary


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit SearchSpaceAdvisor on dashboard trial history."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/.cache/hpo/dashboard_data.json"),
        help="Path to dashboard_data.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/benches/search_space_advisor/audit_latest.json"),
        help="JSON output report path.",
    )
    parser.add_argument(
        "--min-prefix",
        type=int,
        default=8,
        help="Minimum prefix size for rolling backtest.",
    )
    return parser


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()
    cfg = AuditConfig(
        input_path=args.input,
        output_path=args.output,
        min_prefix=max(3, int(args.min_prefix)),
    )
    summary = run_audit(cfg)
    payload = orjson.dumps(summary, option=orjson.OPT_INDENT_2)
    FileManager.write_bytes(payload, cfg.output_path)
    print(payload.decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
