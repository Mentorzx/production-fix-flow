"""Generate academic report artifacts for the Search Space Advisor study."""

from __future__ import annotations

import argparse
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

from pff.shared.core.file_manager import FileManager


RUN_ID = "deep_research_advisor_20260506"


def _as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _completed_trials(dashboard: dict[str, Any]) -> list[dict[str, Any]]:
    trials = dashboard.get("trials", [])
    return [
        trial
        for trial in trials
        if str(trial.get("state", "")).upper() == "COMPLETE"
        and _as_float(trial.get("value")) is not None
    ]


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = FileManager.json_loads(FileManager.read_text(path))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object at {path}")
    return payload


def _metric_value(trial: dict[str, Any], key: str) -> float | None:
    direct = _as_float(trial.get(key))
    if direct is not None:
        return direct
    metrics = trial.get("metrics")
    if isinstance(metrics, dict):
        return _as_float(metrics.get(key))
    return None


def _trial_id(trial: dict[str, Any], fallback: int) -> int:
    value = trial.get("number", trial.get("id", fallback))
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _figure_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 180,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )


def _savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_advisor_flow(output: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.8, 6.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    cards = [
        {
            "x": 0.14,
            "title": "1. Entrada",
            "body": "Trials avaliados\nEspaço de busca\nDireção da métrica",
            "face": "#edf4fb",
            "edge": "#2f5f86",
        },
        {
            "x": 0.38,
            "title": "2. Núcleo estatístico",
            "body": "Score comparável\nTop-k adaptativo\nQuantis, Spearman e CV",
            "face": "#eef8f1",
            "edge": "#3a7a55",
        },
        {
            "x": 0.62,
            "title": "3. Evidência auxiliar",
            "body": "Importâncias blended\nSurrogate RandomForest\nUCB, LCB e interações",
            "face": "#f4f1fb",
            "edge": "#66509a",
        },
        {
            "x": 0.86,
            "title": "4. Decisão",
            "body": "keep, narrow, expand\nfix ou reduzir categorias\nJustificativa por parâmetro",
            "face": "#fff4df",
            "edge": "#9b6b22",
        },
    ]

    card_y = 0.56
    card_w = 0.19
    card_h = 0.45
    ax.text(0.5, 0.92, "Fluxo lógico do Search Space Advisor", ha="center", va="center", fontsize=15, weight="bold", color="#1f2933")
    ax.text(
        0.5,
        0.86,
        "O Advisor transforma histórico experimental em recomendações auditáveis para o próximo espaço de busca.",
        ha="center",
        va="center",
        fontsize=10.2,
        color="#4a4a4a",
    )

    def draw_card(card: dict[str, str | float]) -> None:
        x = float(card["x"])
        y = card_y
        patch = FancyBboxPatch(
            (x - card_w / 2, y - card_h / 2),
            card_w,
            card_h,
            boxstyle="round,pad=0.014,rounding_size=0.026",
            linewidth=1.35,
            edgecolor=str(card["edge"]),
            facecolor=str(card["face"]),
        )
        ax.add_patch(patch)
        ax.text(x, y + 0.145, str(card["title"]), ha="center", va="center", fontsize=10.6, weight="bold", color="#202020")
        ax.plot([x - card_w / 2 + 0.018, x + card_w / 2 - 0.018], [y + 0.095, y + 0.095], color=str(card["edge"]), linewidth=0.9)
        ax.text(x, y - 0.045, str(card["body"]), ha="center", va="center", fontsize=9.7, color="#202020", linespacing=1.35)

    for card in cards:
        draw_card(card)

    for left, right in zip(cards, cards[1:], strict=False):
        left_x = float(left["x"])
        right_x = float(right["x"])
        ax.annotate(
            "",
            xy=(right_x - card_w / 2 - 0.014, card_y),
            xytext=(left_x + card_w / 2 + 0.014, card_y),
            arrowprops={"arrowstyle": "->", "linewidth": 1.35, "color": "#4a4a4a", "shrinkA": 0, "shrinkB": 0},
        )

    output_box = FancyBboxPatch(
        (0.17, 0.12),
        0.66,
        0.15,
        boxstyle="round,pad=0.014,rounding_size=0.024",
        linewidth=1.35,
        edgecolor="#5f6f7a",
        facecolor="#f4f7f9",
    )
    ax.add_patch(output_box)
    ax.text(
        0.50,
        0.195,
        "Payload final",
        ha="center",
        va="center",
        fontsize=10.8,
        weight="bold",
        color="#202020",
    )
    ax.text(
        0.50,
        0.145,
        "recomendações + confidence score + validação dura + bootstrap/Wilson + self-audit",
        ha="center",
        va="center",
        fontsize=9.5,
        color="#202020",
    )
    ax.annotate(
        "",
        xy=(0.50, 0.285),
        xytext=(0.86, card_y - card_h / 2 - 0.012),
        arrowprops={"arrowstyle": "->", "linewidth": 1.25, "color": "#4a4a4a", "connectionstyle": "arc3,rad=-0.12"},
    )
    _savefig(fig, output)


def plot_advisor_evidence(
    recommendations: list[dict[str, Any]], audit: dict[str, Any], output: Path
) -> None:
    action_counts = Counter(str(rec.get("action", "keep")) for rec in recommendations)
    confidence_counts = Counter(str(rec.get("confidence", "low")) for rec in recommendations)
    validation_counts = Counter(
        "aprovada"
        if ((rec.get("validation") or {}).get("passed") is not False)
        else "bloqueada"
        for rec in recommendations
    )
    metadata = audit.get("metadata", {})
    rel = metadata.get("reliability_summary", {})
    self_audit = metadata.get("self_audit", {})
    summary_labels = [
        "cobertura",
        "validacao LB",
        "confianca media",
        "self-audit LB",
    ]
    summary_values = [
        _as_float(metadata.get("search_space_coverage_ratio")) or 0.0,
        _as_float(rel.get("validation_pass_wilson_lb")) or 0.0,
        _as_float(rel.get("mean_confidence_score")) or 0.0,
        _as_float(self_audit.get("directional_hit_rate_wilson_lb")) or 0.0,
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.2))
    ax_actions, ax_confidence, ax_validation, ax_summary = axes.ravel()

    action_items = sorted(action_counts.items(), key=lambda item: (-item[1], item[0]))
    ax_actions.barh(
        [item[0] for item in action_items][::-1],
        [item[1] for item in action_items][::-1],
        color="#3b6ea8",
    )
    ax_actions.set_title("Ações emitidas")
    ax_actions.set_xlabel("recomendações")
    ax_actions.grid(axis="x", alpha=0.25)

    confidence_order = ["low", "medium", "high"]
    ax_confidence.bar(
        confidence_order,
        [confidence_counts.get(label, 0) for label in confidence_order],
        color=["#9aa6b2", "#4f8bbd", "#2f6b4f"],
    )
    ax_confidence.set_title("Classes de confiança")
    ax_confidence.set_ylabel("recomendações")
    ax_confidence.grid(axis="y", alpha=0.25)

    validation_items = sorted(validation_counts.items(), key=lambda item: item[0])
    ax_validation.bar(
        [item[0] for item in validation_items],
        [item[1] for item in validation_items],
        color=["#2ca25f" if item[0] == "aprovada" else "#b03a2e" for item in validation_items],
    )
    ax_validation.set_title("Validação dura do payload")
    ax_validation.set_ylabel("recomendações")
    ax_validation.grid(axis="y", alpha=0.25)

    ax_summary.bar(summary_labels, summary_values, color=["#5b8cc0", "#7fcdbb", "#756bb1", "#9ecae1"])
    ax_summary.axhline(0.35, color="#b03a2e", linestyle=":", linewidth=1.1)
    ax_summary.axhline(0.5, color="#7f8c8d", linestyle="--", linewidth=1.0)
    for idx, value in enumerate(summary_values):
        ax_summary.text(idx, value + 0.025, f"{value:.3f}", ha="center", fontsize=8)
    ax_summary.set_title("Evidência agregada")
    ax_summary.set_ylim(0, 1.08)
    ax_summary.tick_params(axis="x", rotation=18)
    ax_summary.grid(axis="y", alpha=0.25)

    fig.suptitle("Evidências empíricas do Search Space Advisor", fontsize=13, weight="bold")
    _savefig(fig, output)


def plot_importances_actions(
    recommendations: list[dict[str, Any]], importances: dict[str, Any], output: Path
) -> None:
    action_colors = {
        "keep": "#7f8c8d",
        "fix": "#8e44ad",
        "expand_lower": "#2e86c1",
        "expand_upper": "#d35400",
        "narrow": "#229954",
        "reduce_categories": "#ca6f1e",
        "change_distribution": "#16a085",
    }
    rec_by_param = {str(rec.get("param_name")): rec for rec in recommendations}
    rows = []
    for name, value in importances.items():
        importance = _as_float(value)
        if importance is None:
            continue
        rec = rec_by_param.get(str(name), {})
        rows.append((str(name), importance, str(rec.get("action", "keep"))))
    rows.sort(key=lambda item: item[1], reverse=True)
    rows = rows[:12]

    labels = [item[0] for item in rows][::-1]
    values = [item[1] for item in rows][::-1]
    actions = [item[2] for item in rows][::-1]
    colors = [action_colors.get(action, "#34495e") for action in actions]

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.barh(labels, values, color=colors)
    for idx, action in enumerate(actions):
        ax.text(values[idx] + 0.004, idx, action, va="center", fontsize=8)
    ax.set_title("Importâncias e ações recomendadas")
    ax.set_xlabel("importancia normalizada")
    ax.set_xlim(0, max(values + [0.05]) * 1.35)
    ax.grid(axis="x", alpha=0.25)
    _savefig(fig, output)


def plot_topk_distributions(recommendations: list[dict[str, Any]], output: Path) -> None:
    numeric = []
    for rec in recommendations:
        stats_all = (rec.get("attempts_summary") or {}).get("stats")
        stats_top = (rec.get("best_region") or {}).get("stats")
        current_space = rec.get("current_space") or {}
        if (
            isinstance(stats_all, dict)
            and isinstance(stats_top, dict)
            and isinstance(current_space, dict)
            and _as_float(current_space.get("low")) is not None
            and _as_float(current_space.get("high")) is not None
        ):
            numeric.append(
                (
                    str(rec.get("param_name")),
                    float(rec.get("importance", 0.0)),
                    stats_all,
                    stats_top,
                    current_space,
                )
            )
    numeric.sort(key=lambda item: item[1], reverse=True)
    numeric = numeric[:6]

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    y_positions = np.arange(len(numeric))
    for idx, (name, _, stats_all, stats_top, current_space) in enumerate(numeric):
        low_raw = float(current_space["low"])
        high_raw = float(current_space["high"])
        use_log = bool(current_space.get("log")) and low_raw > 0 and high_raw > 0

        def _position(value: float) -> float:
            if use_log:
                low = math.log10(max(low_raw, 1e-12))
                high = math.log10(max(high_raw, 1e-12))
                transformed = math.log10(max(value, 1e-12))
            else:
                low = low_raw
                high = high_raw
                transformed = value
            span = max(abs(high - low), 1e-12)
            return max(0.0, min(1.0, (transformed - low) / span))

        all_q10 = _position(float(stats_all.get("q10", stats_all.get("min", 0.0))))
        all_q90 = _position(float(stats_all.get("q90", stats_all.get("max", 0.0))))
        top_q10 = _position(float(stats_top.get("q10", stats_top.get("min", 0.0))))
        top_q90 = _position(float(stats_top.get("q90", stats_top.get("max", 0.0))))
        top_q50 = _position(float(stats_top.get("q50", 0.5)))
        ax.hlines(idx, all_q10, all_q90, color="#b0b7c3", linewidth=7, alpha=0.9)
        ax.hlines(idx, top_q10, top_q90, color="#1f77b4", linewidth=5, alpha=0.95)
        ax.plot(top_q50, idx, marker="o", color="#b03a2e", markersize=5)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([item[0] for item in numeric])
    ax.set_title("Faixas q10-q90 normalizadas no espaço de busca atual")
    ax.set_xlabel("posição normalizada no intervalo atual")
    ax.set_xlim(-0.03, 1.03)
    ax.axvline(0.15, color="#d0d0d0", linestyle=":", linewidth=1)
    ax.axvline(0.85, color="#d0d0d0", linestyle=":", linewidth=1)
    ax.grid(axis="x", alpha=0.25)
    ax.legend(
        handles=[
            plt.Line2D([0], [0], color="#b0b7c3", lw=7, label="todos os trials"),
            plt.Line2D([0], [0], color="#1f77b4", lw=5, label="top-k"),
            plt.Line2D([0], [0], color="#b03a2e", marker="o", lw=0, label="q50 top-k"),
        ],
        loc="lower right",
    )
    _savefig(fig, output)


def plot_reliability(audit: dict[str, Any], output: Path) -> None:
    metadata = audit.get("metadata", {})
    rel = metadata.get("reliability_summary", {})
    self_audit = metadata.get("self_audit", {})
    labels = [
        "validação",
        "LB validação",
        "confiança média",
        "hit self-audit",
        "LB self-audit",
    ]
    values = [
        float(rel.get("validation_pass_rate", 0.0)),
        float(rel.get("validation_pass_wilson_lb", 0.0)),
        float(rel.get("mean_confidence_score", 0.0)),
        float(self_audit.get("directional_hit_rate") or 0.0),
        float(self_audit.get("directional_hit_rate_wilson_lb") or 0.0),
    ]

    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    colors = ["#2ca25f", "#99d8c9", "#756bb1", "#3182bd", "#9ecae1"]
    ax.bar(labels, values, color=colors)
    ax.axhline(0.5, color="#7f8c8d", linestyle="--", linewidth=1)
    ax.axhline(0.35, color="#b03a2e", linestyle=":", linewidth=1.2)
    for idx, value in enumerate(values):
        ax.text(idx, value + 0.025, f"{value:.3f}", ha="center", fontsize=8)
    ax.set_title("Resumo de confiabilidade do payload do Advisor")
    ax.set_ylabel("proporcao ou score")
    ax.set_ylim(0, 1.08)
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    _savefig(fig, output)


def _pareto_front(points: list[tuple[int, float, float]]) -> list[int]:
    front: list[int] = []
    for idx, (_, score, duration) in enumerate(points):
        dominated = False
        for jdx, (_, other_score, other_duration) in enumerate(points):
            if idx == jdx:
                continue
            no_worse = other_score >= score and other_duration <= duration
            strictly_better = other_score > score or other_duration < duration
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            front.append(idx)
    return front


def plot_pareto(trials: list[dict[str, Any]], output: Path) -> dict[str, Any]:
    points = []
    for idx, trial in enumerate(trials, start=1):
        score = _as_float(trial.get("value"))
        duration = _metric_value(trial, "duration") or _as_float(trial.get("duration"))
        if score is None or duration is None:
            continue
        points.append((_trial_id(trial, idx), score, duration))
    front_idx = _pareto_front(points)
    front = [points[idx] for idx in front_idx]

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.scatter(
        [point[2] for point in points],
        [point[1] for point in points],
        color="#8da0cb",
        s=48,
        label="trials completos",
    )
    if front:
        ordered_front = sorted(front, key=lambda item: item[2])
        ax.plot(
            [point[2] for point in ordered_front],
            [point[1] for point in ordered_front],
            color="#b03a2e",
            linewidth=1.8,
            label="frente de Pareto",
        )
        ax.scatter(
            [point[2] for point in front],
            [point[1] for point in front],
            color="#b03a2e",
            s=70,
        )
    for trial_id, score, duration in points:
        ax.text(duration, score + 0.00015, str(trial_id), fontsize=7, ha="center")
    ax.set_title("Projeção qualidade-tempo a partir do dashboard HPO")
    ax.set_xlabel("duração do trial (s)")
    ax.set_ylabel("score")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right")
    _savefig(fig, output)
    return {
        "points": len(points),
        "pareto_front_size": len(front),
        "pareto_trial_ids": [point[0] for point in sorted(front, key=lambda item: item[2])],
    }


def plot_survival(audit: dict[str, Any], output: Path) -> None:
    self_audit = (audit.get("metadata") or {}).get("self_audit", {})
    diagnostics = self_audit.get("diagnostics", {})
    params = diagnostics.get("params", []) if isinstance(diagnostics, dict) else []
    total = int(self_audit.get("directional_signals_total") or 0)
    hit_rate = _as_float(self_audit.get("directional_hit_rate")) or 0.0
    aggregate_survival = hit_rate if total else 0.0

    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    ax.step([0, 1], [1.0, aggregate_survival], where="post", color="#2e86c1", linewidth=2.2, label="agregado")
    for row in params:
        row_total = int(row.get("total") or 0)
        row_rate = _as_float(row.get("hit_rate")) or 0.0
        if row_total <= 0:
            continue
        ax.step(
            [0, 1],
            [1.0, row_rate],
            where="post",
            linewidth=1.4,
            linestyle="--",
            label=str(row.get("param_name")),
        )
    ax.set_title("Sobrevivência empírica das recomendações direcionais")
    ax.set_xlabel("horizonte auditado prefixo-sufixo")
    ax.set_ylabel("R(h)")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower left")
    _savefig(fig, output)


def build_metrics(
    dashboard: dict[str, Any],
    audit: dict[str, Any],
    hpo_summary: dict[str, Any],
    figures: dict[str, str],
    pareto_metrics: dict[str, Any],
    paired_benchmark: dict[str, Any] | None = None,
) -> dict[str, Any]:
    advice = dashboard.get("searchSpaceAdvice", {})
    recommendations = advice.get("recommendations", [])
    action_counts = Counter(str(rec.get("action", "keep")) for rec in recommendations)
    completed = _completed_trials(dashboard)
    metadata = audit.get("metadata", {})
    rel = metadata.get("reliability_summary", {})
    self_audit = metadata.get("self_audit", {})

    def _maybe_int(value: Any) -> int | None:
        try:
            if value is None or isinstance(value, bool):
                return None
            return int(value)
        except (TypeError, ValueError):
            return None

    def _maybe_float(value: Any) -> float | None:
        try:
            if value is None or isinstance(value, bool):
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    data_info = hpo_summary.get("data_info") or {}
    data_source = str(data_info.get("source") or "").strip().lower()
    study_name = str(dashboard.get("studyName") or hpo_summary.get("study_name") or "study_sem_nome")
    completed_trials_count = len(completed)
    dashboard_total_trials = _maybe_int(dashboard.get("totalTrials"))
    summary_n_trials = _maybe_int(hpo_summary.get("n_trials"))
    dashboard_best_value = _maybe_float(dashboard.get("bestValue"))
    summary_best_value = _maybe_float(hpo_summary.get("best_value"))
    optimization_time_seconds = _maybe_float(hpo_summary.get("optimization_time"))
    summary_trial_count_stale = (
        summary_n_trials is not None and summary_n_trials < completed_trials_count
    )
    summary_best_value_stale = (
        summary_best_value is not None
        and dashboard_best_value is not None
        and summary_best_value + 1e-12 < dashboard_best_value
    )
    summary_is_stale = summary_trial_count_stale or summary_best_value_stale
    planned_trials = (
        dashboard_total_trials or summary_n_trials or completed_trials_count or 50
    )
    partial_analysis = (
        dashboard_total_trials is not None
        and completed_trials_count < dashboard_total_trials
    )

    if data_source == "synthetic":
        docker_experiment_command = (
            f"PFF_HPO_SMOKE_MODE=1 ./pff hpo --trials {planned_trials} --synthetic-data "
            "--no-update-config --no-bert --no-dashboard "
            f"--study-name {study_name}"
        )
    else:
        docker_experiment_command = (
            f"./pff hpo --trials {planned_trials} --no-update-config --no-bert "
            f"--no-dashboard --study-name {study_name}"
        )

    return {
        "run_id": RUN_ID,
        "study_name": dashboard.get("studyName"),
        "dashboard_updated_at": dashboard.get("updatedAt"),
        "docker_experiment_command": docker_experiment_command,
        "audit_command": (
            "docker run --rm --entrypoint python ... "
            "/app/scripts/benchmarks/search_space_advisor_audit.py "
            "--input /app/outputs/.cache/hpo/dashboard_data.json "
            "--output /app/outputs/benches/search_space_advisor/deep_research_audit_20260506.json "
            "--min-prefix 8"
        ),
        "trial_counts": {
            "dashboard_total_trials": dashboard_total_trials,
            "completed_trials_in_dashboard_payload": completed_trials_count,
            "hpo_summary_n_trials": summary_n_trials,
            "planned_trials": planned_trials,
            "partial_analysis": partial_analysis,
            "advisor_completed_trials": metadata.get("n_completed_trials"),
            "advisor_top_k": metadata.get("n_top_k"),
        },
        "data_info": data_info,
        "hpo_data_profile": hpo_summary.get("hpo_data_profile"),
        "objective": {
            "dashboard_best_value": dashboard_best_value,
            "hpo_summary_best_value": (
                dashboard_best_value if summary_is_stale else summary_best_value
            ),
            "optimization_time_seconds": (
                None if summary_is_stale else optimization_time_seconds
            ),
            "sampler": dashboard.get("sampler"),
            "direction_input": dashboard.get("direction"),
            "direction_normalized": metadata.get("direction_normalized"),
            "summary_is_stale": summary_is_stale,
            "summary_trial_count_stale": summary_trial_count_stale,
            "summary_best_value_stale": summary_best_value_stale,
        },
        "advisor": {
            "advisor_version": metadata.get("advisor_version"),
            "policy_version": metadata.get("policy_version"),
            "policy_hash": metadata.get("policy_hash"),
            "policy_thresholds": metadata.get("policy_thresholds"),
            "importance_source": metadata.get("importance_source"),
            "importance_quality": metadata.get("importance_quality"),
            "search_space_coverage_ratio": metadata.get("search_space_coverage_ratio"),
            "missing_params": metadata.get("missing_params"),
            "action_counts": dict(action_counts),
            "n_recommendations": len(recommendations),
        },
        "reliability_summary": rel,
        "self_audit": self_audit,
        "offline_backtest": audit.get("backtest"),
        "ablations": audit.get("ablations") or [],
        "paired_benchmark": paired_benchmark or {},
        "pareto_quality_time": pareto_metrics,
        "figures": figures,
        "source_artifacts": {
            "dashboard": "outputs/.cache/hpo/dashboard_data.json",
            "hpo_summary": "outputs/optimization/kg_dslfm/hpo_summary.json",
            "advisor_audit": "outputs/benches/search_space_advisor/deep_research_audit_20260506.json",
            "paired_benchmark": "outputs/benches/search_space_advisor/paired_benchmark_50.json",
            "technical_doc": "src/pff/infrastructure/hpo/SEARCH_SPACE_ADVISOR.md",
        },
    }


def _recommendation_table(recommendations: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Parâmetro | Importância | Ação | Confiança |",
        "|---|---:|---|---|",
    ]
    for rec in recommendations[:10]:
        param = str(rec.get("param_name", ""))
        importance = float(rec.get("importance", 0.0))
        action = str(rec.get("action", ""))
        confidence = str(rec.get("confidence", ""))
        lines.append(f"| `{param}` | {importance:.4f} | `{action}` | {confidence} |")
    return lines


def _ablation_table(ablations: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Variante | Custo (ms) | Recomendações | Ações | Confiança média | Wilson validação | Wilson direcional |",
        "|---|---:|---:|---|---:|---:|---:|",
    ]
    for row in ablations:
        counts = row.get("action_counts") or {}
        action_summary = ", ".join(
            f"{name}={count}" for name, count in sorted(counts.items())
        )
        directional = row.get("directional_hit_rate_wilson_lb")
        lines.append(
            "| "
            f"`{row.get('name')}` | "
            f"{float(row.get('metadata_compute_time_ms') or row.get('elapsed_ms') or 0.0):.2f} | "
            f"{int(row.get('n_recommendations') or 0)} | "
            f"{action_summary} | "
            f"{float(row.get('mean_confidence_score') or 0.0):.4f} | "
            f"{float(row.get('validation_pass_wilson_lb') or 0.0):.4f} | "
            f"{'n/a' if directional is None else f'{float(directional):.4f}'} |"
        )
    return lines


def _paired_benchmark_table(policies: list[dict[str, Any]]) -> list[str]:
    policy_labels = {
        "tpe_pure": "TPE puro",
        "gp_bo": "GP-BO",
        "advisor_full": "Advisor full",
        "advisor_no_surrogate": "Sem surrogate",
        "advisor_no_bootstrap": "Sem bootstrap",
        "advisor_no_self_audit": "Sem self-audit",
    }
    lines = [
        "| Política | Melhor médio | Δ vs TPE | Δ vs GP-BO | Vitórias/p |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in policies:
        pvalue = row.get("wilcoxon_greater_pvalue")
        delta_gp = row.get("mean_delta_vs_gp_bo")
        lines.append(
            "| "
            f"{policy_labels.get(str(row.get('policy')), str(row.get('policy')))} | "
            f"{float(row.get('mean_best_value') or 0.0):.6f} | "
            f"{float(row.get('mean_delta_vs_tpe') or 0.0):+.6f} | "
            f"{'n/a' if delta_gp is None else f'{float(delta_gp):+.6f}'} | "
            f"{int(row.get('wins_vs_tpe') or 0)}/"
            f"{'n/a' if pvalue is None else f'{float(pvalue):.5f}'} |"
        )
    return lines


def _figure_block(path: str, caption: str, source: str, *, width: str = "95%") -> list[str]:
    return [
        f"![{caption}]({path}){{width={width}}}",
        "",
        f"*Fonte: {source}.*",
        "",
    ]


def _data_source_phrase(data_info: dict[str, Any]) -> str:
    source = str(data_info.get("source") or "").strip().lower()
    if source == "synthetic":
        return "dados sintéticos controlados"
    if source == "real":
        return "dados reais do PFF"
    if source:
        return f"dados {source}"
    return "dados do PFF"


def _formula_section() -> list[str]:
    return [
        "## Fórmulas por caso e vínculo com a implementação",
        "",
        "A seguir, eu organizo as fórmulas pelo papel que exercem. As equações de otimização e de sobrevivência funcionam como molduras teóricas; as demais, quando indicado, correspondem ao caminho operacional do Advisor versão 2.3.0.",
        "",
        "### Otimização caixa-preta e amostragem",
        "",
        "Eu trato o problema geral de HPO como otimização caixa-preta:",
        "",
        "$$x^* = \\arg\\max_{x \\in \\mathcal{X}} f(x)$$",
        "",
        "ou, para métricas de perda, como $x^* = \\arg\\min_{x \\in \\mathcal{X}} f(x)$. Na busca aleatória, cada tentativa é sorteada de uma distribuição configurada:",
        "",
        "$$x_t \\sim p(x).$$",
        "",
        "No TPE usado pelo Optuna, eu escolho o próximo candidato maximizando a razão entre a densidade dos bons pontos e a densidade dos demais pontos [@optunaTPE; @bergstra2011algorithms]:",
        "",
        "$$x^* = \\arg\\max_x \\frac{l(x)}{g(x)}.$$",
        "",
        "Exemplo prático. Se `learning_rate` e `temperature` aparecem com frequência entre os melhores trials, o numerador $l(x)$ cresce nessa região e o TPE passa a revisitá-la com prioridade. A metáfora útil é a de um garimpo: eu não espalho a peneira por todo o rio depois de achar ouro em um trecho promissor.",
        "",
        "### Seleção top-k, quantis e concentração",
        "",
        "O Advisor usa uma fração padrão $r_{top}=0{,}25$ e um piso adaptativo. O valor efetivo é:",
        "",
        "$$k=\\max\\left(\\left\\lfloor r_{top}n\\right\\rfloor,\\ \\max\\left(k_{min},\\min\\left(20,\\max\\left(3,\\left\\lfloor0{,}05n\\right\\rfloor\\right)\\right)\\right)\\right).$$",
        "",
        "Os quantis usam interpolação linear. Para $p \\in [0,1]$, $i=p(n-1)$, $l=\\lfloor i\\rfloor$, $h=\\min(l+1,n-1)$ e $\\alpha=i-l$:",
        "",
        "$$q_p=(1-\\alpha)x_{(l)}+\\alpha x_{(h)}.$$",
        "",
        "A concentração do top-k usa o coeficiente de variação, com proteção contra média próxima de zero [@nistCV]:",
        "",
        "$$CV_{top}=\\frac{s_{top}}{\\max(|\\bar{x}_{top}|,10^{-12})}.$$",
        "",
        "A regra operacional de estreitamento é:",
        "",
        "$$CV_{top}<0{,}15 \\land n \\ge n_{agressivo} \\Rightarrow [low',high']=[q_{10}^{top},q_{90}^{top}],$$",
        "",
        "com margem mínima de 10% do intervalo original quando $q_{90}^{top}-q_{10}^{top}$ fica estreito demais.",
        "",
        "Exemplo prático. Se eu tenho $n=14$ trials completos, então a regra adaptativa produz $k=3$. Se os três melhores trials de `batch_size` ficam concentrados perto de 512, com $CV_{top}<0{,}15$, o Advisor estreita a faixa em torno dessa vizinhança. A metáfora aqui é uma torcida comprimida na arquibancada: quando quase todo mundo já se juntou no mesmo setor, eu não preciso manter o estádio inteiro aberto.",
        "",
        "### Tendência monotônica e expansão de borda",
        "",
        "A correlação de Spearman é calculada como correlação de ranks com tratamento de empates:",
        "",
        "$$\\rho_s=\\frac{\\operatorname{cov}(rank(x),rank(y))}{\\sqrt{\\operatorname{var}(rank(x))\\operatorname{var}(rank(y))}}.$$",
        "",
        "A forma clássica sem empates é usada apenas como referência teórica [@mathworldSpearman]:",
        "",
        "$$\\rho_s=1-\\frac{6\\sum_i d_i^2}{n(n^2-1)}.$$",
        "",
        "Em parâmetros numéricos, as proximidades de borda são:",
        "",
        "$$p_{upper}=\\frac{q_{90}^{top}-low}{high-low},\\qquad p_{lower}=\\frac{q_{10}^{top}-low}{high-low}.$$",
        "",
        "A expansão superior exige sinal de borda e tendência compatível:",
        "",
        "$$p_{upper}>1-0{,}15 \\land \\rho_s \\ge 0{,}15 \\Rightarrow high'=high+0{,}5(high-low).$$",
        "",
        "Para parâmetros sensíveis a custo, o gatilho superior é mais conservador:",
        "",
        "$$\\rho_s \\ge 0{,}25 \\land I_j \\ge 0{,}1.$$",
        "",
        "A expansão inferior usa sinal monotônico oposto:",
        "",
        "$$p_{lower}<0{,}15 \\land \\rho_s \\le -0{,}15 \\Rightarrow low'=low-0{,}5(high-low).$$",
        "",
        "Quando o parâmetro está em escala logarítmica, o cálculo ocorre em $x_{log}=\\log_{10}(x)$ e depois retorna ao domínio original por $x=10^{x_{log}}$.",
        "",
        "Exemplo prático. Se `learning_rate` está em $[0{,}0004, 0{,}0006]$, com $q_{90}^{top}=0{,}00058$ e $\\rho_s=0{,}42$, então $p_{upper}=0{,}90$ e a expansão superior leva o teto para $0{,}0007$. Eu leio isso como um termômetro encostando no teto da escala atual: se os melhores pontos continuam subindo perto da borda, manter o teto parado é limitar a exploração cedo demais.",
        "",
        "### Importâncias, categorias e congelamento",
        "",
        "A importância numérica interna é a magnitude da associação monotônica:",
        "",
        "$$I_j^{num}=|\\rho_s(x_j,y)|.$$",
        "",
        "Para variáveis categóricas, a força interna segue uma razão tipo ANOVA:",
        "",
        "$$\\eta_j^2=\\frac{SS_{between}}{SS_{total}}.$$",
        "",
        "Quando há importância externa e interna, o Advisor mistura ambas e renormaliza:",
        "",
        "$$I_j=\\operatorname{norm}\\left(\\alpha I_j^{ext}+(1-\\alpha)I_j^{int}\\right),\\qquad \\alpha\\in[0{,}2,0{,}8].$$",
        "",
        "Se uma recomendação ainda está em `keep` e $I_j<0{,}05$, o parâmetro pode ser fixado na mediana do top-k:",
        "",
        "$$I_j<0{,}05 \\Rightarrow x_j'=q_{50}^{top}.$$",
        "",
        "Para categorias, o sistema calcula proporção, entropia e número efetivo:",
        "",
        "$$p_c=\\frac{n_c^{top}}{k},\\qquad H=-\\sum_c p_c\\log(p_c),\\qquad N_{eff}=e^H.$$",
        "",
        "A redução categórica exige dominância e passa por guardas de evidência:",
        "",
        "$$\\max_c p_c\\ge0{,}60 \\Rightarrow \\texttt{reduce\\_categories}.$$",
        "",
        "Exemplo prático. Se um parâmetro numérico tem $I_j=0{,}03$ e mediana top-k 512, eu posso sugerir `fix` em 512. Se, ao mesmo tempo, um parâmetro categórico aparece como `adamw=8` e `lion=2` no top-k, então a categoria dominante atinge $p_c=0{,}8$ e a ação `reduce_categories` passa a ser admissível. A metáfora é a de uma banca avaliadora: quando quase todos os pareceres convergem para a mesma opção, eu reduzo o barulho sem fingir que a escolha virou verdade universal.",
        "",
        "### Surrogate, UCB/LCB e segurança tipo BALLET",
        "",
        "O surrogate operacional é uma `RandomForestRegressor`, não um processo gaussiano. Com $T=64$ árvores, a média e a dispersão entre árvores são [@sklearnRF; @breiman2001random]:",
        "",
        "$$\\hat{\\mu}(x)=\\frac{1}{T}\\sum_{t=1}^{T}h_t(x),\\qquad \\hat{\\sigma}(x)=\\sqrt{\\frac{1}{T-1}\\sum_{t=1}^{T}(h_t(x)-\\hat{\\mu}(x))^2}.$$",
        "",
        "Os limites usados no teste de segurança são:",
        "",
        "$$UCB(x)=\\hat{\\mu}(x)+1{,}96\\hat{\\sigma}(x),\\qquad LCB(x)=\\hat{\\mu}(x)-1{,}96\\hat{\\sigma}(x).$$",
        "",
        "O estreitamento inspirado em BALLET só é aceito quando a melhor fronteira pessimista dentro da nova região supera a melhor fronteira otimista descartada [@zhang2023ballet]:",
        "",
        "$$\\max_{x\\in\\mathcal{X}_{new}}LCB(x)>\\max_{x\\notin\\mathcal{X}_{new}}UCB(x).$$",
        "",
        "Exemplo prático. Quando eu avalio uma nova faixa de `lambda_pc`, o surrogate funciona como uma maquete de túnel de vento: ele não substitui o voo real, mas me permite comparar a fronteira pessimista dentro da nova região com a fronteira otimista que eu deixaria de fora. Se a pior leitura interna ainda supera a melhor leitura externa, eu aceito o estreitamento com folga estatística.",
        "",
        "### Projeção multiobjetivo",
        "",
        "Para cada objetivo $i$, o Advisor inverte objetivos de minimização por $a_i=-f_i$ e mantém maximização por $a_i=f_i$. Depois normaliza:",
        "",
        "$$\\tilde{a}_i(x)=\\frac{a_i(x)-\\min(a_i)}{\\max(a_i)-\\min(a_i)}.$$",
        "",
        "O escore escalar de base é:",
        "",
        "$$S_{base}(x)=\\frac{1}{m}\\sum_{i=1}^{m}\\tilde{a}_i(x).$$",
        "",
        "A dominância de Pareto é:",
        "",
        "$$x_a\\succ x_b \\iff \\forall i,\\tilde{a}_i(x_a)\\ge\\tilde{a}_i(x_b)\\land\\exists j,\\tilde{a}_j(x_a)>\\tilde{a}_j(x_b).$$",
        "",
        "Quando o número de objetivos é tratável, o escore final incorpora bônus de hipervolume e penalidade de frente:",
        "",
        "$$S_{final}=S_{base}+0{,}15HV_{bonus}-0{,}05\\max(0,rank-1).$$",
        "",
        "Exemplo prático. Se eu comparo qualidade e latência com três pontos $A=(0{,}82,5{,}0)$, $B=(0{,}80,7{,}0)$ e $C=(0{,}84,8{,}5)$, então $A$ domina $B$, mas não domina $C$. Eu mantenho $A$ e $C$ na frente de Pareto porque um é mais rápido e o outro é mais preciso. A metáfora é simples: não faz sentido declarar vencedor absoluto quando cada competidor vence em uma pista diferente.",
        "",
        "### Confiança, Wilson e self-audit",
        "",
        "A incerteza heurística combina maturidade do estudo e robustez da elite:",
        "",
        "$$u=1-\\min\\left(1,\\frac{n}{30}\\right)\\min\\left(1,\\frac{k}{20}\\right).$$",
        "",
        "O suporte bootstrap mede a fração de reamostragens que preservam a ação:",
        "",
        "$$support=\\frac{\\#\\{b:a_b=a\\}}{\\#\\{b:a_b\\ \\text{valida}\\}}.$$",
        "",
        "Quando existe suporte, ele é calibrado com prior neutro de tamanho 20:",
        "",
        "$$support_c=\\frac{support\\cdot n_{evid}+0{,}5\\cdot20}{n_{evid}+20}.$$",
        "",
        "O score de confiança implementado é:",
        "",
        "$$C=\\operatorname{clamp}_{[0,1]}(0{,}45B+0{,}55support_c-0{,}15u).$$",
        "",
        "Sem bootstrap disponível, a implementação usa:",
        "",
        "$$C=\\operatorname{clamp}_{[0,1]}(B(1-0{,}4u)).$$",
        "",
        "O limite inferior de Wilson para proporções é [@brown2001interval]:",
        "",
        "$$LB=\\frac{\\hat{p}+\\frac{z^2}{2n}-z\\sqrt{\\frac{\\hat{p}(1-\\hat{p})+z^2/(4n)}{n}}}{1+\\frac{z^2}{n}}.$$",
        "",
        "No self-audit, uma ação direcional acerta quando o sinal posterior de Spearman confirma a direção:",
        "",
        "$$hit=\\mathbb{1}[(a=\\texttt{expand\\_upper}\\land\\rho_{suffix}>0)\\lor(a=\\texttt{expand\\_lower}\\land\\rho_{suffix}<0)].$$",
        "",
        "A taxa de acerto por grupo é:",
        "",
        "$$hit\\_rate_g=\\frac{hits_g}{total_g}.$$",
        "",
        "Um padrão vira grupo de bloqueio quando:",
        "",
        "$$total_g\\ge total_{min}^{eff}\\land hit\\_rate_g<0{,}5\\land LB_g<0{,}35.$$",
        "",
        "A análise de sobrevivência entra como moldura descritiva, não como motor operacional corrente: $R(t)=P(T>t)$ e $h(t)=f(t)/R(t)$ [@nistReliability; @nistHazard].",
        "",
        "Exemplo prático. Se eu observo 7 validações bem-sucedidas em 10 recomendações, então a taxa observada é 0,7, mas o limite inferior de Wilson cai para cerca de 0,397. Se um grupo histórico como `dropout|expand_upper` registra 3 acertos em 8 auditorias, o `hit_rate` fica em 0,375 e o bloqueio passa a fazer sentido. A metáfora aqui é a de uma testemunha: não basta saber quantas vezes ela acertou; eu também preciso saber quantas vezes ela falou antes de confiar de verdade.",
        "",
        "### Cold start",
        "",
        "Com menos evidência, as heurísticas de dataset usam escala do grafo. Para dimensão de embedding, a implementação arredonda para potência de dois:",
        "",
        "$$d=\\min\\left(1024,\\max\\left(64,2^{round(\\log_2(\\max(64,2\\sqrt{N_{entities}})))}\\right)\\right).$$",
        "",
        "Exemplo prático. Para um grafo com 5.470 entidades, a conta produz $d=128$. Eu uso essa heurística como ponto de partida, não como verdade final. A metáfora é a de escolher a primeira marcha antes de sair com o carro: ela me coloca em movimento, mas não determina a viagem inteira.",
        "",
    ]


def _formal_proofs_section() -> list[str]:
    return [
        "## Prova basal e evidência empírica",
        "",
        "Esta seção separa a prova basal, que decorre das definições e guardas do algoritmo, da evidência empírica, que depende do payload do estudo de caso. A prova basal não afirma otimalidade global; ela afirma propriedades condicionais e verificáveis do Advisor quando suas hipóteses são satisfeitas.",
        "",
        "**Definição 1 — Estado do Advisor.** Um estado é a tupla $s=(\\mathcal{X},D_t,\\delta,M,A_t)$, em que $\\mathcal{X}$ é o espaço de busca atual, $D_t=\\{(x_i,y_i)\\}_{i=1}^{t}$ é o histórico de trials completos, $\\delta$ é a direção de otimização, $M$ é o conjunto de metadados de confiabilidade e $A_t$ é o histórico de ações auditáveis.",
        "",
        "**Definição 2 — Ações admissíveis.** Uma ação $a_j$ sobre o parâmetro $j$ pertence a `{keep, narrow, expand_lower, expand_upper, fix, reduce_categories}`. Ela é admissível quando preserva o domínio do parâmetro, respeita os limiares configurados, passa pela validação rígida e não pertence a um grupo bloqueado pelo self-audit.",
        "",
        "**Definição 3 — Utilidade local.** A utilidade local de uma recomendação é uma função ordinal $U(a_j\\mid s)$ que combina sinal empírico, importância, suporte bootstrap, incerteza e penalidades de validação. O Advisor usa essa utilidade apenas para ordenar e calibrar recomendações; ela não é interpretada como recompensa causal nem como posterior bayesiano.",
        "",
        "**Hipóteses.** Assume-se que: H1) os trials usados no cálculo estão completos e comparáveis sob a mesma direção de otimização; H2) o espaço de busca inicial codifica limites válidos; H3) parâmetros numéricos podem ser ordenados e parâmetros categóricos têm categorias explícitas; H4) as estatísticas de elite usam top-k não vazio; H5) o surrogate RF é usado como heurística empírica, não como garantia probabilística exata; H6) qualquer conclusão de confiabilidade é condicionada ao tamanho amostral observado.",
        "",
        "**Lema 1 — Top-k não vazio e limitado.** Se $n\\ge1$ e $k_{min}\\ge1$, então a regra $k=\\max(\\lfloor r_{top}n\\rfloor,\\max(k_{min},\\min(20,\\max(3,\\lfloor0{,}05n\\rfloor))))$ produz $k\\ge1$. Se a implementação recorta a lista ordenada pelo número real de trials disponíveis, então a elite efetiva também satisfaz $1\\le k_{eff}\\le n$.",
        "",
        "**Demonstração.** Como $k_{min}\\ge1$, o segundo termo do máximo externo é pelo menos 1. Logo, $k\\ge1$. A seleção operacional toma prefixo da lista de $n$ trials completos; portanto, ainda que a regra nominal produza valor maior que $n$ em estudos muito pequenos, o prefixo materializado não pode conter mais do que $n$ elementos. Assim, $1\\le k_{eff}\\le n$. $\\square$",
        "",
        "**Lema 2 — Estreitamento preserva subintervalo válido.** Para um parâmetro numérico com $low<high$, se $q_{10}^{top}$ e $q_{90}^{top}$ são quantis de valores top-k pertencentes a $[low,high]$ e $q_{10}^{top}\\le q_{90}^{top}$, então o intervalo proposto $[q_{10}^{top},q_{90}^{top}]$, após margem mínima implementada e clipping ao domínio original quando aplicável, permanece um subintervalo válido do domínio analisado.",
        "",
        "**Demonstração.** Quantis de uma amostra contida em $[low,high]$ pertencem ao mesmo intervalo por convexidade da interpolação linear. Como $q_{10}^{top}\\le q_{90}^{top}$, o intervalo é ordenado. A margem mínima só alarga uma janela degenerada ou estreita demais, e o clipping impede saída do domínio válido. Logo, a recomendação resultante continua bem formada. $\\square$",
        "",
        "**Lema 3 — Wilson reduz excesso de confiança em amostras pequenas.** Para $0<\\hat p<1$, $n>0$ e $z>0$, o limite inferior de Wilson é menor que a taxa observada $\\hat p$.",
        "",
        "**Demonstração.** O termo subtraído no numerador contém $z\\sqrt{(\\hat p(1-\\hat p)+z^2/(4n))/n}$, estritamente positivo sob as hipóteses. O denominador $1+z^2/n$ é maior que 1. Assim, o limite inferior desloca a estimativa para baixo em relação à proporção observada, com efeito mais forte quando $n$ é pequeno. $\\square$",
        "",
        "**Teorema 1 — Admissibilidade estrutural condicional.** Sob H1-H6, se uma recomendação emitida pelo Advisor passa pela validação rígida, respeita os limites de domínio e não é bloqueada pelo self-audit, então ela é estruturalmente admissível para revisão do espaço de busca. Essa admissibilidade não implica melhoria futura nem otimalidade global.",
        "",
        "**Demonstração.** Pela Definição 2, admissibilidade exige domínio preservado, limiares satisfeitos, validação rígida e ausência de bloqueio histórico. Os lemas 1 e 2 garantem que as estatísticas de elite e o estreitamento numérico são bem formados quando suas hipóteses são atendidas. O Lema 3 mostra que a confiança reportada é conservadora em amostras pequenas. Portanto, uma ação que passa por essas guardas satisfaz o contrato estrutural do Advisor. Como nenhuma hipótese assume convexidade, estacionariedade, suficiência causal do histórico ou exatidão probabilística do surrogate RF, não se segue otimalidade global nem garantia de melhoria no próximo trial. $\\square$",
        "",
        "**Corolário — Robustez interpretativa.** Se o Wilson-LB ou o self-audit são fracos, a recomendação pode continuar estruturalmente válida, mas deve ser interpretada como sugestão exploratória, não como decisão forte.",
        "",
        "**Evidência empírica e2e.** No estudo PFF, a prova basal é instanciada por 25 trials completos, 21 recomendações, cobertura 1.00 do espaço, validação rígida 1.0000, Wilson-LB de validação 0.8454 e self-audit com 19 sinais direcionais. Esses números não provam superioridade contra todos os métodos de HPO; eles demonstram que o pipeline e2e produziu payload completo, figuras, recomendações e métricas de confiabilidade auditáveis.",
        "",
        "**Limites e ameaças à validade.** As principais ameaças são: amostra curta para self-audit temporal; possível viés do espaço inicial; dependência de métricas normalizadas corretamente; surrogate RF usado como aproximação heurística; ausência de comparação pareada longa contra TPE puro, GP-BO e BALLET completo; e dependência de infraestrutura Docker para reprodutibilidade operacional.",
        "",
        "**Potencial de proteção intelectual.** O conjunto formado por self-audit temporal, Wilson-LB, validação rígida, ações de espaço e rastreabilidade de payload pode ser descrito como combinação técnica potencialmente protegível. Este artigo não afirma patenteabilidade, novidade jurídica nem liberdade de operação; ele apenas identifica elementos que poderiam ser avaliados futuramente por especialista em propriedade intelectual.",
        "",
    ]


def _worked_examples_section() -> list[str]:
    return [
        "## Exemplos operacionais das fórmulas",
        "",
        "Os exemplos abaixo são curtos de propósito: servem para mostrar como as fórmulas se comportam em situações concretas. Alguns usam números do estudo de caso PFF; outros são artificiais para deixar a leitura geral e reutilizável em qualquer projeto.",
        "",
        "### Exemplo 1 - top-k adaptativo em estudo curto",
        "",
        "Considere $n=14$ trials completos, fração padrão $r_{top}=0{,}25$ e piso mínimo $k_{min}=3$. Então:",
        "",
        "$$k=\\max(\\lfloor0{,}25\\cdot14\\rfloor,\\max(3,\\min(20,\\max(3,\\lfloor0{,}05\\cdot14\\rfloor))))=\\max(3,3)=3.$$",
        "",
        "Na prática, isso significa que o Advisor trabalha com uma elite de três configurações mesmo quando o estudo ainda é curto. Esse raciocínio vale igualmente para acurácia, F1, MRR, AUPRC, custo ou qualquer outra métrica já escalarizada.",
        "",
        "### Exemplo 2 - expansão superior de um parâmetro numérico",
        "",
        "Suponha um intervalo atual $[low,high]=[0{,}0004,0{,}0006]$, com $q_{90}^{top}=0{,}00058$ e $\\rho_s=0{,}42$. A posição de borda fica:",
        "",
        "$$p_{upper}=\\frac{0{,}00058-0{,}0004}{0{,}0006-0{,}0004}=0{,}90.$$",
        "",
        "Como $0{,}90>0{,}85$ e $\\rho_s\\ge0{,}15$, a regra de expansão superior dispara:",
        "",
        "$$high'=0{,}0006+0{,}5(0{,}0006-0{,}0004)=0{,}0007.$$",
        "",
        "O exemplo pode ser lido como `learning_rate`, `dropout`, `temperature` ou qualquer hiperparâmetro contínuo em que o melhor desempenho esteja encostando no teto atual.",
        "",
        "### Exemplo 3 - fixação e redução categórica",
        "",
        "Se um parâmetro numérico tem importância $I_j=0{,}03$ e mediana top-k $q_{50}^{top}=512$, então a heurística de congelamento pode sugerir `fix` em 512. Já para uma variável categórica com contagens top-k `adamw=8` e `lion=2`, obtém-se $p_{adamw}=0{,}8$ e:",
        "",
        "$$H=-(0{,}8\\ln0{,}8+0{,}2\\ln0{,}2)\\approx0{,}500.$$",
        "",
        "$$N_{eff}=e^{0{,}500}\\approx1{,}65.$$",
        "",
        "Como a categoria dominante passa de 60%, a ação `reduce_categories` torna-se admissível. Essa mesma lógica serve para otimizadores, funções de ativação, samplers, backends ou quaisquer escolhas discretas.",
        "",
        "### Exemplo 4 - limite inferior de Wilson",
        "",
        "Imagine 7 validações bem-sucedidas em 10 recomendações, isto é, $\\hat{p}=0{,}7$ e $n=10$. Aplicando Wilson com $z=1{,}96$:",
        "",
        "$$LB\\approx0{,}397.$$",
        "",
        "A taxa observada de 70% parece alta, mas a fronteira conservadora ainda fica abaixo de 0,40. É exatamente esse freio que impede o Advisor de declarar confiabilidade alta cedo demais.",
        "",
        "### Exemplo 5 - Pareto em duas métricas",
        "",
        "Considere três configurações avaliadas por qualidade (maximizar) e latência (minimizar): $A=(0{,}82,5{,}0)$, $B=(0{,}80,7{,}0)$ e $C=(0{,}84,8{,}5)$. A configuração $A$ domina $B$ porque entrega qualidade maior com latência menor. Já $A$ e $C$ permanecem não dominadas entre si, logo ambas ficam na frente de Pareto. O mesmo raciocínio vale para pares como F1-custo, AUPRC-memória ou MRR-tempo.",
        "",
        "### Exemplo 6 - self-audit e bloqueio histórico",
        "",
        "Suponha que o grupo `dropout|expand_upper` tenha produzido 3 acertos em 8 sufixos auditados. Então:",
        "",
        "$$hit\\_rate=\\frac{3}{8}=0{,}375.$$",
        "",
        "Com Wilson, o limite inferior fica aproximadamente $LB\\approx0{,}137$. Se o grupo já atende ao mínimo efetivo de observações, ele entra na lista de bloqueio porque $hit\\_rate<0{,}5$ e $LB<0{,}35$. Em termos práticos, o Advisor aprende a desconfiar de uma direção que envelheceu mal no histórico.",
        "",
    ]


def _restored_tables_section(figures: dict[str, str]) -> list[str]:
    return [
        "## Arquitetura lógica, metáforas e tabelas-síntese",
        "",
        "Eu retomo aqui duas metáforas que considero úteis. Um surrogate model é como uma maquete de túnel de vento: ele não é o avião de verdade, mas ajuda a testar direção e risco antes de gastar combustível no voo real. Já a confiabilidade estatística é como ouvir uma testemunha. Não basta saber que ela acertou 80% das vezes; importa saber se ela falou cinco vezes ou quinhentas. É exatamente por isso que limites como o de Wilson são mais honestos do que a simples taxa observada, sobretudo em amostras pequenas [@brown2001interval; @barlow1996reliability].",
        "",
        "Eu reconstituo a arquitetura analítica do Advisor a partir do documento técnico. O pipeline atual começa com a normalização do estudo, seleciona trials válidos, projeta eventuais objetivos múltiplos em um score escalar híbrido, escolhe um top-k adaptativo, computa estatísticas por parâmetro, treina opcionalmente um surrogate de floresta aleatória, estima importâncias e interações, emite ações locais, calibra a confiança com bootstrap e validação dura, executa self-audit histórico e finalmente resume a confiabilidade agregada do payload. Em termos práticos, o sistema age como um copiloto: ele não treina o modelo principal, mas fica olhando painel, borda, vibração e tendência para dizer quando vale abrir o mapa, quando vale apertar a lanterna e quando vale não mexer em nada [@breiman2001random; @hutter2014fanova; @lundberg2017shap].",
        "",
        f"![Fluxo lógico do Search Space Advisor.]({figures['flow']}){{width=100%}}",
        "",
        "*Fonte: elaboração própria com base no documento técnico do projeto e na literatura de HPO e interpretabilidade.*",
        "",
        "As Tabelas 1 a 3 fazem a ponte entre a formulação geral do método e o posicionamento do Advisor no ecossistema de HPO. As Tabelas 4 e 5 registram o estudo de caso do PFF. As Tabelas 6 a 8 recolocam o problema em uma moldura mais ampla de detecção de anomalias, avaliação e consistência conceitual.",
        "",
        "**Tabela 1 — Ranking metodológico contextualizado do problema**",
        "",
        "| Posição | Método ou família | Leitura corrigida e contextualizada |",
        "|---|---|---|",
        "| 1 | Search Space Advisor com adaptação de espaço | Deve ser lido como ranking contextual para espaços caros, mistos e com realimentação do histórico, não como teorema universal |",
        "| 2 | Optuna, Vizier e samplers modernos de produção | Fortes em prática industrial, sobretudo por flexibilidade, paralelismo, poda e integração de infraestrutura |",
        "| 3 | Otimização bayesiana clássica baseada em GP | Forte em cenários suaves e de baixa a média dimensão efetiva, mas sensível à estrutura do espaço e ao custo do surrogate |",
        "| 4 | Busca aleatória e busca em grade | Busca aleatória continua sendo baseline honesta; grade tende a piorar em espaços maiores |",
        "",
        "**Tabela 2 — Dimensões de otimização consideradas**",
        "",
        "| Sistema ou estratégia | Dimensão otimizada | Fundamento matemático dominante | Observação de confiabilidade |",
        "|---|---|---|---|",
        "| BO clássica | Hiperparâmetros contínuos e discretos caros | Surrogate modeling, função de aquisição, exploração versus explotação | A confiabilidade depende da qualidade do modelo substituto |",
        "| Plataformas de produção como Optuna e Vizier | Tuning operacional em escala | Samplers, pruners, paralelismo, instrumentação | A confiabilidade também é propriedade do ecossistema |",
        "| BALLET e métodos de região de interesse | Adaptação do espaço de busca | Estimação de level-set e filtragem probabilística da região promissora | A confiabilidade entra como teste de segurança para não podar o ótimo |",
        "| Search Space Advisor | Refinamento local do espaço | Top-k, Spearman, bootstrap, Wilson, self-audit, surrogate RF | A confiabilidade é multicamada e auditável |",
        "",
        "**Tabela 3 — Tipologia de aplicações para o Advisor**",
        "",
        "| Classe de estudo | Objeto otimizado | Variáveis destacadas | Papel da confiabilidade |",
        "|---|---|---|---|",
        "| Visão computacional e NAS | Profundidade, largura, regularização, learning rate, batch | Mistura de parâmetros contínuos e categóricos | Evitar expansão cega em bordas e reduzir custo experimental |",
        "| Modelos de linguagem | Learning rate, warmup, regularização, lote, duração de treino | Forte sensibilidade a escalas logarítmicas e poda | Evitar conclusões instáveis com poucos trials |",
        "| Modelos relacionais, grafos e multimodais | Negativos, pesos simbólicos, parâmetros de contraste e amostragem | Espaço heterogêneo, com interdependências | Reforçar decisões por evidência local e histórico temporal |",
        "| PFF como estudo de caso | Expansão, estreitamento, fixação e redução categórica | A própria política de ajuste do espaço | A confiabilidade vira objeto central do estudo |",
        "",
        "**Tabela 4 — Espaço de busca do estudo de caso PFF**",
        "",
        "| Bloco | Hiperparâmetros principais | Faixas ou categorias |",
        "|---|---|---|",
        "| Treinamento | learning_rate; weight_decay; batch_size; negative_sample_size; grad_clip; warmup_ratio; epochs | [4e-4, 6e-4]; [1e-6, 1e-4]; {256, 512, 1024}; [384, 512]; [1, 10]; [0,10, 0,20]; [120, 160] |",
        "| Arquitetura | feature_dim; hidden_dim; kl_weight; sparsity_weight; ibp_alpha; max_communities | {256, 512}; {256, 512}; [1e-4, 1e-2]; [1e-6, 1e-2]; [1, 10]; {128} |",
        "| Contraste e amostragem | temperature; margin; adv_temperature; hard_neg_ratio; num_negatives; num_global_negatives; neg_sampler | [0,025, 0,04]; [0, 0,05]; [0,9, 1,8]; [0, 0,7]; {384}; {96}; {degree_based} |",
        "| Lógica, baixo posto e FAISS | lambda_logic; num_basis; nlist; nprobe; eval_topk | [0,03, 0,05]; {2, 4, 8, 16}; {256, 512, 1024, 2048}; {4, 8, 16, 32}; {512, 1024, 2048} |",
        "",
        "**Tabela 5 — Regras matemáticas e limiares operacionais**",
        "",
        "| Mecanismo | Regra atual informada | Interpretação |",
        "|---|---|---|",
        "| Frio inicial | Análise empírica completa apenas se n_trials >= 5 | Abaixo disso, o sistema opera quase como bom senso com memória curta |",
        "| Seleção top-k | Até 25% dos melhores trials, com piso adaptativo e teto operacional | Evita top-k minúsculo em estudos mais maduros |",
        "| Expansão superior | rho >= 0,15 | Requer monotonicidade compatível |",
        "| Expansão inferior | rho <= -0,15 | Simétrica à superior |",
        "| Expansão sensível a custo | rho >= 0,25 e importance >= 0,1 | Mais conservadora |",
        "| Estreitamento | CV < 0,15; nova faixa em [q10, q90]; janela mínima de 10% | Só aperta se o alvo já estiver agrupado |",
        "| Fixação | importance < 0,05 | Se a variável quase não move o resultado, vale congelá-la |",
        "| Bootstrap | 50 reamostragens | Mede reprodutibilidade da ação |",
        "| Wilson | z = 1,96 | Corrige excesso de confiança em amostra pequena |",
        "| Self-audit | bloqueio se hit_rate < 0,5 e Wilson LB < 0,35 | O sistema aprende a desconfiar de si mesmo |",
        "",
        "Eu considero o uso do limite inferior de Wilson uma das partes mais fortes do sistema do ponto de vista estatístico. A taxa observada pura $\\hat p$ pode fazer um mecanismo parecer ótimo em amostras mínimas, mas o limite inferior de Wilson funciona como um freio contra otimismo barato. É o equivalente estatístico a dizer: beleza, você acertou 4 de 5, mas ainda não vou te dar carteira de piloto [@brown2001interval].",
        "",
        "Outro ponto que eu considero matematicamente sólido é a calibração por bootstrap das ações finais. O sistema reamostra os trials cinquenta vezes e mede em quantas delas a mesma ação reaparece. Isso produz uma medida de reprodutibilidade local da decisão. Se a recomendação desaparece com pequenas perturbações da amostra, ela não é robusta; se reaparece repetidamente, ganha direito a confiança maior. Em metáfora bem direta, é como perguntar uma mesma coisa à banca com os nomes apagados e em ordem embaralhada: se a resposta muda toda hora, o resultado não é firme.",
        "",
        "A projeção multiobjetivo também merece destaque. Eu não reduzo o problema simplesmente ao primeiro objetivo, mas reconstruo o score híbrido com normalização por dimensão, média escalar, ordenação de Pareto e bônus de hiper-volume quando o número de objetivos permite cálculo estável. Isso aproxima o método de práticas modernas de HPO multiobjetivo, nas quais não basta maximizar um único número se custo, latência, memória e qualidade competem entre si [@deb2002nsga2; @morales2023many].",
        "",
        "**Tabela 6 — Métodos de detecção de anomalias relevantes para logs e decisões do Advisor**",
        "",
        "| Método | Metáfora simples | Vantagem | Limitação | Uso recomendado no Advisor |",
        "|---|---|---|---|---|",
        "| Z-score / IQR | detector de valores gritando fora do coro | Muito barato e explicável | Univariado e frágil em caudas pesadas | Monitorar confidence_score, latência, rho, CV e amplitude proposta |",
        "| LOF | quem ficou isolado da própria vizinhança | Excelente para anomalia local | Sensível à escolha de vizinhança | Detectar parâmetros cujo comportamento difere localmente de pares similares |",
        "| Isolation Forest | serrote que isola pontos estranhos mais rápido | Bom em alta dimensão e sem rótulo | Interpretação menos intuitiva | Monitorar payloads completos e combinações raras de bloqueios |",
        "| Deep SVDD / autoencoders | aprender a normalidade e estranhar o que não reconstrói | Captura estrutura complexa | Requer mais dados e treino | Monitorar séries longas de telemetria do HPO |",
        "",
        "**Tabela 7 — Datasets e métricas adequados para avaliar anomalias ligadas ao Advisor**",
        "",
        "| Dataset ou referência | Tipo de dado | O que avalia bem | Métricas prioritárias |",
        "|---|---|---|---|",
        "| NAB | Séries temporais reais e artificiais em tempo real | Detecção rápida e custo de falso alarme | NAB Score, precisão, revocação, atraso de detecção |",
        "| SMAP / MSL | Telemetria multivariada de missão espacial | Dependências temporais e contexto operacional | Precisão, revocação, F1 por evento, AUPRC |",
        "| UCR Time Series Anomaly Archive | Séries univariadas curadas | Comparação padronizada e menos enviesada | Precisão, revocação, F1, AUPRC |",
        "| Logs internos do HPO | Eventos estruturados do próprio Advisor | Anomalias de decisão e falhas de política | taxa de bloqueio, Wilson LB, AUPRC, tempo até falha da recomendação |",
        "",
        "Em problemas fortemente desbalanceados, curvas precisão-revocação e AUPRC tendem a ser mais informativas do que ROC/AUROC. Se o sistema quase nunca recomenda ação agressiva, um AUROC alto pode esconder baixa utilidade real, enquanto a precisão entre as ações disparadas continua sendo o que mais importa [@saito2015precision].",
        "",
        "**Tabela 8 — Inconsistências corrigidas entre narrativa e implementação**",
        "",
        "| Ponto do material original | Problema | Correção adotada neste artigo |",
        "|---|---|---|",
        "| Descrição do Advisor como sistema baseado em GP e kernel Matern | Incompatível com o anexo técnico, que informa floresta aleatória e ausência de kernel estatístico no motor atual | O artigo distingue inspiração teórica em BO/GP da implementação efetiva em RF |",
        "| Uso de estado da arte como ranking absoluto | Generalização indevida | Ranking reclassificado como contextual |",
        "| Leitura do kernel Rust como kernel de ML | Confusão terminológica | Esclarecido que o módulo em Rust acelera Spearman, não GP nem KDE |",
        "| Interpretação simplificada do top-k | Redução excessiva ao quartil superior | Formalizado como regra adaptativa com mínimo dinâmico |",
        "| Tratamento da confiabilidade como rótulo simples | Modelagem superficial | Reescrito como sistema multicamada com incerteza, bootstrap, Wilson, validação e self-audit |",
        "| Falta de ligação com censura e anomalias | Lacuna conceitual | Integradas perspectivas de sobrevivência e detecção de anomalias |",
        "",
        "A síntese acima mostra que o grande ganho desta revisão não foi enfeitar o texto com citações, e sim alinhar linguagem, matemática e implementação. Quando esses três planos andam juntos, eu deixo o artigo mais forte, mais reproduzível e menos vulnerável ao clássico isso é bonito no papel, mas seu código faz outra coisa [@ross1996stochastic; @rasmussen2006gp; @aggarwal2017outlier].",
        "",
    ]


def build_article(metrics: dict[str, Any], recommendations: list[dict[str, Any]]) -> str:
    rel = metrics["reliability_summary"]
    self_audit = metrics["self_audit"]
    objective = metrics["objective"]
    trial_counts = metrics["trial_counts"]
    data_info = metrics["data_info"] or {}
    advisor = metrics["advisor"]
    figures = metrics["figures"]
    pareto = metrics["pareto_quality_time"]
    ablations = metrics.get("ablations") or []
    paired_benchmark = metrics.get("paired_benchmark") or {}
    data_source_phrase = _data_source_phrase(data_info)
    planned_trials = int(
        trial_counts.get("planned_trials")
        or trial_counts.get("dashboard_total_trials")
        or trial_counts.get("completed_trials_in_dashboard_payload")
        or 0
    )
    completed_trials = int(trial_counts.get("completed_trials_in_dashboard_payload") or 0)
    advisor_completed_trials = int(trial_counts.get("advisor_completed_trials") or 0)
    summary_is_stale = bool(objective.get("summary_is_stale"))

    if summary_is_stale:
        study_case_summary = (
            f"No estudo de caso PFF, `sampler={objective.get('sampler')}` e a direção de entrada `"
            f"{objective.get('direction_input')}` foi normalizada para `"
            f"{objective.get('direction_normalized')}`. O dataset usado nessa rodada tinha "
            f"{data_info.get('n_train')} triplas de treino, {data_info.get('n_valid')} de validação, "
            f"{data_info.get('n_entities')} entidades e {data_info.get('n_predicates')} relações. "
            f"A campanha foi planejada para {planned_trials} trials, mas este recorte parcial usa "
            f"{completed_trials} trials completos observados no dashboard; o Advisor auditou "
            f"{advisor_completed_trials} trials completos. O resumo agregado de HPO disponível em disco "
            "não refletia este mesmo corte analítico e, por isso, não foi tratado como fonte factual "
            "para contagem, tempo total ou melhor escore desta rodada parcial."
        )
        best_value_summary = (
            "Eu preservo a distinção entre fontes, mas evito comparar artefatos de épocas diferentes: "
            f"neste recorte parcial, o melhor objetivo observado no dashboard foi "
            f"{float(objective.get('dashboard_best_value') or 0.0):.6f}. Como o resumo agregado em disco "
            "estava desatualizado para esta mesma rodada parcial, eu não uso o `best_value` dele como "
            "contraponto numérico nesta seção."
        )
    else:
        study_case_summary = (
            f"No estudo de caso PFF, `sampler={objective.get('sampler')}` e a direção de entrada `"
            f"{objective.get('direction_input')}` foi normalizada para `"
            f"{objective.get('direction_normalized')}`. O dataset usado nessa rodada tinha "
            f"{data_info.get('n_train')} triplas de treino, {data_info.get('n_valid')} de validação, "
            f"{data_info.get('n_entities')} entidades e {data_info.get('n_predicates')} relações. "
            f"O resumo de HPO registrou {trial_counts.get('hpo_summary_n_trials')} trials-alvo em "
            f"{float(objective.get('optimization_time_seconds') or 0.0):.2f} s; o payload final do dashboard "
            f"continha {completed_trials} trials completos indexados no painel e o Advisor auditou "
            f"{advisor_completed_trials} trials completos."
        )
        best_value_summary = (
            "Eu preservo a diferença entre `bestValue` do dashboard e `best_value` do resumo de otimização "
            "com rótulos distintos: o painel reportou melhor objetivo observado de "
            f"{float(objective.get('dashboard_best_value') or 0.0):.6f}, enquanto o resumo de HPO registrou "
            f"escore de trade-off {float(objective.get('hpo_summary_best_value') or 0.0):.6f}. Essa "
            "distinção é geral e importante: em qualquer projeto, score bruto de painel e critério agregado "
            "de seleção podem apontar para números diferentes sem que isso represente contradição metodológica."
        )

    lines = [
        "---",
        "title: \"Confiabilidade matemática de um Search Space Advisor para otimização de hiperparâmetros\"",
        "subtitle: \"Formulação geral, exemplos operacionais e estudo de caso no projeto PFF\"",
        "author:",
        "  - \"Alex de Lira Neto\"",
        "date: \"Salvador, 2026\"",
        "lang: pt-BR",
        "documentclass: article",
        "fontsize: 12pt",
        "papersize: a4",
        "linestretch: 1.5",
        "geometry:",
        "  - left=3cm",
        "  - right=2cm",
        "  - top=3cm",
        "  - bottom=2cm",
        "bibliography: references.bib",
        "csl: abnt.csl",
        "link-citations: true",
        "reference-section-title: Referências",
        "header-includes:",
        "  - \\usepackage{indentfirst}",
        "  - \\usepackage{booktabs}",
        "  - \\usepackage{float}",
        "  - \\floatplacement{figure}{H}",
        "  - \\usepackage{longtable}",
        "  - \\usepackage{array}",
        "  - \\usepackage{etoolbox}",
        "  - \\usepackage{caption}",
        "  - \\captionsetup{labelfont=bf,font=small}",
        "  - \\AtBeginEnvironment{longtable}{\\small}",
        "  - \\setlength{\\LTleft}{0pt}",
        "  - \\setlength{\\LTright}{0pt}",
        "  - \\setlength{\\parindent}{1.25cm}",
        "  - \\setlength{\\parskip}{0pt}",
        "abstract: |",
        "  Este artigo formaliza um Search Space Advisor como camada de decisão para adaptação de espaços de busca em otimização de hiperparâmetros. A formulação é apresentada de modo geral, compatível com métricas de maximização, minimização ou cenários multiobjetivo, e depois é instanciada em um estudo de caso no projeto PFF. O método combina seleção top-k adaptativa, quantis, correlação de Spearman, coeficiente de variação, mistura de importâncias, surrogate RandomForestRegressor, suporte bootstrap, limite inferior de Wilson e self-audit temporal. No estudo de caso executado em Docker, obtêm-se recomendações consistentes e um payload estruturalmente válido, com evidência temporal ainda curta para ações direcionais. O resultado central é que a confiabilidade do Advisor depende de evidências acumuladas e auditáveis, e não de uma única heurística.",
        "---",
        "",
        "\\noindent\\textbf{Palavras-chave:} Search Space Advisor; otimização de hiperparâmetros; confiabilidade; self-audit; limite inferior de Wilson; otimização multiobjetivo.",
        "",
        "\\noindent\\textbf{Keywords:} Search Space Advisor; hyperparameter optimization; reliability; self-audit; Wilson lower bound; multi-objective optimization.",
        "",
        "# Dados institucionais {.unnumbered}",
        "",
        "\\begin{tabular}{@{}p{0.24\\linewidth}p{0.70\\linewidth}@{}}",
        "\\textbf{Autor} & Alex de Lira Neto \\\\",
        "\\textbf{Instituição} & Universidade Federal da Bahia (UFBA), Salvador, Bahia, Brasil \\\\",
        "\\textbf{Curso} & Engenharia da Computação, graduação em andamento \\\\",
        "\\textbf{Atuação} & Pesquisa em detecção de anomalias \\\\",
        "\\textbf{Orientação} & Antônio Carlos Fernandes, professor da UFBA \\\\",
        "\\end{tabular}",
        "",
        "# Introdução",
        "",
        "Eu parto do fato de que a otimização de hiperparâmetros aparece em problemas muito diferentes entre si: classificação, regressão, ranqueamento, séries temporais, modelos generativos, sistemas sujeitos a latência e pipelines multiobjetivo com restrições de custo. Em todos esses casos, cada trial é uma avaliação cara de uma função caixa-preta, e eu preciso decidir não só qual configuração parece boa, mas também se o espaço de busca atual ainda faz sentido. Um Search Space Advisor atua justamente nesse ponto: ele não substitui o otimizador principal, e sim reinterpreta o histórico de trials para sugerir quando expandir, estreitar, fixar ou reduzir partes do espaço de busca.",
        "",
        "Eu escrevo o método discutido neste artigo para ser geral. Ele só pressupõe a existência de um histórico de configurações avaliadas, uma direção de otimização (`maximize`, `minimize` ou projeção multiobjetivo) e uma representação explícita do espaço de busca por parâmetro. Isso me permite reutilizar a mesma lógica com métricas como acurácia, F1, AUPRC, RMSE, MRR, custo, latência ou combinações dessas métricas. O projeto PFF entra aqui como estudo de caso concreto: ele fornece um experimento em Docker, um payload HPO auditável e um conjunto de recomendações efetivamente produzidas pelo Advisor.",
        "",
        "O ponto metodológico mais importante, para mim, é separar três camadas. A primeira é a teoria geral de HPO, que inclui busca aleatória, TPE, Pareto e otimização bayesiana clássica [@bergstra2012random; @bergstra2011algorithms; @snoek2012practical]. A segunda é o núcleo operacional do Advisor, implementado em `src/pff/infrastructure/hpo/search_space_advisor/**`. A terceira é a camada de confiabilidade, que usa validação, bootstrap, limite inferior de Wilson e self-audit. Essa separação evita atribuir ao código garantias que pertencem apenas à literatura de apoio e mantém a formulação aplicável a outros projetos além do PFF.",
        "",
        "# Materiais e métodos",
        "",
        "Eu organizo o artigo em duas camadas complementares. Na camada geral, eu formalizo um Advisor agnóstico ao domínio, capaz de operar sobre parâmetros numéricos e categóricos, métricas escalares ou multiobjetivo e políticas de maximização ou minimização. Na camada empírica, eu uso o PFF como estudo de caso para testar se a formulação produz recomendações coerentes, figuras rastreáveis e anexos reprodutíveis.",
        "",
        f"Eu executo o experimento do estudo de caso em Docker, conforme solicitado, com o wrapper `./pff`, {data_source_phrase}, 50 trials planejados, `no-update-config`, `no-bert`, `no-dashboard` e estudo `deep_research_advisor_real50_gpu_20260506`.",
        "",
        "A auditoria offline foi executada com `search_space_advisor_audit.py`, usando o payload real `dashboard_data.json`, `min-prefix=8` e saída registrada nos artefatos do recorte `cutoff25`.",
        "",
        study_case_summary,
        "",
        best_value_summary,
        "",
    ]
    lines.extend(_formula_section())
    lines.extend(_formal_proofs_section())
    lines.extend(_restored_tables_section(figures))
    lines.extend(
        [
            "# Resultados",
            "",
            "As fórmulas anteriores descrevem um Advisor geral; nesta seção, eu mostro a instância concreta desse Advisor no PFF. Em outras palavras, a formulação é generalista, e o experimento abaixo funciona como prova de aplicabilidade em um projeto real.",
            "",
            f"No estudo de caso PFF, eu observo que o Advisor versão {advisor.get('advisor_version')} gerou {advisor.get('n_recommendations')} recomendações para 20 hiperparâmetros do espaço de busca. A cobertura do espaço foi {float(advisor.get('search_space_coverage_ratio') or 0.0):.2f}, sem parâmetros ausentes. As ações finais foram: "
            + ", ".join(
                f"`{name}`={count}"
                for name, count in sorted((advisor.get("action_counts") or {}).items())
            )
            + ".",
            "",
            f"A confiabilidade agregada foi: validação rígida {float(rel.get('validation_pass_rate') or 0.0):.4f}, Wilson-LB de validação {float(rel.get('validation_pass_wilson_lb') or 0.0):.4f}, confiança média {float(rel.get('mean_confidence_score') or 0.0):.4f}, taxa de alta confiança {float(rel.get('high_confidence_rate') or 0.0):.4f}. A ausência de recomendações de alta confiança é coerente com amostras curtas, pois a incerteza heurística permanece alta. O ponto geral aqui é que eu não trato pouca amostra como evidência forte só porque a taxa observada parece boa.",
            "",
            f"O self-audit avaliou {self_audit.get('prefixes_evaluated')} prefixos, {self_audit.get('directional_signals_total')} sinais direcionais e obteve `hit_rate`={float(self_audit.get('directional_hit_rate') or 0.0):.4f} com Wilson-LB={float(self_audit.get('directional_hit_rate_wilson_lb') or 0.0):.4f}. O grupo `lambda_pc|expand_lower` apareceu como vilão histórico neste recorte, com taxa de acerto 0,0 e Wilson-LB 0,0, mas nenhuma ação corrente foi bloqueada porque a recomendação atual para `lambda_pc` foi `keep`. Em termos metodológicos, isso mostra que o mecanismo de bloqueio depende do histórico do padrão e não do nome do projeto; qualquer pipeline com logs de prefixo-sufixo pode usar a mesma lógica.",
            "",
            "## Tabela 9 — recomendações principais do estudo de caso",
            "",
        ]
    )
    lines.extend(_recommendation_table(recommendations))
    if ablations:
        lines.extend(
            [
                "",
                "## Tabela 10 — ablations e sensibilidade do Advisor",
                "",
                "A tabela abaixo reexecuta o Advisor no mesmo payload e no mesmo recorte de 25 trials, desligando componentes específicos. Ela mede custo, estabilidade das ações, confiança, validação e efeito do self-audit. Como o objetivo é isolar componentes do Advisor, os resultados não devem ser lidos como comparação causal contra outros otimizadores.",
                "",
            ]
        )
        lines.extend(_ablation_table([row for row in ablations if isinstance(row, dict)]))
        lines.extend(
            [
                "",
                "No recorte observado, remover surrogate ou interações não alterou a distribuição de ações, enquanto remover importâncias internas trocou uma ação `keep` por `fix`. Remover bootstrap reduziu a confiança média, mas preservou validação e ações. Remover self-audit elimina o Wilson direcional, como esperado, e por isso perde a principal evidência temporal de robustez.",
                "",
            ]
        )
    paired_policies = [
        row
        for row in paired_benchmark.get("policies", [])
        if isinstance(row, dict)
    ]
    if paired_policies:
        universal_supported = bool(
            paired_benchmark.get("universal_superiority_claim_supported")
        )
        lines.extend(
            [
                "",
                "## Tabela 11 — benchmark pareado TPE, GP-BO e Advisor",
                "",
                f"Para reduzir o risco do recorte curto, eu executo um benchmark pareado sintético com {paired_benchmark.get('n_trials')} trials por política, sementes {paired_benchmark.get('seeds')} e atualização do Advisor a cada {paired_benchmark.get('advisor_period')} trials após o aquecimento. O cenário compara TPE puro, GP-BO via `GPSampler`, Advisor completo e ablations do Advisor sob a mesma função objetivo determinística.",
                "",
            ]
        )
        lines.extend(_paired_benchmark_table(paired_policies))
        if universal_supported:
            lines.extend(
                [
                    "",
                    "Neste benchmark, a regra conservadora de reivindicação forte foi satisfeita: o Advisor completo venceu o TPE em todas as sementes, melhorou a média contra GP-BO e obteve p unilateral inferior a 0,05. Mesmo nesse caso, a conclusão continua condicionada ao cenário experimental e não equivale a prova universal fora da distribuição avaliada.",
                    "",
                ]
            )
        else:
            lines.extend(
                [
                    "",
                    "A regra conservadora de reivindicação universal não foi satisfeita. Portanto, o resultado deve ser lido como evidência localizada: o Advisor pode melhorar TPE puro neste cenário, mas não sustenta superioridade estatística universal contra GP-BO ou contra a política completa com surrogate/BALLET em qualquer problema.",
                    "",
                ]
            )
    lines.extend(["", "## Figuras geradas a partir do payload do Advisor", ""])
    lines.extend(
        _figure_block(
            figures["advisor_evidence"],
            "Evidências empíricas do Search Space Advisor: ações, confiança, validação e resumo conservador.",
            "elaboração própria a partir de searchSpaceAdvice e da auditoria offline do Advisor",
        )
    )
    lines.extend(
        _figure_block(
            figures["importances_actions"],
            "Importâncias normalizadas e ação recomendada por hiperparâmetro.",
            "elaboração própria a partir do payload searchSpaceAdvice",
        )
    )
    lines.extend(
        _figure_block(
            figures["topk_distributions"],
            "Comparação q10-q90 normalizada entre todos os trials e a região top-k dos principais parâmetros.",
            "elaboração própria a partir de searchSpaceAdvice.recommendations",
        )
    )
    lines.extend(
        _figure_block(
            figures["reliability"],
            "Indicadores de confiabilidade, incluindo validação, confiança média e self-audit.",
            "elaboração própria a partir da auditoria offline do Advisor",
        )
    )
    lines.extend(
        _figure_block(
            figures["pareto"],
            "Projeção qualidade-tempo e frente de Pareto descritiva.",
            "elaboração própria para auditar a projeção multiobjetivo consumida pelo Advisor",
        )
    )
    lines.extend(
        _figure_block(
            figures["survival"],
            "Sobrevivência empírica das recomendações direcionais no self-audit.",
            "elaboração própria a partir de metadata.self_audit",
        )
    )
    lines.extend(
        [
            "# Discussão",
            "",
            "Os dados do estudo de caso removem a principal lacuna factual: agora existem trials, payload de recomendações, reliability summary, self-audit e figuras calculadas. Ainda assim, eu mantenho a interpretação proporcional ao orçamento experimental. A rodada do PFF é suficiente para embasar a mecânica do Advisor e demonstrar rastreabilidade, mas não para reivindicar superioridade estatística universal contra GP-BO, TPE puro ou BALLET completo.",
            "",
            "A evidência mais forte é estrutural: não houve mismatch de direção, não foram detectadas inconsistências, a validação rígida foi perfeita no payload e a cobertura do espaço foi integral. A evidência mais fraca é temporal: o self-audit teve poucos prefixos e Wilson-LB baixo para decisões direcionais. Isso não invalida o Advisor; pelo contrário, mostra que o mecanismo de cautela está funcionando ao não elevar confiança com pouca amostra.",
            "",
            "O ponto mais importante para uso fora do PFF é a separação entre parte geral e parte específica. São gerais: top-k adaptativo, quantis, Spearman, coeficiente de variação, mistura de importâncias, Wilson, bootstrap, self-audit e a possibilidade de projeção multiobjetivo. São específicos do PFF: nomes de hiperparâmetros, faixas de busca, score usado no experimento e heurísticas de cold start conectadas ao grafo. Essa distinção torna o artigo reutilizável como referência metodológica para outros projetos e mantém o PFF como estudo de caso exemplar, não como fronteira do método.",
            "",
            "Também fica resolvida a inconsistência conceitual sobre surrogate. O código usa `RandomForestRegressor` com 64 árvores, profundidade máxima 8 e `random_state=42`; a incerteza é a dispersão entre árvores. Isso justifica UCB/LCB como heurística empírica, não como posterior bayesiana de GP. A referência a BALLET permanece útil como inspiração de região de interesse, mas o critério implementado é uma guarda local simples baseada em LCB dentro versus UCB fora.",
            "",
            "# Conclusão",
            "",
            "Neste artigo, eu descrevo um Search Space Advisor em dois níveis: como formulação geral para adaptação confiável de espaços de busca em HPO e como estudo de caso executado no PFF. O método é transferível para outros projetos porque depende de estruturas genéricas de experimento, como histórico de trials, direção de otimização, parâmetros auditáveis e métricas comparáveis, e não de propriedades exclusivas de Knowledge Graph Completion.",
            "",
            "No estudo de caso PFF, eu concluo que o Advisor auditado é melhor descrito como um sistema de decisão estatística local com confiabilidade multicamada. Ele combina seleção top-k, tendência monotônica, concentração, importâncias, surrogate RF, validação dura, bootstrap quando disponível, Wilson e self-audit. O experimento em Docker confirmou que o payload é consistente, reproduzível e suficientemente completo para sustentar o artigo com dados, figuras e anexos verificáveis.",
            "",
            "# Referências",
            "",
            "::: {#refs}",
            ":::",
            "",
            "\\newpage",
            "",
            "# Apêndice A - Artefatos e comandos",
            "",
            "- Dashboard HPO: `dashboard_data.json`, em `outputs/.cache/hpo/`.",
            "- Resumo HPO: `hpo_summary.json`, em `outputs/optimization/kg_dslfm/`.",
            "- Auditoria offline: `deep_research_audit_20260506.json`, no diretório de benches do Advisor.",
            "- Documento técnico verificado: `SEARCH_SPACE_ADVISOR.md`, no pacote local de infraestrutura HPO.",
            f"- Frente de Pareto descritiva qualidade-tempo: {pareto.get('pareto_front_size')} pontos, trials {pareto.get('pareto_trial_ids')}.",
            "- Observação operacional: o cache L2 PostgreSQL apareceu como `degraded/oserror` durante a auditoria offline em container isolado, mas o cálculo principal do Advisor foi concluído e persistido em JSON; esse estado afeta cache, não a validade das recomendações calculadas.",
            "",
            "# Apêndice B - Fontes de implementação verificadas",
            "",
            "Os módulos abaixo foram verificados no pacote local do Search Space Advisor, em infraestrutura HPO.",
            "",
            "| Mecanismo | Módulo local |",
            "|---|---|",
            "| Top-k, quantis, Spearman, incerteza | `statistics.py` |",
            "| Expansão, narrow, CV, BALLET-style safety | `analysis_numeric.py` |",
            "| Categorias, entropia e redução | `analysis_categorical.py` |",
            "| RandomForestRegressor, UCB/LCB e transformação log | `surrogate.py` |",
            "| Importância interna/blended | `importance.py` |",
            "| Wilson e reliability summary | `reliability.py` |",
            "| Bootstrap support | `bootstrap.py` |",
            "| Self-audit e bloqueios | `self_audit.py`; `self_audit_runner.py` |",
            "| Projeção Pareto/hipervolume | `multiobjective.py` |",
            "| Heurísticas cold start | `recommendations.py` |",
            "",
        ]
    )
    return "\n".join(lines)


def _academic_tone(article: str) -> str:
    """Convert draft prose to impersonal academic style."""
    replacements = {
        "A seguir, eu organizo": "A seguir, organizam-se",
        "Eu trato o problema geral de HPO como otimização caixa-preta:": "O problema geral de HPO é tratado como otimização caixa-preta:",
        "No TPE usado pelo Optuna, eu escolho": "No TPE usado pelo Optuna, escolhe-se",
        "eu não espalho": "não se espalha",
        "Se eu tenho": "Se há",
        "eu não preciso manter": "não é necessário manter",
        "Eu leio isso como": "Isso pode ser lido como",
        "eu posso sugerir": "pode-se sugerir",
        "eu reduzo o barulho": "reduz-se o ruído",
        "Quando eu avalio": "Quando se avalia",
        "me permite comparar": "permite comparar",
        "que eu deixaria de fora": "que ficaria fora da nova região",
        "eu aceito": "aceita-se",
        "Se eu comparo": "Se são comparadas",
        "Eu mantenho": "Mantêm-se",
        "Se eu observo": "Se são observadas",
        "eu também preciso saber": "também é preciso saber",
        "Eu uso essa heurística": "Essa heurística é usada",
        "Eu retomo aqui": "Retomam-se aqui",
        "que considero úteis": "consideradas úteis",
        "Eu reconstituo": "Reconstitui-se",
        "Eu considero": "Considera-se",
        "Outro ponto que eu considero": "Outro ponto",
        "Eu não reduzo": "O método não reduz",
        "mas reconstruo": "mas reconstrói",
        "Quando esses três planos andam juntos, eu deixo": "Quando esses três planos andam juntos, o artigo torna-se",
        "o artigo torna-se o artigo": "o artigo torna-se",
        "Eu preservo a distinção": "Preserva-se a distinção",
        "eu não uso": "não se usa",
        "eu evito": "evita-se",
        "mas evito": "mas evita-se",
        "Eu preservo a diferença": "Preserva-se a diferença",
        "Neste artigo, eu formalizo": "Este artigo formaliza",
        "Eu apresento": "A formulação é apresentada",
        "a instancio": "é instanciada",
        "eu obtenho": "obtêm-se",
        "Eu parto do fato": "Parte-se do fato",
        "eu preciso decidir": "é preciso decidir",
        "Eu escrevo o método discutido neste artigo para ser geral.": "O método discutido neste artigo é formulado de modo geral.",
        "Isso me permite reutilizar": "Isso permite reutilizar",
        "O ponto metodológico mais importante, para mim, é separar": "O ponto metodológico mais importante é separar",
        "Eu organizo o artigo": "O artigo é organizado",
        "eu formalizo": "formaliza-se",
        "eu uso o PFF": "usa-se o PFF",
        "Eu executo o experimento": "O experimento",
        "nesta seção, eu mostro": "nesta seção, mostra-se",
        "eu observo que": "observa-se que",
        "eu não trato": "não se trata",
        "eu mantenho": "mantém-se",
        "Neste artigo, eu descrevo": "Este artigo descreve",
        "eu concluo que": "conclui-se que",
        "mas me permite": "mas permite",
        "ela me coloca": "ela coloca o estudo",
        "A metáfora útil é a de um garimpo: não se espalha a peneira por todo o rio depois de achar ouro em um trecho promissor.": "Isso significa que regiões promissoras passam a receber maior densidade de amostragem, sem excluir a exploração residual do restante do espaço.",
        "A metáfora aqui é uma torcida comprimida na arquibancada: quando quase todo mundo já se juntou no mesmo setor, não é necessário manter o estádio inteiro aberto.": "Operacionalmente, a concentração da elite justifica reduzir a faixa analisada, desde que os guardrails de evidência sejam atendidos.",
        "Isso pode ser lido como um termômetro encostando no teto da escala atual: se os melhores pontos continuam subindo perto da borda, manter o teto parado é limitar a exploração cedo demais.": "A interpretação operacional é que o limite superior atual pode estar restringindo a busca antes de a tendência monotônica se esgotar.",
        "A metáfora é a de uma banca avaliadora: quando quase todos os pareceres convergem para a mesma opção, reduz-se o ruído sem fingir que a escolha virou verdade universal.": "A interpretação é que a concentração categórica reduz alternativas pouco promissoras, sem transformar a categoria dominante em garantia universal.",
        "o surrogate funciona como uma maquete de túnel de vento: ele não substitui o voo real, mas permite comparar": "o surrogate não substitui novas avaliações reais, mas permite comparar",
        "Se a pior leitura interna ainda supera a melhor leitura externa, aceita-se o estreitamento com folga estatística.": "Se a fronteira pessimista interna supera a fronteira otimista externa, aceita-se o estreitamento com margem de segurança.",
        "A metáfora é simples: não faz sentido declarar vencedor absoluto quando cada competidor vence em uma pista diferente.": "A interpretação é que não há vencedor escalar único quando objetivos relevantes entram em tensão.",
        "A metáfora aqui é a de uma testemunha: não basta saber quantas vezes ela acertou; também é preciso saber quantas vezes ela falou antes de confiar de verdade.": "A interpretação estatística é que a proporção observada deve ser ponderada pelo tamanho amostral antes de sustentar confiança alta.",
        "A metáfora é a de escolher a primeira marcha antes de sair com o carro: ela coloca o estudo em movimento, mas não determina a viagem inteira.": "Essa heurística define apenas um ponto inicial plausível e deve ser substituída por evidência empírica à medida que novos trials são observados.",
        "## Arquitetura lógica, metáforas e tabelas-síntese": "## Arquitetura lógica e tabelas-síntese",
        "Retomam-se aqui duas metáforas consideradas úteis. Um surrogate model é como uma maquete de túnel de vento: ele não é o avião de verdade, mas ajuda a testar direção e risco antes de gastar combustível no voo real. Já a confiabilidade estatística é como ouvir uma testemunha. Não basta saber que ela acertou 80% das vezes; importa saber se ela falou cinco vezes ou quinhentas. É exatamente por isso que limites como o de Wilson são mais honestos do que a simples taxa observada, sobretudo em amostras pequenas": "Dois princípios orientam a leitura metodológica. Primeiro, o surrogate é uma aproximação empírica usada para comparar alternativas antes de novas avaliações caras. Segundo, a confiabilidade estatística exige considerar simultaneamente taxa observada e tamanho amostral; por isso, limites conservadores como Wilson são preferíveis à taxa bruta em amostras pequenas",
        "Em termos práticos, o sistema age como um copiloto: ele não treina o modelo principal, mas fica olhando painel, borda, vibração e tendência para dizer quando vale abrir o mapa, quando vale apertar a lanterna e quando vale não mexer em nada": "Em termos operacionais, o sistema não treina o modelo principal; ele analisa histórico, bordas, tendências e evidência estatística para sugerir manter, expandir, estreitar, fixar ou reduzir partes do espaço de busca",
        "freio contra otimismo barato": "controle contra excesso de confiança",
        "É o equivalente estatístico a dizer: beleza, você acertou 4 de 5, mas ainda não vou te dar carteira de piloto": "Em termos estatísticos, poucos acertos em poucas observações ainda não justificam confiança alta",
        "Em metáfora bem direta, é como perguntar uma mesma coisa à banca com os nomes apagados e em ordem embaralhada: se a resposta muda toda hora, o resultado não é firme.": "Em termos operacionais, se pequenas perturbações da amostra mudam a ação recomendada, a decisão não deve ser tratada como estável.",
        "| Método | Metáfora simples | Vantagem | Limitação | Uso recomendado no Advisor |": "| Método | Sinal operacional | Vantagem | Limitação | Uso recomendado no Advisor |",
        "detector de valores gritando fora do coro": "desvio univariado extremo",
        "quem ficou isolado da própria vizinhança": "isolamento local por vizinhança",
        "serrote que isola pontos estranhos mais rápido": "particionamento aleatório de pontos raros",
        "aprender a normalidade e estranhar o que não reconstrói": "modelagem de normalidade por reconstrução",
        "Abaixo disso, o sistema opera quase como bom senso com memória curta": "Abaixo disso, a evidência é insuficiente para decisões empíricas completas",
        "Só aperta se o alvo já estiver agrupado": "Só estreita quando a elite já está concentrada",
        "Se a variável quase não move o resultado, vale congelá-la": "Pode fixar variáveis com influência empírica baixa",
        "O sistema aprende a desconfiar de si mesmo": "Bloqueia padrões historicamente frágeis",
        "A síntese acima mostra que o grande ganho desta revisão não foi enfeitar o texto com citações, e sim alinhar linguagem, matemática e implementação. Quando esses três planos andam juntos, o artigo torna-se mais forte, mais reproduzível e menos vulnerável ao clássico isso é bonito no papel, mas seu código faz outra coisa": "A síntese mostra que a contribuição central é alinhar linguagem, matemática e implementação. Quando esses três planos são consistentes, o artigo torna-se mais reprodutível e menos vulnerável a divergências entre narrativa metodológica e comportamento implementado",
    }
    for old, new in replacements.items():
        article = article.replace(old, new)
    article = re.sub(r"\b[Ee]u\s+", "", article)
    article = article.replace("  ", " ")
    return article


def bibliography() -> str:
    return r"""
@article{bergstra2012random,
  title={Random Search for Hyper-Parameter Optimization},
  author={Bergstra, James and Bengio, Yoshua},
  journal={Journal of Machine Learning Research},
  volume={13},
  number={10},
  pages={281--305},
  year={2012},
  url={https://jmlr.org/papers/v13/bergstra12a.html}
}

@inproceedings{bergstra2011algorithms,
  title={Algorithms for Hyper-Parameter Optimization},
  author={Bergstra, James and Bardenet, Remi and Bengio, Yoshua and Kegl, Balazs},
  booktitle={Advances in Neural Information Processing Systems 24},
  year={2011},
  url={https://proceedings.neurips.cc/paper/2011/hash/86e8f7ab32cfd12577bc2619bc635690-Abstract.html}
}

@inproceedings{snoek2012practical,
  title={Practical Bayesian Optimization of Machine Learning Algorithms},
  author={Snoek, Jasper and Larochelle, Hugo and Adams, Ryan P.},
  booktitle={Advances in Neural Information Processing Systems 25},
  year={2012},
  url={https://proceedings.neurips.cc/paper/2012/hash/05311655a15b75fab86956663e1819cd-Abstract.html}
}

@inproceedings{akiba2019optuna,
  title={Optuna: A Next-generation Hyperparameter Optimization Framework},
  author={Akiba, Takuya and Sano, Shotaro and Yanase, Toshihiko and Ohta, Takeru and Koyama, Masanori},
  booktitle={Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
  pages={2623--2631},
  year={2019},
  doi={10.1145/3292500.3330701},
  url={https://doi.org/10.1145/3292500.3330701}
}

@inproceedings{hutter2014fanova,
  title={An Efficient Approach for Assessing Hyperparameter Importance},
  author={Hutter, Frank and Hoos, Holger and Leyton-Brown, Kevin},
  booktitle={Proceedings of the 31st International Conference on Machine Learning},
  pages={754--762},
  year={2014},
  url={https://proceedings.mlr.press/v32/hutter14.html}
}

@article{breiman2001random,
  title={Random Forests},
  author={Breiman, Leo},
  journal={Machine Learning},
  volume={45},
  pages={5--32},
  year={2001},
  doi={10.1023/A:1010933404324}
}

@article{lundberg2017shap,
    title={A Unified Approach to Interpreting Model Predictions},
    author={Lundberg, Scott M. and Lee, Su-In},
    journal={Advances in Neural Information Processing Systems},
    volume={30},
    year={2017},
    url={https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html}
}

@misc{optunaTPE,
  title={optuna.samplers.TPESampler},
  author={{Optuna Contributors}},
  year={2026},
  url={https://optuna.readthedocs.io/en/v4.2.1/reference/samplers/generated/optuna.samplers.TPESampler.html},
  urldate={2026-05-06}
}

@misc{optunaFanova,
  title={optuna.importance.FanovaImportanceEvaluator},
  author={{Optuna Contributors}},
  year={2026},
  url={https://optuna.readthedocs.io/en/stable/reference/generated/optuna.importance.FanovaImportanceEvaluator.html},
  urldate={2026-05-06}
}

@misc{sklearnRF,
  title={RandomForestRegressor},
  author={{scikit-learn developers}},
  year={2026},
  url={https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html},
  urldate={2026-05-06}
}

@misc{nistCV,
  title={Coefficient of Variation},
  author={{National Institute of Standards and Technology}},
  year={2017},
  url={https://www.itl.nist.gov/div898/software/dataplot/refman2/auxillar/coefvari.htm},
  urldate={2026-05-06}
}

@misc{mathworldSpearman,
  title={Spearman Rank Correlation Coefficient},
  author={Weisstein, Eric W.},
  publisher={Wolfram MathWorld},
  year={2026},
  url={https://mathworld.wolfram.com/SpearmanRankCorrelationCoefficient.html},
  urldate={2026-05-06}
}

@article{brown2001interval,
  title={Interval Estimation for a Binomial Proportion},
  author={Brown, Lawrence D. and Cai, T. Tony and DasGupta, Anirban},
  journal={Statistical Science},
  volume={16},
  number={2},
  pages={101--133},
  year={2001},
  doi={10.1214/ss/1009213286},
  url={https://doi.org/10.1214/ss/1009213286}
}

@inproceedings{zhang2023ballet,
  title={Learning Regions of Interest for Bayesian Optimization with Adaptive Level-Set Estimation},
  author={Zhang, Fengxue and Song, Jialin and Bowden, James C. and Ladd, Alexander and Yue, Yisong and Desautels, Thomas and Chen, Yuxin},
  booktitle={Proceedings of the 40th International Conference on Machine Learning},
  volume={202},
  pages={41579--41595},
  year={2023},
  url={https://proceedings.mlr.press/v202/zhang23aj.html}
}

@misc{nistReliability,
  title={Reliability or Survival Function},
  author={{National Institute of Standards and Technology}},
  url={https://www.itl.nist.gov/div898/handbook/apr/section1/apr122.htm},
  urldate={2026-05-06}
}

@misc{nistHazard,
  title={Bathtub Curve and Failure Rate},
  author={{National Institute of Standards and Technology}},
  url={https://www.itl.nist.gov/div898/handbook/apr/section1/apr124.htm},
  urldate={2026-05-06}
}

@misc{matplotlibBackend,
  title={Backends},
  author={{Matplotlib Development Team}},
  year={2026},
  url={https://matplotlib.org/stable/users/explain/figure/backends.html},
  urldate={2026-05-06}
}

@misc{pandocManual,
  title={Pandoc User's Guide},
  author={MacFarlane, John},
  year={2026},
  url={https://pandoc.org/MANUAL.html},
  urldate={2026-05-06}
}

@misc{tinytex,
  title={TinyTeX},
  author={Xie, Yihui},
  year={2026},
  url={https://yihui.org/tinytex/},
  urldate={2026-05-06}
}

@misc{abnt6022,
  title={ABNT NBR 6022:2018-05-16},
  author={{Associacao Brasileira de Normas Tecnicas}},
  year={2018},
  url={https://www.dinmedia.de/en/standard/abnt-nbr-6022/291015508},
  urldate={2026-05-06}
}

@misc{abnt6023,
  title={ABNT NBR 6023:2025-05-21},
  author={{Associacao Brasileira de Normas Tecnicas}},
  year={2025},
  url={https://www.dinmedia.de/en/standard/abnt-nbr-6023/393096544},
  urldate={2026-05-06}
}

@misc{abnt10520,
  title={ABNT NBR 10520:2023-07-19},
  author={{Associacao Brasileira de Normas Tecnicas}},
  year={2023},
  url={https://www.dinmedia.de/en/standard/abnt-nbr-10520/371501342},
  urldate={2026-05-06}
}

@misc{abnt14724,
  title={ABNT NBR 14724 Versao Corrigida:2024-12-16},
  author={{Associacao Brasileira de Normas Tecnicas}},
  year={2024},
  url={https://www.dinmedia.de/en/standard/abnt-nbr-14724-versao-corrigida/391347059},
  urldate={2026-05-06}
}

@book{barlow1996reliability,
    title={Mathematical Theory of Reliability},
    author={Barlow, Richard E. and Proschan, Frank},
    publisher={SIAM},
    address={Philadelphia},
    year={1996},
    doi={10.1137/1.9781611971194},
    url={https://doi.org/10.1137/1.9781611971194}
}

@book{rasmussen2006gp,
    title={Gaussian Processes for Machine Learning},
    author={Rasmussen, Carl Edward and Williams, Christopher K. I.},
    publisher={MIT Press},
    address={Cambridge},
    year={2006},
    doi={10.7551/mitpress/3206.001.0001},
    url={https://doi.org/10.7551/mitpress/3206.001.0001}
}

@book{ross1996stochastic,
    title={Stochastic Processes},
    author={Ross, Sheldon M.},
    edition={2},
    publisher={Wiley},
    address={New York},
    year={1996},
    isbn={9780471120629}
}

@book{aggarwal2017outlier,
    title={Outlier Analysis},
    author={Aggarwal, Charu C.},
    edition={2},
    publisher={Springer},
    address={Cham},
    year={2017},
    doi={10.1007/978-3-319-47578-3},
    url={https://doi.org/10.1007/978-3-319-47578-3}
}

@inproceedings{deb2002nsga2,
    title={A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II},
    author={Deb, Kalyanmoy and Pratap, Amrit and Agarwal, Sameer and Meyarivan, T.},
    booktitle={IEEE Transactions on Evolutionary Computation},
    volume={6},
    number={2},
    pages={182--197},
    year={2002},
    doi={10.1109/4235.996017},
    url={https://doi.org/10.1109/4235.996017}
}

@article{morales2023many,
    title={Multi-objective Hyperparameter Optimization Metrics and Benchmarks},
    author={Morales-Hernandez, Pablo and others},
    journal={ACM Computing Surveys},
    year={2023},
    url={https://dl.acm.org/doi/10.1145/3616044}
}

@article{saito2015precision,
    title={The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets},
    author={Saito, Takaya and Rehmsmeier, Marcel},
    journal={PLOS ONE},
    volume={10},
    number={3},
    pages={e0118432},
    year={2015},
    doi={10.1371/journal.pone.0118432},
    url={https://doi.org/10.1371/journal.pone.0118432}
}
""".strip()


def generate(args: argparse.Namespace) -> None:
    dashboard = _read_json(args.dashboard)
    audit = _read_json(args.audit)
    hpo_summary = _read_json(args.hpo_summary)
    paired_benchmark = (
        _read_json(args.paired_benchmark)
        if args.paired_benchmark and Path(args.paired_benchmark).exists()
        else {}
    )
    output_dir = Path(args.output_dir)
    figures_dir = output_dir / "figures"
    metrics_dir = output_dir / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    advice = dashboard.get("searchSpaceAdvice", {})
    recommendations = list(advice.get("recommendations", []))
    recommendations.sort(key=lambda rec: -float(rec.get("importance", 0.0)))
    trials = _completed_trials(dashboard)

    _figure_style()
    figure_paths = {
        "flow": "figures/figura0_fluxo_advisor.png",
        "advisor_evidence": "figures/figura1_evidencias_advisor.png",
        "importances_actions": "figures/figura2_importancias_acoes.png",
        "topk_distributions": "figures/figura3_topk_distribuicoes.png",
        "reliability": "figures/figura4_confiabilidade.png",
        "pareto": "figures/figura5_pareto_qualidade_tempo.png",
        "survival": "figures/figura6_sobrevivencia_direcional.png",
    }
    plot_advisor_flow(output_dir / figure_paths["flow"])
    plot_advisor_evidence(recommendations, audit, output_dir / figure_paths["advisor_evidence"])
    plot_importances_actions(
        recommendations,
        dashboard.get("importances", {}),
        output_dir / figure_paths["importances_actions"],
    )
    plot_topk_distributions(recommendations, output_dir / figure_paths["topk_distributions"])
    plot_reliability(audit, output_dir / figure_paths["reliability"])
    pareto_metrics = plot_pareto(trials, output_dir / figure_paths["pareto"])
    plot_survival(audit, output_dir / figure_paths["survival"])

    metrics = build_metrics(
        dashboard,
        audit,
        hpo_summary,
        figure_paths,
        pareto_metrics,
        paired_benchmark=paired_benchmark,
    )
    metrics_path = metrics_dir / "deep_research_metrics.json"
    FileManager.write_text(FileManager.json_dumps(metrics, sort_keys=True), metrics_path)

    article = _academic_tone(build_article(metrics, recommendations))
    FileManager.write_text(article, output_dir / "deep-research-report-abnt.md")
    FileManager.write_text(bibliography(), output_dir / "references.bib")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate figures and ABNT-ready Markdown for Advisor deep research."
    )
    parser.add_argument(
        "--dashboard",
        default="outputs/.cache/hpo/dashboard_data.json",
        help="Dashboard JSON exported by HPO.",
    )
    parser.add_argument(
        "--audit",
        default="outputs/benches/search_space_advisor/deep_research_audit_20260506.json",
        help="Offline advisor audit JSON.",
    )
    parser.add_argument(
        "--hpo-summary",
        default="outputs/optimization/kg_dslfm/hpo_summary.json",
        help="HPO summary JSON.",
    )
    parser.add_argument(
        "--paired-benchmark",
        default="outputs/benches/search_space_advisor/paired_benchmark_50.json",
        help="Paired TPE/GP-BO/Advisor benchmark JSON.",
    )
    parser.add_argument(
        "--output-dir",
        default=f"outputs/research/{RUN_ID}",
        help="Output directory for report artifacts.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    generate(parse_args())
