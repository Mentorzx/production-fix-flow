"""Multi-objective scoring utilities for Search Space Advisor."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import product
from typing import Any

from .parsing import normalize_direction


@dataclass(frozen=True)
class MultiObjectiveProjection:
    """Projected single-score view used by advisor internals."""

    scores: list[float | None]
    metadata: dict[str, Any]


def _coerce_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_trial_vector(trial: dict[str, Any]) -> list[float] | None:
    values = trial.get("values")
    if isinstance(values, (list, tuple)) and values:
        vec = [_coerce_float(v) for v in values]
        if all(v is not None for v in vec):
            return [float(v) for v in vec if v is not None]
    value = _coerce_float(trial.get("value"))
    if value is not None:
        return [value]
    return None


def _normalize_directions(
    fallback_direction: str,
    objective_directions: list[str] | None,
    n_objectives: int,
) -> list[str]:
    fallback = normalize_direction(fallback_direction)
    if not objective_directions:
        return [fallback] * max(1, n_objectives)
    resolved = [normalize_direction(d) for d in objective_directions]
    if len(resolved) >= n_objectives:
        return resolved[:n_objectives]
    return resolved + [fallback] * (n_objectives - len(resolved))


def _dominates(left: tuple[float, ...], right: tuple[float, ...]) -> bool:
    return all(
        left_value >= right_value
        for left_value, right_value in zip(left, right, strict=False)
    ) and any(
        left_value > right_value
        for left_value, right_value in zip(left, right, strict=False)
    )


def _pareto_ranks(points: list[tuple[float, ...]]) -> list[int]:
    if not points:
        return []
    dominates_map: list[set[int]] = [set() for _ in points]
    dominated_by_count = [0] * len(points)

    for i, point_i in enumerate(points):
        for j, point_j in enumerate(points):
            if i == j:
                continue
            if _dominates(point_i, point_j):
                dominates_map[i].add(j)
            elif _dominates(point_j, point_i):
                dominated_by_count[i] += 1

    fronts: list[list[int]] = [[]]
    for idx, count in enumerate(dominated_by_count):
        if count == 0:
            fronts[0].append(idx)

    rank = [0] * len(points)
    front_id = 1
    current = fronts[0]
    while current:
        next_front: list[int] = []
        for i in current:
            rank[i] = front_id
            for j in dominates_map[i]:
                dominated_by_count[j] -= 1
                if dominated_by_count[j] == 0:
                    next_front.append(j)
        front_id += 1
        current = next_front

    for idx, r in enumerate(rank):
        if r <= 0:
            rank[idx] = front_id
    return rank


def _hypervolume(points: list[tuple[float, ...]], dims: int) -> float:
    if not points:
        return 0.0
    if dims <= 0:
        return 0.0

    axes: list[list[float]] = []
    for dim in range(dims):
        coords = {0.0}
        for point in points:
            coords.add(max(0.0, min(1.0, float(point[dim]))))
        sorted_coords = sorted(coords)
        if len(sorted_coords) < 2:
            sorted_coords = [0.0, max(sorted_coords[0], 1.0)]
        axes.append(sorted_coords)

    volume = 0.0
    ranges = [range(len(axis) - 1) for axis in axes]
    for index in product(*ranges):
        lows = [axes[d][idx] for d, idx in enumerate(index)]
        highs = [axes[d][idx + 1] for d, idx in enumerate(index)]
        widths = [highs[d] - lows[d] for d in range(dims)]
        if any(w <= 0 for w in widths):
            continue
        center = tuple((lows[d] + highs[d]) * 0.5 for d in range(dims))
        dominated = any(all(p[d] >= center[d] for d in range(dims)) for p in points)
        if dominated:
            cell_volume = 1.0
            for width in widths:
                cell_volume *= width
            volume += cell_volume
    return float(volume)


def build_multiobjective_projection(
    trials_data: list[dict[str, Any]],
    *,
    fallback_direction: str,
    objective_directions: list[str] | None,
) -> MultiObjectiveProjection:
    """Build scalar trial scores from single/multi-objective trial payloads."""
    vectors: list[list[float] | None] = [_extract_trial_vector(t) for t in trials_data]
    n_objectives = max((len(v) for v in vectors if v), default=1)
    directions = _normalize_directions(
        fallback_direction, objective_directions, n_objectives
    )

    adjusted: list[list[float | None]] = []
    for vector in vectors:
        if not vector:
            adjusted.append([None] * n_objectives)
            continue
        arr: list[float | None] = []
        for idx in range(n_objectives):
            if idx >= len(vector):
                arr.append(None)
                continue
            raw = float(vector[idx])
            arr.append(raw if directions[idx] == "maximize" else -raw)
        adjusted.append(arr)

    mins = [math.inf] * n_objectives
    maxs = [-math.inf] * n_objectives
    for row in adjusted:
        for idx, value in enumerate(row):
            if value is None:
                continue
            mins[idx] = min(mins[idx], value)
            maxs[idx] = max(maxs[idx], value)

    normalized: list[list[float | None]] = []
    for row in adjusted:
        out: list[float | None] = []
        for idx, value in enumerate(row):
            if value is None:
                out.append(None)
                continue
            low = mins[idx]
            high = maxs[idx]
            if not math.isfinite(low) or not math.isfinite(high):
                out.append(None)
                continue
            if abs(high - low) <= 1e-12:
                out.append(0.5)
            else:
                out.append((value - low) / (high - low))
        normalized.append(out)

    scalar_scores: list[float | None] = []
    for row in normalized:
        vals = [v for v in row if isinstance(v, (int, float))]
        scalar_scores.append(float(sum(vals)) / float(len(vals)) if vals else None)

    if n_objectives <= 1:
        return MultiObjectiveProjection(
            scores=scalar_scores,
            metadata={
                "objective_count": 1,
                "objective_directions": directions,
                "multiobjective_mode": "single_objective",
                "pareto_front_size": 0,
                "hypervolume": None,
                "hypervolume_computed": False,
            },
        )

    complete_indices: list[int] = []
    complete_points: list[tuple[float, ...]] = []
    for idx, row in enumerate(normalized):
        if all(isinstance(v, (int, float)) for v in row):
            complete_indices.append(idx)
            complete_points.append(tuple(float(v) for v in row if v is not None))

    if len(complete_points) < 2:
        return MultiObjectiveProjection(
            scores=scalar_scores,
            metadata={
                "objective_count": n_objectives,
                "objective_directions": directions,
                "multiobjective_mode": "scalarized_fallback",
                "pareto_front_size": 0,
                "hypervolume": None,
                "hypervolume_computed": False,
            },
        )

    ranks = _pareto_ranks(complete_points)
    front_indices = [idx for idx, rank in enumerate(ranks) if rank == 1]
    front_points = [complete_points[idx] for idx in front_indices]

    hv_computed = False
    total_hv = None
    contrib_norm: dict[int, float] = {}
    if n_objectives <= 3 and 1 <= len(front_points) <= 64:
        hv_computed = True
        total_hv = _hypervolume(front_points, n_objectives)
        contributions: list[float] = []
        for point_idx in range(len(front_points)):
            without = [p for i, p in enumerate(front_points) if i != point_idx]
            hv_without = _hypervolume(without, n_objectives)
            contributions.append(max(0.0, total_hv - hv_without))
        max_contrib = max(contributions) if contributions else 0.0
        for local_idx, value in enumerate(contributions):
            global_idx = complete_indices[front_indices[local_idx]]
            contrib_norm[global_idx] = (
                float(value / max_contrib) if max_contrib > 1e-12 else 0.0
            )

    final_scores = list(scalar_scores)
    complete_rank_by_global = {
        complete_indices[idx]: rank for idx, rank in enumerate(ranks)
    }
    for global_idx, rank in complete_rank_by_global.items():
        base_score = final_scores[global_idx]
        base = float(base_score) if base_score is not None else 0.0
        hv_bonus = contrib_norm.get(global_idx, 0.0)
        rank_penalty = max(0, rank - 1)
        final_scores[global_idx] = float(
            base + (0.15 * hv_bonus) - (0.05 * rank_penalty)
        )

    return MultiObjectiveProjection(
        scores=final_scores,
        metadata={
            "objective_count": n_objectives,
            "objective_directions": directions,
            "multiobjective_mode": (
                "pareto_hypervolume" if hv_computed else "pareto_scalarized"
            ),
            "pareto_front_size": len(front_points),
            "hypervolume": round(float(total_hv), 6) if total_hv is not None else None,
            "hypervolume_computed": hv_computed,
        },
    )


__all__ = [
    "MultiObjectiveProjection",
    "build_multiobjective_projection",
]
