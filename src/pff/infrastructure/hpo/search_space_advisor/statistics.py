"""Statistical utilities used by Search Space Advisor."""

from __future__ import annotations

import heapq
import math
import random
from typing import Any, Protocol, TypeVar

from .parsing import normalize_direction


class _HasValue(Protocol):
    value: float


TValue = TypeVar("TValue", bound=_HasValue)


def select_top_k(
    trials: list[TValue],
    direction: str,
    fraction: float = 0.25,
    min_k: int = 3,
) -> list[TValue]:
    """Select top-k trials according to direction."""
    if not trials:
        return []
    norm_direction = normalize_direction(direction)
    k = max(min_k, int(len(trials) * fraction))
    if norm_direction == "maximize":
        return heapq.nlargest(k, trials, key=lambda t: t.value)
    return heapq.nsmallest(k, trials, key=lambda t: t.value)


def estimate_uncertainty(n_trials: int, top_k_count: int) -> float:
    """Heuristic uncertainty score in [0, 1]."""
    if n_trials <= 0 or top_k_count <= 0:
        return 1.0
    trial_factor = min(1.0, n_trials / 30)
    top_k_factor = min(1.0, top_k_count / 20)
    return round(1.0 - (trial_factor * top_k_factor), 3)


def numeric_stats(values: list[float]) -> dict[str, float]:
    """Compute descriptive stats for numeric vectors."""
    if not values:
        return {}
    n = len(values)
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / max(1, n - 1)
    std = math.sqrt(variance)
    sv = sorted(values)

    def _quantile(q: float) -> float:
        idx = q * (n - 1)
        lo = int(idx)
        hi = min(lo + 1, n - 1)
        frac = idx - lo
        return sv[lo] * (1 - frac) + sv[hi] * frac

    return {
        "mean": mean,
        "std": std,
        "min": sv[0],
        "max": sv[-1],
        "q10": _quantile(0.1),
        "q25": _quantile(0.25),
        "q50": _quantile(0.5),
        "q75": _quantile(0.75),
        "q90": _quantile(0.9),
        "count": n,
    }


def rankdata(values: list[float]) -> list[float]:
    """Compute average ranks with tie handling (SciPy-like)."""
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    idx = 0
    while idx < len(indexed):
        end = idx + 1
        while end < len(indexed) and indexed[end][1] == indexed[idx][1]:
            end += 1
        avg_rank = (idx + end - 1) / 2.0 + 1.0
        for pos in range(idx, end):
            ranks[indexed[pos][0]] = avg_rank
        idx = end
    return ranks


def spearman_rho(
    values: list[float],
    scores: list[float],
    *,
    min_points: int,
    rust_fast_fn: Any | None = None,
    rust_min_len: int = 512,
    np_module: Any | None = None,
) -> float | None:
    """Compute Spearman rho without external dependencies."""
    if len(values) != len(scores) or len(values) < min_points:
        return None
    if len(set(values)) < 2 or len(set(scores)) < 2:
        return None
    if (
        rust_fast_fn is not None
        and np_module is not None
        and len(values) >= max(1, int(rust_min_len))
        and callable(rust_fast_fn)
    ):
        try:
            x = np_module.asarray(values, dtype=float)
            y = np_module.asarray(scores, dtype=float)
            rho_fast = rust_fast_fn(x, y, min_points=min_points)
            if isinstance(rho_fast, (int, float)) and math.isfinite(float(rho_fast)):
                return float(rho_fast)
        except Exception:
            pass
    ranks_values = rankdata(values)
    ranks_scores = rankdata(scores)
    mean_values = sum(ranks_values) / len(ranks_values)
    mean_scores = sum(ranks_scores) / len(ranks_scores)
    cov = sum(
        (x - mean_values) * (y - mean_scores)
        for x, y in zip(ranks_values, ranks_scores, strict=False)
    )
    var_values = sum((x - mean_values) ** 2 for x in ranks_values)
    var_scores = sum((y - mean_scores) ** 2 for y in ranks_scores)
    if var_values <= 1e-12 or var_scores <= 1e-12:
        return None
    return float(cov / math.sqrt(var_values * var_scores))


def reservoir_sample(values: list[float], *, k: int, seed: int) -> list[float]:
    """Deterministic-size reservoir sample."""
    if len(values) <= k:
        return list(values)
    rng = random.Random(seed)
    sample = list(values[:k])
    for idx in range(k, len(values)):
        j = rng.randint(0, idx)
        if j < k:
            sample[j] = values[idx]
    return sample


__all__ = [
    "estimate_uncertainty",
    "numeric_stats",
    "rankdata",
    "reservoir_sample",
    "select_top_k",
    "spearman_rho",
]
