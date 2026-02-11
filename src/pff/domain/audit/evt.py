"""EVT/POT utilities for robust anomaly thresholds.

Design patterns:
    - Builder: fits per-relation EVT parameters and returns compact JSON payloads.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from pff.shared import FileManager
from pff.shared.core.config import AUDIT_CONFIG_PATH
from pff.shared.core.config_loader import load_config
from pff_rust import hash_bytes


@dataclass(frozen=True)
class EVTConfig:
    """EVT configuration loaded from `config/audit/audit.yaml`."""

    threshold_quantile: float = 0.95
    min_exceedances: int = 50
    clip_eps: float = 1e-12

    @staticmethod
    def load(file_manager: FileManager | None = None) -> EVTConfig:
        cfg_obj = load_config(AUDIT_CONFIG_PATH)
        if not cfg_obj:
            return EVTConfig()
        audit_cfg = cfg_obj.get("audit", cfg_obj)
        if not isinstance(audit_cfg, dict):
            return EVTConfig()
        evt_cfg = audit_cfg.get("evt", {})
        if not isinstance(evt_cfg, dict):
            return EVTConfig()
        return EVTConfig(
            threshold_quantile=float(evt_cfg.get("threshold_quantile", 0.95)),
            min_exceedances=int(evt_cfg.get("min_exceedances", 50)),
            clip_eps=float(evt_cfg.get("clip_eps", 1e-12)),
        )


def fit_gpd_pot(
    scores: np.ndarray, *, config: EVTConfig | None = None
) -> dict[str, Any] | None:
    """Fit a Generalized Pareto Distribution (GPD) with POT.

    Args:
        scores: 1D array of anomaly scores (higher == more anomalous).
        config: Optional EVTConfig.

    Returns:
        Parameter dict containing threshold `u`, `shape` and `scale`, or None if
        insufficient data to fit.
    """

    cfg = config or EVTConfig.load()
    x = np.asarray(scores, dtype=np.float64).ravel()
    if x.size == 0:
        return None
    q = float(cfg.threshold_quantile)
    q = min(max(q, 0.50), 0.999)
    u = float(np.quantile(x, q))
    exceed = x[x > u] - u
    if exceed.size < int(cfg.min_exceedances):
        return None

    try:
        from scipy.stats import genpareto
    except Exception as exc:
        raise RuntimeError(f"scipy unavailable for EVT fit: {exc}") from exc

    shape, loc, scale = genpareto.fit(exceed, floc=0.0)
    params: dict[str, Any] = {
        "evt_version": 1,
        "threshold_quantile": q,
        "u": u,
        "shape": float(shape),
        "scale": float(scale),
        "n_total": int(x.size),
        "n_exceed": int(exceed.size),
    }
    params["params_hash"] = (
        f"{hash_bytes(FileManager.json_dumps(params, sort_keys=True)):x}"
    )
    return params


def evt_p_value(
    score: float, *, params: dict[str, Any], clip_eps: float = 1e-12
) -> float:
    """Compute an EVT tail p-value for an anomaly score given fitted params."""

    u = float(params["u"])
    if score <= u:
        return 1.0

    shape = float(params["shape"])
    scale = float(params["scale"])

    try:
        from scipy.stats import genpareto
    except Exception as exc:
        raise RuntimeError(f"scipy unavailable for EVT p-value: {exc}") from exc

    excess = float(score - u)
    p = float(genpareto.sf(excess, c=shape, loc=0.0, scale=scale))
    eps = float(clip_eps)
    if eps > 0:
        p = max(eps, min(1.0, p))
    return p


def evt_p_values(
    scores: np.ndarray, *, params: dict[str, Any], clip_eps: float = 1e-12
) -> np.ndarray:
    """Vectorized EVT tail p-values for anomaly scores given fitted params."""

    u = float(params["u"])
    shape = float(params["shape"])
    scale = float(params["scale"])

    scores_arr = np.asarray(scores, dtype=np.float64).ravel()
    p = np.ones_like(scores_arr, dtype=np.float64)
    mask = scores_arr > u
    if not np.any(mask):
        return p

    try:
        from scipy.stats import genpareto
    except Exception as exc:
        raise RuntimeError(f"scipy unavailable for EVT p-values: {exc}") from exc

    excess = scores_arr[mask] - u
    p[mask] = genpareto.sf(excess, c=shape, loc=0.0, scale=scale)

    eps = float(clip_eps)
    if eps > 0:
        p = np.clip(p, eps, 1.0)
    return p


def fit_evt_by_relation(
    *,
    anomaly_scores: np.ndarray,
    relations: np.ndarray,
    config: EVTConfig | None = None,
) -> dict[str, dict[str, Any]]:
    """Fit EVT params per relation (and global fallback) for anomaly scores."""

    cfg = config or EVTConfig.load()
    scores_arr = np.asarray(anomaly_scores, dtype=np.float64).ravel()
    rel_arr = np.asarray(relations).astype(str)

    result: dict[str, dict[str, Any]] = {}

    global_params = fit_gpd_pot(scores_arr, config=cfg)
    if global_params is not None:
        result["__global__"] = global_params

    for relation in sorted(set(rel_arr.tolist())):
        mask = rel_arr == relation
        params = fit_gpd_pot(scores_arr[mask], config=cfg)
        if params is None:
            continue
        result[relation] = params

    return result
