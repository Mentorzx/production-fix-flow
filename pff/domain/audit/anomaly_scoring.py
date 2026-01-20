"""End-to-end anomaly scoring from calibrated probabilities + EVT p-values.

Design patterns:
    - Strategy: selects per-relation/global calibrators and EVT params.
    - Builder: produces per-triple evidence dicts for `audit_report.json`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from pff.domain.audit.calibration import CalibrationConfig, calibrator_from_dict
from pff.domain.audit.evt import EVTConfig, evt_p_values
from pff.shared import FileManager
from pff.shared.core.config import AUDIT_CONFIG_PATH


@dataclass(frozen=True)
class AnomalyScoringConfig:
    """Configuration for anomaly finding thresholds."""

    p_value_warning: float = 0.05
    p_value_error: float = 0.01
    max_findings: int = 200

    @staticmethod
    def load(file_manager: FileManager | None = None) -> AnomalyScoringConfig:
        fm = file_manager or FileManager()
        try:
            cfg_obj = fm.read(AUDIT_CONFIG_PATH, return_native=True)
        except FileNotFoundError:
            return AnomalyScoringConfig()
        if not isinstance(cfg_obj, dict):
            return AnomalyScoringConfig()
        audit_cfg = cfg_obj.get("audit", cfg_obj)
        if not isinstance(audit_cfg, dict):
            return AnomalyScoringConfig()
        anomaly_cfg = audit_cfg.get("anomaly", {})
        if not isinstance(anomaly_cfg, dict):
            return AnomalyScoringConfig()
        return AnomalyScoringConfig(
            p_value_warning=float(anomaly_cfg.get("p_value_warning", 0.05)),
            p_value_error=float(anomaly_cfg.get("p_value_error", 0.01)),
            max_findings=int(anomaly_cfg.get("max_findings", 200)),
        )


def score_with_calibration_and_evt(
    *,
    scores: np.ndarray,
    relations: np.ndarray,
    calibrators_by_relation: dict[str, dict[str, Any]],
    evt_params_by_relation: dict[str, dict[str, Any]],
    calibration_config: CalibrationConfig | None = None,
    evt_config: EVTConfig | None = None,
) -> list[dict[str, Any]]:
    """Compute calibrated probabilities, anomaly scores and EVT p-values.

    Args:
        scores: Raw model scores (higher == more plausible).
        relations: Relation identifiers aligned with `scores`.
        calibrators_by_relation: Output from `fit_per_relation_calibrators(...)`.
        evt_params_by_relation: Output from `fit_evt_by_relation(...)`.
        calibration_config: Optional calibration config.
        evt_config: Optional EVT config.

    Returns:
        List of evidence dicts (one per input score).
    """

    cal_cfg = calibration_config or CalibrationConfig.load()
    evt_cfg = evt_config or EVTConfig.load()

    scores_arr = np.asarray(scores, dtype=np.float64).ravel()
    rel_arr = np.asarray(relations).astype(str)

    if len(scores_arr) != len(rel_arr):
        raise ValueError("scores and relations must have the same length")

    global_cal_payload = calibrators_by_relation.get("__global__", {}).get("model")
    if not isinstance(global_cal_payload, dict):
        raise ValueError("Missing global calibrator payload under key='__global__'")
    global_cal = calibrator_from_dict(global_cal_payload)

    global_evt = evt_params_by_relation.get("__global__")
    if global_evt is None:
        raise ValueError("Missing global EVT params under key='__global__'")

    n_samples = scores_arr.size
    out_p_calibrated = np.zeros(n_samples, dtype=np.float64)
    out_anomaly_scores = np.zeros(n_samples, dtype=np.float64)
    out_evt_p_values = np.zeros(n_samples, dtype=np.float64)

    if n_samples:
        unique_rel, rel_codes = np.unique(rel_arr, return_inverse=True)
        order = np.argsort(rel_codes)
        sorted_codes = rel_codes[order]
        boundaries = np.flatnonzero(sorted_codes[1:] != sorted_codes[:-1]) + 1
        starts = np.concatenate(([0], boundaries))
        ends = np.concatenate((boundaries, [order.size]))

        for start, end in zip(starts, ends):
            code = sorted_codes[start]
            relation = str(unique_rel[code])

            cal_payload = calibrators_by_relation.get(relation, {}).get("model")
            calibrator = (
                calibrator_from_dict(cal_payload) if isinstance(cal_payload, dict) else global_cal
            )

            evt_params = evt_params_by_relation.get(relation) or global_evt

            group_indices = order[start:end]
            group_scores = scores_arr[group_indices]

            probs = calibrator.transform(group_scores)
            probs = np.clip(probs, float(cal_cfg.clip_eps), 1.0 - float(cal_cfg.clip_eps))
            anomaly_scores = -np.log(probs)
            p_vals = evt_p_values(
                anomaly_scores, params=evt_params, clip_eps=float(evt_cfg.clip_eps)
            )

            out_p_calibrated[group_indices] = probs
            out_anomaly_scores[group_indices] = anomaly_scores
            out_evt_p_values[group_indices] = p_vals

    if n_samples == 0:
        return []

    results = _build_results_vectorized(
        rel_arr, scores_arr, out_p_calibrated, out_anomaly_scores, out_evt_p_values
    )

    if not results:
        raise RuntimeError("Failed to score all items (empty result)")
    return results


def _build_results_vectorized(
    rel_arr: np.ndarray,
    scores_arr: np.ndarray,
    p_calibrated: np.ndarray,
    anomaly_scores: np.ndarray,
    evt_p_values: np.ndarray,
) -> list[dict[str, Any]]:
    """Build result dicts using vectorized operations for 2-4x speedup.

    Avoids per-element Python float()/str() calls by converting arrays in batch.
    """
    relations_list = rel_arr.tolist()
    scores_list = scores_arr.tolist()
    p_cal_list = p_calibrated.tolist()
    anom_list = anomaly_scores.tolist()
    evt_list = evt_p_values.tolist()

    return [
        {
            "relation": r,
            "score": s,
            "p_calibrated": p,
            "anomaly_score": a,
            "evt_p_value": e,
        }
        for r, s, p, a, e in zip(relations_list, scores_list, p_cal_list, anom_list, evt_list)
    ]
