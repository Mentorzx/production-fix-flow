"""Statistical profiling + drift for canonical audit records.

Design patterns:
    - Builder: constructs profile and drift dictionaries with stable keys.

The profile is intentionally small and JSON-serializable so it can be persisted
in PostgreSQL (JSONB) without generating intermediate files.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from pff.domain.audit.canonicalize import CanonicalRecord
from pff.shared import FileManager
from pff.shared.core.config import AUDIT_CONFIG_PATH
from pff.shared.hash import hash_bytes

MIN_ARRAY_SIZE = 2


@dataclass(frozen=True)
class AuditProfileConfig:
    """Configuration for profiling and drift metrics."""

    top_k: int = 10
    num_bins: int = 10
    eps: float = 1e-12
    drift_thresholds: dict[str, float] | None = None

    @staticmethod
    def load(file_manager: FileManager | None = None) -> AuditProfileConfig:
        fm = file_manager or FileManager()
        try:
            cfg_obj = fm.read(AUDIT_CONFIG_PATH, return_native=True)
        except FileNotFoundError:
            return AuditProfileConfig(drift_thresholds={})
        if not isinstance(cfg_obj, dict):
            return AuditProfileConfig(drift_thresholds={})
        audit_cfg = cfg_obj.get("audit", cfg_obj)
        if not isinstance(audit_cfg, dict):
            return AuditProfileConfig(drift_thresholds={})
        profile_cfg = audit_cfg.get("profile", {})
        if not isinstance(profile_cfg, dict):
            return AuditProfileConfig(drift_thresholds={})
        thresholds = profile_cfg.get("drift_thresholds", {})
        if not isinstance(thresholds, dict):
            thresholds = {}
        parsed_thresholds = {}
        for key, value in thresholds.items():
            try:
                parsed_thresholds[str(key)] = float(value)
            except (TypeError, ValueError):
                continue
        return AuditProfileConfig(
            top_k=int(profile_cfg.get("top_k", 10)),
            num_bins=int(profile_cfg.get("num_bins", 10)),
            eps=float(profile_cfg.get("eps", 1e-12)),
            drift_thresholds=parsed_thresholds,
        )


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _ensure_strictly_increasing(edges: np.ndarray) -> np.ndarray:
    fixed = np.asarray(edges, dtype=np.float64).copy()
    if fixed.size == 0:
        return fixed
    min_step = 1e-9
    for idx in range(1, fixed.size):
        if fixed[idx] <= fixed[idx - 1]:
            fixed[idx] = fixed[idx - 1] + min_step
    return fixed


def _numeric_histogram(
    values: np.ndarray,
    *,
    num_bins: int,
    edges: list[float] | None = None,
) -> tuple[list[float], list[int]]:
    if values.size == 0:
        return [], []

    if edges is None:
        num_bins = max(1, int(num_bins))
        probs = np.linspace(0.0, 1.0, num_bins + 1)
        computed = np.quantile(values, probs)
        computed = _ensure_strictly_increasing(computed)
        if computed.size < MIN_ARRAY_SIZE:
            v = float(values[0])
            computed = np.array([v, v + 1e-9], dtype=np.float64)
        counts, used_edges = np.histogram(values, bins=computed)
        return [float(x) for x in used_edges.tolist()], [int(x) for x in counts.tolist()]

    edges_np = _ensure_strictly_increasing(np.asarray(edges, dtype=np.float64))
    if edges_np.size < MIN_ARRAY_SIZE:
        v = float(values[0])
        edges_np = np.array([v, v + 1e-9], dtype=np.float64)
    counts, used_edges = np.histogram(values, bins=edges_np)
    return [float(x) for x in used_edges.tolist()], [int(x) for x in counts.tolist()]


def _psi(p: np.ndarray, q: np.ndarray, *, eps: float) -> float:
    eps = float(eps)
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = (p + eps) / float(p.sum() + eps * p.size)
    q = (q + eps) / float(q.sum() + eps * q.size)
    return float(np.sum((p - q) * np.log(p / q)))


def _js_divergence(p: np.ndarray, q: np.ndarray, *, eps: float) -> float:
    eps = float(eps)
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = (p + eps) / float(p.sum() + eps * p.size)
    q = (q + eps) / float(q.sum() + eps * q.size)
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))
    return float(0.5 * (kl_pm + kl_qm))


def build_profile(
    records: list[CanonicalRecord],
    *,
    config: AuditProfileConfig | None = None,
    numeric_bin_edges_by_field: dict[str, list[float]] | None = None,
) -> dict[str, Any]:
    """Build a statistical profile for canonical leaf records.

    Args:
        records: Canonical leaf records.
        config: Optional profile configuration.

    Returns:
        JSON-serializable profile dict.
    """

    cfg = config or AuditProfileConfig.load()
    grouped: dict[str, list[CanonicalRecord]] = {}
    for rec in records:
        grouped.setdefault(rec.field_path, []).append(rec)

    fields: dict[str, Any] = {}
    for field_path in sorted(grouped.keys()):
        group = grouped[field_path]
        n = len(group)
        type_counts: dict[str, int] = {}
        values: list[str] = []
        numeric_values: list[float] = []
        null_count = 0

        for rec in group:
            type_counts[rec.value_type] = type_counts.get(rec.value_type, 0) + 1
            values.append(rec.normalized_value)
            if rec.value_type == "null":
                null_count += 1
                continue
            if rec.value_type in ("int", "float"):
                num = _safe_float(rec.raw_value)
                if num is not None:
                    numeric_values.append(num)

        unique_count = len(set(values))
        missing_pct = float(null_count) / float(n) if n else 0.0

        top_values: list[dict[str, Any]] = []
        other_count = 0
        if n:
            counts: dict[str, int] = {}
            for v in values:
                counts[v] = counts.get(v, 0) + 1
            sorted_values = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[: cfg.top_k]
            top_value_keys = {k for k, _ in sorted_values}
            other_count = sum(c for v, c in counts.items() if v not in top_value_keys)
            top_values = [
                {"value": v, "count": c, "pct": float(c) / float(n)} for v, c in sorted_values
            ]

        field_entry: dict[str, Any] = {
            "n": n,
            "types": dict(sorted(type_counts.items())),
            "missing_pct": missing_pct,
            "unique_count": unique_count,
            "top_values": top_values,
            "other_count": other_count,
        }

        if numeric_values:
            arr = np.asarray(numeric_values, dtype=np.float64)
            edges_override = None
            if numeric_bin_edges_by_field is not None:
                edges_override = numeric_bin_edges_by_field.get(field_path)
            edges, counts = _numeric_histogram(arr, num_bins=cfg.num_bins, edges=edges_override)
            field_entry["numeric_summary"] = {
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "p05": float(np.quantile(arr, 0.05)),
                "p50": float(np.quantile(arr, 0.50)),
                "p95": float(np.quantile(arr, 0.95)),
            }
            field_entry["numeric_hist"] = {
                "edges": edges,
                "counts": counts,
                "bins": int(cfg.num_bins),
            }

        fields[field_path] = field_entry

    profile = {
        "profile_version": 1,
        "total_records": len(records),
        "total_fields": len(fields),
        "fields": fields,
    }
    profile_hash = f"{hash_bytes(FileManager.json_dumps(profile, sort_keys=True)):x}"
    profile["profile_hash"] = profile_hash
    return profile


def compute_drift(
    *,
    baseline_profile: dict[str, Any],
    current_profile: dict[str, Any],
    config: AuditProfileConfig | None = None,
) -> dict[str, Any]:
    """Compute drift metrics between a baseline and a current profile.

    Args:
        baseline_profile: Baseline profile dict (from build_profile()).
        current_profile: Current profile dict.
        config: Optional profile configuration.

    Returns:
        Drift report dict keyed by field_path.
    """

    cfg = config or AuditProfileConfig.load()
    eps = float(cfg.eps)

    base_fields: dict[str, Any] = (
        baseline_profile.get("fields", {}) if isinstance(baseline_profile, dict) else {}
    )
    cur_fields: dict[str, Any] = (
        current_profile.get("fields", {}) if isinstance(current_profile, dict) else {}
    )

    field_paths = sorted(set(base_fields.keys()) | set(cur_fields.keys()))
    drift_fields: dict[str, Any] = {}

    for field_path in field_paths:
        base_entry = base_fields.get(field_path)
        cur_entry = cur_fields.get(field_path)
        if not isinstance(base_entry, dict) or not isinstance(cur_entry, dict):
            drift_fields[field_path] = {
                "status": ("missing_in_baseline" if base_entry is None else "missing_in_current")
            }
            continue

        base_missing = float(base_entry.get("missing_pct", 0.0))
        cur_missing = float(cur_entry.get("missing_pct", 0.0))
        missing_delta = cur_missing - base_missing

        drift_entry: dict[str, Any] = {"missing_delta": missing_delta}

        base_hist = base_entry.get("numeric_hist")
        cur_hist = cur_entry.get("numeric_hist")
        if isinstance(base_hist, dict) and isinstance(cur_hist, dict):
            base_edges = base_hist.get("edges")
            cur_edges = cur_hist.get("edges")
            if base_edges != cur_edges:
                drift_entry["status"] = "numeric_bins_mismatch"
                drift_fields[field_path] = drift_entry
                continue
            base_counts = np.asarray(base_hist.get("counts", []), dtype=np.float64)
            cur_counts = np.asarray(cur_hist.get("counts", []), dtype=np.float64)
            if base_counts.size and cur_counts.size and base_counts.size == cur_counts.size:
                drift_entry["psi"] = _psi(base_counts, cur_counts, eps=eps)
        else:
            base_top = base_entry.get("top_values", [])
            cur_top = cur_entry.get("top_values", [])
            base_other = int(base_entry.get("other_count", 0))
            cur_other = int(cur_entry.get("other_count", 0))
            if isinstance(base_top, list) and isinstance(cur_top, list):
                base_counts_map = {
                    str(v["value"]): int(v["count"]) for v in base_top if isinstance(v, dict)
                }
                cur_counts_map = {
                    str(v["value"]): int(v["count"]) for v in cur_top if isinstance(v, dict)
                }
                keys = sorted(set(base_counts_map.keys()) | set(cur_counts_map.keys()))
                base_vec = np.array(
                    [base_counts_map.get(k, 0) for k in keys] + [base_other],
                    dtype=np.float64,
                )
                cur_vec = np.array(
                    [cur_counts_map.get(k, 0) for k in keys] + [cur_other],
                    dtype=np.float64,
                )
                drift_entry["js"] = _js_divergence(base_vec, cur_vec, eps=eps)

        drift_fields[field_path] = drift_entry

    drift = {
        "drift_version": 1,
        "baseline_profile_hash": baseline_profile.get("profile_hash"),
        "current_profile_hash": current_profile.get("profile_hash"),
        "fields": drift_fields,
    }
    drift_hash = f"{hash_bytes(FileManager.json_dumps(drift, sort_keys=True)):x}"
    drift["drift_hash"] = drift_hash
    return drift
