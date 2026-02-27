"""Policy versioning helpers for Search Space Advisor decisions."""

from __future__ import annotations

from typing import Any

from pff.shared import stable_hash

POLICY_VERSION = "1.0.0"


def build_policy_metadata(
    *,
    advisor_version: str,
    direction: str,
    effective_cfg: dict[str, Any],
    decision_thresholds: dict[str, Any],
) -> dict[str, Any]:
    """Build deterministic policy metadata with stable hash."""
    normalized_cfg = {str(k): effective_cfg[k] for k in sorted(effective_cfg)}
    normalized_thresholds = {
        str(k): decision_thresholds[k] for k in sorted(decision_thresholds)
    }
    signature = {
        "policy_version": POLICY_VERSION,
        "advisor_version": str(advisor_version),
        "direction": str(direction),
        "config": normalized_cfg,
        "thresholds": normalized_thresholds,
    }
    digest_int = int(stable_hash(signature, truncate=96))
    policy_hash = f"{digest_int & ((1 << 96) - 1):024x}"[:24]
    return {
        "policy_version": POLICY_VERSION,
        "policy_hash": policy_hash,
        "policy_thresholds": normalized_thresholds,
    }


def policy_stub(*, version: str, policy_hash: str) -> dict[str, str]:
    """Build compact policy ref for per-recommendation traceability."""
    return {"version": str(version), "hash": str(policy_hash)}


__all__ = [
    "POLICY_VERSION",
    "build_policy_metadata",
    "policy_stub",
]
