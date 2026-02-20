"""Root-cause selection for audit findings.

Design patterns:
    - Builder: constructs `summary.root_causes[]` entries from findings.
"""

from __future__ import annotations

import heapq
import math
from typing import Any


def _severity_weight(severity: str) -> float:
    """Execute severity weight.



    Args:

        severity: Input value used by this callable.



    Returns:

        Return value produced by the callable.

    """

    sev = str(severity).lower()
    if sev == "error":
        return 6.0
    if sev == "warning":
        return 3.0
    return 1.0


def _finding_risk(finding: dict[str, Any]) -> float:
    """Execute finding risk.



    Args:

        finding: Input value used by this callable.



    Returns:

        Return value produced by the callable.

    """

    base = _severity_weight(str(finding.get("severity", "info")))
    evidence = finding.get("evidence")
    if isinstance(evidence, dict):
        p_val = evidence.get("evt_p_value")
        if isinstance(p_val, (int, float)) and 0.0 < float(p_val) <= 1.0:
            return base * (-math.log(float(p_val)))
    return base


def select_root_causes(
    findings: list[dict[str, Any]],
    *,
    max_causes: int = 3,
) -> list[dict[str, Any]]:
    """Select a small set of JSON Pointers that explain most findings.

    This is a deterministic greedy coverage heuristic. It is designed to be
    stable and fast, and to provide actionable pointers even when full causal
    recomputation is unavailable.
    """

    by_pointer: dict[str, dict[str, Any]] = {}
    for finding in findings:
        ptr = finding.get("json_pointer")
        if not isinstance(ptr, str) or not ptr:
            continue
        entry = by_pointer.setdefault(
            ptr,
            {"json_pointer": ptr, "delta_risk": 0.0, "layers_impacted": set()},
        )
        entry["delta_risk"] += float(_finding_risk(finding))
        layer = finding.get("layer")
        if isinstance(layer, str) and layer:
            entry["layers_impacted"].add(layer)

    ordered = heapq.nsmallest(
        max(0, int(max_causes)),
        by_pointer.values(),
        key=lambda e: (-float(e["delta_risk"]), str(e["json_pointer"])),
    )
    selected = []
    for entry in ordered:
        selected.append(
            {
                "json_pointer": entry["json_pointer"],
                "delta_risk": float(entry["delta_risk"]),
                "layers_impacted": sorted(entry["layers_impacted"]),
            }
        )
    return selected
