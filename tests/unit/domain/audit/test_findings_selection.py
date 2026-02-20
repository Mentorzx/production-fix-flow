"""Regression tests for findings/root-causes selection complexity optimizations."""

from __future__ import annotations

from pff.domain.audit.findings import neuro_symbolic_scores_to_findings
from pff.domain.audit.root_causes import select_root_causes


def test_neuro_symbolic_selection_returns_lowest_p_values_first() -> None:
    scored = [
        {"relation": "r1", "evt_p_value": 0.30, "p_calibrated": 0.2},
        {"relation": "r2", "evt_p_value": 0.10, "p_calibrated": 0.1},
        {"relation": "r3", "evt_p_value": 0.20, "p_calibrated": 0.3},
    ]

    findings = neuro_symbolic_scores_to_findings(
        scored,
        p_value_warning=0.5,
        p_value_error=0.15,
        max_findings=2,
    )
    assert len(findings) == 2
    assert findings[0]["evidence"]["evt_p_value"] == 0.10
    assert findings[1]["evidence"]["evt_p_value"] == 0.20


def test_root_causes_selects_top_k_by_delta_risk() -> None:
    findings = [
        {
            "severity": "error",
            "layer": "schema",
            "json_pointer": "/a",
            "evidence": {"evt_p_value": 0.5},
        },
        {
            "severity": "warning",
            "layer": "profile",
            "json_pointer": "/b",
            "evidence": {"evt_p_value": 0.01},
        },
        {
            "severity": "warning",
            "layer": "profile",
            "json_pointer": "/a",
            "evidence": {"evt_p_value": 0.6},
        },
    ]

    causes = select_root_causes(findings, max_causes=1)
    assert len(causes) == 1
    assert causes[0]["json_pointer"] == "/b"
