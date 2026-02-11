"""Findings builders for audit_report.json.

Design patterns:
    - Builder: converts low-level reports (schema/profile) into `findings[]`
      entries that conform to `config/audit/audit_report.schema.v1.json`.
"""

from __future__ import annotations

from typing import Any


def schema_report_to_findings(
    schema_report: list[dict[str, Any]],
    *,
    max_findings: int = 500,
) -> list[dict[str, Any]]:
    """Convert a schema validation report into audit findings."""

    findings: list[dict[str, Any]] = []
    for item in schema_report[: max(0, int(max_findings))]:
        if not isinstance(item, dict):
            continue
        json_pointer = item.get("json_pointer", "")
        message = item.get("message") or "JSON Schema validation error"
        findings.append(
            {
                "severity": "error",
                "layer": "schema",
                "message": str(message),
                "json_pointer": str(json_pointer),
                "evidence": {
                    "validator": item.get("validator"),
                    "validator_value": item.get("validator_value"),
                    "error_code": item.get("error_code"),
                },
                "broken_invariants": [{"name": "json_schema"}],
            }
        )
    return findings


def drift_to_findings(
    drift_report: dict[str, Any],
    *,
    thresholds: dict[str, float],
    max_findings: int = 500,
) -> list[dict[str, Any]]:
    """Convert a drift report into audit findings using configured thresholds."""

    fields = drift_report.get("fields", {})
    if not isinstance(fields, dict):
        return []

    psi_warn = float(thresholds.get("psi_warning", 0.10))
    psi_err = float(thresholds.get("psi_error", 0.25))
    js_warn = float(thresholds.get("js_warning", 0.10))
    js_err = float(thresholds.get("js_error", 0.25))
    miss_warn = float(thresholds.get("missing_delta_warning", 0.05))
    miss_err = float(thresholds.get("missing_delta_error", 0.10))
    min_count = int(thresholds.get("min_count", 0))

    candidates: list[tuple[str, dict[str, Any]]] = []
    for field_path, entry in fields.items():
        if not isinstance(entry, dict):
            continue
        candidates.append((str(field_path), entry))

    candidates.sort(key=lambda kv: kv[0])

    findings: list[dict[str, Any]] = []
    for field_path, entry in candidates:
        if len(findings) >= max_findings:
            break
        missing_delta = float(entry.get("missing_delta", 0.0))
        psi = entry.get("psi")
        js = entry.get("js")

        metrics: list[tuple[str, float, float, float]] = []
        if isinstance(psi, (int, float)):
            metrics.append(("psi", float(psi), psi_warn, psi_err))
        if isinstance(js, (int, float)):
            metrics.append(("js", float(js), js_warn, js_err))
        metrics.append(("missing_delta", abs(missing_delta), miss_warn, miss_err))

        worst = max(metrics, key=lambda m: m[1])
        name, value, warn_thr, err_thr = worst

        if value < warn_thr:
            continue

        severity = "warning" if value < err_thr else "error"
        findings.append(
            {
                "severity": severity,
                "layer": "profile",
                "message": (
                    f"Drift detected: metric={name} value={value:.6f} field_path={field_path}"
                ),
                "json_pointer": field_path,
                "evidence": {
                    "metric": name,
                    "value": value,
                    "missing_delta": missing_delta,
                    "psi": psi,
                    "js": js,
                    "min_count": min_count,
                },
                "broken_invariants": [{"name": "profile_drift"}],
            }
        )
    return findings


def neuro_symbolic_scores_to_findings(
    scored_items: list[dict[str, Any]],
    *,
    p_value_warning: float,
    p_value_error: float,
    max_findings: int = 200,
) -> list[dict[str, Any]]:
    """Convert scored items (p_calibrated/anomaly_score/evt_p_value) into findings."""

    candidates: list[dict[str, Any]] = []
    for item in scored_items:
        if not isinstance(item, dict):
            continue
        p_val = item.get("evt_p_value")
        if not isinstance(p_val, (int, float)):
            continue
        if float(p_val) <= float(p_value_warning):
            candidates.append(item)

    candidates.sort(key=lambda it: float(it.get("evt_p_value", 1.0)))
    findings: list[dict[str, Any]] = []

    for item in candidates[: max(0, int(max_findings))]:
        p_val = float(item["evt_p_value"])
        severity = "warning" if p_val > float(p_value_error) else "error"
        relation = item.get("relation", "")
        message = (
            "Anomaly detected: "
            f"relation={relation} evt_p_value={p_val:.6g} "
            f"p_calibrated={float(item.get('p_calibrated', 0.0)):.6g}"
        )
        finding: dict[str, Any] = {
            "severity": severity,
            "layer": "neuro_symbolic",
            "message": message,
            "evidence": {
                "relation": relation,
                "p_calibrated": item.get("p_calibrated"),
                "anomaly_score": item.get("anomaly_score"),
                "evt_p_value": p_val,
            },
            "broken_invariants": [{"name": "evt_tail_risk"}],
        }
        json_pointer = item.get("json_pointer")
        if isinstance(json_pointer, str):
            finding["json_pointer"] = json_pointer
        findings.append(finding)

    return findings


def graph_validation_report_to_findings(
    validation_report: list[dict[str, Any]],
    *,
    max_findings: int = 500,
) -> list[dict[str, Any]]:
    """Convert SHACL-like graph validation report entries into audit findings."""

    findings: list[dict[str, Any]] = []
    for item in validation_report[: max(0, int(max_findings))]:
        if not isinstance(item, dict):
            continue
        message = item.get("message") or "Graph constraint violation"
        finding: dict[str, Any] = {
            "severity": "error",
            "layer": "graph",
            "message": str(message),
            "evidence": {
                "focus_node": item.get("focus_node"),
                "result_path": item.get("result_path"),
                "value": item.get("value"),
                "constraint": item.get("constraint"),
            },
            "broken_invariants": [{"name": "graph_constraint"}],
        }
        ptr = item.get("json_pointer")
        if isinstance(ptr, str) and ptr:
            finding["json_pointer"] = ptr
        findings.append(finding)
    return findings
