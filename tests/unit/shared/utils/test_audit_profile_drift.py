"""Provide module-level functionality for the PFF codebase.



Notes:

    File: tests/unit/shared/utils/test_audit_profile_drift.py

"""

from __future__ import annotations

from pff.domain.audit import canonicalize_json_document
from pff.domain.audit.profile import build_profile, compute_drift


def test_audit_profile_and_drift_are_deterministic() -> None:
    """Execute test audit profile and drift are deterministic.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    baseline_doc = {"a": {"y": [1, 2], "x": True}, "b": "foo"}
    current_doc = {"a": {"y": [100, 200], "x": False}, "b": "bar"}

    baseline_records = canonicalize_json_document(baseline_doc, document_id="doc")
    current_records = canonicalize_json_document(current_doc, document_id="doc")

    baseline_profile = build_profile(baseline_records)
    edges_map = {}
    for field_path, entry in baseline_profile["fields"].items():
        hist = entry.get("numeric_hist")
        if isinstance(hist, dict) and isinstance(hist.get("edges"), list):
            edges_map[str(field_path)] = list(hist["edges"])

    current_profile = build_profile(current_records, numeric_bin_edges_by_field=edges_map)
    drift_first = compute_drift(baseline_profile=baseline_profile, current_profile=current_profile)
    drift_second = compute_drift(baseline_profile=baseline_profile, current_profile=current_profile)

    assert drift_first == drift_second

    num_entry = drift_first["fields"]["/a/y/*"]
    assert "psi" in num_entry or num_entry.get("status") != "numeric_bins_mismatch"


def test_audit_profile_top_values_order_is_stable_on_ties() -> None:
    """Use lexical tie-break to keep top_values deterministic."""

    doc = {"vals": ["b", "a", "c", "a", "b", "d"]}
    records = canonicalize_json_document(doc, document_id="doc")
    profile = build_profile(records)
    top_values = profile["fields"]["/vals/*"]["top_values"]

    assert top_values[0]["value"] == "a"
    assert top_values[1]["value"] == "b"
