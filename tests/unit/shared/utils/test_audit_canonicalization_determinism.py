"""Provide module-level functionality for the PFF codebase.



Notes:

    File: tests/unit/shared/utils/test_audit_canonicalization_determinism.py

"""

from __future__ import annotations

from pff.domain.audit import (
    build_audit_run_ids,
    canonicalize_json_document,
    records_to_triples,
)


def test_audit_canonicalization_is_deterministic() -> None:
    """Execute test audit canonicalization is deterministic.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    document = {
        "b": 1,
        "a": {"y": [2, 3], "x": True},
        "c": None,
    }
    run_ids = build_audit_run_ids(
        document=document,
        baseline_key={"baseline": 1},
        schema_version="1",
    )

    records_first = canonicalize_json_document(document, document_id=run_ids.document_id)
    records_second = canonicalize_json_document(document, document_id=run_ids.document_id)

    assert [r.json_pointer for r in records_first] == [
        "/a/x",
        "/a/y/0",
        "/a/y/1",
        "/b",
        "/c",
    ]
    assert [r.field_path for r in records_first] == [
        "/a/x",
        "/a/y/*",
        "/a/y/*",
        "/b",
        "/c",
    ]
    assert [r.record_hash for r in records_first] == [r.record_hash for r in records_second]
    assert [r.normalized_value for r in records_first] == [
        r.normalized_value for r in records_second
    ]

    triples_first = records_to_triples(records_first, run_id=run_ids.run_id)
    triples_second = records_to_triples(records_second, run_id=run_ids.run_id)
    assert [t.triple_hash for t in triples_first] == [t.triple_hash for t in triples_second]
