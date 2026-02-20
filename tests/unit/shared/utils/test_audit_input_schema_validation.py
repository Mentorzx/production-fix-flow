"""Provide module-level functionality for the PFF codebase.



Notes:

    File: tests/unit/shared/utils/test_audit_input_schema_validation.py

"""

from __future__ import annotations

from pff.domain.audit.input_validation import AuditInputSchemaValidator


def test_audit_input_schema_validation_returns_json_pointers() -> None:
    """Execute test audit input schema validation returns json pointers.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {"a": {"type": "integer"}},
        "required": ["a"],
        "additionalProperties": False,
    }
    validator = AuditInputSchemaValidator(schema=schema)
    report = validator.validate({"a": "x"})

    assert report
    assert report[0]["json_pointer"] == "/a"
    assert report[0]["error_code"] == "json_schema_validation_error"
