from __future__ import annotations

from pff.domain.audit.input_validation import AuditInputSchemaValidator
from pff.domain.audit.json_patch import (
    apply_json_patch,
    suggest_repairs_from_schema_report,
)


def test_apply_json_patch_add_replace_remove() -> None:
    doc = {"a": {"b": 1}}
    patched = apply_json_patch(doc, [{"op": "add", "path": "/a/c", "value": 2}])
    assert patched["a"]["c"] == 2

    patched = apply_json_patch(doc, [{"op": "replace", "path": "/a/b", "value": 3}])
    assert patched["a"]["b"] == 3

    patched = apply_json_patch(doc, [{"op": "remove", "path": "/a/b"}])
    assert "b" not in patched["a"]


def test_suggest_repairs_from_schema_report_validates_patch() -> None:
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {"a": {"type": ["integer", "null"]}},
        "required": ["a"],
        "additionalProperties": False,
    }
    doc = {}
    report = AuditInputSchemaValidator(schema=schema).validate(doc)
    assert report

    repairs = suggest_repairs_from_schema_report(document=doc, schema=schema, schema_report=report)
    assert repairs

    patched = apply_json_patch(doc, repairs[0]["ops"])
    assert AuditInputSchemaValidator(schema=schema).validate(patched) == []
