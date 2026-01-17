from __future__ import annotations

from pathlib import Path

from pff.shared.core.file_manager import FileManager
from pff.domain.audit.report import AuditReportBuilder
from pff.domain.audit.schema import AuditReportSchemaValidator


def test_audit_report_schema_smoke_is_deterministic() -> None:
    fm = FileManager()
    outputs_root = Path("outputs") / "temp_tests"
    fm.ensure_dir(outputs_root)

    schema_path = Path("config") / "audit" / "audit_report.schema.v1.json"
    validator = AuditReportSchemaValidator(schema_path=schema_path, file_manager=fm)

    builder = AuditReportBuilder(
        outputs_dir=outputs_root, schema_validator=validator, file_manager=fm
    )

    document = {"id": 123, "payload": {"x": 1, "y": "abc"}}
    baseline_key = {"name": "unit_test", "window": "synthetic"}
    schema_version = "v1"
    findings = [
        {
            "severity": "warning",
            "layer": "schema",
            "message": "Example warning for schema smoke test",
            "json_pointer": "/payload/x",
        }
    ]

    report1, ids1, paths1 = builder.build_report(
        document=document,
        baseline_key=baseline_key,
        schema_version=schema_version,
        findings=findings,
        meta_overrides={"source_system": "pytest"},
    )
    report2, ids2, paths2 = builder.build_report(
        document=document,
        baseline_key=baseline_key,
        schema_version=schema_version,
        findings=findings,
        meta_overrides={"source_system": "pytest"},
    )

    assert report1["schema_version"] == 1
    assert report1["meta"]["document_id"] == ids1.document_id
    assert report1["meta"]["baseline_id"] == ids1.baseline_id
    assert report1["meta"]["run_id"] == ids1.run_id

    assert ids1 == ids2
    assert paths1 == paths2

    written = builder.write_report(report1, paths=paths1)
    assert written.exists()
    assert written == paths1.report_path

    loaded = fm.read(written, return_native=True)
    assert isinstance(loaded, dict)
    validator.validate(loaded)

    fm.delete_directory(paths1.run_root, ignore_errors=True)
