from __future__ import annotations

from pathlib import Path
from typing import Any

from pff.application.services.business_service import BusinessService
from pff.domain.audit import AuditReportBuilder, AuditReportSchemaValidator
from pff.drivers.api.deps import get_validator_service
from pff.shared.core.file_manager import FileManager


class _FakeAuditStorage:
    def __init__(self) -> None:
        self.persisted: dict[str, Any] | None = None

    async def persist_canonicalization(
        self,
        *,
        run_id: str,
        document_id: str,
        baseline_id: str,
        records: list[Any],
        triples: list[Any],
    ) -> Any:
        self.persisted = {
            "run_id": run_id,
            "document_id": document_id,
            "baseline_id": baseline_id,
            "records": records,
            "triples": triples,
        }
        return {"inserted_records": len(records), "inserted_triples": len(triples)}


class _FakeAuditAnalysisRepo:
    def __init__(self) -> None:
        self.schema_reports: dict[str, list[dict[str, Any]]] = {}
        self.baselines: dict[str, dict[str, Any]] = {}
        self.run_profiles: dict[str, dict[str, Any]] = {}

    async def save_schema_report(
        self,
        *,
        run_id: str,
        schema_report: list[dict[str, Any]],
        schema_id: str | None = None,
        schema_version: str | int | None = None,
    ) -> None:
        self.schema_reports[run_id] = schema_report

    async def load_baseline_profile(self, *, baseline_id: str) -> dict[str, Any] | None:
        return self.baselines.get(baseline_id)

    async def save_baseline_profile(
        self,
        *,
        baseline_id: str,
        profile: dict[str, Any],
        digest: dict[str, Any],
    ) -> None:
        self.baselines[baseline_id] = profile

    async def save_run_profile(
        self,
        *,
        run_id: str,
        profile_current: dict[str, Any],
        drift: dict[str, Any],
    ) -> None:
        self.run_profiles[run_id] = {"profile_current": profile_current, "drift": drift}


class _FakeAuditReportsRepo:
    def __init__(self) -> None:
        self.reports: dict[str, dict[str, Any]] = {}

    async def save_report(self, *, run_id: str, report: dict[str, Any]) -> None:
        self.reports[run_id] = report


def test_business_service_is_context_manager_and_models_guard_exists() -> None:
    with BusinessService() as service:
        service._ensure_models_loaded()


def test_api_deps_validator_service_yields_business_service() -> None:
    gen = get_validator_service()
    service = next(gen)
    assert isinstance(service, BusinessService)
    gen.close()


def test_business_service_audit_document_is_postgres_first_by_default(
    tmp_path: Path,
) -> None:
    fake_storage = _FakeAuditStorage()
    fake_analysis = _FakeAuditAnalysisRepo()
    fake_reports = _FakeAuditReportsRepo()
    fm = FileManager()
    builder = AuditReportBuilder(
        outputs_dir=tmp_path,
        schema_validator=AuditReportSchemaValidator(file_manager=fm),
        file_manager=fm,
    )

    with BusinessService(
        audit_storage=fake_storage,
        audit_analysis_repo=fake_analysis,
        audit_reports_repo=fake_reports,
        audit_report_builder=builder,
    ) as service:
        document = {"id": 1, "payload": {"x": 1, "y": "abc"}}
        input_schema = {
            "type": "object",
            "properties": {"id": {"type": "integer"}, "payload": {"type": "object"}},
            "required": ["id"],
            "additionalProperties": True,
        }

        result = service.audit_document(
            document,
            baseline_key={"baseline": "unit_test"},
            schema_version=1,
            input_schema=input_schema,
            schema_id="unit_test",
            export_outputs=False,
        )

        assert result.run_id == result.report["meta"]["run_id"]
        assert result.report["schema_version"] == 1
        assert result.report["meta"]["schema_version"] == 1
        assert result.run_id in fake_reports.reports
        assert fake_storage.persisted is not None

        run_dir = tmp_path / "audit" / result.run_id
        assert not run_dir.exists()
