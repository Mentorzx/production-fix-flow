"""Audit use case orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from collections.abc import Mapping

from pff.application.ports.storage import StoragePort
from pff.domain.audit.artifacts import AuditArtifactPaths
from pff.domain.audit.report import AuditReportBuilder


class AuditUseCase:
    """Generate audit reports with deterministic IDs and schema validation."""

    def __init__(
        self,
        *,
        outputs_dir: Path | None = None,
        storage: StoragePort | None = None,
        report_builder: AuditReportBuilder | None = None,
    ) -> None:
        """Initialize the audit use case.

        Args:
            outputs_dir: Optional outputs directory override.
            storage: Optional storage port implementation.
            report_builder: Optional report builder override.
        """
        self._outputs_dir = outputs_dir
        self._storage = storage
        self._report_builder = report_builder or AuditReportBuilder.default()

    def execute(
        self,
        *,
        document: Any,
        baseline_key: Any,
        schema_version: str | int,
        findings: list[dict[str, Any]] | None = None,
        meta_overrides: Mapping[str, Any] | None = None,
    ) -> Path:
        """Build and persist an audit report.

        Args:
            document: Input JSON-like document.
            baseline_key: Baseline identifier key.
            schema_version: Schema version for the input document.
            findings: Optional list of findings.
            meta_overrides: Optional metadata overrides.

        Returns:
            Path to the persisted report JSON.
        """
        report, run_ids, paths = self._report_builder.build_report(
            document=document,
            baseline_key=baseline_key,
            schema_version=schema_version,
            findings=findings,
            meta_overrides=meta_overrides,
        )

        if self._outputs_dir is not None:
            paths = AuditArtifactPaths.for_run(
                outputs_dir=self._outputs_dir,
                run_id=run_ids.run_id,
            )

        self._storage.ensure_dir(paths.report_dir)
        return self._storage.save_json(report, paths.report_path)
