"""Audit report builder utilities.

This module provides a minimal Builder for `audit_report.json` that:
- assigns deterministic ids (document_id/baseline_id/run_id),
- writes artifacts under outputs/audit/<run_id>/,
- validates the payload against the versioned JSON Schema before persisting.
"""

from __future__ import annotations

import heapq
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from pff.shared import logger

from .artifacts import AuditArtifactPaths
from .ids import AuditRunIds, build_audit_run_ids
from .root_causes import select_root_causes
from .schema import AuditReportSchemaValidator


class AuditReportStoragePort(Protocol):
    """Minimal storage protocol for audit report persistence."""

    def ensure_dir(self, path: Path | str, **kwargs: Any) -> None:
        """Ensure directory exists."""
        ...

    def save(self, data: Any, path: Path | str, **kwargs: Any) -> None:
        """Persist payload at path."""
        ...


def _count_severities(findings: list[dict[str, Any]]) -> dict[str, int]:
    """Execute count severities.



    Args:

        findings: Input value used by this callable.



    Returns:

        Return value produced by the callable.

    """

    counts = {"info": 0, "warning": 0, "error": 0}
    for finding in findings:
        sev = str(finding.get("severity", "")).lower()
        if sev in counts:
            counts[sev] += 1
    return counts


def _top_json_pointers(findings: list[dict[str, Any]], *, limit: int = 20) -> list[str]:
    """Execute top json pointers.



    Args:

        findings: Input value used by this callable.

        limit: Optional input value.



    Returns:

        Return value produced by the callable.

    """

    freq: dict[str, int] = {}
    for finding in findings:
        ptr = finding.get("json_pointer")
        if not isinstance(ptr, str) or not ptr:
            continue
        freq[ptr] = freq.get(ptr, 0) + 1
    ordered = heapq.nsmallest(limit, freq.items(), key=lambda kv: (-kv[1], kv[0]))
    return [k for k, _ in ordered]


@dataclass
class AuditReportBuilder:
    """Builder for schema-valid audit reports.

    Args:
        outputs_dir: Root outputs directory.
        schema_validator: Validator enforcing the report contract.
        file_manager: Storage adapter for persistence operations.
    """

    outputs_dir: Path
    schema_validator: AuditReportSchemaValidator
    file_manager: AuditReportStoragePort | None = None

    def build_report(
        self,
        *,
        document: Any,
        baseline_key: Any,
        schema_version: str | int,
        findings: list[dict[str, Any]] | None = None,
        meta_overrides: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, Any], AuditRunIds, AuditArtifactPaths]:
        """Build a schema-valid audit report payload.

        Args:
            document: Input JSON-like object.
            baseline_key: Baseline identifier key.
            schema_version: Input document schema version.
            findings: Optional list of finding dictionaries.
            meta_overrides: Optional metadata overrides.

        Returns:
            Tuple with (report_dict, run_ids, artifact_paths).
        """
        run_ids = build_audit_run_ids(
            document=document,
            baseline_key=baseline_key,
            schema_version=schema_version,
        )
        paths = AuditArtifactPaths.for_run(
            outputs_dir=self.outputs_dir, run_id=run_ids.run_id
        )

        meta: dict[str, Any] = {
            "document_id": run_ids.document_id,
            "baseline_id": run_ids.baseline_id,
            "run_id": run_ids.run_id,
            "schema_version": schema_version,
            "deterministic": True,
        }
        if meta_overrides:
            meta.update(dict(meta_overrides))

        normalized_findings = list(findings or [])
        report: dict[str, Any] = {
            "schema_version": 1,
            "meta": meta,
            "findings": normalized_findings,
            "summary": {
                "counts": _count_severities(normalized_findings),
                "top_json_pointers": _top_json_pointers(normalized_findings),
                "root_causes": select_root_causes(normalized_findings),
            },
        }

        self.schema_validator.validate(report)
        return report, run_ids, paths

    def write_report(
        self,
        report: dict[str, Any],
        *,
        paths: AuditArtifactPaths,
    ) -> Path:
        """Validate and persist an audit report to the canonical location.

        Args:
            report: Audit report payload.
            paths: Artifact layout for the run.

        Returns:
            Path to the persisted report JSON.
        """
        if self.file_manager is None:
            raise RuntimeError(
                "AuditReportStoragePort is required to persist audit report"
            )
        fm = self.file_manager
        paths.ensure(fm.ensure_dir)  # type: ignore[arg-type]
        self.schema_validator.validate(report)
        fm.save(report, paths.report_path)
        logger.info(f"laudo_persistido path={paths.report_path}")
        return paths.report_path
