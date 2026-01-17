"""Audit utilities for JSON→Graph→JSON validation workflows.

This package provides small, deterministic building blocks for the audit
pipeline described in `DSLFM_ANOMALY_NEUROSYMBOLIC_ROADMAP.md`.

Design patterns:
    - Builder: `AuditReportBuilder` creates a schema-valid audit report payload.
    - Adapter: report/paths helpers adapt internal evidence to the external JSON contract.
"""

from .artifacts import AuditArtifactPaths
from .canonicalize import (
    CanonicalRecord,
    CanonicalTriple,
    canonicalize_json_document,
    records_to_triples,
)
from .ids import AuditRunIds, build_audit_run_ids
from .report import AuditReportBuilder
from .schema import AuditReportSchemaValidator

__all__ = [
    "AuditArtifactPaths",
    "AuditReportBuilder",
    "AuditReportSchemaValidator",
    "AuditRunIds",
    "build_audit_run_ids",
    "CanonicalRecord",
    "CanonicalTriple",
    "canonicalize_json_document",
    "records_to_triples",
]
