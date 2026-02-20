"""Persistence ports for audit canonicalization and report storage."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from pff.domain.audit.canonicalize import CanonicalRecord, CanonicalTriple


class AuditStoragePort(Protocol):
    """Port for persisting raw audit canonicalization data."""

    async def persist_canonicalization(
        self,
        *,
        run_id: str,
        document_id: str,
        baseline_id: str,
        records: list[CanonicalRecord],
        triples: list[CanonicalTriple],
    ) -> Any:
        """Persist canonicalized records and triples for an audit run."""
        ...


class AuditAnalysisPort(Protocol):
    """Port for audit analysis artifacts (profiles, schema reports, and drift)."""

    async def save_schema_report(
        self,
        *,
        run_id: str,
        schema_report: list[dict[str, Any]],
        schema_id: str | None,
        schema_version: str | int,
    ) -> None:
        """Persist schema validation report artifacts."""
        ...

    async def load_baseline_profile(self, *, baseline_id: str) -> dict[str, Any] | None:
        """Load a baseline profile for drift comparison."""
        ...

    async def save_baseline_profile(
        self,
        *,
        baseline_id: str,
        profile: dict[str, Any],
        digest: dict[str, Any],
    ) -> None:
        """Persist baseline profile and its digest summary."""
        ...

    async def save_run_profile(
        self,
        *,
        run_id: str,
        profile_current: dict[str, Any],
        drift: dict[str, Any],
    ) -> None:
        """Persist run profile and drift metrics against baseline."""
        ...


class AuditReportsPort(Protocol):
    """Port for persisting final audit reports."""

    async def save_report(self, *, run_id: str, report: dict[str, Any]) -> None:
        """Persist the final report artifact for a run identifier."""
        ...
