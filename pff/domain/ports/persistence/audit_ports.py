from typing import Any, Protocol


class AuditStoragePort(Protocol):
    """Port for persisting raw audit canonicalization data."""

    async def persist_canonicalization(
        self,
        run_id: str,
        document_id: str,
        baseline_id: str,
        records: list[dict[str, Any]],
        triples: list[Any],
    ) -> None: ...


class AuditAnalysisPort(Protocol):
    """Port for audit analysis artifacts (profiles, schema reports, etc)."""

    async def save_schema_report(
        self,
        run_id: str,
        schema_report: list[dict[str, Any]],
        schema_id: str | None,
        schema_version: str | int,
    ) -> None: ...

    async def load_baseline_profile(self, baseline_id: str) -> dict[str, Any] | None: ...

    async def save_baseline_profile(
        self,
        baseline_id: str,
        profile: dict[str, Any],
        digest: dict[str, Any],
    ) -> None: ...

    async def save_run_profile(
        self,
        run_id: str,
        profile_current: dict[str, Any],
        drift: dict[str, Any],
    ) -> None: ...


class AuditReportsPort(Protocol):
    """Port for persisting final audit reports."""

    async def save_report(self, run_id: str, report: dict[str, Any]) -> None: ...
