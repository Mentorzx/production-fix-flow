"""Audit storage adapters.

Design patterns:
    - Adapter: exposes a small persistence surface for the audit pipeline while
      delegating concrete storage concerns to PostgreSQL repositories.
"""

from __future__ import annotations

from dataclasses import dataclass

from pff.infrastructure.persistence.db.repositories import AuditArtifactsRepository
from pff.shared.core.logger import logger
from pff.domain.audit.canonicalize import CanonicalRecord, CanonicalTriple


@dataclass(frozen=True)
class AuditPersistenceResult:
    """Persistence outcome for canonicalization artifacts."""

    inserted_records: int
    inserted_triples: int


class AuditPostgresStorage:
    """PostgreSQL-backed storage for audit artifacts."""

    def __init__(self, *, repository: AuditArtifactsRepository | None = None) -> None:
        self._repo = repository or AuditArtifactsRepository()

    async def persist_canonicalization(
        self,
        *,
        run_id: str,
        document_id: str,
        baseline_id: str,
        records: list[CanonicalRecord],
        triples: list[CanonicalTriple],
    ) -> AuditPersistenceResult:
        """Persist canonical records + triples to PostgreSQL.

        Args:
            run_id: Audit run identifier.
            document_id: Input document identifier.
            baseline_id: Baseline identifier for historical comparisons.
            records: Canonical leaf records.
            triples: Canonical triples derived from records.

        Returns:
            Inserted row counts (excluding conflicts).
        """

        await self._repo.save_run(
            run_id=run_id,
            document_id=document_id,
            baseline_id=baseline_id,
            meta={"artifact": "canonicalization"},
        )
        inserted_records = await self._repo.save_canonical_records(
            run_id=run_id, records=records
        )
        inserted_triples = await self._repo.save_triples(run_id=run_id, triples=triples)

        logger.info(
            "canonicalizacao_persistida "
            f"run_id={run_id} registros={inserted_records:,} triplas={inserted_triples:,}"
        )
        return AuditPersistenceResult(
            inserted_records=inserted_records,
            inserted_triples=inserted_triples,
        )
