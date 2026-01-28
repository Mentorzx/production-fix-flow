"""
Database Ingestion Module - Sprint 3

Ingest correct.parquet (14.5k JSON records) into PostgreSQL:
- telecom_data table (raw JSON data)
- kg_triples table (extracted triples)

Performance target: <10min for full ingest (batch insert 1000 records/time)
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, cast

import asyncpg
import pyarrow.parquet as pq

from pff.domain.kg.builder import KGBuilder
from pff.infrastructure.persistence.db.config import get_postgres_config
from pff.shared import FileManager, logger, progress_bar
from pff.shared.core.config import INGESTION_CONFIG_PATH, settings
from pff.shared.core.file_manager import ParquetBundle
from pff.shared.core.file_manager.handlers.parquet import iter_parquet_as_json

DATABASE_URL = get_postgres_config().dsn_asyncpg


def _load_ingestion_config() -> dict[str, Any]:
    base_defaults: dict[str, Any] = {
        "correct_zip_path": settings.DATA_DIR / "models" / "correct.parquet",
        "batch_size": 1000,
        "temp_output_dir": settings.OUTPUTS_DIR / "temp" / "kg_ingestion",
        "progress": {
            "telecom": "Ingesting telecom_data",
            "triples": "Ingesting kg_triples",
        },
    }
    try:
        cfg_raw = FileManager.read(INGESTION_CONFIG_PATH, return_native=True)
        if isinstance(cfg_raw, dict):
            cfg: dict[str, Any] = cfg_raw.get("ingestion", cfg_raw)
            if not isinstance(cfg, dict):
                return base_defaults
            merged = dict(base_defaults)
            merged.update(cfg)
            progress_cfg = cfg.get("progress") or {}
            if isinstance(progress_cfg, dict):
                merged_progress = dict(cast(dict, base_defaults["progress"]))
                merged_progress.update(progress_cfg)
                merged["progress"] = merged_progress
            return merged
    except Exception as exc:
        logger.debug(f"Using default ingestion config (reason: {exc})")
    return base_defaults


INGESTION_CONFIG = _load_ingestion_config()


class TelecomDataIngestion:
    """Ingest telecom data from correct.parquet into PostgreSQL.

    Design Patterns Applied:
        - **Facade Pattern:** Provides a simplified interface to the complex
          subsystem of ZIP extraction, data transformation, and PostgreSQL
          batch insertion.
        - **Adapter Pattern:** Adapts different data formats (ZIP/CSV) to
          the PostgreSQL schema expected by downstream services.
        - **Template Method:** The `ingest()` method defines the skeleton
          of the ingestion algorithm with customizable extraction/transform steps.

    Performance Optimizations:
        - Async batch insertion with configurable batch_size.
        - Connection pooling via shared asyncpg.Pool.
        - FileManager used for I/O operations (AGENTS.md compliance).
    """

    _pool: asyncpg.Pool | None = None

    def __init__(self, zip_path: Path | None = None, batch_size: int | None = None):
        """
        Initialize ingestion.

        Args:
            zip_path: Path to correct.parquet
            batch_size: Number of records to insert per batch
        """
        cfg = INGESTION_CONFIG
        resolved_zip = Path(zip_path) if zip_path is not None else Path(cfg["correct_zip_path"])
        if not resolved_zip.is_absolute():
            resolved_zip = (settings.ROOT_DIR / resolved_zip).resolve()
        self.zip_path = resolved_zip
        self.batch_size = batch_size or int(cfg["batch_size"])
        temp_dir = Path(cfg["temp_output_dir"])
        self.temp_output_dir = (
            temp_dir if temp_dir.is_absolute() else (settings.ROOT_DIR / temp_dir)
        )
        self.temp_output_dir.mkdir(parents=True, exist_ok=True)
        self.progress_labels = cfg.get("progress", {})
        self.stats = {
            "total_files": 0,
            "telecom_inserted": 0,
            "triples_inserted": 0,
            "errors": 0,
        }

    async def run(self):
        """Execute full ingestion pipeline."""
        logger.info(f"component_name=ingestion message='Iniciando ingestão de {self.zip_path}'")

        if not self.zip_path.exists():
            raise FileNotFoundError(f"correct.parquet not found at {self.zip_path}")

        TelecomDataIngestion._pool = await asyncpg.create_pool(
            DATABASE_URL, min_size=2, max_size=10
        )

        try:
            await self._ingest_telecom_data(TelecomDataIngestion._pool)

            await self._ingest_kg_triples(TelecomDataIngestion._pool)

            self._report_stats()

        finally:
            await TelecomDataIngestion._pool.close()
            TelecomDataIngestion._pool = None

    async def _ingest_telecom_data(self, pool: asyncpg.Pool):
        """
        Ingest raw JSON data into telecom_data table.

        Uses batch insert for performance (1000 records per transaction).
        Supports both legacy parquets with _raw_json and optimized parquets
        with struct columns only.
        """
        logger.info("component_name=ingestion message='Etapa 1/2: importando telecom_data...'")

        batch: list[tuple[str, str]] = []
        bundle = FileManager.read(self.zip_path)
        telecom_desc = self.progress_labels.get("telecom", "Ingesting telecom_data")

        if not isinstance(bundle, ParquetBundle):
            raise RuntimeError("Expected parquet bundle for ingestion pipeline")
        if bundle.parsed_kind != "tabular":
            raise RuntimeError("Expected tabular parquet bundle for ingestion pipeline")

        parquet_path = bundle.parsed_parquet_path or bundle.source_path
        parquet_file = pq.ParquetFile(parquet_path)
        total_rows = parquet_file.metadata.num_rows if parquet_file.metadata else None
        self.stats["total_files"] = int(total_rows) if total_rows else 0

        def _extract_msisdn(name: str | None, external_id: Any) -> str | None:
            if isinstance(name, str) and name:
                token = name.rsplit("_", 1)[-1].split(".", 1)[0]
                if token:
                    return token
            if external_id is not None:
                return str(external_id)
            return None

        read_batch_size = max(self.batch_size, 1024)
        row_iter = iter_parquet_as_json(parquet_path, batch_size=read_batch_size)

        for source_name, external_id, json_str in progress_bar(
            row_iter,
            desc=telecom_desc,
            total=total_rows,
        ):
            msisdn = _extract_msisdn(source_name, external_id)
            if not msisdn:
                self.stats["errors"] += 1
                continue
            batch.append((msisdn, json_str))

            if len(batch) >= self.batch_size:
                await self._insert_telecom_batch(pool, batch)
                batch = []

        if batch:
            await self._insert_telecom_batch(pool, batch)

        logger.info(
            f"component=ingestion evento=telecom_concluido n={self.stats['telecom_inserted']}"
        )

    async def _insert_telecom_batch(self, pool: asyncpg.Pool, batch: list[tuple[str, str]]):
        """
        Batch insert into telecom_data table.

        Args:
            pool: Database connection pool
            batch: List of (msisdn, json_data) tuples where json_data is a JSON string
        """
        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    """
                    CREATE TEMP TABLE IF NOT EXISTS tmp_telecom_ingest (
                        msisdn TEXT,
                        data JSONB
                    ) ON COMMIT DROP
                    """
                )

                await conn.copy_records_to_table(
                    table_name="tmp_telecom_ingest",
                    columns=("msisdn", "data"),
                    records=batch,
                )

                await conn.execute(
                    """
                    INSERT INTO telecom_data (msisdn, data)
                    SELECT msisdn, data
                    FROM tmp_telecom_ingest
                    ON CONFLICT (msisdn) DO UPDATE SET
                        data = EXCLUDED.data,
                        updated_at = CURRENT_TIMESTAMP
                    """
                )

                await conn.execute("TRUNCATE tmp_telecom_ingest")

        self.stats["telecom_inserted"] += len(batch)

    async def _ingest_kg_triples(self, pool: asyncpg.Pool):
        """
        Extract KG triples from telecom_data and insert into kg_triples table.

        Reuses KGBuilder logic for triple extraction.
        """
        logger.info(
            "component_name=ingestion message='Etapa 2/2: extraindo e importando triplicas do KG...'"
        )

        builder = KGBuilder(
            source_path=self.zip_path,
            output_dir=self.temp_output_dir,
            max_members=None,
            parallel=True,
            disk_cache=False,
        )

        triples = await builder.extract_triples()

        logger.info(f"Extraidas {len(triples)} triplas de {len(triples) // 100} clientes (media)")

        batch = []
        triples_desc = self.progress_labels.get("triples", "Ingesting kg_triples")
        for s, p, o in progress_bar(triples, desc=triples_desc):
            batch.append((s, p, o, "correct.parquet", 1.0))

            if len(batch) >= self.batch_size:
                await self._insert_triples_batch(pool, batch)
                batch = []

        if batch:
            await self._insert_triples_batch(pool, batch)

        logger.info(
            f"component_name=ingestion stop_reason=step_completion message='Triplas KG ingeridas: {self.stats['triples_inserted']} triplas'"
        )

    async def _insert_triples_batch(self, pool: asyncpg.Pool, batch: list[tuple]):
        """
        Batch insert into kg_triples table.

        Args:
            pool: Database connection pool
            batch: List of (subject, predicate, object, source, confidence) tuples
        """
        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    """
                    CREATE TEMP TABLE IF NOT EXISTS tmp_kg_triples (
                        subject TEXT,
                        predicate TEXT,
                        object TEXT,
                        source TEXT,
                        confidence DOUBLE PRECISION
                    ) ON COMMIT DROP
                    """
                )

                await conn.copy_records_to_table(
                    table_name="tmp_kg_triples",
                    columns=("subject", "predicate", "object", "source", "confidence"),
                    records=batch,
                )

                await conn.execute(
                    """
                    INSERT INTO kg_triples (subject, predicate, object, source, confidence)
                    SELECT subject, predicate, object, source, confidence
                    FROM tmp_kg_triples
                    ON CONFLICT (subject, predicate, object) DO UPDATE SET
                        confidence = GREATEST(kg_triples.confidence, EXCLUDED.confidence)
                    """
                )

                await conn.execute("TRUNCATE tmp_kg_triples")

        self.stats["triples_inserted"] += len(batch)

    def _report_stats(self):
        """Print ingestion statistics."""
        logger.info(
            "component_name=ingestion stop_reason=ingestion_complete message='Importação concluída com sucesso!'"
        )
        logger.info(
            f"component_name=ingestion message='Arquivos processados: {self.stats['total_files']}'"
        )
        logger.info(
            f"component_name=ingestion message='Registros telecom inseridos: {self.stats['telecom_inserted']}'"
        )
        logger.info(
            f"component_name=ingestion message='Triplas KG inseridas: {self.stats['triples_inserted']}'"
        )
        logger.info(
            f"component_name=ingestion message='Quantidade de falhas: {self.stats['errors']}'"
        )


async def main():
    """CLI entrypoint for ingestion."""
    import argparse

    parser = argparse.ArgumentParser(description="Ingest correct.parquet into PostgreSQL")
    parser.add_argument(
        "--zip-path",
        type=Path,
        default=Path(INGESTION_CONFIG["correct_zip_path"]),
        help="Path to correct.parquet",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(INGESTION_CONFIG["batch_size"]),
        help="Records per batch insert",
    )
    args = parser.parse_args()

    ingestion = TelecomDataIngestion(zip_path=args.zip_path, batch_size=args.batch_size)

    await ingestion.run()


if __name__ == "__main__":
    asyncio.run(main())
