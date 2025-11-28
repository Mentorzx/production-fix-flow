"""
Database Ingestion Module - Sprint 3

Ingest correct.zip (14.5k JSON files) into PostgreSQL:
- telecom_data table (raw JSON data)
- kg_triples table (extracted triples)

Performance target: <10min for full ingest (batch insert 1000 records/time)
"""

from __future__ import annotations

import asyncio
import zipfile
from pathlib import Path
from typing import Any

import asyncpg
from pff import settings
from pff.config import INGESTION_CONFIG_PATH
from pff.utils import FileManager, logger, progress_bar
from pff.utils.db import get_postgres_config
from pff.validators.kg.builder import KGBuilder

# Database connection string (using centralized Postgres config)
DATABASE_URL = get_postgres_config().dsn_asyncpg


def _load_ingestion_config() -> dict[str, Any]:
    base_defaults = {
        "correct_zip_path": settings.DATA_DIR / "models" / "correct.zip",
        "batch_size": 1000,
        "temp_output_dir": settings.OUTPUTS_DIR / "temp" / "kg_ingestion",
        "progress": {
            "telecom": "Ingesting telecom_data",
            "triples": "Ingesting kg_triples",
        },
    }
    try:
        cfg = FileManager.read(INGESTION_CONFIG_PATH)
        if isinstance(cfg, dict):
            cfg = cfg.get("ingestion", cfg)
            if not isinstance(cfg, dict):
                return base_defaults
            merged = base_defaults | cfg
            progress_cfg = cfg.get("progress") or {}
            merged["progress"] = base_defaults["progress"] | progress_cfg if isinstance(progress_cfg, dict) else base_defaults["progress"]
            return merged
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"Using default ingestion config (reason: {exc})")
    return base_defaults


INGESTION_CONFIG = _load_ingestion_config()


class TelecomDataIngestion:
    """Ingest telecom data from correct.zip into PostgreSQL.

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

    _pool: asyncpg.Pool | None = None  # Shared connection pool for graceful shutdown

    def __init__(self, zip_path: Path | None = None, batch_size: int | None = None):
        """
        Initialize ingestion.

        Args:
            zip_path: Path to correct.zip
            batch_size: Number of records to insert per batch
        """
        cfg = INGESTION_CONFIG
        resolved_zip = Path(zip_path) if zip_path is not None else Path(cfg["correct_zip_path"])
        if not resolved_zip.is_absolute():
            resolved_zip = (settings.ROOT_DIR / resolved_zip).resolve()
        self.zip_path = resolved_zip
        self.batch_size = batch_size or int(cfg["batch_size"])
        temp_dir = Path(cfg["temp_output_dir"])
        self.temp_output_dir = temp_dir if temp_dir.is_absolute() else (settings.ROOT_DIR / temp_dir)
        self.temp_output_dir.mkdir(parents=True, exist_ok=True)
        self.progress_labels = cfg.get("progress", {})
        self.stats = {
            "total_files": 0,
            "telecom_inserted": 0,
            "triples_inserted": 0,
            "errors": 0
        }

    async def run(self):
        """Execute full ingestion pipeline."""
        logger.info(f"Iniciando ingestao de {self.zip_path}")

        # Validate zip exists
        if not self.zip_path.exists():
            raise FileNotFoundError(f"correct.zip not found at {self.zip_path}")

        # Create database connection pool (stored in class variable for graceful shutdown)
        TelecomDataIngestion._pool = await asyncpg.create_pool(DATABASE_URL, min_size=2, max_size=10)

        try:
            # Step 1: Ingest raw telecom data
            await self._ingest_telecom_data(TelecomDataIngestion._pool)

            # Step 2: Extract and ingest KG triples
            await self._ingest_kg_triples(TelecomDataIngestion._pool)

            # Step 3: Report statistics
            self._report_stats()

        finally:
            await TelecomDataIngestion._pool.close()
            TelecomDataIngestion._pool = None

    async def _ingest_telecom_data(self, pool: asyncpg.Pool):
        """
        Ingest raw JSON data into telecom_data table.

        Uses batch insert for performance (1000 records per transaction).
        """
        logger.info("Etapa 1/2: importando telecom_data...")

        batch = []

        with zipfile.ZipFile(self.zip_path, 'r') as zf:
            filenames = [name for name in zf.namelist() if name.endswith('.txt')]
            self.stats["total_files"] = len(filenames)
            telecom_desc = self.progress_labels.get("telecom", "Ingesting telecom_data")

            for filename in progress_bar(filenames, desc=telecom_desc):
                try:
                    # Read JSON from zip
                    content = zf.read(filename).decode('utf-8')
                    # Sprint 16.5: Use FileManager for 2-3x faster deserialization (msgspec)
                    data = FileManager.json_loads(content)

                    # Extract MSISDN from filename
                    # customer_enquiry_5511910001706.txt → 5511910001706
                    msisdn = filename.split('_')[-1].replace('.txt', '')

                    # Add to batch (JSONB requires JSON string)
                    # Sprint 16.5: Use FileManager for 2-3x faster serialization (msgspec)
                    batch.append((msisdn, FileManager.json_dumps(data)))

                    # Insert batch when full
                    if len(batch) >= self.batch_size:
                        await self._insert_telecom_batch(pool, batch)
                        batch = []

                except Exception as e:
                    logger.error(f"Error processing {filename}: {e}")
                    self.stats["errors"] += 1

            # Insert remaining batch
            if batch:
                await self._insert_telecom_batch(pool, batch)

        logger.info(f" Telecom data ingested: {self.stats['telecom_inserted']} records")

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
        logger.info("Etapa 2/2: extraindo e importando triplicas do KG...")

        # Use KGBuilder to parse triples from correct.zip
        builder = KGBuilder(
            source_path=self.zip_path,
            output_dir=self.temp_output_dir,
            max_members=None,  # Process all
            parallel=True,
            disk_cache=False
        )

        triples = await builder.extract_triples()

        logger.info(f"Extraidas {len(triples)} triplas de {len(triples) // 100} clientes (media)")

        # Batch insert triples
        batch = []
        triples_desc = self.progress_labels.get("triples", "Ingesting kg_triples")
        for s, p, o in progress_bar(triples, desc=triples_desc):
            batch.append((s, p, o, "correct.zip", 1.0))  # source, confidence

            if len(batch) >= self.batch_size:
                await self._insert_triples_batch(pool, batch)
                batch = []

        # Insert remaining
        if batch:
            await self._insert_triples_batch(pool, batch)

        logger.info(f" Triplas KG ingeridas: {self.stats['triples_inserted']} triplas")

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
        logger.info("="*60)
        logger.info("Importacao concluida com sucesso!")
        logger.info("="*60)
        logger.info(f"Arquivos processados: {self.stats['total_files']}")
        logger.info(f"Registros telecom inseridos: {self.stats['telecom_inserted']}")
        logger.info(f"Triplas KG inseridas: {self.stats['triples_inserted']}")
        logger.info(f"Erros: {self.stats['errors']}")
        logger.info("="*60)


# CLI interface
async def main():
    """CLI entrypoint for ingestion."""
    import argparse

    parser = argparse.ArgumentParser(description="Ingest correct.zip into PostgreSQL")
    parser.add_argument(
        "--zip-path",
        type=Path,
        default=Path(INGESTION_CONFIG["correct_zip_path"]),
        help="Path to correct.zip"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(INGESTION_CONFIG["batch_size"]),
        help="Records per batch insert"
    )
    args = parser.parse_args()

    ingestion = TelecomDataIngestion(
        zip_path=args.zip_path,
        batch_size=args.batch_size
    )

    await ingestion.run()


if __name__ == "__main__":
    asyncio.run(main())
