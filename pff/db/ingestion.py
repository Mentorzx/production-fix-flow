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
from pff.config import settings
from pff.utils import logger, progress_bar, FileManager
from pff.validators.kg.builder import KGBuilder

# Database connection string (using DATABASE_URL_ASYNC from settings)
DATABASE_URL = settings.DATABASE_URL.replace("postgresql://", "").replace("postgresql+asyncpg://", "")
DATABASE_URL = f"postgresql://{DATABASE_URL}"  # asyncpg format

# Configuration
BATCH_SIZE = 1000  # Records per batch insert
DEFAULT_CORRECT_ZIP = Path("data/models/correct.zip")


class TelecomDataIngestion:
    """Ingest telecom data from correct.zip into PostgreSQL."""

    _pool: asyncpg.Pool | None = None  # Shared connection pool for graceful shutdown

    def __init__(self, zip_path: Path = DEFAULT_CORRECT_ZIP, batch_size: int = BATCH_SIZE):
        """
        Initialize ingestion.

        Args:
            zip_path: Path to correct.zip
            batch_size: Number of records to insert per batch
        """
        self.zip_path = zip_path
        self.batch_size = batch_size
        self.stats = {
            "total_files": 0,
            "telecom_inserted": 0,
            "triples_inserted": 0,
            "errors": 0
        }

    async def run(self):
        """Execute full ingestion pipeline."""
        logger.info(f"Starting ingestion from {self.zip_path}")

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
        logger.info("Step 1/2: Ingesting telecom_data...")

        batch = []

        with zipfile.ZipFile(self.zip_path, 'r') as zf:
            filenames = [name for name in zf.namelist() if name.endswith('.txt')]
            self.stats["total_files"] = len(filenames)

            for filename in progress_bar(filenames, desc="Ingesting telecom_data"):
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

        logger.info(f"✅ Telecom data ingested: {self.stats['telecom_inserted']} records")

    async def _insert_telecom_batch(self, pool: asyncpg.Pool, batch: list[tuple[str, str]]):
        """
        Batch insert into telecom_data table.

        Args:
            pool: Database connection pool
            batch: List of (msisdn, json_data) tuples where json_data is a JSON string
        """
        async with pool.acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO telecom_data (msisdn, data)
                VALUES ($1, $2)
                ON CONFLICT (msisdn) DO UPDATE SET
                    data = EXCLUDED.data,
                    updated_at = CURRENT_TIMESTAMP
                """,
                batch
            )

        self.stats["telecom_inserted"] += len(batch)

    async def _ingest_kg_triples(self, pool: asyncpg.Pool):
        """
        Extract KG triples from telecom_data and insert into kg_triples table.

        Reuses KGBuilder logic for triple extraction.
        """
        logger.info("Step 2/2: Extracting and ingesting KG triples...")

        # Use KGBuilder to parse triples from correct.zip
        builder = KGBuilder(
            source_path=self.zip_path,
            output_dir=Path("/tmp/kg_temp"),  # Temporary (não vamos salvar em disco)
            max_members=None,  # Process all
            parallel=True,
            disk_cache=False
        )

        # Load and parse triples (reutiliza builder._load_and_parse())
        await builder._load_and_parse()
        triples = builder._triples

        logger.info(f"Extracted {len(triples)} triples from {len(triples) // 100} customers (avg)")

        # Batch insert triples
        batch = []
        for s, p, o in progress_bar(triples, desc="Ingesting kg_triples"):
            batch.append((s, p, o, "correct.zip", 1.0))  # source, confidence

            if len(batch) >= self.batch_size:
                await self._insert_triples_batch(pool, batch)
                batch = []

        # Insert remaining
        if batch:
            await self._insert_triples_batch(pool, batch)

        logger.info(f"✅ KG triples ingested: {self.stats['triples_inserted']} triples")

    async def _insert_triples_batch(self, pool: asyncpg.Pool, batch: list[tuple]):
        """
        Batch insert into kg_triples table.

        Args:
            pool: Database connection pool
            batch: List of (subject, predicate, object, source, confidence) tuples
        """
        async with pool.acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO kg_triples (subject, predicate, object, source, confidence)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (subject, predicate, object) DO UPDATE SET
                    confidence = GREATEST(kg_triples.confidence, EXCLUDED.confidence)
                """,
                batch
            )

        self.stats["triples_inserted"] += len(batch)

    def _report_stats(self):
        """Print ingestion statistics."""
        logger.info("="*60)
        logger.info("Ingestion Complete!")
        logger.info("="*60)
        logger.info(f"Total files processed: {self.stats['total_files']}")
        logger.info(f"Telecom records inserted: {self.stats['telecom_inserted']}")
        logger.info(f"KG triples inserted: {self.stats['triples_inserted']}")
        logger.info(f"Errors: {self.stats['errors']}")
        logger.info("="*60)


# CLI interface
async def main():
    """CLI entrypoint for ingestion."""
    import argparse

    parser = argparse.ArgumentParser(description="Ingest correct.zip into PostgreSQL")
    parser.add_argument(
        "--zip-path",
        type=Path,
        default=DEFAULT_CORRECT_ZIP,
        help="Path to correct.zip"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
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
