"""
MLModelsRepository - Repository Pattern for ML Models Storage.

Design Patterns Applied:
- Repository Pattern: Encapsulates data access logic
- Versioning: Automatic versioning with timestamps
- Compression: gzip compression for large models (2-3x size reduction)
- Metadata: JSONB for flexible metrics and hyperparameters

Performance:
- Save model: 3.8 MB → 2 MB gzip (~0.5s)
- Load model: 2 MB → 3.8 MB ungzip (~0.3s)
- Versioning: Query by version or get latest
"""

import gzip
from datetime import datetime

from pff.infrastructure.persistence.db.repositories.base import PostgresRepository
from pff.shared.core.logging import logger


class MLModelsRepository(PostgresRepository):
    """
    Repository for managing ML models with versioning and compression.

    Pattern: Repository Pattern + Versioning
    """

    async def save_model(
        self,
        model_name: str,
        model_type: str,
        model_data: bytes,
        model_version: str | None = None,
        metrics: dict | None = None,
        hyperparameters: dict | None = None,
        compress: bool = True,
    ) -> int:
        """
        Save ML model to PostgreSQL with compression and versioning.

        Args:
            model_name: Model identifier ('dslfm', 'ensemble')
            model_type: Type ('checkpoint', 'binary', 'metadata')
            model_data: Model binary data
            model_version: Optional version (default: timestamp)
            metrics: Optional metrics dict (MRR, Hits@1, AUC, etc.)
            hyperparameters: Optional hyperparameters dict
            compress: Whether to gzip compress (default: True)

        Returns:
            Model ID

        Pattern: Automatic Versioning + Compression
        """
        await self._ensure_pool()

        if model_version is None:
            model_version = datetime.now().strftime("%Y%m%d_%H%M%S")

        if compress:
            original_size = len(model_data)
            model_data = gzip.compress(model_data, compresslevel=6)
            compressed_size = len(model_data)
            compression_ratio = compressed_size / original_size
            logger.info(
                f"component_name=MLModelsRepository message='Comprimindo modelo: {original_size / 1024:.1f} KB -> "
                f"{compressed_size / 1024:.1f} KB ({compression_ratio:.1%})'"
            )

        logger.info(
            f"component_name=MLModelsRepository key_parameters={{'model': '{model_name}', 'type': '{model_type}', 'v': '{model_version}'}} "
            f"message='Salvando modelo ({len(model_data) / 1024:.1f} KB)...'"
        )

        async with self.pool.acquire() as conn:
            query = """
                INSERT INTO ml_models
                    (model_name, model_type, model_data, model_version, metrics, hyperparameters)
                VALUES ($1, $2, $3, $4, $5::jsonb, $6::jsonb)
                ON CONFLICT (model_name, model_type, model_version)
                DO UPDATE SET
                    model_data = EXCLUDED.model_data,
                    metrics = EXCLUDED.metrics,
                    hyperparameters = EXCLUDED.hyperparameters,
                    created_at = NOW()
                RETURNING id
            """

            metrics_json = self._file_manager.json_dumps(metrics) if metrics is not None else None
            hyperparams_json = (
                self._file_manager.json_dumps(hyperparameters)
                if hyperparameters is not None
                else None
            )

            model_id = await conn.fetchval(
                query,
                model_name,
                model_type,
                model_data,
                model_version,
                metrics_json,
                hyperparams_json,
            )

        logger.success(f" Modelo salvo no PostgreSQL (ID: {model_id})")

        return model_id

    async def load_model(
        self,
        model_name: str,
        model_type: str,
        model_version: str | None = None,
        decompress: bool = True,
    ) -> bytes | None:
        """
        Load ML model from PostgreSQL.

        Args:
            model_name: Model identifier
            model_type: Type ('checkpoint', 'binary', 'metadata')
            model_version: Optional version (None = latest)
            decompress: Whether to gzip decompress (default: True)

        Returns:
            Model binary data (decompressed if requested), or None if not found

        Pattern: Load Latest or Specific Version
        """
        await self._ensure_pool()

        logger.info(f" Carregando modelo {model_name}/{model_type} v{model_version or 'latest'}...")

        async with self.pool.acquire() as conn:
            if model_version:
                query = """
                    SELECT model_data
                    FROM ml_models
                    WHERE model_name = $1
                    AND model_type = $2
                    AND model_version = $3
                """
                row = await conn.fetchrow(query, model_name, model_type, model_version)
            else:
                query = """
                    SELECT model_data
                    FROM ml_models
                    WHERE model_name = $1
                    AND model_type = $2
                    ORDER BY created_at DESC
                    LIMIT 1
                """
                row = await conn.fetchrow(query, model_name, model_type)

        if row is None:
            logger.warning(f"Model {model_name}/{model_type} not found in PostgreSQL")
            return None

        model_data = bytes(row["model_data"])

        if decompress:
            try:
                compressed_size = len(model_data)
                model_data = gzip.decompress(model_data)
                decompressed_size = len(model_data)
                logger.info(
                    f" Descomprimindo modelo: {compressed_size / 1024:.1f} KB → "
                    f"{decompressed_size / 1024:.1f} KB"
                )
            except gzip.BadGzipFile:
                logger.warning("  Model is not compressed (gzip), returning raw data")

        logger.success(f" Modelo carregado do PostgreSQL ({len(model_data) / 1024:.1f} KB)")

        return model_data

    async def load_model_with_metadata(
        self, model_name: str, model_type: str, model_version: str | None = None
    ) -> dict | None:
        """
        Load model with full metadata (data, metrics, hyperparameters).

        Args:
            model_name: Model identifier
            model_type: Type
            model_version: Optional version (None = latest)

        Returns:
            Dictionary with model data and metadata, or None
        """
        await self._ensure_pool()

        async with self.pool.acquire() as conn:
            if model_version:
                query = """
                    SELECT model_data, model_version, metrics, hyperparameters, created_at
                    FROM ml_models
                    WHERE model_name = $1
                    AND model_type = $2
                    AND model_version = $3
                """
                row = await conn.fetchrow(query, model_name, model_type, model_version)
            else:
                query = """
                    SELECT model_data, model_version, metrics, hyperparameters, created_at
                    FROM ml_models
                    WHERE model_name = $1
                    AND model_type = $2
                    ORDER BY created_at DESC
                    LIMIT 1
                """
                row = await conn.fetchrow(query, model_name, model_type)

        if row is None:
            return None

        return {
            "data": bytes(row["model_data"]),
            "version": row["model_version"],
            "metrics": row["metrics"],
            "hyperparameters": row["hyperparameters"],
            "created_at": row["created_at"],
        }

    async def list_models(
        self, model_name: str | None = None, model_type: str | None = None
    ) -> list[dict]:
        """
        List available models with metadata (versioning history).

        Args:
            model_name: Optional name filter
            model_type: Optional type filter

        Returns:
            List of model metadata dictionaries
        """
        await self._ensure_pool()

        logger.info(f" Listando modelos ({model_name or 'all'}/{model_type or 'all'})...")

        async with self.pool.acquire() as conn:
            if model_name and model_type:
                query = """
                    SELECT id, model_name, model_type, model_version,
                           metrics, hyperparameters, created_at,
                           pg_size_pretty(length(model_data)::bigint) as size
                    FROM ml_models
                    WHERE model_name = $1
                    AND model_type = $2
                    ORDER BY created_at DESC
                """
                rows = await conn.fetch(query, model_name, model_type)
            elif model_name:
                query = """
                    SELECT id, model_name, model_type, model_version,
                           metrics, hyperparameters, created_at,
                           pg_size_pretty(length(model_data)::bigint) as size
                    FROM ml_models
                    WHERE model_name = $1
                    ORDER BY created_at DESC
                """
                rows = await conn.fetch(query, model_name)
            elif model_type:
                query = """
                    SELECT id, model_name, model_type, model_version,
                           metrics, hyperparameters, created_at,
                           pg_size_pretty(length(model_data)::bigint) as size
                    FROM ml_models
                    WHERE model_type = $1
                    ORDER BY created_at DESC
                """
                rows = await conn.fetch(query, model_type)
            else:
                query = """
                    SELECT id, model_name, model_type, model_version,
                           metrics, hyperparameters, created_at,
                           pg_size_pretty(length(model_data)::bigint) as size
                    FROM ml_models
                    ORDER BY created_at DESC
                """
                rows = await conn.fetch(query)

        models = []
        for row in rows:
            models.append(
                {
                    "id": row["id"],
                    "model_name": row["model_name"],
                    "model_type": row["model_type"],
                    "model_version": row["model_version"],
                    "metrics": row["metrics"],
                    "hyperparameters": row["hyperparameters"],
                    "created_at": row["created_at"],
                    "size": row["size"],
                }
            )

        logger.success(f" {len(models)} modelos encontrados")

        return models

    async def delete_model(
        self, model_name: str, model_type: str, model_version: str | None = None
    ) -> int:
        """
        Delete model(s) from PostgreSQL.

        Args:
            model_name: Model identifier
            model_type: Type
            model_version: Optional version (None = delete all versions)

        Returns:
            Number of models deleted
        """
        await self._ensure_pool()

        async with self.pool.acquire() as conn:
            if model_version:
                query = """
                    DELETE FROM ml_models
                    WHERE model_name = $1
                    AND model_type = $2
                    AND model_version = $3
                """
                result = await conn.execute(query, model_name, model_type, model_version)
            else:
                query = """
                    DELETE FROM ml_models
                    WHERE model_name = $1
                    AND model_type = $2
                """
                result = await conn.execute(query, model_name, model_type)

        deleted = int(result.split()[-1]) if result else 0

        logger.info(f"  {deleted} modelo(s) deletado(s) do PostgreSQL")

        return deleted

    async def get_latest_version(self, model_name: str, model_type: str) -> str | None:
        """
        Get the latest version of a model.

        Args:
            model_name: Model identifier
            model_type: Type

        Returns:
            Latest version string, or None
        """
        await self._ensure_pool()

        async with self.pool.acquire() as conn:
            query = """
                SELECT model_version
                FROM ml_models
                WHERE model_name = $1
                AND model_type = $2
                ORDER BY created_at DESC
                LIMIT 1
            """
            version = await conn.fetchval(query, model_name, model_type)

        return version
