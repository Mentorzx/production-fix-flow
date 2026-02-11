"""
KG Data Loader - Strategy Pattern for Loading from PostgreSQL.

Design Patterns Applied:
- Strategy Pattern: Repository-backed loading
- Facade Pattern: Simplified interface for data loading

Performance:
- PostgreSQL: 0.5s for 23K triples
"""

from pathlib import Path

import numpy as np
import polars as pl

from pff.domain.ports.persistence.kg_ports import KGMappingsPort, KGSplitsPort
from pff.shared import logger


class KGDataLoader:
    """
    Facade for loading KG data with PostgreSQL-only strategy.

    Pattern: Facade
    """

    def __init__(self, splits_repo: KGSplitsPort | None = None):
        """Initialize data loader with repositories."""
        self.splits_repo = splits_repo
        self.mappings_repo: KGMappingsPort | None = None

    async def load_split(
        self, split_name: str, split_type: str = "raw", disk_path: Path | None = None
    ) -> pl.DataFrame | None:
        try:
            if self.splits_repo is not None:
                df = await self.splits_repo.load_split(split_name, split_type)
                if df is not None:
                    logger.success(f"{split_name} carregado do PostgreSQL (0.5s)")
                    return df
        except Exception as e:
            raise RuntimeError(f"PostgreSQL split load failed: {split_name}/{split_type}") from e

        raise RuntimeError(f"PostgreSQL split not found: {split_name}/{split_type}")

    async def load_all_splits(
        self, split_type: str = "raw", disk_dir: Path | None = None
    ) -> tuple[pl.DataFrame | None, pl.DataFrame | None, pl.DataFrame | None]:
        """
        Load train, valid, test splits with PostgreSQL-first strategy.

        Args:
            split_type: 'raw' or 'homogenized'
            disk_dir: Directory for fallback files

        Returns:
            Tuple of (train_df, valid_df, test_df)
        """
        train_path = disk_dir / "train.parquet" if disk_dir else None
        valid_path = disk_dir / "valid.parquet" if disk_dir else None
        test_path = disk_dir / "test.parquet" if disk_dir else None

        train_df = await self.load_split("train", split_type, train_path)
        valid_df = await self.load_split("valid", split_type, valid_path)
        test_df = await self.load_split("test", split_type, test_path)

        return train_df, valid_df, test_df

    async def load_mappings(
        self, mapping_type: str, disk_path: Path | None = None
    ) -> dict[str, int] | None:
        """
        Load entity/relation mappings with PostgreSQL-first strategy.

        Args:
            mapping_type: 'entity' or 'relation'
            disk_path: Fallback parquet file

        Returns:
            Dictionary {label: id} or None
        """

        try:
            if self.mappings_repo is not None:
                mappings = await self.mappings_repo.load_mappings(mapping_type, use_cache=True)
                if mappings is not None:
                    logger.success(f"{mapping_type} mappings carregados do PostgreSQL (cached)")
                    return mappings
        except Exception as e:
            raise RuntimeError(f"PostgreSQL mappings load failed: {mapping_type}") from e

        raise RuntimeError(f"PostgreSQL mappings not found: {mapping_type}")

    async def check_data_availability(self) -> dict:
        """
        Check what data is available in PostgreSQL vs disk.

        Returns:
            Dictionary with availability status
        """
        status = {"postgresql": {"splits": {}, "mappings": {}}, "disk": {}}

        for split_name in ["train", "valid", "test"]:
            for split_type in ["raw", "homogenized"]:
                exists = False
                if self.splits_repo is not None:
                    exists = await self.splits_repo.split_exists(split_name, split_type)
                status["postgresql"]["splits"][f"{split_name}/{split_type}"] = exists

        for mapping_type in ["entity", "relation"]:
            exists = False
            if self.mappings_repo is not None:
                exists = await self.mappings_repo.mapping_exists(mapping_type)
            status["postgresql"]["mappings"][mapping_type] = exists

        return status

    def load_indexed_data(self, numpy_path: Path) -> np.ndarray:
        """
        Load indexed data from a NumPy file.

        Args:
            numpy_path: Path to the NumPy file

        Returns:
            NumPy array with indexed data

        Raises:
            FileNotFoundError: If file does not exist

        Note:
            This is a compatibility method for StandardDataLoader interface.
            NumPy files are not yet migrated to PostgreSQL.
        """
        raise RuntimeError(
            "NumPy indexed data loading is not supported in PostgreSQL-only mode. "
            "Migrate the dataset to PostgreSQL and use load_split()."
        )

    def load_triples_from_parquet(self, parquet_path: Path) -> list[list[str]]:
        """
        Load triples from a Parquet file.

        Args:
            parquet_path: Path to the Parquet file

        Returns:
            List of triples as [subject, predicate, object]

        Raises:
            ValueError: If required columns are missing

        Note:
            This is a compatibility method for StandardDataLoader interface.
            Use load_split() for PostgreSQL-first loading.
        """
        raise RuntimeError(
            "Parquet triple loading is not supported in PostgreSQL-only mode. "
            "Use load_split() after ingesting data into PostgreSQL."
        )
