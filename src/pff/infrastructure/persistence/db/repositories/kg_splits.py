"""
KGSplitsRepositoryLance - LanceDB Implementation for KG Splits Data.

Design Patterns Applied:
- Repository Pattern: Encapsulates data access logic
- Data Lakehouse: Uses LanceDB for embedded storage
- Zero-Copy: Integration with Polars/Arrow
- Versioning: Built-in dataset versioning

Performance Target:
- >50x faster writes than Postgres
- Zero-copy reads to Polars
"""

from pathlib import Path
from typing import Any, cast

try:
    import lancedb

    LANCEDB_AVAILABLE = True
except ImportError:
    lancedb = None  # type: ignore[assignment]
    LANCEDB_AVAILABLE = False

import polars as pl

from pff.infrastructure.persistence.db.repositories.kg_mappings import (
    KGMappingsRepository,
)
from pff.shared.core.logging import logger

LANCE_DB_PATH = "data/lancedb"
SPLITS_TABLE = "kg_splits"


class KGSplitsRepositoryLance:
    """
    Repository for managing KG split data (train/valid/test) using LanceDB.
    """

    def __init__(self, db_path: str = LANCE_DB_PATH):
        """Initialize repository with LanceDB connection."""
        if not LANCEDB_AVAILABLE:
            raise ImportError(
                "lancedb is required for KGSplitsRepositoryLance. "
                "Install it with: pip install lancedb"
            )
        self.db_path = db_path
        self.db: Any = lancedb.connect(self.db_path)
        self.mappings_repo = KGMappingsRepository()
        self._table: Any = None

        Path(self.db_path).mkdir(parents=True, exist_ok=True)

    def _get_table(self):
        """Lazy load table reference."""
        if self._table is not None:
            return self._table

        if SPLITS_TABLE in self.db.list_tables().tables:
            self._table = self.db.open_table(SPLITS_TABLE)
        return self._table

    async def save_split(
        self,
        split_name: str,
        split_type: str,
        df: pl.DataFrame,
        source: str = "correct.parquet",
        batch_size: int = 10000,
    ) -> int:
        """
        Save KG split data to LanceDB.

        Args:
            split_name: 'train', 'valid', or 'test'
            split_type: 'raw' or 'homogenized'
            df: DataFrame with columns [s, p, o, sample_id (optional)]
            source: Data source identifier
            batch_size: Ignored for Lance (writes whole arrow table)

        Returns:
            Number of records inserted
        """
        required_cols = ["s", "p", "o"]
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"DataFrame must have columns: {required_cols}")

        total_rows = len(df)
        logger.debug(f"saving_triples n={total_rows:,} split={split_name}/{split_type}")

        df_to_save = df.with_columns(
            [
                pl.lit(split_name).alias("split_name"),
                pl.lit(split_type).alias("split_type"),
                pl.lit(source).alias("source"),
                pl.col("s").cast(pl.Utf8),
                pl.col("p").cast(pl.Utf8),
                pl.col("o").cast(pl.Utf8),
            ]
        )

        if "sample_id" not in df_to_save.columns:
            df_to_save = df_to_save.with_columns(
                pl.lit(None, dtype=pl.Utf8).alias("sample_id")
            )
        else:
            df_to_save = df_to_save.with_columns(pl.col("sample_id").cast(pl.Utf8))

        table = self._get_table()
        if table is not None:
            table.delete(f"split_name = '{split_name}' AND split_type = '{split_type}'")

            table.add(df_to_save.to_arrow())
        else:
            self._table = self.db.create_table(SPLITS_TABLE, df_to_save.to_arrow())

        logger.debug(f"triplas_salvas n={total_rows:,} split={split_name}/{split_type}")
        return total_rows

    async def load_split(
        self,
        split_name: str,
        split_type: str,
        map_to_ints: bool = True,
    ) -> pl.DataFrame | None:
        """
        Load KG split data from LanceDB.

        Returns:
            DataFrame with columns [s, p, o, sample_id] or None if not found
        """
        logger.debug(f"triplas_carregando split={split_name}/{split_type}")

        table = self._get_table()
        if table is None:
            logger.warning(f"Table {SPLITS_TABLE} not found")
            return None

        arrow_table = (
            table.search()
            .where(f"split_name = '{split_name}' AND split_type = '{split_type}'")
            .select(["s", "p", "o", "sample_id"])
            .to_arrow()
        )

        if arrow_table.num_rows == 0:
            logger.warning(f"Split {split_name}/{split_type} not found in LanceDB")
            return None

        df_loaded: pl.DataFrame = cast(pl.DataFrame, pl.from_arrow(arrow_table))

        logger.debug(
            f"triplas_carregadas (lance) n={len(df_loaded):,} split={split_name}/{split_type}"
        )

        if map_to_ints:
            df_final, _, _ = await self._map_to_ints(
                df_loaded, f"{split_name}_{split_type}"
            )
            return df_final

        return df_loaded

    async def _map_to_ints(
        self, df: pl.DataFrame, source_key: str
    ) -> tuple[pl.DataFrame, dict[str, int], dict[str, int]]:
        """Map subject/predicate/object to contiguous ints."""

        if df.is_empty():
            return df, {}, {}

        int_types = {
            pl.Int8,
            pl.Int16,
            pl.Int32,
            pl.Int64,
            pl.UInt8,
            pl.UInt16,
            pl.UInt32,
            pl.UInt64,
        }
        if (
            df.schema["s"] in int_types
            and df.schema["p"] in int_types
            and df.schema["o"] in int_types
        ):
            return df, {}, {}

        entities = sorted(pl.concat([df["s"], df["o"]]).unique().to_list())
        relations = sorted(df["p"].unique().to_list())
        entity_map = {val: idx for idx, val in enumerate(entities)}
        relation_map = {val: idx for idx, val in enumerate(relations)}

        mapped = (
            df.lazy()
            .with_columns(
                pl.col("s")
                .replace_strict(entity_map, default=0)
                .cast(pl.Int64)
                .alias("s"),
                pl.col("o")
                .replace_strict(entity_map, default=0)
                .cast(pl.Int64)
                .alias("o"),
                pl.col("p")
                .replace_strict(relation_map, default=0)
                .cast(pl.Int64)
                .alias("p"),
            )
            .collect(engine="streaming")
        )
        logger.info(
            f"Split {source_key} mapeado para IDs inteiros contiguos "
            f"(entidades={len(entities):,}, relacoes={len(relations)})"
        )
        try:
            await self.mappings_repo.save_mappings(
                "entity", entity_map, source=source_key
            )
            await self.mappings_repo.save_mappings(
                "relation", relation_map, source=source_key
            )
        except Exception as exc:
            logger.warning(f"Failed to persist mappings for {source_key}: {exc}")

        return mapped, entity_map, relation_map

    async def split_exists(self, split_name: str, split_type: str) -> bool:
        """Check if a split exists in LanceDB."""
        table = self._get_table()
        if table is None:
            return False

        count = (
            table.search()
            .where(f"split_name = '{split_name}' AND split_type = '{split_type}'")
            .limit(1)
            .to_arrow()
            .num_rows
        )

        return count > 0

    async def delete_split(self, split_name: str, split_type: str) -> int:
        """Delete a specific split from LanceDB."""
        table = self._get_table()
        if table is None:
            return 0

        table.delete(f"split_name = '{split_name}' AND split_type = '{split_type}'")
        return 1

    async def delete_all(self) -> int:
        """Delete all splits (drops table)."""
        if SPLITS_TABLE in self.db.list_tables().tables:
            self.db.drop_table(SPLITS_TABLE)
            self._table = None
            logger.info(" Todas as triplas deletadas do LanceDB (Drop Table)")
            return 1
        return 0

    async def truncate_all(self) -> int:
        return await self.delete_all()

    async def vacuum_full(self) -> None:
        """Run compaction/cleanup."""
        table = self._get_table()
        if table:
            table.compact_files()
            table.cleanup_old_versions()

    async def get_statistics(self) -> dict:
        """Get statistics about stored splits."""
        table = self._get_table()
        if table is None:
            return {}

        arrow_table = table.to_arrow()  # noqa: F841

        import duckdb

        query = """
        SELECT split_name, split_type, COUNT(*) as count, COUNT(DISTINCT sample_id) as unique_samples
        FROM arrow_table
        GROUP BY split_name, split_type
        ORDER BY split_name, split_type
        """

        try:
            df_stats = duckdb.query(query).to_df()
            stats = {}
            for _, row in df_stats.iterrows():
                key = f"{row['split_name']}/{row['split_type']}"
                stats[key] = {
                    "count": int(row["count"]),
                    "unique_samples": int(row["unique_samples"]),
                }
            return stats
        except Exception as e:
            logger.warning(f"Failed to get statistics: {e}")
            return {}

    async def save_preprocessed_splits(
        self,
        train_df: pl.DataFrame,
        valid_df: pl.DataFrame,
        test_df: pl.DataFrame | None = None,
        source: str = "preprocessing_pipeline",
        batch_size: int = 10000,
    ) -> dict[str, int]:
        """Save preprocessed KG splits to LanceDB."""
        results = {}

        logger.info("Salvando splits preprocessados no LanceDB...")

        if train_df is not None and len(train_df) > 0:
            results["train"] = await self.save_split(
                "train", "preprocessed", train_df, source, batch_size
            )

        if valid_df is not None and len(valid_df) > 0:
            results["valid"] = await self.save_split(
                "valid", "preprocessed", valid_df, source, batch_size
            )

        if test_df is not None and len(test_df) > 0:
            results["test"] = await self.save_split(
                "test", "preprocessed", test_df, source, batch_size
            )

        total = sum(results.values())
        logger.success(
            f"Splits preprocessados salvos: {total:,} triplas "
            f"(train={results.get('train', 0):,}, "
            f"valid={results.get('valid', 0):,}, "
            f"test={results.get('test', 0):,})"
        )

        return results

    async def load_preprocessed_splits(
        self,
        fallback_to_raw: bool = True,
        map_to_ints: bool = True,
    ) -> tuple[pl.DataFrame | None, pl.DataFrame | None, pl.DataFrame | None, dict]:
        metadata = {"source": None, "splits_loaded": []}

        train_exists = await self.split_exists("train", "preprocessed")
        valid_exists = await self.split_exists("valid", "preprocessed")

        if train_exists and valid_exists:
            logger.info("Carregando splits preprocessados do LanceDB...")
            metadata["source"] = "preprocessed"

            train_df = await self.load_split(
                "train", "preprocessed", map_to_ints=map_to_ints
            )
            valid_df = await self.load_split(
                "valid", "preprocessed", map_to_ints=map_to_ints
            )
            test_df = await self.load_split(
                "test", "preprocessed", map_to_ints=map_to_ints
            )

            metadata["splits_loaded"] = ["train", "valid"]
            if test_df is not None:
                metadata["splits_loaded"].append("test")

            train_len = len(train_df) if train_df is not None else 0
            valid_len = len(valid_df) if valid_df is not None else 0
            logger.success(
                f"Splits preprocessados carregados: train={train_len:,}, valid={valid_len:,}"
            )

            return train_df, valid_df, test_df, metadata

        elif fallback_to_raw:
            logger.debug("Preprocessed splits not found, loading raw splits...")
            metadata["source"] = "raw"
            metadata["needs_preprocessing"] = True

            train_df = await self.load_split("train", "raw", map_to_ints=map_to_ints)
            valid_df = await self.load_split("valid", "raw", map_to_ints=map_to_ints)
            test_df = await self.load_split("test", "raw", map_to_ints=map_to_ints)

            if train_df is not None:
                metadata["splits_loaded"].append("train")
            if valid_df is not None:
                metadata["splits_loaded"].append("valid")
            if test_df is not None:
                metadata["splits_loaded"].append("test")

            return train_df, valid_df, test_df, metadata

        else:
            logger.debug("Preprocessed splits not found")
            return None, None, None, metadata

    async def preprocessed_exists(self) -> bool:
        train_exists = await self.split_exists("train", "preprocessed")
        valid_exists = await self.split_exists("valid", "preprocessed")
        return train_exists and valid_exists

    async def delete_preprocessed(self) -> int:
        total_deleted = 0
        for split_name in ["train", "valid", "test"]:
            await self.delete_split(split_name, "preprocessed")
            total_deleted += 1

        return total_deleted
