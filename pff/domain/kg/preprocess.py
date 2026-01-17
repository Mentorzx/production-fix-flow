import asyncio
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pff.domain.ports.persistence.kg_ports import KGMappingsPort, KGSplitsPort

import numpy as np
import polars as pl

from pff.shared import FileManager, logger
from pff.shared.core.file_manager import ParquetBundle
from pff.shared.core.cache import CacheManager
from pff.shared.hash import stable_hash

from .config import ConfigurationInterface

try:
    from pff.domain.kg.preprocessing import (
        KGPreprocessingPipeline,
        PreprocessingConfig,
    )

    HAS_PREPROCESSING_MODULE = True
except ImportError:
    HAS_PREPROCESSING_MODULE = False
    logger.debug("Centralized preprocessing module not available")

"""
Data preprocessing module for the KGC pipeline.

This module handles data homogenization, entity/relation indexing,
and preparation of data for efficient processing.

Design Pattern: Strategy + Facade
- Uses centralized pff.domain.kg.preprocessing module when available
- Falls back to legacy preprocessing for backward compatibility
"""

file_manager = FileManager()


class DataPreprocessorInterface(ABC):
    """Interface for data preprocessing operations."""

    @abstractmethod
    def run(self) -> None:
        """Execute the complete preprocessing workflow."""
        pass


class DataHomogenizer:
    """Handle data homogenization operations."""

    DATE_PATTERN = r"^\d{4}-\d{2}-\d{2}"

    def homogenize_dataframe(
        self,
        dataframe: pl.DataFrame,
        relation_statistics: pl.DataFrame,
        homogeneity_level: float,
        total_training_triples: int,
        *,
        use_map_elements: bool = False,
    ) -> pl.DataFrame:
        """
        Homogenizes a dataframe by applying specific transformations to object values.

        Args:
            dataframe (pl.DataFrame): Input dataframe containing s,p,o triples
            relation_statistics (pl.DataFrame): DataFrame containing statistics about relations
            homogeneity_level (float): Threshold level for determining support cutoff
            total_training_triples (int): Total training triples for support threshold
        Returns:
            pl.DataFrame: Homogenized dataframe
        """
        support_threshold = max(1, int(total_training_triples * homogeneity_level))

        normalized = dataframe.with_columns(
            [
                pl.col("s").cast(pl.Utf8),
                pl.col("p").cast(pl.Utf8),
                pl.col("o").cast(pl.Utf8),
            ]
        )
        if use_map_elements:
            date_re = re.compile(self.DATE_PATTERN)

            def _homogenize_struct(row: dict) -> str | None:
                obj = row.get("o")
                pred = row.get("p")
                support = row.get("support", 0)
                if obj is None:
                    return None
                if date_re.match(str(obj)):
                    return str(obj)[:4]
                if support is not None and support > support_threshold:
                    return f"{pred}_CATEGORY"
                return str(obj)

            homogenized_dataframe = (
                normalized.lazy()
                .join(relation_statistics.lazy(), on="p", how="left")
                .with_columns(
                    pl.struct(["o", "p", "support"])
                    .map_elements(_homogenize_struct, return_dtype=pl.Utf8)
                    .alias("o_homogenized")
                )
                .select(["s", "p", pl.col("o_homogenized").alias("o")])
            )
            logger.info("Homogeneização usando map_elements")
        else:
            homogenized_dataframe = (
                normalized.lazy()
                .join(relation_statistics.lazy(), on="p", how="left")
                .with_columns(
                    pl.when(pl.col("o").str.contains(r"^\d{4}-") & ~pl.col("o").is_null())
                    .then(pl.col("o").str.slice(0, 4))
                    .when(pl.col("support") > support_threshold)
                    .then(pl.col("p") + "_CATEGORY")
                    .otherwise(pl.col("o"))
                    .alias("o_homogenized")
                )
                .select(["s", "p", pl.col("o_homogenized").alias("o")])
            )

        try:
            import torch

            if torch.cuda.is_available():
                result = homogenized_dataframe.collect(engine="gpu")
                logger.info("Homogeneizacao executada com engine=gpu (polars)")
                return result
        except Exception:
            pass

        return homogenized_dataframe.collect()


class EntityRelationIndexer:
    """Handle entity and relation indexing operations."""

    def __init__(self, cache_manager: CacheManager | None = None) -> None:
        self.cache_manager = cache_manager

    def create_entity_map(self, unique_entities: list[str] | pl.Series) -> pl.DataFrame:
        """
        Create entity to index mapping.

        Args:
            unique_entities: List of unique entity labels

        Returns:
            DataFrame with entity mappings
        """
        return pl.DataFrame({"label": unique_entities}).unique().with_row_index("id")

    def create_relation_map(self, unique_relations: list[str] | pl.Series) -> pl.DataFrame:
        """
        Create relation to index mapping.

        Args:
            unique_relations: List of unique relation labels

        Returns:
            DataFrame with relation mappings
        """
        return pl.DataFrame({"label": unique_relations}).unique().with_row_index("id")

    def index_triples(
        self,
        triples_dataframe: pl.DataFrame,
        entity_map: pl.DataFrame,
        relation_map: pl.DataFrame,
    ) -> np.ndarray:
        """
        Convert triple strings to numeric indices.
        """
        entity_to_id = self._get_cached_mapping("entity", entity_map)
        relation_to_id = self._get_cached_mapping("relation", relation_map)

        if entity_to_id is not None and relation_to_id is not None:
            indexed_lazy = triples_dataframe.lazy().select(
                [
                    pl.col("s").replace(entity_to_id).alias("s_id"),
                    pl.col("p").replace(relation_to_id).alias("p_id"),
                    pl.col("o").replace(entity_to_id).alias("o_id"),
                ]
            )
        else:
            entity_map_lazy = entity_map.lazy()
            relation_map_lazy = relation_map.lazy()

            indexed_lazy = (
                triples_dataframe.lazy()
                .join(entity_map_lazy.rename({"id": "s_id", "label": "s"}), on="s")
                .join(relation_map_lazy.rename({"id": "p_id", "label": "p"}), on="p")
                .join(entity_map_lazy.rename({"id": "o_id", "label": "o"}), on="o")
                .select(["s_id", "p_id", "o_id"])
            )

        try:
            import torch

            if torch.cuda.is_available():
                indexed_dataframe = indexed_lazy.collect(engine="gpu")
                logger.info("Indexacao executada com engine=gpu (polars)")
            else:
                indexed_dataframe = indexed_lazy.collect()
        except Exception:
            indexed_dataframe = indexed_lazy.collect()

        indexed_np = indexed_dataframe.to_numpy(order="c")
        if indexed_np.dtype != np.uint32:
            indexed_np = indexed_np.astype(np.uint32)

        return indexed_np

    def _get_cached_mapping(self, label: str, mapping_df: pl.DataFrame) -> dict[str, int] | None:
        if self.cache_manager is None:
            return None
        if "label" not in mapping_df.columns or "id" not in mapping_df.columns:
            return None

        cache_key = f"kg_indexer:{label}:{stable_hash(mapping_df, truncate=16)}"
        cached = self.cache_manager.get(cache_key)
        if isinstance(cached, dict):
            return cached

        mapping = dict(zip(mapping_df["label"], mapping_df["id"]))
        self.cache_manager.set(cache_key, mapping)
        return mapping


class KGPreprocessor(DataPreprocessorInterface):
    """
    Standard implementation of data preprocessing.

    Unifies indexing and homogenization of the knowledge graph.
    """

    def __init__(
        self,
        configuration: ConfigurationInterface,
        splits_repo: "KGSplitsPort | None" = None,
        mappings_repo: "KGMappingsPort | None" = None,
    ):
        """
        Initialize the preprocessor.

        Args:
            configuration: Configuration object
            splits_repo: Optional splits repository port
            mappings_repo: Optional mappings repository port
        """
        self.configuration = configuration
        self.splits_repo = splits_repo
        self.mappings_repo = mappings_repo
        parameters = configuration.get_preprocessing_parameters()

        self.homogeneity_level = parameters.get("homogeneity_level", 0.5)
        self.minimum_support = parameters.get("min_support", 3)
        self.use_map_elements_homogenizer = bool(
            parameters.get("use_map_elements_homogenizer", False)
        )

        self.use_centralized_preprocessing = parameters.get("use_centralized_preprocessing", False)

        self.homogenizer = DataHomogenizer()
        self.cache_manager = CacheManager()
        self.indexer = EntityRelationIndexer(cache_manager=self.cache_manager)

        logger.info(
            f"DataPreprocessor inicializado com: "
            f"homogeneity_level={self.homogeneity_level}, "
            f"min_support={self.minimum_support}, "
            f"centralized_preprocessing={self.use_centralized_preprocessing}"
        )

    def run(self) -> None:
        """Execute the complete preprocessing workflow."""
        if self.use_centralized_preprocessing and HAS_PREPROCESSING_MODULE:
            logger.info("Usando modulo centralizado de preprocessing...")
            success = self._run_centralized_preprocessing()
            if success:
                return
            logger.warning("Centralized preprocessing failed, falling back to legacy")

        self._run_legacy_preprocessing()

    def _run_centralized_preprocessing(self) -> bool:
        """Run preprocessing using the centralized pff.domain.kg.preprocessing module."""
        try:
            config_path = Path("config/preprocessing.yaml")
            if config_path.exists():
                config = PreprocessingConfig.from_yaml(config_path)
                logger.info(f"Configuracao de preprocessing carregada de {config_path}")
            else:
                config = PreprocessingConfig()
                logger.info("Usando configuracao de preprocessing padrao")

            pipeline = KGPreprocessingPipeline(config)

            raw_splits = self._load_raw_parquet_splits()
            if not raw_splits:
                logger.error("No Parquet data files found")
                return False

            if len(raw_splits) == 3:
                logger.debug("preprocess_splits mode=existing")
                result = pipeline.preprocess_splits(
                    raw_splits["train"],
                    raw_splits["valid"],
                    raw_splits["test"],
                )
            else:
                logger.debug("preprocess_splits mode=create")
                combined = pl.concat(list(raw_splits.values()))
                result = pipeline.preprocess_and_split(combined)

            output_dir = self.configuration.get_mappings_directory()
            file_manager.save(
                result.train,
                output_dir / "train.preprocessed.parquet",
                compression="zstd",
                statistics=True,
                row_group_size=512000,
            )
            if result.valid is not None:
                file_manager.save(
                    result.valid,
                    output_dir / "valid.preprocessed.parquet",
                    compression="zstd",
                    statistics=True,
                    row_group_size=512000,
                )
            if result.test is not None:
                file_manager.save(
                    result.test,
                    output_dir / "test.preprocessed.parquet",
                    compression="zstd",
                    statistics=True,
                    row_group_size=512000,
                )

            stats_path = output_dir / "preprocessing_stats.json"
            file_manager.save(result.stats, stats_path)

            preprocessed_splits = {
                "train": result.train,
                "valid": result.valid if result.valid is not None else pl.DataFrame(),
                "test": result.test if result.test is not None else pl.DataFrame(),
            }

            preprocessed_splits = {k: v for k, v in preprocessed_splits.items() if len(v) > 0}

            self._save_preprocessed_to_postgres(preprocessed_splits)

            homogenized_splits, entity_map, relation_map = self._homogenize_and_map(
                preprocessed_splits
            )
            self._save_mappings(entity_map, relation_map)
            self._index_and_save_numpy(homogenized_splits, entity_map, relation_map)

            logger.success("Preprocessing centralizado concluido com sucesso!")
            return True

        except Exception as e:
            logger.error(f"Centralized preprocessing error: {e}")
            return False

    def _save_preprocessed_to_postgres(self, splits: dict[str, pl.DataFrame]) -> None:
        """
        Save preprocessed splits to PostgreSQL for HPO/pipeline consistency.
        """
        if self.splits_repo is None:
            logger.debug("splits_repo not available; skipping postgres save")
            return

        async def _save():
            await self.splits_repo.delete_preprocessed()

            train_df = splits.get("train")
            valid_df = splits.get("valid")
            test_df = splits.get("test")

            await self.splits_repo.save_preprocessed_splits(
                train_df=train_df if train_df is not None else pl.DataFrame(),
                valid_df=valid_df if valid_df is not None else pl.DataFrame(),
                test_df=test_df if test_df is not None else pl.DataFrame(),
                source="pff_learn_preprocessing",
            )

        try:
            asyncio.run(_save())
            logger.success("Dados preprocessados salvos no PostgreSQL (fonte única para HPO)")
        except Exception as e:
            logger.warning(f"Could not save to PostgreSQL (non-critical): {e}")

    def _run_legacy_preprocessing(self) -> None:
        """Run legacy preprocessing workflow."""
        raise RuntimeError(
            "Legacy preprocessing path is disabled. Use the centralized preprocessing pipeline."
        )

    def _load_raw_parquet_splits(self) -> dict[str, pl.DataFrame]:
        """Load raw Parquet files for all data splits."""
        splits = {}

        for split_name in ["train", "valid", "test"]:
            split_path = self.configuration.get_split_path(split_name)

            if split_path.exists():
                logger.debug(f"split_loading name={split_name} path={split_path}")
                payload = file_manager.read(split_path, lazy=True, streaming=True)
                if isinstance(payload, ParquetBundle):
                    lf = payload.lazyframe()
                    cols = lf.collect_schema().names()
                    if all(c in cols for c in ["s", "p", "o"]):
                        lf = lf.select(["s", "p", "o"])
                    splits[split_name] = lf.collect(engine="streaming")
                else:
                    splits[split_name] = payload

        return splits

    def _filter_orphan_entities(self, splits: dict[str, pl.DataFrame]) -> dict[str, pl.DataFrame]:
        """
        Remove triplas que contenham entidades não presentes no conjunto de treino.

        Args:
            splits: Dicionário com DataFrames de train, valid e test

        Returns:
            Dicionário com DataFrames filtrados
        """
        if "train" not in splits:
            logger.warning("Train set not found. Skipping orphan filtering.")
            return splits

        train_entities = pl.concat([splits["train"]["s"], splits["train"]["o"]]).unique().sort()
        logger.info(f"Entidades únicas no treino: {len(train_entities):,}")

        filtered_splits = {"train": splits["train"]}

        for split_name in ["valid", "test"]:
            if split_name not in splits:
                continue

            original_df = splits[split_name]
            original_count = len(original_df)

            filtered_df = original_df.filter(
                pl.col("s").is_in(train_entities) & pl.col("o").is_in(train_entities)
            )

            filtered_count = len(filtered_df)
            removed_count = original_count - filtered_count

            filtered_splits[split_name] = filtered_df

            logger.info(
                f"Split '{split_name}': {original_count:,} → {filtered_count:,} triplas "
                f"({removed_count:,} órfãs removidas)"
            )

        train_df = splits["train"]
        valid_df = splits.get("valid", pl.DataFrame())
        test_df = splits.get("test", pl.DataFrame())

        train_entities = pl.concat([train_df["s"], train_df["o"]]).unique()
        empty_entities = pl.Series([], dtype=train_entities.dtype)
        valid_entities = (
            pl.concat([valid_df["s"], valid_df["o"]]).unique()
            if len(valid_df) > 0
            else empty_entities
        )
        test_entities = (
            pl.concat([test_df["s"], test_df["o"]]).unique() if len(test_df) > 0 else empty_entities
        )

        train_valid_overlap = int(train_entities.is_in(valid_entities).sum())
        train_test_overlap = int(train_entities.is_in(test_entities).sum())

        logger.info(f"Overlap train-valid: {train_valid_overlap}")
        logger.info(f"Overlap train-test: {train_test_overlap}")

        return filtered_splits

    def _homogenize_and_map(
        self, raw_splits: dict[str, pl.DataFrame]
    ) -> tuple[dict[str, pl.DataFrame], pl.DataFrame, pl.DataFrame]:
        """Orchestrate homogenization and mapping creation."""
        filtered_splits = self._filter_orphan_entities(raw_splits)
        combined_raw = pl.concat(raw_splits.values())
        all_relations = combined_raw.select("p").unique().get_column("p")
        relation_map = self.indexer.create_relation_map(all_relations)
        train_dataframe = filtered_splits["train"]
        total_training_triples = len(train_dataframe)
        relation_statistics = (
            train_dataframe.group_by("p")
            .len()
            .rename({"len": "support"})
            .filter(pl.col("support") >= self.minimum_support)
        )
        homogenized_splits = {}
        homogenized_entity_series: list[pl.Series] = []

        for split_name, dataframe in filtered_splits.items():
            homogenized_dataframe = self.homogenizer.homogenize_dataframe(
                dataframe,
                relation_statistics,
                self.homogeneity_level,
                total_training_triples,
                use_map_elements=self.use_map_elements_homogenizer,
            )
            homogenized_splits[split_name] = homogenized_dataframe

            output_path = (
                self.configuration.get_mappings_directory() / f"{split_name}.homogenized.parquet"
            )
            if len(homogenized_dataframe) > 100_000:
                homogenized_dataframe.lazy().sink_parquet(
                    output_path,
                    compression="zstd",
                    row_group_size=512000,
                )
            else:
                file_manager.save(
                    homogenized_dataframe,
                    output_path,
                    compression="zstd",
                    statistics=True,
                    row_group_size=512000,
                )

            homogenized_entity_series.append(homogenized_dataframe["s"])
            homogenized_entity_series.append(homogenized_dataframe["o"])

        unique_entities = (
            pl.concat(homogenized_entity_series).unique()
            if homogenized_entity_series
            else pl.Series([], dtype=pl.Utf8)
        )
        entity_map = self.indexer.create_entity_map(unique_entities)

        return homogenized_splits, entity_map, relation_map

    def _save_mappings(self, entity_map: pl.DataFrame, relation_map: pl.DataFrame) -> None:
        """Save entity and relation mappings."""
        mappings_directory = self.configuration.get_mappings_directory()

        entity_map_path = mappings_directory / "entity_map.parquet"
        relation_map_path = mappings_directory / "relation_map.parquet"

        file_manager.save(entity_map, entity_map_path)
        file_manager.save(relation_map, relation_map_path)

        logger.info(f"Mapas finais de entidades e relações salvos em {mappings_directory}")

        self._persist_mappings_to_database(entity_map, relation_map)

    def _persist_mappings_to_database(
        self, entity_map: pl.DataFrame, relation_map: pl.DataFrame
    ) -> None:
        """Persist mappings to PostgreSQL for reproducibility."""
        if self.mappings_repo is None:
            logger.debug("mappings_repo not available; skipping postgres save")
            return

        async def _persist() -> None:
            await self.mappings_repo.save_mappings_from_dataframe(
                "entity", entity_map, source="preprocess"
            )
            await self.mappings_repo.save_mappings_from_dataframe(
                "relation", relation_map, source="preprocess"
            )

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(_persist())
        else:
            loop.create_task(_persist())

    def _index_and_save_numpy(
        self,
        homogenized_splits: dict[str, pl.DataFrame],
        entity_map: pl.DataFrame,
        relation_map: pl.DataFrame,
    ) -> None:
        """Convert dataframes to numeric indices and save as NumPy arrays."""
        logger.info("Iniciando indexação para arquivos .npy...")

        for split_name, dataframe in homogenized_splits.items():
            numpy_array = self.indexer.index_triples(dataframe, entity_map, relation_map)

            output_path = getattr(self.configuration, f"{split_name}_numpy_path")

            file_manager.save(numpy_array, output_path)

            logger.info(f" Salvo {split_name}.npy com {len(numpy_array)} triplas indexadas.")

    def update_maps_and_reindex_from_rules(self) -> None:
        """
        Updates entity and relation maps based on rules and re-indexes the data.
        """
        logger.info("Iniciando atualização de mapas e re-indexação com base nas regras...")

        entity_bundle = file_manager.read(self.configuration.get_entity_map_path(), streaming=True)
        relation_bundle = file_manager.read(
            self.configuration.get_relation_map_path(), streaming=True
        )
        entity_map = (
            entity_bundle.lazyframe().collect(engine="streaming")
            if isinstance(entity_bundle, ParquetBundle)
            else entity_bundle
        )
        relation_map = (
            relation_bundle.lazyframe().collect(engine="streaming")
            if isinstance(relation_bundle, ParquetBundle)
            else relation_bundle
        )

        self.configuration.get_rules_path()
        rule_literals = set()

        existing_entities = set(entity_map["label"].to_list())
        new_from_rules = rule_literals - existing_entities

        if new_from_rules:
            logger.info(f"Adicionando {len(new_from_rules)} novas entidades do arquivo de regras.")
            new_df = pl.DataFrame({"label": list(new_from_rules)})

            # Ensure last_id is an integer even if entity_map is empty or has nulls
            max_val = entity_map["id"].max()
            last_id = int(max_val) if max_val is not None else -1

            new_df = new_df.with_row_index("id", offset=last_id + 1)
            entity_map = pl.concat([entity_map, new_df])

        self._save_mappings(entity_map, relation_map)

        homogenized_splits = {}
        for split in ["train", "valid", "test"]:
            split_path = (
                self.configuration.get_mappings_directory() / f"{split}.homogenized.parquet"
            )
            if not split_path.exists():
                continue
            split_bundle = file_manager.read(split_path, streaming=True)
            homogenized_splits[split] = (
                split_bundle.lazyframe().collect(engine="streaming")
                if isinstance(split_bundle, ParquetBundle)
                else split_bundle
            )
        self._index_and_save_numpy(homogenized_splits, entity_map, relation_map)
        logger.info(" Mapas e índices finais atualizados com sucesso.")
