import re
from abc import ABC, abstractmethod

import numpy as np
import polars as pl

from pff.utils import FileManager, logger

from .anyburl import RuleParser
from .config import ConfigurationInterface

"""
Data preprocessing module for the KGC pipeline.

This module handles data homogenization, entity/relation indexing,
and preparation of data for efficient processing.
"""

# Initialize file manager
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
    ) -> pl.DataFrame:
        """
        Homogenizes a dataframe by applying specific transformations to object values based on patterns and thresholds.
        This function performs the following homogenization steps:
        1. For date patterns: extracts only the year (first 4 characters)
        2. For special dates (1970-01-01 or 9999-12-31): replaces with "SPECIAL_DATE"
        3. For relations with support above threshold: appends "_CATEGORY" to predicate
        4. Otherwise: keeps original object value
        Args:
            dataframe (pl.DataFrame): Input dataframe containing s,p,o triples
            relation_statistics (pl.DataFrame): DataFrame containing statistics about relations
            homogeneity_level (float): Threshold level for determining support cutoff
            total_training_triples (int): Total number of training triples used to calculate support threshold
        Returns:
            pl.DataFrame: Homogenized dataframe with transformed object values
        Notes:
            - The support threshold is calculated as max(1, total_training_triples * homogeneity_level)
            - Special dates like 1970-01-01 and 9999-12-31 are treated differently from regular dates
            - Relations with high support are categorized by appending "_CATEGORY" to predicate
        """
        support_threshold = max(1, int(total_training_triples * homogeneity_level))
        
        homogenized_dataframe = (
            dataframe.lazy()
            .join(relation_statistics.lazy(), on="p", how="left")
            .with_columns(
                pl.when(
                    pl.col("o").str.contains(self.DATE_PATTERN) & 
                    ~pl.col("o").str.contains("1970-01-01") &  
                    ~pl.col("o").str.contains("9999-12-31") &  # ADICIONE ESTA LINHA
                    ~pl.col("o").is_null()
                )
                .then(pl.col("o").str.slice(0, 4))
                .when(
                    # ADICIONE ESTE BLOCO PARA SUBSTITUIR DATAS ESPECIAIS
                    pl.col("o").str.contains("1970-01-01") | 
                    pl.col("o").str.contains("9999-12-31")
                )
                .then(pl.lit("SPECIAL_DATE"))
                .when(pl.col("support") > support_threshold)
                .then(pl.col("p") + "_CATEGORY")
                .otherwise(pl.col("o"))
                .alias("o_homogenized")
            )
            .select(["s", "p", pl.col("o_homogenized").alias("o")])
            .collect()
        )

        return homogenized_dataframe


class EntityRelationIndexer:
    """Handle entity and relation indexing operations."""

    def create_entity_map(self, unique_entities: list[str]) -> pl.DataFrame:
        """
        Create entity to index mapping.

        Args:
            unique_entities: List of unique entity labels

        Returns:
            DataFrame with entity mappings
        """
        return pl.DataFrame({"label": unique_entities}).unique().with_row_index("id")

    def create_relation_map(self, unique_relations: list[str]) -> pl.DataFrame:
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

        Args:
            triples_dataframe: DataFrame with string triples
            entity_map: Entity to index mapping
            relation_map: Relation to index mapping

        Returns:
            NumPy array with indexed triples
        """
        entity_map_lazy = entity_map.lazy()
        relation_map_lazy = relation_map.lazy()

        indexed_dataframe = (
            triples_dataframe.lazy()
            .join(entity_map_lazy.rename({"id": "s_id", "label": "s"}), on="s")
            .join(relation_map_lazy.rename({"id": "p_id", "label": "p"}), on="p")
            .join(entity_map_lazy.rename({"id": "o_id", "label": "o"}), on="o")
            .select(["s_id", "p_id", "o_id"])
            .collect()
        )

        return indexed_dataframe.to_numpy().astype(np.uint32)


class KGPreprocessor(DataPreprocessorInterface):
    """
    Standard implementation of data preprocessing.

    Unifies indexing and homogenization of the knowledge graph.
    Reads raw Parquet files, applies homogenization, and saves
    entity/relation maps and indexed triple files.
    """

    def __init__(self, configuration: ConfigurationInterface):
        """
        Initialize the preprocessor.

        Args:
            configuration: Configuration object
        """
        self.configuration = configuration
        parameters = configuration.get_preprocessing_parameters()

        self.homogeneity_level = parameters.get("homogeneity_level", 0.5)
        self.minimum_support = parameters.get("min_support", 3)

        self.homogenizer = DataHomogenizer()
        self.indexer = EntityRelationIndexer()

        logger.info(
            f"DataPreprocessor inicializado com: "
            f"homogeneity_level={self.homogeneity_level}, "
            f"min_support={self.minimum_support}"
        )

    def run(self) -> None:
        """Execute the complete preprocessing workflow."""
        logger.info("Iniciando o fluxo de pré-processamento de dados...")

        # Load raw data splits
        raw_splits = self._load_raw_parquet_splits()
        if not raw_splits:
            logger.error("Nenhum arquivo de dados Parquet encontrado. Abortando.")
            return

        # Homogenize and create mappings
        homogenized_splits, entity_map, relation_map = self._homogenize_and_map(
            raw_splits
        )

        # Save mappings
        self._save_mappings(entity_map, relation_map)

        # Index and save as NumPy arrays
        self._index_and_save_numpy(homogenized_splits, entity_map, relation_map)

        logger.info("✅ Fluxo de pré-processamento concluído com sucesso!")

    def _load_raw_parquet_splits(self) -> dict[str, pl.DataFrame]:
        """Load raw Parquet files for all data splits."""
        splits = {}

        for split_name in ["train", "valid", "test"]:
            split_path = self.configuration.get_split_path(split_name)

            if split_path.exists():
                logger.info(f"Carregando split '{split_name}' de {split_path}")
                splits[split_name] = file_manager.read(split_path)

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
            logger.warning("Conjunto de treino não encontrado. Pulando filtragem de órfãs.")
            return splits
        
        train_entities = set()
        train_entities.update(splits["train"]["s"].to_list())
        train_entities.update(splits["train"]["o"].to_list())
        
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
        
        train_entities = set(train_df['s'].unique()) | set(train_df['o'].unique())
        valid_entities = set(valid_df['s'].unique()) | set(valid_df['o'].unique()) if len(valid_df) > 0 else set()
        test_entities = set(test_df['s'].unique()) | set(test_df['o'].unique()) if len(test_df) > 0 else set()
        
        logger.info(f"Overlap train-valid: {len(train_entities & valid_entities)}")
        logger.info(f"Overlap train-test: {len(train_entities & test_entities)}")
            
        return filtered_splits

    def _homogenize_and_map(
        self, raw_splits: dict[str, pl.DataFrame]
    ) -> tuple[dict[str, pl.DataFrame], pl.DataFrame, pl.DataFrame]:
        """Orchestrate homogenization and mapping creation."""
        filtered_splits = self._filter_orphan_entities(raw_splits)
        combined_raw = pl.concat(raw_splits.values())
        all_relations = combined_raw.select("p").unique()["p"].to_list()
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
        all_homogenized_entities = []

        for split_name, dataframe in filtered_splits.items():
            homogenized_dataframe = self.homogenizer.homogenize_dataframe(
                dataframe,
                relation_statistics,
                self.homogeneity_level,
                total_training_triples,
            )
            homogenized_splits[split_name] = homogenized_dataframe
            homogenized_dataframe.write_parquet(
                self.configuration.get_pyclause_directory()
                / f"{split_name}.homogenized.parquet"
            )
            all_homogenized_entities.extend(homogenized_dataframe["s"].to_list())
            all_homogenized_entities.extend(homogenized_dataframe["o"].to_list())

        logger.info(
            "Extraindo entidades literais do arquivo de regras para garantir a consistência do dicionário..."
        )
        rules_path = self.configuration.get_rules_path()
        if rules_path.exists():
            rule_parser = RuleParser()
            rules_as_strings, _ = rule_parser.parse_rules_file(rules_path)
            literal_pattern = re.compile(
                r"\'(.*?)\'|\"([^\"]*)\"|(?<=,)([\w\-\:]+)(?=\))"
            )
            rule_literals = set()
            for rule_string in rules_as_strings:
                found_literals = literal_pattern.findall(rule_string)
                for lit_tuple in found_literals:
                    literal = next((lit for lit in lit_tuple if lit), None)
                    if literal:
                        rule_literals.add(literal)

            if rule_literals:
                logger.info(
                    f"Encontradas {len(rule_literals)} entidades literais únicas nas regras."
                )
                existing_entities = set(all_homogenized_entities)
                new_from_rules = rule_literals - existing_entities

                if new_from_rules:
                    logger.info(
                        f"Adicionando {len(new_from_rules)} novas entidades do arquivo de regras ao mapa de entidades."
                    )
                    all_homogenized_entities.extend(list(new_from_rules))

        unique_entities = list(set(all_homogenized_entities))
        entity_map = self.indexer.create_entity_map(unique_entities)

        return homogenized_splits, entity_map, relation_map

    def _save_mappings(
        self, entity_map: pl.DataFrame, relation_map: pl.DataFrame
    ) -> None:
        """Save entity and relation mappings."""
        pyclause_directory = self.configuration.get_pyclause_directory()

        entity_map_path = pyclause_directory / "entity_map.parquet"
        relation_map_path = pyclause_directory / "relation_map.parquet"

        file_manager.save(entity_map, entity_map_path)
        file_manager.save(relation_map, relation_map_path)

        logger.info(
            f"Mapas finais de entidades e relações salvos em {pyclause_directory}"
        )

    def _index_and_save_numpy(
        self,
        homogenized_splits: dict[str, pl.DataFrame],
        entity_map: pl.DataFrame,
        relation_map: pl.DataFrame,
    ) -> None:
        """Convert dataframes to numeric indices and save as NumPy arrays."""
        logger.info("Iniciando indexação para arquivos .npy...")

        for split_name, dataframe in homogenized_splits.items():
            numpy_array = self.indexer.index_triples(
                dataframe, entity_map, relation_map
            )

            # Get output path from configuration
            output_path = getattr(self.configuration, f"{split_name}_numpy_path")

            # Save as NumPy array
            np.save(output_path, numpy_array)

            logger.info(
                f"✅ Salvo {split_name}.npy com {len(numpy_array)} triplas indexadas."
            )

    def update_maps_and_reindex_from_rules(self) -> None:
        """
        Updates entity and relation maps based on rules and re-indexes the data.
        This method performs the following steps:
        1. Loads existing entity and relation maps
        2. Extracts literals from rules file if it exists
        3. Adds any new entities found in rules to the entity map
        4. Re-indexes all data splits using the updated maps
        5. Saves the final indexed data and updated maps
        The method expects the following files to exist:
        - Entity map file
        - Relation map file
        - Homogenized data splits (train/valid/test)
        Raises:
            FileNotFoundError: If initial entity/relation maps are not found
        Returns:
            None
        """
        logger.info(
            "Iniciando atualização de mapas e re-indexação com base nas regras..."
        )

        entity_map = file_manager.read(self.configuration.get_entity_map_path())
        relation_map = file_manager.read(self.configuration.get_relation_map_path())

        rules_path = self.configuration.get_rules_path()
        rule_literals = set()
        if rules_path.exists():
            rules, _ = RuleParser().parse_rules_file(rules_path)
            pattern = re.compile(r"\'(.*?)\'|\"([^\"]*)\"|(?<=,)([\w\-\:]+)(?=\))")
            for rule in rules:
                for match in pattern.findall(rule):
                    literal = next((lit for lit in match if lit), None)
                    if literal:
                        rule_literals.add(literal)

        existing_entities = set(entity_map["label"].to_list())
        new_from_rules = rule_literals - existing_entities

        if new_from_rules:
            logger.info(
                f"Adicionando {len(new_from_rules)} novas entidades do arquivo de regras."
            )
            new_df = pl.DataFrame({"label": list(new_from_rules)})
            last_id = entity_map["id"].max() if len(entity_map) > 0 else -1
            new_df = new_df.with_row_index("id", offset=last_id + 1)
            entity_map = pl.concat([entity_map, new_df])

        self._save_mappings(entity_map, relation_map)

        homogenized_splits = {
            split: file_manager.read(
                self.configuration.get_pyclause_directory()
                / f"{split}.homogenized.parquet"
            )
            for split in ["train", "valid", "test"]
            if (
                self.configuration.get_pyclause_directory()
                / f"{split}.homogenized.parquet"
            ).exists()
        }
        self._index_and_save_numpy(homogenized_splits, entity_map, relation_map)
        logger.info("✅ Mapas e índices finais atualizados com sucesso.")
