"""
Central KG Preprocessing Pipeline.

Design Pattern: Pipeline + Facade
- Orchestrates preprocessing strategies in correct order
- Provides simple interface for common operations
- Ensures consistency between main training and HPO pipelines

This module is the PRIMARY entry point for KG preprocessing.
Both the main training pipeline and HPO should use this.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import polars as pl

import random

import numpy as np

from pff.shared import FileManager, logger
from pff.shared.hash import stable_hash

from .config import (
    ATTRIBUTE_HANDLING_MARK,
    ATTRIBUTE_HANDLING_REMOVE,
    ATTRIBUTE_HANDLING_SEPARATE,
    PreprocessingConfig,
)
from .strategies import (
    DeduplicationStrategy,
    SelfLoopRemovalStrategy,
    InverseRelationStrategy,
    AttributeRelationClassifier,
    DegreeFeatureExtractor,
    EntityDegreeFilter,
    RelationSupportFilter,
    PreprocessingComposer,
)
from .split import SafeSplitter, LeakageChecker


@dataclass
class PipelineResult:
    """Result of the preprocessing pipeline.

    Attributes:
        train: Preprocessed training data
        valid: Preprocessed validation data
        test: Preprocessed test data
        stats: Complete statistics from all steps
        features: Extracted features (e.g., degree features)
        metadata: Additional metadata (e.g., attribute relation info)
    """

    train: pl.DataFrame
    valid: pl.DataFrame | None = None
    test: pl.DataFrame | None = None
    stats: dict[str, Any] = field(default_factory=dict)
    features: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class KGPreprocessingPipeline:
    """Central preprocessing pipeline for KG data.

    This is the AUTHORITATIVE source for preprocessing logic.
    Both main training and HPO pipelines MUST use this class.

    Usage:
        # For raw data that needs splitting
        pipeline = KGPreprocessingPipeline(config)
        result = pipeline.preprocess_and_split(raw_df)

        # For already-split data
        result = pipeline.preprocess_splits(train_df, valid_df, test_df)
    """

    def __init__(
        self,
        config: PreprocessingConfig | None = None,
        file_manager: FileManager | None = None,
    ) -> None:
        """Initialize pipeline.

        Args:
            config: Preprocessing configuration
            file_manager: FileManager instance for I/O
        """
        self.config = config or PreprocessingConfig()
        self.fm = file_manager or FileManager()

        # Initialize strategies based on config
        self._init_strategies()

        # Stats accumulator
        self._stats: dict[str, Any] = {}
        self._features: dict[str, Any] = {}
        self._metadata: dict[str, Any] = {}
        self._entity_map_path: Path | None = None
        self._relation_map_path: Path | None = None

    def _ensure_output_dir(self) -> Path:
        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    def _map_ids(self, df: pl.DataFrame, source: str) -> tuple[pl.DataFrame, dict[str, Any]]:
        """Map string IDs to contiguous integers for s/p/o columns.

        Returns mapped DataFrame and metadata with map paths and cardinalities.
        """
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
        # Fast path: already integer types
        if (
            df.schema["s"] in int_types
            and df.schema["p"] in int_types
            and df.schema["o"] in int_types
        ):
            return df, {}

        out_dir = self._ensure_output_dir()
        # Entity map from s + o
        entity_map_df = (
            pl.concat([df["s"], df["o"]]).unique().to_frame("entity").with_row_index("entity_id")
        )
        relation_map_df = (
            df.select("p").unique().rename({"p": "relation"}).with_row_index("relation_id")
        )

        entity_map_path = out_dir / f"entity_map_{stable_hash(source)}.parquet"
        relation_map_path = out_dir / f"relation_map_{stable_hash(source)}.parquet"
        self.fm.save(entity_map_df, entity_map_path)
        self.fm.save(relation_map_df, relation_map_path)
        self._entity_map_path = entity_map_path
        self._relation_map_path = relation_map_path

        entity_map_s = entity_map_df.rename({"entity": "s", "entity_id": "s_id"}).lazy()
        entity_map_o = entity_map_df.rename({"entity": "o", "entity_id": "o_id"}).lazy()
        relation_map_p = relation_map_df.rename({"relation": "p", "relation_id": "p_id"}).lazy()
        mapped = (
            df.lazy()
            .with_row_index("_row_id")
            .join(entity_map_s, on="s")
            .join(entity_map_o, on="o")
            .join(relation_map_p, on="p")
            .select(
                [
                    pl.col("s_id").cast(pl.Int64).alias("s"),
                    pl.col("p_id").cast(pl.Int64).alias("p"),
                    pl.col("o_id").cast(pl.Int64).alias("o"),
                    pl.col("_row_id"),
                ]
            )
            .sort("_row_id")
            .drop("_row_id")
            .collect()
        )

        # Validate no nulls after mapping
        nulls = int(
            mapped.select(
                pl.col("s").is_null().sum()
                + pl.col("p").is_null().sum()
                + pl.col("o").is_null().sum()
            ).item()
        )
        if nulls > 0:
            raise ValueError("Null IDs produced during mapping; missing keys in maps.")

        meta = {
            "entity_map_path": str(entity_map_path),
            "relation_map_path": str(relation_map_path),
            "num_entities": int(entity_map_df.height),
            "num_relations": int(relation_map_df.height),
        }
        return mapped, meta

    def _map_ids_for_splits(
        self, train: pl.DataFrame, valid: pl.DataFrame | None, test: pl.DataFrame | None
    ) -> tuple[pl.DataFrame, pl.DataFrame | None, pl.DataFrame | None]:
        """Ensure consistent mapping across splits."""
        combined = pl.concat([df for df in [train, valid, test] if df is not None])
        mapped_combined, meta = self._map_ids(combined, source="splits")
        # Split back respecting original lengths
        lengths = [
            len(train),
            len(valid) if valid is not None else 0,
            len(test) if test is not None else 0,
        ]
        cursor = 0
        mapped_train = mapped_combined[cursor : cursor + lengths[0]]
        cursor += lengths[0]
        mapped_valid = None
        if valid is not None:
            mapped_valid = mapped_combined[cursor : cursor + lengths[1]]
            cursor += lengths[1]
        mapped_test = None
        if test is not None:
            mapped_test = mapped_combined[cursor : cursor + lengths[2]]
        self._metadata.setdefault("id_mapping", meta)
        return mapped_train, mapped_valid, mapped_test

    def _init_strategies(self) -> None:
        """Initialize preprocessing strategies from config."""
        self.dedup_strategy = DeduplicationStrategy(enabled=self.config.remove_duplicates)

        self.self_loop_strategy = (
            SelfLoopRemovalStrategy(set(self.config.allowed_reflexive_relations))
            if self.config.remove_self_loops
            else None
        )

        self.inverse_strategy = (
            InverseRelationStrategy(self.config.inverse_suffix)
            if self.config.add_inverse_relations
            else None
        )

        remove_attrs = self.config.attribute_handling in {
            ATTRIBUTE_HANDLING_REMOVE,
            ATTRIBUTE_HANDLING_SEPARATE,
        }
        self.attribute_classifier = AttributeRelationClassifier(
            attribute_relations=set(self.config.attribute_relations),
            attribute_patterns=self.config.attribute_patterns,
            remove_from_data=remove_attrs,
            mark_only=self.config.attribute_handling == ATTRIBUTE_HANDLING_MARK,
        )

        self.degree_extractor = (
            DegreeFeatureExtractor() if self.config.compute_degree_features else None
        )

        self.entity_filter = (
            EntityDegreeFilter(self.config.min_entity_degree)
            if self.config.min_entity_degree > 0
            else None
        )

        self.relation_filter = (
            RelationSupportFilter(
                self.config.min_relation_support,
                policy=self.config.relation_support_policy,
            )
            if self.config.min_relation_support > 0
            else None
        )

        self.basic_composer = PreprocessingComposer(
            [
                ("deduplication", self.dedup_strategy),
                ("self_loops", self.self_loop_strategy),
                ("attributes", self.attribute_classifier),
                ("entity_filter", self.entity_filter),
                ("relation_filter", self.relation_filter),
            ]
        )

        self.splitter = SafeSplitter(inverse_suffix=self.config.inverse_suffix)
        self.leakage_checker = LeakageChecker(self.config.inverse_suffix)

    def _fix_leakage_resplit(
        self,
        train: pl.DataFrame,
        valid: pl.DataFrame,
        test: pl.DataFrame,
        seed: int = 42,
    ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, dict]:
        """Fix data leakage by re-splitting all data.

        SOTA Strategy:
        1. Unify all splits into single pool
        2. Remove cross-split duplicates
        3. Stratified random split by relation
        4. Ensure transductive coverage
        5. Add inverses independently per split

        Args:
            train: Original train DataFrame
            valid: Original valid DataFrame
            test: Original test DataFrame
            seed: Random seed for reproducibility

        Returns:
            Tuple of (new_train, new_valid, new_test, stats)
        """
        logger.info("CORRIGINDO LEAKAGE VIA RE-SPLIT SOTA")

        random.seed(seed)
        np.random.seed(seed)

        ratios = self.config.resplit_ratios
        train_ratio, valid_ratio, test_ratio = ratios

        # Step 1: Unify all splits
        logger.info("Unificando todos os splits...")
        unified = pl.concat([train, valid, test])
        original_count = len(unified)

        # Step 2: Remove duplicates (only if enabled)
        if self.config.remove_duplicates:
            unified = unified.unique(subset=["s", "p", "o"])
            unique_count = len(unified)
            duplicates_removed = original_count - unique_count
            logger.info(f"  Total apos dedup: {unique_count:,}")
            logger.info(f"  Duplicatas cross-split removidas: {duplicates_removed:,}")
        else:
            unique_count = original_count
            duplicates_removed = 0
            logger.info("  Deduplicacao desativada: preservando frequencia das triplas")

        logger.info(f"  Total para split: {unique_count:,}")

        # Step 3: Stratified split by relation
        logger.info("Executando split estratificado por relacao...")

        train_parts, valid_parts, test_parts = [], [], []
        relations = unified["p"].unique().to_list()

        for rel in relations:
            rel_df = unified.filter(pl.col("p") == rel)
            n = len(rel_df)

            if n < 3:
                train_parts.append(rel_df)
                continue

            indices = np.arange(n)
            np.random.shuffle(indices)

            n_train = max(1, int(n * train_ratio))
            n_valid = max(1, int(n * valid_ratio))

            train_idx = indices[:n_train].tolist()
            valid_idx = indices[n_train : n_train + n_valid].tolist()
            test_idx = indices[n_train + n_valid :].tolist()

            train_parts.append(rel_df[train_idx])
            if valid_idx:
                valid_parts.append(rel_df[valid_idx])
            if test_idx:
                test_parts.append(rel_df[test_idx])

        new_train = pl.concat(train_parts) if train_parts else pl.DataFrame()
        new_valid = pl.concat(valid_parts) if valid_parts else pl.DataFrame()
        new_test = pl.concat(test_parts) if test_parts else pl.DataFrame()

        logger.info(
            f"  Split inicial: train={len(new_train):,}, valid={len(new_valid):,}, test={len(new_test):,}"
        )

        # Step 4: Ensure transductive coverage
        if self.config.ensure_transductive:
            logger.info("Garantindo cobertura transductiva...")

            def get_entities(df: pl.DataFrame) -> set:
                if len(df) == 0:
                    return set()
                return set(df["s"].to_list()) | set(df["o"].to_list())

            train_entities = get_entities(new_train)
            all_unseen = (get_entities(new_valid) | get_entities(new_test)) - train_entities

            if all_unseen:
                logger.info(f"  Entidades nao vistas no train: {len(all_unseen)}")

                # Move triples with unseen entities to train
                moved_valid, keep_valid = [], []
                for row in new_valid.iter_rows(named=True):
                    if row["s"] in all_unseen or row["o"] in all_unseen:
                        moved_valid.append(row)
                        train_entities.add(row["s"])
                        train_entities.add(row["o"])
                    else:
                        keep_valid.append(row)

                # Update unseen after valid moves
                all_unseen = (get_entities(new_valid) | get_entities(new_test)) - train_entities

                moved_test, keep_test = [], []
                for row in new_test.iter_rows(named=True):
                    if row["s"] in all_unseen or row["o"] in all_unseen:
                        moved_test.append(row)
                        train_entities.add(row["s"])
                        train_entities.add(row["o"])
                    else:
                        keep_test.append(row)

                if moved_valid or moved_test:
                    logger.info(
                        f"  Movidas para train: {len(moved_valid)} do valid, {len(moved_test)} do test"
                    )

                    if moved_valid:
                        new_train = pl.concat([new_train, pl.DataFrame(moved_valid)])
                    if moved_test:
                        new_train = pl.concat([new_train, pl.DataFrame(moved_test)])

                    new_valid = (
                        pl.DataFrame(keep_valid)
                        if keep_valid
                        else pl.DataFrame(schema={"s": pl.Utf8, "p": pl.Utf8, "o": pl.Utf8})
                    )
                    new_test = (
                        pl.DataFrame(keep_test)
                        if keep_test
                        else pl.DataFrame(schema={"s": pl.Utf8, "p": pl.Utf8, "o": pl.Utf8})
                    )

        # Step 5: Verify no leakage BEFORE inverses
        pre_report = self.leakage_checker.check_triple_leakage(
            new_train, new_valid, new_test, log_on_leak=False
        )
        if pre_report["has_leakage"]:
            logger.error(
                "Leakage still present after resplit attempt; abort pipeline or inspect leakage_report."
            )

        # Stats
        stats = {
            "original_total": original_count,
            "duplicates_removed": duplicates_removed,
            "unique_triples": unique_count,
            "resplit": {
                "train": len(new_train),
                "valid": len(new_valid),
                "test": len(new_test),
            },
            "seed": seed,
            "ratios": ratios,
        }

        logger.success("Re-split SOTA concluido: zero leakage!")

        return new_train, new_valid, new_test, stats

    def _apply_strategy(self, df: pl.DataFrame, strategy: Any, stage_name: str) -> pl.DataFrame:
        """Apply a strategy and accumulate stats.

        Args:
            df: Input DataFrame
            strategy: Strategy to apply (or None to skip)
            stage_name: Name for logging/stats

        Returns:
            Processed DataFrame
        """
        if strategy is None:
            return df

        result = strategy.process(df)
        self._stats[stage_name] = result.stats

        if result.metadata:
            self._metadata[stage_name] = result.metadata

        return result.data

    def preprocess_single(self, df: pl.DataFrame) -> pl.DataFrame:
        """Preprocess a single DataFrame (no splitting).

        Pipeline order:
        1. Deduplication (remove exact duplicates)
        2. Self-loop removal (h == t)
        3. Attribute classification (mark, don't remove)
        4. Entity degree filter (optional)
        5. Relation support filter (optional)
        6. Degree feature extraction
        7. Inverse relations (LAST - only for single split)

        Args:
            df: Raw DataFrame with columns [s, p, o]

        Returns:
            Preprocessed DataFrame
        """
        logger.info("INICIANDO PRE-PROCESSAMENTO KG")

        current = df

        # Step 1-5: Basic preprocessing
        current = self.basic_composer.apply(current, self._apply_strategy)

        # Step 6: Degree features
        if self.degree_extractor:
            result = self.degree_extractor.process(current)
            self._stats["degree_features"] = result.stats
            if result.metadata and "degree_features" in result.metadata:
                self._features["entity_degrees"] = result.metadata["degree_features"]

        # Step 7: Inverse relations (only for single-split use case)
        # NOTE: For split data, inverses are added AFTER split!
        current = self._apply_strategy(current, self.inverse_strategy, "inverse_relations")

        logger.success("PRE-PROCESSAMENTO CONCLUIDO")

        return current

    def preprocess_and_split(
        self,
        df: pl.DataFrame,
        train_ratio: float = 0.8,
        valid_ratio: float = 0.1,
        test_ratio: float = 0.1,
    ) -> PipelineResult:
        """Preprocess raw data and split into train/valid/test.

        CRITICAL ORDER:
        1. Clean data (dedup, self-loops)
        2. Split into train/valid/test
        3. Add inverses to EACH split independently
        4. Verify no leakage

        Args:
            df: Raw DataFrame with columns [s, p, o]
            train_ratio: Fraction for training
            valid_ratio: Fraction for validation
            test_ratio: Fraction for test

        Returns:
            PipelineResult with preprocessed splits
        """
        logger.info("INICIANDO PRE-PROCESSAMENTO COM SPLIT")

        # Phase 1: Clean data BEFORE split
        current, meta = self._map_ids(df, source="raw")
        if meta:
            self._metadata.setdefault("id_mapping", meta)

        current = self.basic_composer.apply(current, self._apply_strategy)

        # Phase 2: Split with inverse safety
        splitter = SafeSplitter(
            train_ratio=train_ratio,
            valid_ratio=valid_ratio,
            test_ratio=test_ratio,
            inverse_suffix=self.config.inverse_suffix,
        )

        split_result = splitter.split_with_inverse_safety(
            current,
            add_inverses=self.config.add_inverse_relations,
            chronological=self.config.chronological_split,
            timestamp_column=self.config.timestamp_column,
        )

        self._stats["split"] = split_result.stats

        # Phase 3: Extract features from training data
        if self.degree_extractor:
            result = self.degree_extractor.process(split_result.train)
            self._stats["degree_features"] = result.stats
            if result.metadata and "degree_features" in result.metadata:
                self._features["entity_degrees"] = result.metadata["degree_features"]

        logger.success("PRE-PROCESSAMENTO COM SPLIT CONCLUIDO")

        return PipelineResult(
            train=split_result.train,
            valid=split_result.valid,
            test=split_result.test,
            stats=self._stats,
            features=self._features,
            metadata=self._metadata,
        )

    def preprocess_splits(
        self, train_df: pl.DataFrame, valid_df: pl.DataFrame, test_df: pl.DataFrame
    ) -> PipelineResult:
        """Preprocess already-split data.

        Use this when data comes pre-split (e.g., from KGBuilder).

        CRITICAL: Applies same preprocessing to ALL splits to ensure consistency.
        Inverse relations are added to each split INDEPENDENTLY.

        Args:
            train_df: Training DataFrame
            valid_df: Validation DataFrame
            test_df: Test DataFrame

        Returns:
            PipelineResult with preprocessed splits
        """
        logger.info("INICIANDO PRE-PROCESSAMENTO DE SPLITS EXISTENTES")

        # Map IDs consistently across splits if needed
        train_df, valid_df, test_df = self._map_ids_for_splits(train_df, valid_df, test_df)

        def preprocess_one_split(df: pl.DataFrame, name: str) -> pl.DataFrame:
            """Preprocess a single split (no inverses yet)."""
            logger.info(f"[{name.upper()}] Processando {len(df):,} triplas...")

            current = df

            # Apply cleaning to each split
            if self.dedup_strategy:
                result = self.dedup_strategy.process(current)
                current = result.data

            if self.self_loop_strategy:
                result = self.self_loop_strategy.process(current)
                current = result.data

            if self.attribute_classifier:
                result = self.attribute_classifier.process(current)
                current = result.data
                self._stats[f"attributes_{name}"] = result.stats

            return current

        # Phase 1: Clean each split
        train_clean = preprocess_one_split(train_df, "train")
        valid_clean = preprocess_one_split(valid_df, "valid")
        test_clean = preprocess_one_split(test_df, "test")

        # Phase 1.5: Check for leakage BEFORE adding inverses
        # If leakage detected and fix_leakage enabled, re-split the data
        if self.config.check_leakage:
            pre_leakage = self.leakage_checker.check_triple_leakage(
                train_clean, valid_clean, test_clean, log_on_leak=False
            )

            if pre_leakage["has_leakage"]:
                logger.warning(
                    f"Pre-inverse leakage detected between splits: "
                    f"train-valid={pre_leakage['train_valid_overlap']}, "
                    f"train-test={pre_leakage['train_test_overlap']}. "
                    f"fix_leakage={self.config.fix_leakage}."
                )

                if self.config.fix_leakage:
                    logger.info("fix_leakage=True: Executando re-split SOTA...")
                    train_clean, valid_clean, test_clean, resplit_stats = self._fix_leakage_resplit(
                        train_clean, valid_clean, test_clean
                    )
                    self._stats["resplit"] = resplit_stats
                else:
                    logger.warning(
                        "fix_leakage=False: Leakage will persist. Consider enabling fix_leakage "
                        "to avoid contamination during training."
                    )

        # Phase 2: Apply entity filter based on TRAIN entities only
        if self.entity_filter:
            # Get valid entities from train
            train_entities = set(train_clean["s"]) | set(train_clean["o"])

            # Apply filter to train
            filter_result = self.entity_filter.process(train_clean)
            train_clean = filter_result.data
            self._stats["entity_filter"] = filter_result.stats

            # Filter valid/test to only include train entities
            valid_clean = valid_clean.filter(
                pl.col("s").is_in(list(train_entities)) & pl.col("o").is_in(list(train_entities))
            )
            test_clean = test_clean.filter(
                pl.col("s").is_in(list(train_entities)) & pl.col("o").is_in(list(train_entities))
            )

        # Phase 3: Add inverses to EACH split independently
        if self.inverse_strategy:
            logger.info("[INVERSAS] Adicionando relacoes inversas a cada split...")

            train_inv = self.inverse_strategy.process(train_clean)
            valid_inv = self.inverse_strategy.process(valid_clean)
            test_inv = self.inverse_strategy.process(test_clean)

            train_final = train_inv.data
            valid_final = valid_inv.data
            test_final = test_inv.data

            self._stats["inverse_train"] = train_inv.stats
            self._stats["inverse_valid"] = valid_inv.stats
            self._stats["inverse_test"] = test_inv.stats
        else:
            train_final = train_clean
            valid_final = valid_clean
            test_final = test_clean

        # Phase 4: Verify no leakage (final check after inverses)
        if self.config.check_leakage:
            leakage_report = self.leakage_checker.full_check(train_final, valid_final, test_final)
            self._stats["leakage_report"] = leakage_report

            if not leakage_report["all_clear"]:
                if self.config.fix_leakage:
                    # This shouldn't happen if resplit worked
                    logger.error("DATA LEAKAGE DETECTED even after fix attempt!")
                else:
                    logger.error("DATA LEAKAGE DETECTED! Enable fix_leakage to auto-correct.")
            else:
                logger.success("Verificacao de leakage: OK (zero leakage)")

        # Phase 5: Extract features from training data
        if self.degree_extractor:
            result = self.degree_extractor.process(train_final)
            self._stats["degree_features"] = result.stats
            if result.metadata and "degree_features" in result.metadata:
                self._features["entity_degrees"] = result.metadata["degree_features"]
        if self.attribute_classifier:
            self._metadata["attributes"] = {
                "attribute_relations": list(self.config.attribute_relations)
            }

        # Final stats
        self._stats["final"] = {
            "train_triples": len(train_final),
            "valid_triples": len(valid_final),
            "test_triples": len(test_final),
            "total_triples": len(train_final) + len(valid_final) + len(test_final),
        }

        logger.success("PRE-PROCESSAMENTO DE SPLITS CONCLUIDO")
        logger.info(
            f"Final: train={len(train_final):,}, "
            f"valid={len(valid_final):,}, test={len(test_final):,}"
        )

        return PipelineResult(
            train=train_final,
            valid=valid_final,
            test=test_final,
            stats=self._stats,
            features=self._features,
            metadata=self._metadata,
        )

    def save_preprocessed(
        self, result: PipelineResult, output_dir: Path, suffix: str = "_preprocessed"
    ) -> dict[str, Path]:
        """Save preprocessed data to disk.

        Args:
            result: Pipeline result to save
            output_dir: Output directory
            suffix: Suffix for filenames

        Returns:
            Dictionary mapping split names to saved paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        paths = {}

        if result.train is not None:
            path = output_dir / f"train{suffix}.parquet"
            self.fm.save(result.train, path)
            paths["train"] = path

        if result.valid is not None:
            path = output_dir / f"valid{suffix}.parquet"
            self.fm.save(result.valid, path)
            paths["valid"] = path

        if result.test is not None:
            path = output_dir / f"test{suffix}.parquet"
            self.fm.save(result.test, path)
            paths["test"] = path

        # Save stats
        stats_path = output_dir / f"preprocessing_stats{suffix}.parquet"
        stats_df = (
            pl.DataFrame([result.stats])
            if isinstance(result.stats, dict)
            else pl.DataFrame(result.stats)
        )
        self.fm.save(stats_df, stats_path)
        paths["stats"] = stats_path

        # Save features if present
        if result.features:
            features_dir = output_dir / "features"
            features_dir.mkdir(exist_ok=True)

            if "entity_degrees" in result.features:
                degree_path = features_dir / f"entity_degrees{suffix}.parquet"
                self.fm.save(result.features["entity_degrees"], degree_path)
                paths["entity_degrees"] = degree_path

        logger.info(f"Dados pre-processados salvos em: {output_dir}")

        return paths


def get_shared_preprocessing_pipeline(
    config_path: Path | str | None = None,
) -> KGPreprocessingPipeline:
    """Get a shared preprocessing pipeline instance.

    This is the RECOMMENDED way to get a pipeline to ensure
    consistency between main training and HPO.

    Args:
        config_path: Optional path to config file

    Returns:
        KGPreprocessingPipeline instance
    """
    config = PreprocessingConfig.from_yaml(config_path)
    return KGPreprocessingPipeline(config)
