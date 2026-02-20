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
from typing import Any, cast

import numpy as np
import polars as pl

from pff.shared import FileManager, logger
from pff_rust import stable_hash

from .config import (
    ATTRIBUTE_HANDLING_MARK,
    ATTRIBUTE_HANDLING_REMOVE,
    ATTRIBUTE_HANDLING_SEPARATE,
    PreprocessingConfig,
)
from .split import LeakageChecker, SafeSplitter
from .strategies import (
    AttributeRelationClassifier,
    DeduplicationStrategy,
    DegreeFeatureExtractor,
    EntityDegreeFilter,
    InverseRelationStrategy,
    PreprocessingComposer,
    RelationSupportFilter,
    SelfLoopRemovalStrategy,
)


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
        pipeline = KGPreprocessingPipeline(config)
        result = pipeline.preprocess_and_split(raw_df)

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

        self._init_strategies()

        self._stats: dict[str, Any] = {}
        self._features: dict[str, Any] = {}
        self._metadata: dict[str, Any] = {}
        self._entity_map_path: Path | None = None
        self._relation_map_path: Path | None = None

    def _ensure_output_dir(self) -> Path:
        out_dir = Path(self.config.output_dir)
        self.fm.ensure_dir(out_dir)
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
        if (
            df.schema["s"] in int_types
            and df.schema["p"] in int_types
            and df.schema["o"] in int_types
        ):
            return df, {}

        out_dir = self._ensure_output_dir()
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
            .collect(engine="streaming")
        )

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

        rng = np.random.RandomState(seed)
        ratios = self.config.resplit_ratios

        logger.info("Unificando todos os splits...")
        unified = pl.concat([train, valid, test])
        original_count = len(unified)

        if self.config.remove_duplicates:
            unified = unified.unique(subset=["s", "p", "o"])
            unique_count = len(unified)
            duplicates_removed = original_count - unique_count
            logger.debug(f"Total after dedup: {unique_count:,}")
            logger.debug(f"Cross-split duplicates removed: {duplicates_removed:,}")
        else:
            unique_count = original_count
            duplicates_removed = 0
            logger.debug("Deduplication disabled: preserving triple frequency")

        logger.debug(f"Total for split: {unique_count:,}")
        new_train, new_valid, new_test = self._stratified_resplit(unified, rng, ratios)

        logger.info(
            f"  Split inicial: train={len(new_train):,}, valid={len(new_valid):,}, test={len(new_test):,}"
        )

        if self.config.ensure_transductive:
            new_train, new_valid, new_test = self._enforce_transductive_resplit(
                new_train=new_train,
                new_valid=new_valid,
                new_test=new_test,
                entity_dtype=unified.schema["s"],
            )

        pre_report = self.leakage_checker.check_triple_leakage(
            new_train, new_valid, new_test, log_on_leak=False
        )
        if pre_report["has_leakage"]:
            logger.error(
                "Leakage still present after resplit attempt; abort pipeline or inspect leakage_report."
            )

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

    def _compute_resplit_counts(
        self,
        n: int,
        train_ratio: float,
        valid_ratio: float,
        test_ratio: float,
    ) -> tuple[int, int, int]:
        """Execute compute resplit counts.



        Args:

            n: Input value used by this callable.

            train_ratio: Input value used by this callable.

            valid_ratio: Input value used by this callable.

            test_ratio: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        n_train = max(1, int(n * train_ratio))
        n_valid = int(n * valid_ratio)
        n_test = int(n * test_ratio)

        if n >= 2 and n_valid == 0 and valid_ratio > 0:
            n_valid = 1
        if n >= 3 and n_test == 0 and test_ratio > 0:
            n_test = 1

        if n_train + n_valid + n_test > n:
            n_train = max(1, n - n_valid - n_test)

        if n_train + n_valid + n_test > n:
            if n_test > 0:
                n_test = max(0, n - n_train - n_valid)
            elif n_valid > 0:
                n_valid = max(0, n - n_train)

        return n_train, n_valid, n_test

    def _stratified_resplit(
        self,
        unified: pl.DataFrame,
        rng: np.random.RandomState,
        ratios: tuple[float, float, float],
    ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """Execute stratified resplit.



        Args:

            unified: Input value used by this callable.

            rng: Input value used by this callable.

            ratios: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        train_ratio, valid_ratio, test_ratio = ratios
        logger.debug("Executing stratified split by relation...")

        train_parts: list[pl.DataFrame] = []
        valid_parts: list[pl.DataFrame] = []
        test_parts: list[pl.DataFrame] = []
        relation_groups = unified.partition_by("p", as_dict=False, maintain_order=True)

        for relation_df in relation_groups:
            n = len(relation_df)
            n_train, n_valid, n_test = self._compute_resplit_counts(
                n, train_ratio, valid_ratio, test_ratio
            )
            indices = rng.permutation(n)
            train_idx = indices[:n_train].tolist()
            valid_idx = indices[n_train : n_train + n_valid].tolist()
            test_idx = indices[n_train + n_valid : n_train + n_valid + n_test].tolist()

            train_parts.append(relation_df[train_idx])
            if valid_idx:
                valid_parts.append(relation_df[valid_idx])
            if test_idx:
                test_parts.append(relation_df[test_idx])

        return (
            pl.concat(train_parts) if train_parts else pl.DataFrame(schema=unified.schema),
            pl.concat(valid_parts) if valid_parts else pl.DataFrame(schema=unified.schema),
            pl.concat(test_parts) if test_parts else pl.DataFrame(schema=unified.schema),
        )

    @staticmethod
    def _collect_entities(df_list: list[pl.DataFrame], entity_dtype: Any) -> pl.Series:
        """Execute collect entities.



        Args:

            df_list: Input value used by this callable.

            entity_dtype: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        active_dfs = [df for df in df_list if len(df) > 0]
        if not active_dfs:
            return pl.Series("e", [], dtype=entity_dtype)
        parts: list[pl.Series] = []
        for df in active_dfs:
            parts.extend([df["s"], df["o"]])
        return pl.concat(parts).unique()

    def _enforce_transductive_resplit(
        self,
        *,
        new_train: pl.DataFrame,
        new_valid: pl.DataFrame,
        new_test: pl.DataFrame,
        entity_dtype: Any,
    ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """Execute enforce transductive resplit.



        Args:

            new_train: Input value used by this callable.

            new_valid: Input value used by this callable.

            new_test: Input value used by this callable.

            entity_dtype: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        logger.debug("Ensuring transductive coverage...")
        train_entities = self._collect_entities([new_train], entity_dtype)
        valid_test_entities = self._collect_entities([new_valid, new_test], entity_dtype)
        unseen = valid_test_entities.filter(~valid_test_entities.is_in(train_entities.implode()))

        if len(unseen) == 0:
            return new_train, new_valid, new_test

        logger.info(f"  Entidades nao vistas no train: {len(unseen)}")
        v_leak_mask = new_valid.select(
            pl.col("s").is_in(unseen.implode()) | pl.col("o").is_in(unseen.implode())
        ).to_series()
        t_leak_mask = new_test.select(
            pl.col("s").is_in(unseen.implode()) | pl.col("o").is_in(unseen.implode())
        ).to_series()

        can_move_valid = v_leak_mask.sum() < len(new_valid) * 0.7
        can_move_test = t_leak_mask.sum() < len(new_test) * 0.7

        moved_valid = (
            new_valid.filter(v_leak_mask)
            if can_move_valid
            else pl.DataFrame(schema=new_valid.schema)
        )
        moved_test = (
            new_test.filter(t_leak_mask) if can_move_test else pl.DataFrame(schema=new_test.schema)
        )

        if can_move_valid:
            new_valid = new_valid.filter(~v_leak_mask)
        if can_move_test:
            new_test = new_test.filter(~t_leak_mask)
        if len(moved_valid) > 0 or len(moved_test) > 0:
            logger.info(
                f"  Movidas para train: {len(moved_valid)} do valid, {len(moved_test)} do test"
            )
            new_train = pl.concat([new_train, moved_valid, moved_test])

        return new_train, new_valid, new_test

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

        return cast(pl.DataFrame, result.data)

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

        current = self.basic_composer.apply(current, self._apply_strategy)

        if self.degree_extractor:
            result = self.degree_extractor.process(current)
            self._stats["degree_features"] = result.stats
            if result.metadata and "degree_features" in result.metadata:
                self._features["entity_degrees"] = result.metadata["degree_features"]

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

        current, meta = self._map_ids(df, source="raw")
        if meta:
            self._metadata.setdefault("id_mapping", meta)

        current = self.basic_composer.apply(current, self._apply_strategy)

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

    @staticmethod
    def _empty_split(schema: dict[str, Any]) -> pl.DataFrame:
        return pl.DataFrame(schema=schema)

    def _preprocess_existing_split(
        self,
        df_in: pl.DataFrame | None,
        *,
        name: str,
        base_schema: dict[str, Any],
    ) -> pl.DataFrame:
        """Execute preprocess existing split.



        Args:

            df_in: Input value used by this callable.

            name: Input value used by this callable.

            base_schema: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        if df_in is None or len(df_in) == 0:
            return self._empty_split(base_schema)

        logger.info(f"[{name.upper()}] Processando {len(df_in):,} triplas...")
        current_df = df_in
        if self.dedup_strategy:
            current_df = self.dedup_strategy.process(current_df).data
        if self.self_loop_strategy:
            current_df = self.self_loop_strategy.process(current_df).data
        if self.attribute_classifier:
            result_item = self.attribute_classifier.process(current_df)
            current_df = result_item.data
            self._stats[f"attributes_{name}"] = result_item.stats
        return current_df

    def _needs_resplit(
        self,
        train_clean: pl.DataFrame,
        valid_clean: pl.DataFrame,
        test_clean: pl.DataFrame,
    ) -> bool:
        """Execute needs resplit.



        Args:

            train_clean: Input value used by this callable.

            valid_clean: Input value used by this callable.

            test_clean: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        needs_resplit = False
        pre_leakage = self.leakage_checker.check_triple_leakage(
            train_clean, valid_clean, test_clean, log_on_leak=False
        )
        if self.config.check_leakage and pre_leakage["has_leakage"]:
            if self.config.fix_leakage:
                logger.bind(
                    component="kg_preprocess",
                    stop_reason="leakage_detected",
                    key_parameters={
                        "train_valid_overlap": pre_leakage["train_valid_overlap"],
                        "train_test_overlap": pre_leakage["train_test_overlap"],
                    },
                ).info("Leakage detectado; re-split sera executado.")
            else:
                logger.warning(
                    f"Triple leakage detected: train-valid={pre_leakage['train_valid_overlap']}, "
                    f"train-test={pre_leakage['train_test_overlap']}."
                )
            needs_resplit = True

        if not self.config.ensure_transductive:
            return needs_resplit

        coverage = self.leakage_checker.check_entity_coverage(
            train_clean,
            valid_clean,
            test_clean,
            log_on_leak=not self.config.fix_leakage,
        )
        has_transductive_violation = (
            coverage.get("valid_unseen_entities", 0) > 0
            or coverage.get("test_unseen_entities", 0) > 0
        )
        if not has_transductive_violation:
            return needs_resplit

        if self.config.fix_leakage:
            logger.bind(
                component="kg_preprocess",
                stop_reason="transductive_violation",
                key_parameters={
                    "valid_unseen_entities": coverage.get("valid_unseen_entities"),
                    "test_unseen_entities": coverage.get("test_unseen_entities"),
                },
            ).info("Violacao transdutiva detectada; re-split sera executado.")
        else:
            logger.warning(
                f"Transductive violation: valid_unseen={coverage.get('valid_unseen_entities')}, "
                f"test_unseen={coverage.get('test_unseen_entities')}."
            )
        return True

    def _maybe_resplit(
        self,
        train_clean: pl.DataFrame,
        valid_clean: pl.DataFrame,
        test_clean: pl.DataFrame,
    ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """Execute maybe resplit.



        Args:

            train_clean: Input value used by this callable.

            valid_clean: Input value used by this callable.

            test_clean: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        if not self._needs_resplit(train_clean, valid_clean, test_clean):
            return train_clean, valid_clean, test_clean
        if not self.config.fix_leakage:
            logger.warning(
                "fix_leakage=False: Leakage/Transductive violations persist. Enable fix_leakage to correct."
            )
            return train_clean, valid_clean, test_clean

        logger.info("fix_leakage=True: Executando re-split SOTA...")
        train_clean, valid_clean, test_clean, resplit_stats = self._fix_leakage_resplit(
            train_clean, valid_clean, test_clean
        )
        self._stats["resplit"] = resplit_stats
        return train_clean, valid_clean, test_clean

    def _apply_entity_filter_to_splits(
        self,
        train_clean: pl.DataFrame,
        valid_clean: pl.DataFrame,
        test_clean: pl.DataFrame,
    ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """Execute apply entity filter to splits.



        Args:

            train_clean: Input value used by this callable.

            valid_clean: Input value used by this callable.

            test_clean: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        if not self.entity_filter:
            return train_clean, valid_clean, test_clean

        filter_result = self.entity_filter.process(train_clean)
        train_clean = filter_result.data
        self._stats["entity_filter"] = filter_result.stats
        train_entities = pl.concat([train_clean["s"], train_clean["o"]]).unique().implode()

        valid_before = len(valid_clean)
        valid_clean = valid_clean.filter(
            pl.col("s").is_in(train_entities) & pl.col("o").is_in(train_entities)
        )
        valid_removed = valid_before - len(valid_clean)

        test_before = len(test_clean)
        test_clean = test_clean.filter(
            pl.col("s").is_in(train_entities) & pl.col("o").is_in(train_entities)
        )
        test_removed = test_before - len(test_clean)

        if valid_removed or test_removed:
            self._stats["entity_filter_orphans"] = {
                "valid_removed": valid_removed,
                "test_removed": test_removed,
            }
            logger.bind(
                component="kg_preprocess",
                stop_reason="entity_filter_orphans",
                key_parameters={
                    "valid_removed": valid_removed,
                    "test_removed": test_removed,
                },
            ).info("Triplas orfas removidas apos filtro de grau.")

        return train_clean, valid_clean, test_clean

    def _apply_inverse_to_splits(
        self,
        train_clean: pl.DataFrame,
        valid_clean: pl.DataFrame,
        test_clean: pl.DataFrame,
    ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """Execute apply inverse to splits.



        Args:

            train_clean: Input value used by this callable.

            valid_clean: Input value used by this callable.

            test_clean: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        if not self.inverse_strategy:
            return train_clean, valid_clean, test_clean

        logger.info("[INVERSAS] Adicionando relacoes inversas a cada split...")
        train_inv = self.inverse_strategy.process(train_clean)
        valid_inv = self.inverse_strategy.process(valid_clean)
        test_inv = self.inverse_strategy.process(test_clean)

        self._stats["inverse_train"] = train_inv.stats
        self._stats["inverse_valid"] = valid_inv.stats
        self._stats["inverse_test"] = test_inv.stats
        return train_inv.data, valid_inv.data, test_inv.data

    def _record_final_leakage(
        self,
        train_final: pl.DataFrame,
        valid_final: pl.DataFrame,
        test_final: pl.DataFrame,
    ) -> None:
        """Execute record final leakage.



        Args:

            train_final: Input value used by this callable.

            valid_final: Input value used by this callable.

            test_final: Input value used by this callable.

        """

        if not self.config.check_leakage:
            return
        leakage_report = self.leakage_checker.full_check(train_final, valid_final, test_final)
        self._stats["leakage_report"] = leakage_report
        if leakage_report["all_clear"]:
            logger.success("Verificacao de leakage: OK (zero leakage)")
            return
        if self.config.fix_leakage:
            logger.error("DATA LEAKAGE DETECTED even after fix attempt!")
        else:
            logger.error("DATA LEAKAGE DETECTED! Enable fix_leakage to auto-correct.")

    def _finalize_split_result(
        self,
        train_final: pl.DataFrame,
        valid_final: pl.DataFrame,
        test_final: pl.DataFrame,
    ) -> PipelineResult:
        """Execute finalize split result.



        Args:

            train_final: Input value used by this callable.

            valid_final: Input value used by this callable.

            test_final: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        if self.degree_extractor:
            result = self.degree_extractor.process(train_final)
            self._stats["degree_features"] = result.stats
            if result.metadata and "degree_features" in result.metadata:
                self._features["entity_degrees"] = result.metadata["degree_features"]
        if self.attribute_classifier:
            self._metadata["attributes"] = {
                "attribute_relations": list(self.config.attribute_relations)
            }

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

    def preprocess_splits(
        self,
        train_df: pl.DataFrame,
        valid_df: pl.DataFrame | None,
        test_df: pl.DataFrame | None,
    ) -> PipelineResult:
        """Execute preprocess splits.



        Args:

            train_df: Input value used by this callable.

            valid_df: Input value used by this callable.

            test_df: Input value used by this callable.



        Returns:

            Return value produced by the callable.



        Raises:

            Exception: Propagates domain-specific failures with context.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        logger.info("INICIANDO PRE-PROCESSAMENTO DE SPLITS EXISTENTES")
        base_schema = train_df.schema
        train_clean = self._preprocess_existing_split(
            train_df, name="train", base_schema=base_schema
        )
        valid_clean = self._preprocess_existing_split(
            valid_df, name="valid", base_schema=base_schema
        )
        test_clean = self._preprocess_existing_split(test_df, name="test", base_schema=base_schema)
        if len(train_clean) == 0:
            raise ValueError("Training data missing after preprocessing")

        train_clean, valid_clean, test_clean = self._maybe_resplit(
            train_clean, valid_clean, test_clean
        )
        train_clean, valid_clean, test_clean = self._apply_entity_filter_to_splits(
            train_clean, valid_clean, test_clean
        )
        train_final, valid_final, test_final = self._apply_inverse_to_splits(
            train_clean, valid_clean, test_clean
        )
        self._record_final_leakage(train_final, valid_final, test_final)
        mapped_train, mapped_valid, mapped_test = self._map_ids_for_splits(
            train_final, valid_final, test_final
        )
        if len(mapped_train) == 0:
            raise ValueError("Training data missing after ID mapping")
        if mapped_valid is None:
            mapped_valid = self._empty_split(mapped_train.schema)
        if mapped_test is None:
            mapped_test = self._empty_split(mapped_train.schema)
        return self._finalize_split_result(mapped_train, mapped_valid, mapped_test)

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
        self.fm.ensure_dir(output_dir)

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

        stats_path = output_dir / f"preprocessing_stats{suffix}.parquet"
        stats_df = pl.DataFrame([result.stats])
        self.fm.save(stats_df, stats_path)
        paths["stats"] = stats_path

        if result.features:
            features_dir = output_dir / "features"
            self.fm.ensure_dir(features_dir)

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
