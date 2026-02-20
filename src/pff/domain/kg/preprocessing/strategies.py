"""
Preprocessing Strategies for KG Data.

Design Pattern: Strategy Pattern
- Each preprocessing step is encapsulated as a strategy
- Strategies are composable and independently testable
- Allows easy addition of new preprocessing steps

Each strategy implements a `process()` method that takes a DataFrame
and returns a processed DataFrame + statistics dict.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import polars as pl

from pff.shared import logger


@dataclass
class ProcessingResult:
    """Result of a preprocessing step.

    Attributes:
        data: Processed DataFrame
        stats: Statistics about the processing step
        metadata: Additional metadata (e.g., mappings, features)
    """

    data: pl.DataFrame
    stats: dict[str, Any]
    metadata: dict[str, Any] | None = None


class PreprocessingStrategy(ABC):
    """Abstract base class for preprocessing strategies."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the strategy for logging."""
        pass

    @abstractmethod
    def process(self, df: pl.DataFrame) -> ProcessingResult:
        """Process the DataFrame.

        Args:
            df: Input DataFrame with columns [s, p, o]

        Returns:
            ProcessingResult with processed data and statistics
        """
        pass


class BasicPreprocessingStrategy(PreprocessingStrategy):
    """Marker base class for basic preprocessing strategies."""


class AdvancedPreprocessingStrategy(PreprocessingStrategy):
    """Marker base class for advanced preprocessing strategies."""


class OptimizationPreprocessingStrategy(PreprocessingStrategy):
    """Marker base class for optimization-focused preprocessing strategies."""


class PreprocessingComposer:
    """Compose preprocessing strategies into an ordered pipeline."""

    def __init__(
        self,
        steps: list[tuple[str, PreprocessingStrategy | None]],
    ) -> None:
        """Execute init.



        Args:

            steps: Input value used by this callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        self._steps = steps

    def apply(
        self,
        df: pl.DataFrame,
        apply_fn: Callable[[pl.DataFrame, PreprocessingStrategy | None, str], pl.DataFrame],
    ) -> pl.DataFrame:
        """Apply all configured steps using the provided apply function."""
        current = df
        for name, strategy in self._steps:
            current = apply_fn(current, strategy, name)
        return current


class DeduplicationStrategy(PreprocessingStrategy):
    """Remove exact duplicate triples.

    This is critical for KG quality when uniqueness is desired.
    If disabled, preserves frequency signal (multiple interactions between entities).
    """

    def __init__(self, enabled: bool = True) -> None:
        """Initialize strategy.

        Args:
            enabled: If False, this strategy becomes a no-op (preserves frequency).
        """
        self.enabled = enabled

    @property
    def name(self) -> str:
        """Execute name.



        Returns:

            Return value produced by the callable.

        """

        return "deduplication"

    def process(self, df: pl.DataFrame) -> ProcessingResult:
        """Remove duplicate (s, p, o) triples if enabled.

        Args:
            df: DataFrame with columns [s, p, o]

        Returns:
            ProcessingResult with unique triples (if enabled) or original data
        """
        initial_count = len(df)

        if not self.enabled:
            logger.debug("dedup disabled - preserving interaction frequency")
            stats = {
                "initial_triples": initial_count,
                "final_triples": initial_count,
                "duplicates_removed": 0,
                "duplicate_percentage": 0.0,
                "enabled": False,
            }
            return ProcessingResult(data=df, stats=stats)

        result_df = df.unique(subset=["s", "p", "o"])
        final_count = len(result_df)

        removed = initial_count - final_count
        pct_removed = (removed / initial_count * 100) if initial_count > 0 else 0

        stats = {
            "initial_triples": initial_count,
            "final_triples": final_count,
            "duplicates_removed": removed,
            "duplicate_percentage": pct_removed,
            "enabled": True,
        }

        logger.debug(f"dedup removed={removed:,} pct={pct_removed:.1f}%")

        return ProcessingResult(data=result_df, stats=stats)


class SelfLoopRemovalStrategy(PreprocessingStrategy):
    """Remove self-loops (triples where subject == object).

    Self-loops are semantically meaningless for most relations and
    can confuse the embedding model during training.

    Exception: Some relations ARE reflexive (e.g., "sameAs"). These can
    be specified in the allowed_reflexive_relations set.
    """

    def __init__(self, allowed_reflexive_relations: set[str] | None = None) -> None:
        """Initialize with optional reflexive relations.

        Args:
            allowed_reflexive_relations: Relations where self-loops are allowed
        """
        self.allowed_reflexive = allowed_reflexive_relations or set()

    @property
    def name(self) -> str:
        """Execute name.



        Returns:

            Return value produced by the callable.

        """

        return "self_loop_removal"

    def process(self, df: pl.DataFrame) -> ProcessingResult:
        """Remove triples where s == o (except for allowed reflexive relations).

        Args:
            df: DataFrame with columns [s, p, o]

        Returns:
            ProcessingResult without self-loops
        """
        initial_count = len(df)

        if self.allowed_reflexive:
            is_self_loop = pl.col("s") == pl.col("o")
            is_reflexive = pl.col("p").is_in(list(self.allowed_reflexive))
            result_df = df.filter(~is_self_loop | is_reflexive)
        else:
            result_df = df.filter(pl.col("s") != pl.col("o"))

        final_count = len(result_df)
        removed = initial_count - final_count
        pct_removed = (removed / initial_count * 100) if initial_count > 0 else 0

        stats = {
            "initial_triples": initial_count,
            "final_triples": final_count,
            "self_loops_removed": removed,
            "self_loop_percentage": pct_removed,
        }

        logger.debug(f"self_loops removed={removed:,} pct={pct_removed:.1f}%")

        return ProcessingResult(data=result_df, stats=stats)


class InverseRelationStrategy(PreprocessingStrategy):
    """Add inverse relations for each triple.

    For each (h, r, t), creates (t, r_inv, h). This:
    - Doubles the training signal
    - Enables learning of inverse patterns
    - Improves tail prediction (becomes head prediction for inverse)
    - Matches WN18RR/FB15k benchmark preprocessing

    CRITICAL: Must be applied AFTER split to avoid leakage!
    """

    def __init__(self, suffix: str = "_inv") -> None:
        """Initialize with inverse suffix.

        Args:
            suffix: Suffix to append to relation name for inverse
        """
        self.suffix = suffix

    @property
    def name(self) -> str:
        """Execute name.



        Returns:

            Return value produced by the callable.

        """

        return "inverse_relations"

    def process(self, df: pl.DataFrame) -> ProcessingResult:
        """Add inverse triples (t, r_inv, h) for each (h, r, t).

        Args:
            df: DataFrame with columns [s, p, o]

        Returns:
            ProcessingResult with original + inverse triples
        """
        initial_count = len(df)

        inverse_df = df.select(
            [
                pl.col("o").alias("s"),
                (pl.col("p").cast(pl.Utf8) + pl.lit(self.suffix)).alias("p"),
                pl.col("s").alias("o"),
            ]
        )

        result_df = pl.concat([df, inverse_df])

        final_count = len(result_df)
        original_relations = df["p"].n_unique()
        final_relations = result_df["p"].n_unique()

        stats = {
            "initial_triples": initial_count,
            "final_triples": final_count,
            "inverse_triples_added": len(inverse_df),
            "original_relations": original_relations,
            "final_relations": final_relations,
        }

        logger.debug(
            f"inversas added={len(inverse_df):,} relations={original_relations}->{final_relations}"
        )

        return ProcessingResult(data=result_df, stats=stats)


class AttributeRelationClassifier(PreprocessingStrategy):
    """Classify and optionally filter attribute relations.

    In telecom KGs, many relations are actually attributes (IDs, timestamps,
    literal values) rather than structural relationships. These:
    - Should NOT be targets for link prediction evaluation
    - May be used as features for entities instead
    - Can dominate training if not handled properly
    """

    def __init__(
        self,
        attribute_relations: set[str],
        attribute_patterns: tuple[str, ...] = tuple(),
        remove_from_data: bool = False,
        mark_only: bool = True,
    ) -> None:
        """Initialize classifier.

        Args:
            attribute_relations: Set of relation names to classify as attributes
            attribute_patterns: Regex or substring patterns to match attribute relations
            remove_from_data: If True, remove attribute triples from data
            mark_only: If True, only add metadata marking (don't modify data)
        """
        self.attribute_relations = attribute_relations
        self.attribute_patterns = attribute_patterns
        self.remove_from_data = remove_from_data
        self.mark_only = mark_only
        self._compiled_patterns = [
            re.compile(p, flags=re.IGNORECASE) for p in attribute_patterns if p
        ]
        self._pattern_union = "|".join(f"(?:{p})" for p in attribute_patterns if p)

    @property
    def name(self) -> str:
        """Execute name.



        Returns:

            Return value produced by the callable.

        """

        return "attribute_classification"

    def process(self, df: pl.DataFrame) -> ProcessingResult:
        """Classify and optionally filter attribute relations.

        Args:
            df: DataFrame with columns [s, p, o]

        Returns:
            ProcessingResult with classification metadata
        """
        initial_count = len(df)
        relation_col = "p"
        if df.schema.get(relation_col) != pl.Utf8:
            df = df.with_columns(pl.col(relation_col).cast(pl.Utf8))

        is_attribute = pl.col(relation_col).is_in(list(self.attribute_relations))
        if self._pattern_union:
            pattern_mask = pl.col(relation_col).str.contains(self._pattern_union, literal=False)
            is_attribute = is_attribute | pattern_mask
        attribute_count = int(df.filter(is_attribute).height)

        if self.remove_from_data and not self.mark_only:
            result_df = df.filter(~is_attribute)
        else:
            result_df = df

        final_count = len(result_df)

        relation_stats = (
            df.group_by(relation_col)
            .agg(pl.len().alias("count"))
            .with_columns(
                pl.col(relation_col)
                .is_in(list(self.attribute_relations))
                .alias("is_attribute_list"),
                (
                    pl.col(relation_col).str.contains(self._pattern_union, literal=False)
                    if self._pattern_union
                    else pl.lit(False)
                ).alias("is_attribute_pattern"),
            )
            .sort("count", descending=True)
        )
        relation_stats = relation_stats.with_columns(
            (pl.col("is_attribute_list") | pl.col("is_attribute_pattern")).alias("is_attribute")
        )

        stats = {
            "initial_triples": initial_count,
            "final_triples": final_count,
            "attribute_triples": int(attribute_count),
            "attribute_percentage": (
                float(attribute_count / initial_count * 100) if initial_count > 0 else 0
            ),
            "structural_triples": initial_count - int(attribute_count),
        }

        metadata = {
            "attribute_relations": list(self.attribute_relations),
            "relation_stats": relation_stats.to_dicts(),
        }

        logger.debug(
            f"atributos n={attribute_count:,} pct={stats['attribute_percentage']:.1f}% "
            f"estruturais={stats['structural_triples']:,}"
        )

        return ProcessingResult(data=result_df, stats=stats, metadata=metadata)


class DegreeFeatureExtractor(PreprocessingStrategy):
    """Extract degree-based features for entities.

    DSLFM benefits from degree features as they encode:
    - Hub vs peripheral entity status
    - Relation type patterns (1-to-N, N-to-1, etc.)
    - Community membership signals

    Features computed:
    - in_degree: Number of incoming edges
    - out_degree: Number of outgoing edges
    - total_degree: in + out
    - log_degree: log1p(total_degree) for normalization
    - relation_diversity: Number of unique relations
    """

    @property
    def name(self) -> str:
        """Execute name.



        Returns:

            Return value produced by the callable.

        """

        return "degree_features"

    def process(self, df: pl.DataFrame) -> ProcessingResult:
        """Extract degree features for all entities.

        Args:
            df: DataFrame with columns [s, p, o]

        Returns:
            ProcessingResult with degree features in metadata
        """
        out_degree = (
            df.group_by("s")
            .agg(
                [
                    pl.len().alias("out_degree"),
                    pl.n_unique("p").alias("out_relation_diversity"),
                ]
            )
            .rename({"s": "entity"})
        )

        in_degree = (
            df.group_by("o")
            .agg(
                [
                    pl.len().alias("in_degree"),
                    pl.n_unique("p").alias("in_relation_diversity"),
                ]
            )
            .rename({"o": "entity"})
        )

        degree_features = (
            out_degree.join(in_degree, on="entity", how="full", coalesce=True)
            .with_columns(
                [
                    pl.col("out_degree").fill_null(0),
                    pl.col("in_degree").fill_null(0),
                    pl.col("out_relation_diversity").fill_null(0),
                    pl.col("in_relation_diversity").fill_null(0),
                ]
            )
            .with_columns(
                [
                    (pl.col("out_degree") + pl.col("in_degree")).alias("total_degree"),
                    (pl.col("out_relation_diversity") + pl.col("in_relation_diversity")).alias(
                        "relation_diversity"
                    ),
                ]
            )
            .with_columns(
                [
                    (pl.col("total_degree") + 1).log().alias("log_degree"),
                ]
            )
        )

        n_entities = len(degree_features)
        avg_degree = degree_features["total_degree"].mean()
        max_degree = degree_features["total_degree"].max()
        min_degree = degree_features["total_degree"].min()
        median_degree = degree_features["total_degree"].median()

        hub_threshold = degree_features["total_degree"].quantile(0.99)
        if hub_threshold is not None:
            n_hubs = len(degree_features.filter(pl.col("total_degree") >= hub_threshold))
        else:
            n_hubs = 0

        n_singletons = len(degree_features.filter(pl.col("total_degree") == 1))

        stats = {
            "n_entities": n_entities,
            "avg_degree": float(avg_degree) if avg_degree else 0.0,  # type: ignore[arg-type]
            "max_degree": int(max_degree) if max_degree else 0,  # type: ignore[arg-type]
            "min_degree": int(min_degree) if min_degree else 0,  # type: ignore[arg-type]
            "median_degree": float(median_degree) if median_degree else 0.0,  # type: ignore[arg-type]
            "n_hubs": n_hubs,
            "hub_threshold": float(hub_threshold) if hub_threshold else 0.0,
            "n_singletons": n_singletons,
            "singleton_percentage": (
                float(n_singletons / n_entities * 100) if n_entities > 0 else 0
            ),
        }

        metadata = {
            "degree_features": degree_features,
        }

        logger.debug(
            f"grau entidades={n_entities:,} avg={avg_degree or 0:.2f} "  # type: ignore[str-bytes-safe]
            f"singletons={n_singletons:,} hubs={n_hubs:,}"
        )

        return ProcessingResult(data=df, stats=stats, metadata=metadata)


class EntityDegreeFilter(PreprocessingStrategy):
    """Filter entities by minimum degree.

    Removes entities with too few connections, which:
    - Have insufficient context for embedding learning
    - May be noise or data quality issues
    - Reduce training effectiveness
    """

    def __init__(self, min_degree: int = 2) -> None:
        """Initialize with minimum degree threshold.

        Args:
            min_degree: Minimum total degree to keep entity
        """
        self.min_degree = min_degree

    @property
    def name(self) -> str:
        """Execute name.



        Returns:

            Return value produced by the callable.

        """

        return "entity_degree_filter"

    def process(self, df: pl.DataFrame) -> ProcessingResult:
        """Filter out entities with degree < min_degree.

        Args:
            df: DataFrame with columns [s, p, o]

        Returns:
            ProcessingResult with filtered data
        """
        initial_count = len(df)

        entity_degrees = (
            pl.concat(
                [
                    df.select(pl.col("s").alias("entity")),
                    df.select(pl.col("o").alias("entity")),
                ]
            )
            .group_by("entity")
            .len()
            .rename({"len": "degree"})
        )

        valid_entities = entity_degrees.filter(pl.col("degree") >= self.min_degree).select("entity")

        result_df = (
            df.lazy()
            .join(valid_entities.lazy(), left_on="s", right_on="entity", how="semi")
            .join(valid_entities.lazy(), left_on="o", right_on="entity", how="semi")
            .select(["s", "p", "o"])
            .collect(engine="streaming")
        )

        final_count = len(result_df)
        initial_entities = len(entity_degrees)
        final_entities = len(valid_entities)

        stats = {
            "initial_triples": initial_count,
            "final_triples": final_count,
            "triples_removed": initial_count - final_count,
            "initial_entities": initial_entities,
            "final_entities": final_entities,
            "entities_removed": initial_entities - final_entities,
        }

        logger.info(
            f"[FILTRO GRAU] Removidas {stats['entities_removed']:,} entidades "
            f"(grau < {self.min_degree}), "
            f"{stats['triples_removed']:,} triplas"
        )

        return ProcessingResult(data=result_df, stats=stats)


class RelationSupportFilter(PreprocessingStrategy):
    """Filter or flag relations by minimum support (triple count).

    Policy-driven to accommodate sparse DSLFM training:
    - warn: keep all relations, log sparse ones
    - drop: remove relations with support < min_support
    """

    def __init__(self, min_support: int = 50, policy: str = "warn") -> None:
        """Initialize with minimum support threshold.

        Args:
            min_support: Minimum number of triples per relation.
            policy: 'warn' to keep and log sparse relations; 'drop' to filter them out.
        """
        self.min_support = min_support
        self.policy = policy

    @property
    def name(self) -> str:
        """Execute name.



        Returns:

            Return value produced by the callable.

        """

        return "relation_support_filter"

    def process(self, df: pl.DataFrame) -> ProcessingResult:
        """Filter out or warn on relations with support < min_support.

        Args:
            df: DataFrame with columns [s, p, o]

        Returns:
            ProcessingResult with filtered data
        """
        initial_count = len(df)

        relation_support = df.group_by("p").len().rename({"len": "support"})

        if self.min_support <= 0 or self.policy == "warn":
            rare = (
                relation_support.filter(pl.col("support") < self.min_support)
                if self.min_support > 0
                else pl.DataFrame()
            )
            if self.min_support > 0 and len(rare) > 0:
                logger.warning(
                    f"Sparse relations detected (policy=warn): "
                    f"min_support={self.min_support}, rare={len(rare)}"
                )
            stats = {
                "initial_triples": initial_count,
                "final_triples": initial_count,
                "triples_removed": 0,
                "initial_relations": len(relation_support),
                "final_relations": len(relation_support),
                "relations_removed": 0,
                "rare_relations": len(rare) if self.min_support > 0 else 0,
            }
            return ProcessingResult(data=df, stats=stats)

        valid_relations = relation_support.filter(pl.col("support") >= self.min_support).select("p")
        result_df = (
            df.lazy()
            .join(valid_relations.lazy(), on="p", how="semi")
            .select(["s", "p", "o"])
            .collect(engine="streaming")
        )

        final_count = len(result_df)
        initial_relations = len(relation_support)
        final_relations = len(valid_relations)

        stats = {
            "initial_triples": initial_count,
            "final_triples": final_count,
            "triples_removed": initial_count - final_count,
            "initial_relations": initial_relations,
            "final_relations": final_relations,
            "relations_removed": initial_relations - final_relations,
            "rare_relations": initial_relations - final_relations,
        }

        logger.info(
            f"[FILTRO RELACOES] Removidas {stats['relations_removed']:,} relacoes "
            f"(suporte < {self.min_support}), "
            f"{stats['triples_removed']:,} triplas"
        )

        return ProcessingResult(data=result_df, stats=stats)
