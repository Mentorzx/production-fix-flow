import polars as pl
from pathlib import Path

from dataclasses import dataclass
from typing import Any
from collections.abc import Mapping

from pff.config import KG_PIPELINE_CONFIG_PATH
from pff.shared import FileManager, logger
from pff.shared.core.file_manager import ParquetBundle
from pff.shared.observer import NullObserver, ProgressObserver


@dataclass
class OptimizationConfig:
    """Configuration for sparse data optimization."""

    # Sparsity filters
    min_entity_degree: int  # Entities must have >= threshold connections
    min_relation_support: int  # Relations must have >= threshold examples

    # Balancing
    max_entities_to_keep: int | None = None  # Limit of entities (None = no limit)
    balance_relations: bool = True  # Balance relation distribution

    # Performance
    preserve_original: bool = True  # Keep a backup of the original data
    log_statistics: bool = True  # Detailed log of transformations

    # Specific telephony data
    focus_on_active_users: bool = True  # Prioritize users with more activity
    min_product_interactions: int = 2  # Users must have >= 2 products/services

    # Data quality preprocessing (NEW)
    remove_duplicates: bool = True  # Remove duplicate triples
    remove_self_loops: bool = True  # Remove triples where s == o
    add_inverse_relations: bool = True  # Add r_inv for each relation r
    inverse_relation_suffix: str = "_inv"  # Suffix for inverse relations

    def __post_init__(self) -> None:
        if self.min_entity_degree is None or self.min_relation_support is None:
            raise ValueError(
                "OptimizationConfig requires min_entity_degree and min_relation_support "
                "from config/models/kg.yaml under data_optimizer.",
            )

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "OptimizationConfig":
        """Create an OptimizationConfig from a mapping with safe fallbacks."""
        missing = [
            key
            for key in ("min_entity_degree", "min_relation_support")
            if key not in data
        ]
        if missing:
            raise ValueError(
                f"Missing required data_optimizer keys in config: {missing}",
            )
        return cls(
            min_entity_degree=int(data["min_entity_degree"]),
            min_relation_support=int(data["min_relation_support"]),
            max_entities_to_keep=data.get(
                "max_entities_to_keep", cls.max_entities_to_keep
            ),
            balance_relations=bool(
                data.get("balance_relations", cls.balance_relations)
            ),
            preserve_original=bool(
                data.get("preserve_original", cls.preserve_original)
            ),
            log_statistics=bool(data.get("log_statistics", cls.log_statistics)),
            focus_on_active_users=bool(
                data.get("focus_on_active_users", cls.focus_on_active_users)
            ),
            min_product_interactions=int(
                data.get("min_product_interactions", cls.min_product_interactions)
            ),
            remove_duplicates=bool(
                data.get("remove_duplicates", cls.remove_duplicates)
            ),
            remove_self_loops=bool(
                data.get("remove_self_loops", cls.remove_self_loops)
            ),
            add_inverse_relations=bool(
                data.get("add_inverse_relations", cls.add_inverse_relations)
            ),
            inverse_relation_suffix=str(
                data.get("inverse_relation_suffix", cls.inverse_relation_suffix)
            ),
        )


def _load_data_optimizer_settings(
    config_path: Path = KG_PIPELINE_CONFIG_PATH,
) -> dict[str, Any]:
    """Load data optimizer settings from the KG pipeline config."""
    try:
        payload = FileManager().read(config_path)
        cfg = (
            payload.to_native() if isinstance(payload, ParquetBundle) else payload or {}
        )
        settings = cfg.get("data_optimizer", {})
        return settings if isinstance(settings, Mapping) else {}
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(
            f"Failed to load data optimizer config from {config_path}: {exc}",
            exc_info=True,
        )
        return {}


class TelecomDataOptimizer:
    """Specific optimizer for sparse telephony data.

    Pattern: Strategy + Observer-ready
    - Acts as a Strategy for data filtering/reshaping steps.
    - Provides hooks (`optimization_stats`) for future Observer/reporting integrations.
    """

    def __init__(
        self,
        config: OptimizationConfig | None = None,
        config_path: Path | None = None,
        observers: list[ProgressObserver] | None = None,
    ):
        settings_path = config_path or KG_PIPELINE_CONFIG_PATH
        self.config = config or OptimizationConfig.from_mapping(
            _load_data_optimizer_settings(settings_path)
        )
        self.file_manager = FileManager()
        self.optimization_stats = {}
        self._observers: list[ProgressObserver] = observers or [NullObserver()]

    def register_observer(self, observer: ProgressObserver) -> None:
        """Register a progress observer."""
        self._observers.append(observer)

    def _notify(self, method: str, context: dict | None = None) -> None:
        for obs in self._observers:
            fn = getattr(obs, method, None)
            if callable(fn):
                try:
                    fn(context)
                except Exception as exc:  # noqa: BLE001 - observers must not break flow
                    logger.debug(f"Observer error on {method}: {exc}")

    def analyze_data_quality(self, train_df: pl.DataFrame | pl.LazyFrame) -> dict:
        """Analyzes the quality and density of the data."""
        logger.debug("analise_qualidade iniciado")

        df = (
            train_df.collect(engine="streaming")
            if isinstance(train_df, pl.LazyFrame)
            else train_df
        )
        if df.is_empty():
            return {
                "num_triples": 0,
                "num_entities": 0,
                "num_relations": 0,
                "density": 0.0,
                "avg_degree": 0.0,
                "median_degree": 0.0,
                "min_degree": 0.0,
                "max_degree": 0.0,
                "relation_distribution": pl.DataFrame(),
                "low_degree_entities": 0,
                "rare_relations": 0,
            }

        # Basic statistics
        num_triples = len(df)
        entities_s = set(df["s"].unique())
        entities_o = set(df["o"].unique())
        all_entities = entities_s | entities_o
        relations = set(df["p"].unique())

        # Graph density
        max_possible_triples = len(all_entities) * len(all_entities) * len(relations)
        density = num_triples / max_possible_triples if max_possible_triples > 0 else 0

        # Degree distribution
        degree_stats = (
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

        # Relation statistics
        relation_stats = df.group_by("p").len().sort("len", descending=True)

        stats = {
            "num_triples": num_triples,
            "num_entities": len(all_entities),
            "num_relations": len(relations),
            "density": density,
            "avg_degree": degree_stats["degree"].mean(),
            "median_degree": degree_stats["degree"].median(),
            "min_degree": degree_stats["degree"].min(),
            "max_degree": degree_stats["degree"].max(),
            "relation_distribution": relation_stats,
            "low_degree_entities": len(
                degree_stats.filter(pl.col("degree") < self.config.min_entity_degree)
            ),
            "rare_relations": len(
                relation_stats.filter(pl.col("len") < self.config.min_relation_support)
            ),
        }

        if self.config.log_statistics:
            self._log_analysis(stats)

        return stats

    def _log_analysis(self, stats: dict):
        """Detailed log of the analysis."""
        logger.info(f"  Triplas: {stats['num_triples']:,}")
        logger.info(f"  Entidades: {stats['num_entities']:,}")
        logger.info(f"  Relacoes: {stats['num_relations']}")
        logger.info(
            f"  Densidade: {stats['density']:.8f} ({stats['density'] * 100:.6f}%)"
        )
        logger.info(f"  Grau medio: {stats['avg_degree']:.2f}")
        logger.info(
            f"  Entidades esparsas (grau < {self.config.min_entity_degree}): {stats['low_degree_entities']:,}"
        )
        logger.info(
            f"  Relacoes raras (< {self.config.min_relation_support} exemplos): {stats['rare_relations']}"
        )

        # Top 10 relations - consolidated to debug level
        top_relations = stats["relation_distribution"].head(10)
        top_rel_str = ", ".join(
            f"{r['p']}:{r['len']:,}" for r in top_relations.iter_rows(named=True)
        )
        logger.debug(f"Top 10 relacoes: {top_rel_str}")

    def filter_sparse_entities(
        self, train_df: pl.DataFrame | pl.LazyFrame
    ) -> pl.LazyFrame:
        lf = train_df.lazy() if isinstance(train_df, pl.DataFrame) else train_df

        entity_degrees = (
            pl.concat(
                [
                    lf.select(pl.col("s").alias("entity")),
                    lf.select(pl.col("o").alias("entity")),
                ]
            )
            .group_by("entity")
            .len()
            .rename({"len": "degree"})
        )

        valid_entities = entity_degrees.filter(
            pl.col("degree") >= self.config.min_entity_degree
        ).select("entity")

        return (
            lf.join(valid_entities, left_on="s", right_on="entity", how="semi")
            .join(valid_entities, left_on="o", right_on="entity", how="semi")
            .select(["s", "p", "o"])
        )

    def balance_relations(self, train_df: pl.DataFrame | pl.LazyFrame) -> pl.LazyFrame:
        lf = train_df.lazy() if isinstance(train_df, pl.DataFrame) else train_df
        relation_counts = lf.group_by("p").len().rename({"len": "count"})
        valid_relations = relation_counts.filter(
            pl.col("count") >= self.config.min_relation_support
        ).select("p")
        return lf.join(valid_relations, on="p", how="semi").select(["s", "p", "o"])

    def remove_duplicates(self, train_df: pl.DataFrame | pl.LazyFrame) -> pl.LazyFrame:
        lf = train_df.lazy() if isinstance(train_df, pl.DataFrame) else train_df
        return lf.unique(subset=["s", "p", "o"])

    def remove_self_loops(self, train_df: pl.DataFrame | pl.LazyFrame) -> pl.LazyFrame:
        lf = train_df.lazy() if isinstance(train_df, pl.DataFrame) else train_df
        return lf.filter(pl.col("s") != pl.col("o"))

    def add_inverse_relations(
        self, train_df: pl.DataFrame | pl.LazyFrame
    ) -> pl.LazyFrame:
        lf = train_df.lazy() if isinstance(train_df, pl.DataFrame) else train_df
        suffix = self.config.inverse_relation_suffix
        inverse_df = lf.select(
            [
                pl.col("o").alias("s"),
                (pl.col("p").cast(pl.Utf8) + pl.lit(suffix)).alias("p"),
                pl.col("s").alias("o"),
            ]
        )
        return pl.concat([lf, inverse_df])

    def optimize_telecom_data(self, train_path: Path) -> tuple[pl.DataFrame, dict]:
        self._notify("on_start", {"stage": "optimize", "path": str(train_path)})

        original_bundle = self.file_manager.read(train_path)
        if isinstance(original_bundle, ParquetBundle):
            current_lf = original_bundle.lazyframe()
        elif isinstance(original_bundle, pl.DataFrame):
            current_lf = original_bundle.lazy()
        else:
            raise ValueError("Expected tabular data for optimization pipeline")

        if self.config.remove_duplicates:
            current_lf = self.remove_duplicates(current_lf)

        if self.config.remove_self_loops:
            current_lf = self.remove_self_loops(current_lf)

        if self.config.add_inverse_relations:
            current_lf = self.add_inverse_relations(current_lf)

        current_lf = self.filter_sparse_entities(current_lf)

        if self.config.balance_relations:
            current_lf = self.balance_relations(current_lf)

        result_df = current_lf.collect(engine="streaming")

        optimized_path = train_path.with_name(
            train_path.stem + "_optimized" + train_path.suffix
        )
        self.file_manager.save(result_df, optimized_path)

        return result_df, {}


def quick_optimize_training_data(
    train_path: Path | None = None,
    min_entity_degree: int | None = None,
    min_relation_support: int | None = None,
    config_path: Path | None = None,
    add_inverse_relations: bool | None = False,
) -> tuple[pl.DataFrame, dict]:
    """
    Utility function for quick optimization.

    Args:
        train_path: Path to the training data (if None, uses default)
        min_entity_degree: Minimum degree to keep entities (None = config)
        min_relation_support: Minimum number of examples to keep relations (None = config)
        config_path: Optional config path override
        add_inverse_relations: Whether to add inverse relations during quick optimize (defaults to False to avoid size growth in smoke runs)

    Returns:
        A tuple with (optimized_data, statistics)
    """
    settings_path = config_path or KG_PIPELINE_CONFIG_PATH
    optimizer_settings = _load_data_optimizer_settings(settings_path)
    if min_entity_degree is not None:
        optimizer_settings["min_entity_degree"] = min_entity_degree
    if min_relation_support is not None:
        optimizer_settings["min_relation_support"] = min_relation_support
    if add_inverse_relations is not None:
        optimizer_settings["add_inverse_relations"] = add_inverse_relations

    if train_path is None:
        # Use default path based on KG configuration
        from pff.domain.kg.config import KGConfig  # noqa: PLC0415

        kg_config = KGConfig(settings_path)
        train_path = kg_config.get_split_path("train")

    config = OptimizationConfig.from_mapping(optimizer_settings)
    optimizer = TelecomDataOptimizer(config, settings_path)
    return optimizer.optimize_telecom_data(train_path)


# Convenience function for easy integration
def optimize_if_needed(
    force_optimization: bool = False, config_path: Path | None = None
) -> bool:
    """
    Optimizes training data if necessary or forced.

    Args:
        force_optimization: If True, forces optimization even if an optimized file exists
        config_path: Optional config path override

    Returns:
        True if optimization was performed, False otherwise
    """
    from pff.domain.kg.config import KGConfig  # noqa: PLC0415

    settings_path = config_path or KG_PIPELINE_CONFIG_PATH
    kg_config = KGConfig(settings_path)
    train_path = kg_config.get_split_path("train")
    optimized_path = train_path.with_name(
        train_path.stem + "_optimized" + train_path.suffix
    )

    # Check if an optimized version already exists
    if FileManager.exists(optimized_path) and not force_optimization:
        logger.info(f"Dados otimizados ja existem: {optimized_path}")
        return False

    # Run optimization
    logger.info("Executando otimizacao automatica dos dados...")
    optimized_df, stats = quick_optimize_training_data(train_path)

    logger.success("Otimizacao concluida!")
    return True
