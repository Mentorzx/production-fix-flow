import polars as pl
from pathlib import Path

from dataclasses import dataclass
from typing import Any, Mapping

from pff.config import KG_PIPELINE_CONFIG_PATH
from pff.utils import FileManager, logger
from pff.utils.observer import NullObserver, ProgressObserver


@dataclass
class OptimizationConfig:
    """Configuration for sparse data optimization."""
    
    # Sparsity filters
    min_entity_degree: int = 3          # Entities must have >= 3 connections
    min_relation_support: int = 50      # Relations must have >= 50 examples
    
    # Balancing
    max_entities_to_keep: int | None = None  # Limit of entities (None = no limit)
    balance_relations: bool = True       # Balance relation distribution
    
    # Performance
    preserve_original: bool = True       # Keep a backup of the original data
    log_statistics: bool = True          # Detailed log of transformations
    
    # Specific telephony data
    focus_on_active_users: bool = True   # Prioritize users with more activity
    min_product_interactions: int = 2    # Users must have >= 2 products/services

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "OptimizationConfig":
        """Create an OptimizationConfig from a mapping with safe fallbacks."""
        return cls(
            min_entity_degree=int(data.get("min_entity_degree", cls.min_entity_degree)),
            min_relation_support=int(data.get("min_relation_support", cls.min_relation_support)),
            max_entities_to_keep=data.get("max_entities_to_keep", cls.max_entities_to_keep),
            balance_relations=bool(data.get("balance_relations", cls.balance_relations)),
            preserve_original=bool(data.get("preserve_original", cls.preserve_original)),
            log_statistics=bool(data.get("log_statistics", cls.log_statistics)),
            focus_on_active_users=bool(data.get("focus_on_active_users", cls.focus_on_active_users)),
            min_product_interactions=int(data.get("min_product_interactions", cls.min_product_interactions)),
        )


def _load_data_optimizer_settings(config_path: Path = KG_PIPELINE_CONFIG_PATH) -> dict[str, Any]:
    """Load data optimizer settings from the KG pipeline config."""
    try:
        cfg = FileManager().read(config_path) or {}
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
        logger.info("[ANALISE] ANALISANDO QUALIDADE DOS DADOS")

        df = (
            train_df.collect(engine="streaming")
            if isinstance(train_df, pl.LazyFrame)
            else train_df
        )
        if df.is_empty():
            return {
                'num_triples': 0,
                'num_entities': 0,
                'num_relations': 0,
                'density': 0.0,
                'avg_degree': 0.0,
                'median_degree': 0.0,
                'min_degree': 0.0,
                'max_degree': 0.0,
                'relation_distribution': pl.DataFrame(),
                'low_degree_entities': 0,
                'rare_relations': 0,
            }

        # Basic statistics
        num_triples = len(df)
        entities_s = set(df['s'].unique())
        entities_o = set(df['o'].unique()) 
        all_entities = entities_s | entities_o
        relations = set(df['p'].unique())
        
        # Graph density
        max_possible_triples = len(all_entities) * len(all_entities) * len(relations)
        density = num_triples / max_possible_triples if max_possible_triples > 0 else 0
        
        # Degree distribution
        degree_stats = pl.concat([
            df.select(pl.col("s").alias("entity")),
            df.select(pl.col("o").alias("entity"))
        ]).group_by("entity").len().rename({"len": "degree"})
        
        # Relation statistics
        relation_stats = df.group_by('p').len().sort('len', descending=True)
        
        stats = {
            'num_triples': num_triples,
            'num_entities': len(all_entities),
            'num_relations': len(relations),
            'density': density,
            'avg_degree': degree_stats['degree'].mean(),
            'median_degree': degree_stats['degree'].median(),
            'min_degree': degree_stats['degree'].min(),
            'max_degree': degree_stats['degree'].max(),
            'relation_distribution': relation_stats,
            'low_degree_entities': len(degree_stats.filter(pl.col('degree') < self.config.min_entity_degree)),
            'rare_relations': len(relation_stats.filter(pl.col('len') < self.config.min_relation_support))
        }
        
        if self.config.log_statistics:
            self._log_analysis(stats)
            
        return stats
    
    def _log_analysis(self, stats: dict):
        """Detailed log of the analysis."""
        logger.info(f"  Triplas: {stats['num_triples']:,}")
        logger.info(f"  Entidades: {stats['num_entities']:,}")
        logger.info(f"  Relacoes: {stats['num_relations']}")
        logger.info(f"  Densidade: {stats['density']:.8f} ({stats['density']*100:.6f}%)")
        logger.info(f"  Grau medio: {stats['avg_degree']:.2f}")
        logger.info(f"  Entidades esparsas (grau < {self.config.min_entity_degree}): {stats['low_degree_entities']:,}")
        logger.info(f"  Relacoes raras (< {self.config.min_relation_support} exemplos): {stats['rare_relations']}")
        
        # Top 10 relations
        logger.info("  Top 10 relacoes mais frequentes:")
        for row in stats['relation_distribution'].head(10).iter_rows(named=True):
            logger.info(f"    - {row['p']}: {row['len']:,} triplas")
    
    def filter_sparse_entities(self, train_df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
        """Removes entities with few connections."""
        logger.info(f"[FILTRO] Filtrando entidades com grau < {self.config.min_entity_degree}")

        lf = train_df.lazy() if isinstance(train_df, pl.DataFrame) else train_df

        entity_degrees = pl.concat([
            lf.select(pl.col("s").alias("entity")),
            lf.select(pl.col("o").alias("entity"))
        ]).group_by("entity").len().rename({"len": "degree"})

        valid_entities = entity_degrees.filter(
            pl.col("degree") >= self.config.min_entity_degree
        ).select("entity")

        filtered_df = (
            lf.join(valid_entities, left_on="s", right_on="entity", how="semi")
            .join(valid_entities, left_on="o", right_on="entity", how="semi")
            .select(["s", "p", "o"])
        )

        result_df = filtered_df.collect(engine="streaming")
        initial_entities = entity_degrees.collect(engine="streaming")
        initial_entity_count = len(initial_entities)
        unique_entities = set(result_df["s"].to_list()) | set(result_df["o"].to_list())
        logger.info(f"  Entidades mantidas: {len(unique_entities):,} / {initial_entity_count:,}")
        logger.info(f"  Triplas mantidas: {len(result_df):,}")
        return result_df

    def balance_relations(self, train_df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
        """Removes relations with few examples to balance the dataset."""
        logger.info(f"[BALANCEAMENTO] Filtrando relacoes com suporte < {self.config.min_relation_support}")

        lf = train_df.lazy() if isinstance(train_df, pl.DataFrame) else train_df

        # Count examples per relation
        relation_counts = lf.group_by('p').len().rename({'len': 'count'})

        valid_relations = relation_counts.filter(
            pl.col('count') >= self.config.min_relation_support
        ).select('p')

        filtered_df = lf.join(valid_relations, on='p', how='semi').select(['s', 'p', 'o'])

        result_df = filtered_df.collect(engine="streaming")
        initial_relations = lf.select(pl.col("p").n_unique()).collect(engine="streaming")[0, 0]
        final_relations = result_df['p'].n_unique()
        logger.info(f"  Relacoes mantidas: {final_relations:,} / {initial_relations:,}")
        logger.info(f"  Triplas mantidas: {len(result_df):,}")
        return result_df
    
    
    def optimize_telecom_data(self, train_path: Path) -> tuple[pl.DataFrame, dict]:
        """Complete optimization pipeline for telephony data."""
        logger.info("[OTIMIZACAO] INICIANDO OTIMIZACAO DE DADOS DE TELEFONIA")
        logger.info("=" * 60)

        self._notify("on_start", {"stage": "optimize", "path": str(train_path)})
        
        # Load original data
        original_lf = self.file_manager.read(train_path, lazy=True, encoding="utf8")
        original_df = original_lf.collect(engine="streaming")
        logger.info(f"Dados originais carregados: {len(original_df):,} triplas")
        
        # Backup if necessary
        backup_path = None
        if self.config.preserve_original:
            backup_path = train_path.with_suffix('.backup' + train_path.suffix)
            if not FileManager.exists(backup_path):
                self.file_manager.save(original_df, backup_path)
                logger.info(f"Backup salvo em: {backup_path}")
        
        # Initial analysis
        initial_stats = self.analyze_data_quality(original_df)
        
        # Step 1: Filter sparse entities
        step1_df = self.filter_sparse_entities(original_lf)
        self._notify("on_step", {"stage": "filter_entities"})
        
        # Step 2: Balance relations
        step2_df = self.balance_relations(step1_df) if self.config.balance_relations else step1_df
        self._notify("on_step", {"stage": "balance_relations"})
        
        # Final analysis
        logger.info("\n" + "=" * 60)
        logger.info("[RESULTADO] ANALISE FINAL:")
        final_stats = self.analyze_data_quality(step2_df)
        
        # Comparison
        improvement_density = final_stats['density'] / initial_stats['density'] if initial_stats['density'] > 0 else float('inf')
        improvement_avg_degree = final_stats['avg_degree'] / initial_stats['avg_degree'] if initial_stats['avg_degree'] > 0 else float('inf')
        
        logger.info("[MELHORIAS]:")
        logger.info(f"  Densidade: {improvement_density:.2f}x melhor")
        logger.info(f"  Grau medio: {improvement_avg_degree:.2f}x melhor")
        logger.info(f"  Reducao de tamanho: {len(step2_df)/len(original_df):.2%} dos dados originais")
        
        # Save optimized data
        optimized_path = train_path.with_name(train_path.stem + '_optimized' + train_path.suffix)
        self.file_manager.save(step2_df, optimized_path)
        logger.info(f"Dados otimizados salvos em: {optimized_path}")
        
        # Compile statistics
        optimization_summary = {
            'original_stats': initial_stats,
            'final_stats': final_stats,
            'improvements': {
                'density_improvement': improvement_density,
                'avg_degree_improvement': improvement_avg_degree,
                'size_reduction': len(step2_df) / len(original_df)
            },
            'paths': {
                'original': str(train_path),
                'backup': str(backup_path) if self.config.preserve_original else None,
                'optimized': str(optimized_path)
            }
        }
        
        self.optimization_stats = optimization_summary
        self._notify("on_complete", {"stage": "optimize", "final_triples": len(step2_df)})
        return step2_df, optimization_summary


def quick_optimize_training_data(
    train_path: Path | None = None, 
    min_entity_degree: int | None = None,
    min_relation_support: int | None = None,
    config_path: Path | None = None,
) -> tuple[pl.DataFrame, dict]:
    """
    Utility function for quick optimization.
    
    Args:
        train_path: Path to the training data (if None, uses default)
        min_entity_degree: Minimum degree to keep entities (None = config)
        min_relation_support: Minimum number of examples to keep relations (None = config)
        config_path: Optional config path override
    
    Returns:
        A tuple with (optimized_data, statistics)
    """
    settings_path = config_path or KG_PIPELINE_CONFIG_PATH
    optimizer_settings = _load_data_optimizer_settings(settings_path)
    if min_entity_degree is not None:
        optimizer_settings["min_entity_degree"] = min_entity_degree
    if min_relation_support is not None:
        optimizer_settings["min_relation_support"] = min_relation_support

    if train_path is None:
        # Use default path based on KG configuration
        from pff.validators.kg.config import KGConfig
        kg_config = KGConfig(settings_path)
        train_path = kg_config.get_split_path("train")
    
    config = OptimizationConfig.from_mapping(optimizer_settings)
    optimizer = TelecomDataOptimizer(config, settings_path)
    return optimizer.optimize_telecom_data(train_path)


# Convenience function for easy integration
def optimize_if_needed(force_optimization: bool = False, config_path: Path | None = None) -> bool:
    """
    Optimizes training data if necessary or forced.
    
    Args:
        force_optimization: If True, forces optimization even if an optimized file exists
        config_path: Optional config path override
        
    Returns:
        True if optimization was performed, False otherwise
    """
    from pff.validators.kg.config import KGConfig
    
    settings_path = config_path or KG_PIPELINE_CONFIG_PATH
    kg_config = KGConfig(settings_path)
    train_path = kg_config.get_split_path("train")
    optimized_path = train_path.with_name(train_path.stem + '_optimized' + train_path.suffix)
    
    # Check if an optimized version already exists
    if FileManager.exists(optimized_path) and not force_optimization:
        logger.info(f"Dados otimizados ja existem: {optimized_path}")
        return False
    
    # Run optimization
    logger.info("Executando otimizacao automatica dos dados...")
    optimized_df, stats = quick_optimize_training_data(train_path)
    
    logger.success("Otimizacao concluida!")
    return True
