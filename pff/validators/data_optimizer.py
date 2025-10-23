import polars as pl
from pathlib import Path

from dataclasses import dataclass

from pff.utils import logger, FileManager
from pff.config import settings


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


class TelecomDataOptimizer:
    """Specific optimizer for sparse telephony data."""
    
    def __init__(self, config: OptimizationConfig | None = None):
        self.config = config or OptimizationConfig()
        self.file_manager = FileManager()
        self.optimization_stats = {}
        
    def analyze_data_quality(self, train_df: pl.DataFrame) -> dict:
        """Analyzes the quality and density of the data."""
        logger.info("📊 ANALISANDO QUALIDADE DOS DADOS")
        
        # Basic statistics
        num_triples = len(train_df)
        entities_s = set(train_df['s'].unique())
        entities_o = set(train_df['o'].unique()) 
        all_entities = entities_s | entities_o
        relations = set(train_df['p'].unique())
        
        # Graph density
        max_possible_triples = len(all_entities) * len(all_entities) * len(relations)
        density = num_triples / max_possible_triples if max_possible_triples > 0 else 0
        
        # Degree distribution
        degree_stats = pl.concat([
            train_df.select(pl.col("s").alias("entity")),
            train_df.select(pl.col("o").alias("entity"))
        ]).group_by("entity").len().rename({"len": "degree"})
        
        # Relation statistics
        relation_stats = train_df.group_by('p').len().sort('len', descending=True)
        
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
        logger.info(f"  📈 Triplas: {stats['num_triples']:,}")
        logger.info(f"  👥 Entidades: {stats['num_entities']:,}")
        logger.info(f"  🔗 Relações: {stats['num_relations']}")
        logger.info(f"  📊 Densidade: {stats['density']:.8f} ({stats['density']*100:.6f}%)")
        logger.info(f"  📐 Grau médio: {stats['avg_degree']:.2f}")
        logger.info(f"  ⚠️  Entidades esparsas (grau < {self.config.min_entity_degree}): {stats['low_degree_entities']:,}")
        logger.info(f"  ⚠️  Relações raras (< {self.config.min_relation_support} exemplos): {stats['rare_relations']}")
        
        # Top 10 relations
        logger.info("  🔝 Top 10 relações mais frequentes:")
        for row in stats['relation_distribution'].head(10).iter_rows(named=True):
            logger.info(f"     - {row['p']}: {row['len']:,} triplas")
    
    def filter_sparse_entities(self, train_df: pl.DataFrame) -> pl.DataFrame:
        """Removes entities with few connections."""
        logger.info(f"🔄 Filtrando entidades com grau < {self.config.min_entity_degree}")
        
        # Calculate degrees
        entity_degrees = pl.concat([
            train_df.select(pl.col("s").alias("entity")),
            train_df.select(pl.col("o").alias("entity"))
        ]).group_by("entity").len().rename({"len": "degree"})
        
        initial_entity_count = entity_degrees.height

        # Valid entities
        valid_entities = entity_degrees.filter(
            pl.col("degree") >= self.config.min_entity_degree
        ).get_column("entity").to_list()
        # Filter DataFrame
        filtered_df = train_df.filter(
            pl.col('s').is_in(valid_entities) & 
            pl.col('o').is_in(valid_entities)
        )
        
        logger.info(f"  - Entidades mantidas: {len(valid_entities):,} / {initial_entity_count:,}")
        logger.info(f"  - Triplas mantidas: {len(filtered_df):,} / {len(train_df):,}")
        
        return filtered_df

    def balance_relations(self, train_df: pl.DataFrame) -> pl.DataFrame:
        """Removes relations with few examples to balance the dataset."""
        logger.info(f"🔄 Filtrando relações com suporte < {self.config.min_relation_support}")

        # Count examples per relation
        relation_counts = train_df.group_by('p').len().rename({'len': 'count'})
        
        # Identify relations to keep
        valid_relations = relation_counts.filter(
            pl.col('count') >= self.config.min_relation_support
        ).get_column('p')

        initial_triples = len(train_df)
        initial_relations = train_df['p'].n_unique()

        # Filter the DataFrame
        filtered_df = train_df.filter(pl.col('p').is_in(valid_relations))

        final_triples = len(filtered_df)
        final_relations = filtered_df['p'].n_unique()

        logger.info(f"  - Relações mantidas: {final_relations:,} / {initial_relations:,}")
        logger.info(f"  - Triplas mantidas: {final_triples:,} / {initial_triples:,}")

        return filtered_df
    
    
    def optimize_telecom_data(self, train_path: Path) -> tuple[pl.DataFrame, dict]:
        """Complete optimization pipeline for telephony data."""
        logger.info("🚀 INICIANDO OTIMIZAÇÃO DE DADOS DE TELEFONIA")
        logger.info("=" * 60)
        
        # Load original data
        original_df = self.file_manager.read(train_path)
        logger.info(f"📂 Dados originais carregados: {len(original_df):,} triplas")
        
        # Backup if necessary
        backup_path = None
        if self.config.preserve_original:
            backup_path = train_path.with_suffix('.backup' + train_path.suffix)
            if not backup_path.exists():
                self.file_manager.save(original_df, backup_path)
                logger.info(f"💾 Backup salvo em: {backup_path}")
        
        # Initial analysis
        initial_stats = self.analyze_data_quality(original_df)
        
        # Step 1: Filter sparse entities
        step1_df = self.filter_sparse_entities(original_df)
        
        # Step 2: Balance relations
        step2_df = self.balance_relations(step1_df) if self.config.balance_relations else step1_df
        
        # Final analysis
        logger.info("\n" + "=" * 60)
        logger.info("📊 ANÁLISE FINAL:")
        final_stats = self.analyze_data_quality(step2_df)
        
        # Comparison
        improvement_density = final_stats['density'] / initial_stats['density'] if initial_stats['density'] > 0 else float('inf')
        improvement_avg_degree = final_stats['avg_degree'] / initial_stats['avg_degree'] if initial_stats['avg_degree'] > 0 else float('inf')
        
        logger.info("🎯 MELHORIAS:")
        logger.info(f"  - Densidade: {improvement_density:.2f}x melhor")
        logger.info(f"  - Grau médio: {improvement_avg_degree:.2f}x melhor")
        logger.info(f"  - Redução de tamanho: {len(step2_df)/len(original_df):.2%} dos dados originais")
        
        # Save optimized data
        optimized_path = train_path.with_name(train_path.stem + '_optimized' + train_path.suffix)
        self.file_manager.save(step2_df, optimized_path)
        logger.info(f"✅ Dados otimizados salvos em: {optimized_path}")
        
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
        return step2_df, optimization_summary


def quick_optimize_training_data(
    train_path: Path | None = None, 
    min_entity_degree: int = 3,
    min_relation_support: int = 50
) -> tuple[pl.DataFrame, dict]:
    """
    Utility function for quick optimization.
    
    Args:
        train_path: Path to the training data (if None, uses default)
        min_entity_degree: Minimum degree to keep entities
        min_relation_support: Minimum number of examples to keep relations
    
    Returns:
        A tuple with (optimized_data, statistics)
    """
    if train_path is None:
        # Use default path based on KG configuration
        from pff.validators.kg.config import KGConfig
        kg_config = KGConfig(settings.CONFIG_DIR / "kg.yaml")
        train_path = kg_config.get_split_path("train")
    
    config = OptimizationConfig(
        min_entity_degree=min_entity_degree,
        min_relation_support=min_relation_support,
        preserve_original=True,
        log_statistics=True
    )
    
    optimizer = TelecomDataOptimizer(config)
    return optimizer.optimize_telecom_data(train_path)


# Convenience function for easy integration
def optimize_if_needed(force_optimization: bool = False) -> bool:
    """
    Optimizes training data if necessary or forced.
    
    Args:
        force_optimization: If True, forces optimization even if an optimized file exists
        
    Returns:
        True if optimization was performed, False otherwise
    """
    from pff.validators.kg.config import KGConfig
    
    kg_config = KGConfig(settings.CONFIG_DIR / "kg.yaml")
    train_path = kg_config.get_split_path("train")
    optimized_path = train_path.with_name(train_path.stem + '_optimized' + train_path.suffix)
    
    # Check if an optimized version already exists
    if optimized_path.exists() and not force_optimization:
        logger.info(f"✅ Dados otimizados já existem: {optimized_path}")
        return False
    
    # Run optimization
    logger.info("🔄 Executando otimização automática dos dados...")
    optimized_df, stats = quick_optimize_training_data(train_path)
    
    logger.success("✅ Otimização concluída!")
    return True