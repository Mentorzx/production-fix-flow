from __future__ import annotations

import asyncio
from typing import Any

import numpy as np
import polars as pl

from pff import settings
from pff.utils import CacheManager, FileManager, logger
from pff.utils.global_interrupt_manager import get_interrupt_manager, should_stop
from pff.validators.kg.builder import KGBuilder
from pff.validators.kg.config import KGConfig

"""
TransE Data Preprocessing Module

This module is responsible for optimizing knowledge graph data for TransE training.
It implements sophisticated filtering techniques to ensure high-quality training data
with at least 10K triples after optimization.
"""


class TransEPreprocessor:
    """
    Advanced preprocessing pipeline for TransE knowledge graph embeddings.

    This class implements state-of-the-art optimization techniques for knowledge
    graph data, including entity degree filtering, relation support filtering,
    and intelligent data augmentation to achieve optimal triple counts.
    """

    def __init__(self, kg_config: KGConfig, target_triples: int = 10000):
        """
        Initialize the TransE preprocessor.

        Args:
            kg_config: Knowledge graph configuration object
            target_triples: Target number of triples after optimization (default: 10000)
        """
        self.config = kg_config
        self.file_manager = FileManager()
        self.cache_manager = CacheManager()
        self.interrupt_manager = get_interrupt_manager()
        self.target_triples = target_triples
        self.graph_dir = self.config.graph_directory
        self.output_dir = settings.OUTPUTS_DIR / "transe"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Optimization parameters
        self.min_entity_degree = 2  # Minimum connections per entity
        self.min_relation_support = 20  # Minimum examples per relation
        self.degree_percentile = 5  # Keep entities above this percentile

        # Statistics tracking
        self.stats: dict[str, Any] = {}

        # Register interrupt handler
        self.interrupt_manager.register_callback(self._cleanup_on_interrupt)
        self._build_raw_df_cached = self.cache_manager.disk_cache(ttl=3600)(
            self._build_raw_df
        )

    def _cleanup_on_interrupt(self) -> None:
        """Cleanup callback for interrupt handling."""
        logger.info("🧹 TransE Preprocessor: Limpeza por interrupção...")
        # Save any intermediate results if needed
        if hasattr(self, "stats") and self.stats:
            stats_path = self.output_dir / "preprocessing_stats.json"
            self.file_manager.save(self.stats, stats_path)
            logger.info(f"💾 Estatísticas salvas em: {stats_path}")

    async def run(self) -> dict[str, Any]:
        """
        Execute the complete preprocessing pipeline.

        Returns:
            Dictionary containing preprocessing statistics and file paths
        """
        logger.info("🚀 Iniciando otimização de dados para TransE...")
        logger.info("=" * 60)

        # Check for interruption
        if should_stop():
            logger.warning("🛑 Preprocessamento cancelado antes de iniciar")
            return {"status": "cancelled"}

        # Ensure raw data exists
        await self._ensure_raw_data_exists()

        # Load and analyze data
        raw_data = self._load_raw_data()
        self._analyze_data_quality(raw_data, "DADOS ORIGINAIS")

        # Apply optimization strategies
        optimized_data = self._optimize_data(raw_data)

        # Generate consistent splits
        train, valid, test = self._create_optimized_splits(optimized_data)

        # Create and save mappings
        entity_map, relation_map = self._create_mappings(train, valid, test)

        # Convert to indexed arrays and save
        self._save_indexed_arrays(train, valid, test, entity_map, relation_map)

        # Final statistics
        self._log_final_statistics()

        return self.stats

    async def _ensure_raw_data_exists(self) -> None:
        """Ensure raw parquet files exist, building them if necessary."""
        required_files = ["train.parquet", "valid.parquet", "test.parquet"]
        missing_files = [f for f in required_files if not (self.graph_dir / f).exists()]

        if missing_files:
            logger.warning(f"Arquivos faltando: {missing_files}")
            logger.info("Executando KGBuilder para gerar dados brutos...")

            builder = KGBuilder(
                source_path=settings.DATA_DIR / "models" / "correct.zip",
                output_dir=self.graph_dir,
            )
            await builder.run()
            logger.success("✅ Dados brutos gerados com sucesso")

    def _build_raw_df(self) -> pl.DataFrame:
        """Builds and returns a unique concatenated DataFrame from available train, valid, and test Parquet files in the graph directory."""
        frames = [
            self.file_manager.read(self.graph_dir / f"{split}.parquet")
            for split in ("train", "valid", "test")
            if (self.graph_dir / f"{split}.parquet").exists()
        ]
        if not frames:
            raise FileNotFoundError("Nenhum arquivo de dados encontrado!")

        df = pl.concat(frames).unique(subset=["s", "p", "o"])

        return df

    def _load_raw_data(self) -> pl.DataFrame:
        """Loads raw data into a Polars DataFrame, updates statistics, and logs the process."""
        logger.info("📂 Carregando dados brutos (TTL = 1 h)…")

        df = self._build_raw_df_cached()  # <- basta isso

        self.stats["original_triples"] = len(df)
        logger.info("✅ Dados obtidos.")
        return df

    def _analyze_data_quality(self, df: pl.DataFrame, phase: str) -> dict[str, Any]:
        """Analyze and log data quality metrics."""
        logger.info(f"\n📊 ANÁLISE DE QUALIDADE - {phase}")
        logger.info("-" * 50)

        # Basic statistics
        num_triples = len(df)
        entities = set(df["s"].to_list() + df["o"].to_list())
        num_entities = len(entities)
        num_relations = df["p"].n_unique()

        # Calculate entity degrees
        entity_degrees = {}
        for entity in entities:
            degree = len(df.filter((pl.col("s") == entity) | (pl.col("o") == entity)))
            entity_degrees[entity] = degree

        # Degree statistics
        degrees = list(entity_degrees.values())
        avg_degree = np.mean(degrees)
        median_degree = np.median(degrees)

        # Sparse entities (degree < threshold)
        sparse_entities = sum(1 for d in degrees if d < self.min_entity_degree)

        # Relation support
        relation_counts = df.group_by("p").count().sort("count", descending=True)
        rare_relations = len(
            relation_counts.filter(pl.col("count") < self.min_relation_support)
        )

        # Graph density
        max_possible_edges = num_entities * (num_entities - 1) * num_relations
        density = num_triples / max_possible_edges if max_possible_edges > 0 else 0

        # Log statistics
        logger.info(f"  📈 Triplas: {num_triples:,}")
        logger.info(f"  👥 Entidades: {num_entities:,}")
        logger.info(f"  🔗 Relações: {num_relations}")
        logger.info(f"  📊 Densidade: {density:.8f} ({density * 100:.6f}%)")
        logger.info(f"  📐 Grau médio: {avg_degree:.2f} (mediana: {median_degree:.1f})")
        logger.info(
            f"  ⚠️  Entidades esparsas (grau < {self.min_entity_degree}): {sparse_entities:,}"
        )
        logger.info(
            f"  ⚠️  Relações raras (< {self.min_relation_support} exemplos): {rare_relations}"
        )

        # Top relations
        logger.info("\n  🔝 Top 10 relações mais frequentes:")
        for row in relation_counts.head(10).iter_rows(named=True):
            logger.info(f"     - {row['p']}: {row['count']:,} triplas")

        # Store metrics
        metrics = {
            "num_triples": num_triples,
            "num_entities": num_entities,
            "num_relations": num_relations,
            "density": density,
            "avg_degree": avg_degree,
            "sparse_entities": sparse_entities,
            "rare_relations": rare_relations,
            "entity_degrees": entity_degrees,
        }

        self.stats[f"{phase.lower().replace(' ', '_')}_metrics"] = metrics
        return metrics

    def _optimize_data(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply sophisticated optimization strategies to the data."""
        logger.info("\n🔧 APLICANDO OTIMIZAÇÕES")
        logger.info("-" * 50)

        # Check for interruption
        if should_stop():
            logger.warning("🛑 Otimização interrompida")
            return df

        # Save backup
        backup_path = self.graph_dir / "train.backup.parquet"
        if not backup_path.exists():
            self.file_manager.save(df, backup_path)
            logger.info(f"💾 Backup salvo em: {backup_path}")

        optimized = df

        # Phase 1: Filter by entity degree
        optimized = self._filter_by_entity_degree(optimized)

        if should_stop():
            return optimized

        # Phase 2: Filter by relation support
        optimized = self._filter_by_relation_support(optimized)

        if should_stop():
            return optimized

        # Phase 3: Apply advanced filtering if still too sparse
        if len(optimized) < self.target_triples * 0.8:
            optimized = self._apply_advanced_filtering(optimized, df)

        # Phase 4: Data augmentation if needed
        if len(optimized) < self.target_triples:
            optimized = self._augment_data(optimized, df)

        return optimized

    def _filter_by_entity_degree(self, df: pl.DataFrame) -> pl.DataFrame:
        """Filter entities by degree using percentile-based thresholding."""
        logger.info(
            f"🔄 Filtrando entidades por grau (percentil {self.degree_percentile})"
        )

        # Calculate entity degrees
        entity_counts = {}
        for entity in set(df["s"].to_list() + df["o"].to_list()):
            count = len(df.filter((pl.col("s") == entity) | (pl.col("o") == entity)))
            entity_counts[entity] = count

        # Dynamic threshold based on percentile
        degrees = list(entity_counts.values())
        threshold = max(
            self.min_entity_degree, int(np.percentile(degrees, self.degree_percentile))
        )

        # Keep entities above threshold
        keep_entities = {e for e, d in entity_counts.items() if d >= threshold}

        # Filter dataframe
        filtered = df.filter(
            pl.col("s").is_in(keep_entities) & pl.col("o").is_in(keep_entities)
        )

        logger.info(f"  - Limite dinâmico: {threshold:.1f}")
        logger.info(
            f"  - Entidades mantidas: {len(keep_entities):,} / {len(entity_counts):,}"
        )
        logger.info(f"  - Triplas mantidas: {len(filtered):,} / {len(df):,}")

        return filtered

    def _filter_by_relation_support(self, df: pl.DataFrame) -> pl.DataFrame:
        """Filter relations by support count."""
        logger.info(f"🔄 Filtrando relações com suporte < {self.min_relation_support}")

        # Count relation occurrences
        relation_counts = df.group_by("p").count()
        keep_relations = relation_counts.filter(
            pl.col("count") >= self.min_relation_support
        )["p"].to_list()

        # Filter dataframe
        filtered = df.filter(pl.col("p").is_in(keep_relations))

        logger.info(
            f"  - Relações mantidas: {len(keep_relations)} / {relation_counts.height}"
        )
        logger.info(f"  - Triplas mantidas: {len(filtered):,} / {len(df):,}")

        return filtered

    def _apply_advanced_filtering(
        self, current: pl.DataFrame, original: pl.DataFrame
    ) -> pl.DataFrame:
        """Apply advanced filtering strategies to reach target triple count."""
        logger.info("🎯 Aplicando filtragem avançada para atingir meta de triplas")

        # Try progressively less strict filtering
        for entity_threshold in [1.5, 1.0, 0.5]:
            for relation_threshold in [15, 10, 5]:
                # Recalculate with new thresholds
                self.min_entity_degree = max(
                    1, int(self.min_entity_degree * entity_threshold)
                )
                self.min_relation_support = max(
                    1, int(self.min_relation_support * relation_threshold)
                )

                filtered = self._filter_by_entity_degree(original)
                filtered = self._filter_by_relation_support(filtered)

                if len(filtered) >= self.target_triples * 0.8:
                    logger.info(
                        f"  ✅ Meta atingida com limites: entidade>={self.min_entity_degree}, relação>={self.min_relation_support}"
                    )
                    return filtered

        logger.warning("  ⚠️ Não foi possível atingir a meta com filtragem")
        return current

    def _augment_data(self, df: pl.DataFrame, original: pl.DataFrame) -> pl.DataFrame:
        """Augment data using inverse relations and sampling."""
        logger.info("🔄 Aumentando dados para atingir meta de triplas")

        current_count = len(df)
        needed = self.target_triples - current_count

        if needed <= 0:
            return df

        augmented_triples = []

        # Strategy 1: Add inverse relations for symmetric relations
        symmetric_relations = self._identify_symmetric_relations(df)
        for rel in symmetric_relations:
            rel_triples = df.filter(pl.col("p") == rel)
            inverse_triples = rel_triples.select(
                [pl.col("o").alias("s"), pl.col("p"), pl.col("s").alias("o")]
            )

            # Remove duplicates
            inverse_triples = inverse_triples.join(df, on=["s", "p", "o"], how="anti")

            augmented_triples.append(
                inverse_triples.head(needed // len(symmetric_relations))
            )

        # Strategy 2: Sample from filtered entities in original data
        if len(pl.concat(augmented_triples)) < needed:
            entities_in_filtered = set(df["s"].to_list() + df["o"].to_list())
            candidates = original.filter(
                (pl.col("s").is_in(entities_in_filtered))
                | (pl.col("o").is_in(entities_in_filtered))
            )
            candidates = candidates.join(df, on=["s", "p", "o"], how="anti")

            remaining_needed = needed - len(pl.concat(augmented_triples))
            sampled = candidates.sample(n=min(remaining_needed, len(candidates)))
            augmented_triples.append(sampled)

        # Combine results
        if augmented_triples:
            augmented = pl.concat([df] + augmented_triples)
            logger.info(
                f"  ✅ Dados aumentados: {current_count:,} → {len(augmented):,} triplas"
            )
            return augmented

        return df

    def _identify_symmetric_relations(self, df: pl.DataFrame) -> list[str]:
        """Identify potentially symmetric relations."""
        symmetric = []

        for relation in df["p"].unique().to_list():
            rel_df = df.filter(pl.col("p") == relation)

            # Check symmetry by looking for inverse patterns
            inverse_count = 0
            for row in rel_df.sample(min(20, len(rel_df))).iter_rows(named=True):
                inverse_exists = (
                    len(
                        df.filter(
                            (pl.col("s") == row["o"])
                            & (pl.col("p") == relation)
                            & (pl.col("o") == row["s"])
                        )
                    )
                    > 0
                )
                if inverse_exists:
                    inverse_count += 1

            # If >50% have inverses, consider symmetric
            if inverse_count / min(20, len(rel_df)) > 0.5:
                symmetric.append(relation)

        return symmetric

    def _create_optimized_splits(
        self, df: pl.DataFrame
    ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """Create train/valid/test splits with no data leakage."""
        logger.info("\n🔧 Gerando splits otimizados sem data leakage")

        # Shuffle data
        shuffled = df.sample(fraction=1.0, shuffle=True, seed=42)

        # Calculate split sizes
        n_total = len(shuffled)
        n_test = int(n_total * 0.15)
        n_valid = int(n_total * 0.15)
        n_train = n_total - n_test - n_valid

        # Create splits
        test = shuffled.head(n_test)
        valid = shuffled.slice(n_test, n_valid)
        train = shuffled.slice(n_test + n_valid, n_train)

        # Verify no overlap
        train_set = set(
            train.select(
                pl.concat_str([pl.col("s"), pl.col("p"), pl.col("o")], separator="|")
            ).to_series()
        )
        valid_set = set(
            valid.select(
                pl.concat_str([pl.col("s"), pl.col("p"), pl.col("o")], separator="|")
            ).to_series()
        )
        test_set = set(
            test.select(
                pl.concat_str([pl.col("s"), pl.col("p"), pl.col("o")], separator="|")
            ).to_series()
        )

        overlap_stats = {
            "train_valid": len(train_set & valid_set),
            "train_test": len(train_set & test_set),
            "valid_test": len(valid_set & test_set),
        }

        logger.info("✅ Splits criados:")
        logger.info(f"   Treino: {len(train):,} triplas")
        logger.info(f"   Validação: {len(valid):,} triplas")
        logger.info(f"   Teste: {len(test):,} triplas")
        logger.info(f"🔍 Verificação de vazamento: {overlap_stats}")

        if any(overlap_stats.values()):
            raise RuntimeError(f"DATA LEAKAGE DETECTADO: {overlap_stats}")

        logger.success("✅ VERIFICAÇÃO PASSOU: Splits completamente limpos!")

        # Save optimized splits
        for split_name, split_df in [
            ("train", train),
            ("valid", valid),
            ("test", test),
        ]:
            path = self.graph_dir / f"{split_name}_optimized.parquet"
            self.file_manager.save(split_df, path)

        self.stats["splits"] = {
            "train": len(train),
            "valid": len(valid),
            "test": len(test),
            "overlap": overlap_stats,
        }

        return train, valid, test

    def _create_mappings(
        self, train: pl.DataFrame, valid: pl.DataFrame, test: pl.DataFrame
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Create entity and relation mappings."""
        logger.info("\n🔄 Criando mapeamentos globais")

        # Collect all unique entities and relations
        all_entities = set()
        all_relations = set()

        for df in [train, valid, test]:
            all_entities.update(df["s"].to_list())
            all_entities.update(df["o"].to_list())
            all_relations.update(df["p"].to_list())

        # Create mappings with sequential IDs
        entity_map = pl.DataFrame(
            {"id": range(len(all_entities)), "label": sorted(list(all_entities))}
        )

        relation_map = pl.DataFrame(
            {"id": range(len(all_relations)), "label": sorted(list(all_relations))}
        )

        # Save mappings
        entity_path = self.output_dir / "transe_entity_map.parquet"
        relation_path = self.output_dir / "transe_relation_map.parquet"

        self.file_manager.save(entity_map, entity_path)
        self.file_manager.save(relation_map, relation_path)

        logger.info("✅ Mapeamentos criados:")
        logger.info(f"   Entidades: {len(entity_map):,}")
        logger.info(f"   Relações: {len(relation_map):,}")

        self.stats["mappings"] = {
            "entities": len(entity_map),
            "relations": len(relation_map),
            "entity_path": str(entity_path),
            "relation_path": str(relation_path),
        }

        return entity_map, relation_map

    def _save_indexed_arrays(
        self,
        train: pl.DataFrame,
        valid: pl.DataFrame,
        test: pl.DataFrame,
        entity_map: pl.DataFrame,
        relation_map: pl.DataFrame,
    ) -> None:
        """Convert splits to indexed numpy arrays and save."""
        logger.info("\n🔄 Convertendo para arrays indexados")

        # Create lookup dictionaries
        entity_to_idx = dict(zip(entity_map["label"], entity_map["id"]))
        relation_to_idx = dict(zip(relation_map["label"], relation_map["id"]))

        # Convert each split
        for split_name, split_df in [
            ("train", train),
            ("valid", valid),
            ("test", test),
        ]:
            # Convert to indices
            indexed = []
            for row in split_df.iter_rows(named=True):
                indexed.append(
                    [
                        entity_to_idx[row["s"]],
                        relation_to_idx[row["p"]],
                        entity_to_idx[row["o"]],
                    ]
                )

            # Save as numpy array
            array = np.array(indexed, dtype=np.int64)
            array_path = self.output_dir / f"{split_name}_indexed.npy"
            np.save(array_path, array)

            logger.info(f"   {split_name}: {array.shape} salvo em {array_path}")

    def _log_final_statistics(self) -> None:
        """Log final optimization statistics."""
        logger.info("\n" + "=" * 60)
        logger.info("📊 ESTATÍSTICAS FINAIS DA OTIMIZAÇÃO")
        logger.info("=" * 60)

        original = self.stats.get("original_triples", 0)
        final = self.stats.get("splits", {})
        # Only sum the counts for train, valid, and test splits (exclude 'overlap' dict)
        total_final = sum(final[k] for k in ("train", "valid", "test") if k in final and isinstance(final[k], int)) if isinstance(final, dict) else 0

        logger.info("🎯 MELHORIAS:")
        logger.info(
            f"   - Redução de tamanho: {total_final / original * 100:.1f}% dos dados originais"
        )
        logger.info(
            f"   - Triplas finais: {total_final:,} (meta: {self.target_triples:,})"
        )

        if total_final >= self.target_triples:
            logger.success(
                f"✅ META ATINGIDA: {total_final:,} >= {self.target_triples:,}"
            )
        else:
            logger.warning(
                f"⚠️ Meta não atingida: {total_final:,} < {self.target_triples:,}"
            )

        logger.info(f"\n✅ Dados otimizados salvos em: {self.output_dir}")
        logger.success("✅ Otimização concluída com sucesso!")


async def main():
    """Example usage of TransE preprocessor."""
    from pff.validators.kg.config import KGConfig

    # Load configuration
    kg_config = KGConfig(settings.CONFIG_DIR / "kg.yaml")

    # Run preprocessing
    preprocessor = TransEPreprocessor(kg_config, target_triples=10000)
    stats = await preprocessor.run()

    # Display results
    logger.info("\n📊 Resultados do pré-processamento:")
    logger.info(f"   Total de triplas otimizadas: {sum(stats['splits'].values()):,}")
    logger.info(f"   Entidades únicas: {stats['mappings']['entities']:,}")
    logger.info(f"   Relações únicas: {stats['mappings']['relations']:,}")


if __name__ == "__main__":
    asyncio.run(main())
