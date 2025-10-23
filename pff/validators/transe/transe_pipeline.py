from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch

from pff import settings
from pff.utils import FileManager, logger, progress_bar
from pff.validators.kg import KGConfig, KGPipeline
from pff.validators.kg.builder import KGBuilder
from pff.validators.kg.pipeline import MetricsCalculator
from pff.validators.transe.core import TransEManager
from pff.validators.transe.lightgbm_trainer import TransELightGBMTrainer
from pff.validators.transe.mapping_utils import (
    convert_graph_to_indices,
)
from pff.validators.transe.transe_preprocessor import TransEPreprocessor
from pff.validators.transe.transe_service import TransEScorerService

"""
TransE Pipeline Implementation

This module provides the main pipeline for TransE knowledge graph completion,
including data preparation, model training, and evaluation.
"""


class TransEPipeline(KGPipeline):
    """
    TransE pipeline for knowledge graph completion.

    This pipeline extends KGPipeline to provide TransE-specific functionality
    while maintaining compatibility with the existing infrastructure.
    """

    def __init__(self, kg_config_path: str | Path):
        """
        Initialize TransE pipeline.

        Args:
            kg_config_path: Path to knowledge graph configuration file
        """
        self.config = KGConfig(kg_config_path)
        self.file_manager = FileManager()

        # Initialize builder
        builder_config = self.config.get_builder_config()
        self.builder = KGBuilder(
            source_path=builder_config.get("source_path", settings.DATA_DIR),
            output_dir=self.config.graph_directory,
        )

        # Initialize preprocessor
        self.preprocessor = TransEPreprocessor(self.config, target_triples=10000)

        # Components to be initialized later
        self.transe_manager: TransEManager | None = None
        self.scorer_service: TransEScorerService | None = None
        self._metrics_calculator: MetricsCalculator | None = None

        # Note: Using Dask instead of Ray for Windows compatibility
        # ConcurrencyManager will automatically handle this

        logger.info("✅ TransE Pipeline inicializado")

    @property
    def metrics_calculator(self) -> MetricsCalculator:
        """Lazy loading of MetricsCalculator."""
        if self._metrics_calculator is None:
            logger.info("Inicializando MetricsCalculator no TransEPipeline")
            self._metrics_calculator = MetricsCalculator(top_k=10)
        return self._metrics_calculator

    async def run_data_preparation(self) -> None:
        """
        Run complete data preparation pipeline.

        This includes:
        1. Building raw parquet files if needed
        2. Running optimization and preprocessing
        3. Creating entity/relation mappings
        4. Generating indexed arrays
        """
        logger.info("🚀 Iniciando preparação de dados para TransE...")

        # Check if raw parquets exist
        raw_files = ["train.parquet", "valid.parquet", "test.parquet"]
        missing_raw = [
            f for f in raw_files if not (self.config.graph_directory / f).exists()
        ]

        if missing_raw:
            logger.warning(f"Arquivos brutos faltando: {missing_raw}")
            logger.info("Executando KGBuilder...")
            await self.builder.run()
            logger.success("✅ Arquivos Parquet brutos construídos")

        # Run preprocessing to create optimized data and mappings
        await self.preprocessor.run()

        logger.success("✅ Preparação de dados concluída")

    async def train_transe(
        self, transe_config_path: Path | None = None, force_retrain: bool = False
    ) -> dict[str, Any]:
        """
        Train TransE model.

        Args:
            transe_config_path: Optional path to TransE config (uses default if None)
            force_retrain: Force retraining even if model exists

        Returns:
            Dictionary with training statistics
        """
        logger.info("🤖 Iniciando pipeline de treinamento TransE...")

        # Use default config if not provided
        if transe_config_path is None:
            transe_config_path = settings.CONFIG_DIR / "transe.yaml"

        # Check if model already exists
        checkpoint_path = Path("checkpoints") / "transe" / "best_model.pt"
        if checkpoint_path.exists() and not force_retrain:
            logger.info("✅ Modelo TransE já treinado encontrado")
            logger.info("Use force_retrain=True para retreinar")
            return {"status": "already_trained", "checkpoint": str(checkpoint_path)}

        # Ensure data is prepared
        await self.run_data_preparation()

        # Initialize TransE manager
        self.transe_manager = TransEManager(
            transe_config_path=transe_config_path,
            kg_config_path=self.config.configuration_path,
        )

        # Setup data
        self.transe_manager._setup_data()

        # Train model
        logger.info("🏋️ Treinando modelo TransE...")
        training_stats = self.transe_manager.train()

        # Log results
        if training_stats.get("status") == "cancelled":
            logger.warning("⚠️ Treinamento cancelado")
        else:
            best = training_stats.get("final_metrics", {})
            hits1 = best.get("hits@1", 0.0)
            hits10 = best.get("hits@10", 0.0)

            logger.success(
                f"✅ Treinamento concluído em {training_stats['training_time']:.1f}s"
            )
            logger.info(
                f"   Melhor época: {training_stats['best_epoch']}, "
                f"MRR: {training_stats['best_val_mrr']:.4f}, "
                f"Hits@1: {hits1:.4f}, "
                f"Hits@10: {hits10:.4f}"
            )

        return training_stats

    def rank_and_evaluate_transe(self) -> dict[str, Any]:
        """
        Evaluate TransE model and optionally train hybrid model.

        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("\n" + "=" * 80)
        logger.info("RANKING E AVALIAÇÃO DO MODELO TRANSE")
        logger.info("=" * 80)

        # Initialize scorer service
        transe_config_path = settings.CONFIG_DIR / "transe.yaml"
        self.scorer_service = TransEScorerService(
            self.config, transe_config_path, load_best_model=True
        )

        # Check if model is loaded
        if (
            self.scorer_service.transe_manager is None
            or self.scorer_service.transe_manager.model is None
        ):
            logger.error("❌ Modelo TransE não carregado")
            return {
                "transe_metrics": {"mrr": 0.0, "hits@1": 0.0, "hits@10": 0.0},
                "hybrid_metrics": None,
            }

        # Load test data
        logger.info("📂 Carregando conjunto de teste...")

        test_path = self.config.graph_directory / "test_optimized.parquet"
        if not test_path.exists():
            test_path = self.config.graph_directory / "test.parquet"

        if not test_path.exists():
            logger.error(f"❌ Arquivo de teste não encontrado: {test_path}")
            return {
                "transe_metrics": {"mrr": 0.0, "hits@1": 0.0, "hits@10": 0.0},
                "hybrid_metrics": None,
            }

        # Convert test data to indexed format
        test_array = convert_graph_to_indices(
            test_path,
            self.scorer_service.entity_to_idx,
            self.scorer_service.relation_to_idx,
            use_optimized=True,
        )

        logger.info(f"✅ Dados de teste carregados: {len(test_array):,} triplas")

        # Evaluate TransE
        logger.info("\n📊 Avaliando modelo TransE puro...")
        transe_metrics = self._evaluate_transe(test_array)

        logger.info("\n📊 Resultados da Avaliação TransE:")
        logger.info(f"   MRR: {transe_metrics['mrr']:.4f}")
        logger.info(f"   Hits@1: {transe_metrics['hits@1']:.4f}")
        logger.info(f"   Hits@10: {transe_metrics['hits@10']:.4f}")

        # Train and evaluate hybrid model
        logger.info("\n" + "=" * 80)
        logger.info("TREINAMENTO DO MODELO HÍBRIDO TRANSE + LightGBM")
        logger.info("=" * 80)

        hybrid_metrics = None
        try:
            hybrid_trainer = TransELightGBMTrainer(self.scorer_service.transe_manager)
            hybrid_metrics = hybrid_trainer.train_hybrid_model()

            logger.info("\n📊 Resultados do Modelo Híbrido:")
            if hybrid_metrics:
                for metric, value in hybrid_metrics.items():
                    logger.info(f"   {metric}: {value:.4f}")
        except Exception as e:
            logger.error(f"❌ Erro no treinamento híbrido: {e}")
            import traceback

            logger.debug(traceback.format_exc())

        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("RESUMO DOS RESULTADOS")
        logger.info("=" * 80)
        logger.info(f"TransE Puro - MRR: {transe_metrics['mrr']:.4f}")
        if hybrid_metrics:
            logger.info(f"Híbrido - AUC: {hybrid_metrics.get('auc', 0):.4f}")

        return {"transe_metrics": transe_metrics, "hybrid_metrics": hybrid_metrics}

    def _evaluate_transe(self, test_triples: np.ndarray) -> dict[str, float]:
        """
        Evaluate TransE model on test set.

        Args:
            test_triples: Test triples in indexed format

        Returns:
            Dictionary with evaluation metrics
        """
        mrr_sum = 0.0
        hits_at_1 = 0
        hits_at_10 = 0

        # Create filter for known triples (train + valid + test)
        logger.info("🔄 Criando filtro de triplas conhecidas...")

        known_triples = set()
        for split in ["train", "valid", "test"]:
            path = settings.OUTPUTS_DIR / "transe" / f"{split}_indexed.npy"
            if path.exists():
                data = np.load(path)
                for triple in data:
                    known_triples.add(tuple(triple))

        logger.info(f"   Total de triplas conhecidas: {len(known_triples):,}")

        # Evaluate each test triple
        num_evaluated = 0
        with torch.no_grad():
            for i in progress_bar(
                range(len(test_triples)),
                total=len(test_triples),
                desc="Avaliando...",
                enabled=True,
            ):
                head_idx, rel_idx, true_tail_idx = test_triples[i]

                # Get all candidate tail scores
                transe_manager = getattr(self.scorer_service, "transe_manager", None)
                if (
                    transe_manager is None
                    or getattr(transe_manager, "model", None) is None
                    or not hasattr(transe_manager.model, "num_entities")
                    or not hasattr(transe_manager, "device")
                ):
                    logger.error(
                        "❌ TransE manager/model não está inicializado corretamente."
                    )
                    continue

                all_tails = torch.arange(transe_manager.model.num_entities).to(
                    transe_manager.device
                )

                heads = torch.full_like(all_tails, head_idx)
                relations = torch.full_like(all_tails, rel_idx)

                scores = transe_manager.model(heads, relations, all_tails).cpu().numpy()

                # Filter out known triples (except the true one)
                for tail_idx in range(len(scores)):
                    if tail_idx != true_tail_idx:
                        triple = (head_idx, rel_idx, tail_idx)
                        if triple in known_triples:
                            scores[tail_idx] = -float("inf")

                # Rank entities
                sorted_indices = np.argsort(scores)[::-1]  # Descending order

                # Find rank of true tail
                rank = np.where(sorted_indices == true_tail_idx)[0]
                if len(rank) > 0:
                    rank = rank[0] + 1

                    # Update metrics
                    mrr_sum += 1.0 / rank
                    if rank == 1:
                        hits_at_1 += 1
                    if rank <= 10:
                        hits_at_10 += 1

                    num_evaluated += 1

        # Calculate final metrics
        if num_evaluated > 0:
            metrics = {
                "mrr": mrr_sum / num_evaluated,
                "hits@1": hits_at_1 / num_evaluated,
                "hits@10": hits_at_10 / num_evaluated,
            }
        else:
            metrics = {"mrr": 0.0, "hits@1": 0.0, "hits@10": 0.0}

        logger.info(
            f"✅ Avaliação concluída: {num_evaluated}/{len(test_triples)} triplas"
        )

        return metrics

    async def run_complete_pipeline(self) -> dict[str, Any]:
        """
        Run the complete TransE pipeline.

        This includes:
        1. Data preparation
        2. Model training
        3. Evaluation

        Returns:
            Dictionary with complete pipeline results
        """
        logger.info("🚀 Executando pipeline completo do TransE")

        results = {}

        # Step 1: Data preparation
        await self.run_data_preparation()
        results["data_preparation"] = "completed"

        # Step 2: Training
        training_stats = await self.train_transe()
        results["training"] = training_stats

        # Step 3: Evaluation
        eval_results = self.rank_and_evaluate_transe()
        results["evaluation"] = eval_results

        logger.success("✅ Pipeline TransE concluído com sucesso!")

        return results

    # Compatibility methods for backward compatibility

    async def train_hgt(self, **kwargs) -> dict[str, Any]:
        """Compatibility method - redirects to train_transe."""
        logger.warning("⚠️ train_hgt() deprecado, usando train_transe()")
        return await self.train_transe(**kwargs)

    def rank_and_evaluate_hgt(self) -> dict[str, Any]:
        """Compatibility method - redirects to rank_and_evaluate_transe."""
        logger.warning(
            "⚠️ rank_and_evaluate_hgt() deprecado, usando rank_and_evaluate_transe()"
        )
        return self.rank_and_evaluate_transe()

    async def _ensure_raw_triples_exist(self) -> None:
        """Ensure raw triple files exist for compatibility."""
        logger.info("🔍 Verificando arquivos raw...")

        raw_dir = self.config.graph_directory
        raw_files = ["train_raw.txt", "valid_raw.txt", "test_raw.txt"]

        missing = [f for f in raw_files if not (raw_dir / f).exists()]

        if missing:
            logger.info(f"Gerando arquivos raw faltantes: {missing}")

            # Convert parquet to raw format
            for split in ["train", "valid", "test"]:
                parquet_path = raw_dir / f"{split}.parquet"
                raw_path = raw_dir / f"{split}_raw.txt"

                if parquet_path.exists() and not raw_path.exists():
                    df = pl.read_parquet(parquet_path)

                    # Write as TSV
                    with open(raw_path, "w", encoding="utf-8") as f:
                        for row in df.iter_rows():
                            f.write(f"{row[0]}\t{row[1]}\t{row[2]}\n")

                    logger.info(f"   ✅ {raw_path.name} criado")

    async def _ensure_transe_artifacts_exist(self) -> None:
        """Ensure TransE-specific artifacts exist."""
        logger.info("🔍 Verificando artefatos TransE...")

        maps_dir = settings.OUTPUTS_DIR / "transe"
        required_files = [
            "transe_entity_map.parquet",
            "transe_relation_map.parquet",
            "train_indexed.npy",
            "valid_indexed.npy",
        ]

        missing = [f for f in required_files if not (maps_dir / f).exists()]

        if missing:
            logger.warning(f"Artefatos TransE faltantes: {missing}")
            logger.info("Executando pré-processamento...")
            await self.preprocessor.run()
