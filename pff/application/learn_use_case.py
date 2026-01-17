"""Learning use case for DSLFM-KGC workflows."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pff.domain.ports.persistence.kg_ports import PipelineCheckpointsPort, KGSplitsPort

from pff.shared.core.config import KG_PIPELINE_CONFIG_PATH
from pff.shared import logger
from pff.shared.ops.global_interrupt_manager import check_interruption
from pff.application.errors import PreprocessedDataMissingError, StrategyResolutionError
from pff.application.strategy_registry import get_strategy_registry
from pff.domain.kg.factory import KGComponentFactory


class TrainingStrategy(ABC):
    """
    Abstract base class for training strategies.

    Pattern: Strategy Pattern + Template Method
    """

    def __init__(
        self,
        config_path: Path,
        checkpoints_repo: PipelineCheckpointsPort | None = None,
        splits_repo: KGSplitsPort | None = None,
    ):
        self.config_path = config_path
        self.checkpoints_repo = checkpoints_repo
        self.splits_repo = splits_repo

    async def execute(self) -> None:
        """Template method for executing a training strategy."""
        self.check_interruption()
        await self._execute()
        self.check_interruption()

    @abstractmethod
    async def _execute(self) -> None:
        """Strategy-specific execution logic."""

    def check_interruption(self) -> None:
        """Check if user requested interruption."""
        check_interruption()


@get_strategy_registry().register("kg")
class KGTrainingStrategy(TrainingStrategy):
    """Strategy for Knowledge Graph training."""

    async def _execute(self) -> None:
        """Train KG model."""
        from pff.domain.kg.config import KGConfig  # noqa: PLC0415

        logger.info("Executando pipeline do Knowledge Graph (KG)...")

        self.check_interruption()
        kg_pipeline = KGComponentFactory().create_pipeline(
            KGConfig(self.config_path),
            checkpoints_repo=self.checkpoints_repo,
            splits_repo=self.splits_repo,
        )

        await kg_pipeline.run_build_and_preprocess()
        self.check_interruption()

        logger.info(" Extração de regras externas desabilitada (modo DSLFM+PC).")
        logger.success(" Pipeline do KG concluída (somente preprocess).")


@get_strategy_registry().register("kgc")
class KGCTrainingStrategy(TrainingStrategy):
    """Strategy for DSLFM-KGC training with BERT encoder for relations."""

    async def _execute(self) -> None:
        """Train DSLFM-KGC model with BERT relation embeddings."""
        import numpy as np  # noqa: PLC0415
        import polars as pl  # noqa: PLC0415

        from pff.domain.learning.dslfm.kgc_manager import (
            train_dslfm_kgc,
        )  # noqa: PLC0415
        from pff.domain.kg.config import KGConfig  # noqa: PLC0415
        from pff.shared.core.file_manager import FileManager  # noqa: PLC0415

        logger.info("Executando pipeline DSLFM-KGC com BERT encoder...")

        self.check_interruption()

        # Load entity and relation maps
        from pff.shared.core.config import settings  # noqa: PLC0415

        kg_output = settings.OUTPUTS_DIR / "kg"
        entity_map_path = kg_output / "mappings" / "entity_map.parquet"
        relation_map_path = kg_output / "mappings" / "relation_map.parquet"
        train_path = kg_output / "train.parquet"
        valid_path = kg_output / "valid.parquet"

        kg_config = KGConfig(self.config_path)

        # Garantir que os splits preprocessados estejam no PostgreSQL e materializados
        await self._ensure_preprocessed_data(
            kg_config,
            train_path,
            valid_path,
            entity_map_path,
            relation_map_path,
        )

        entity_bundle = FileManager.read(entity_map_path)
        relation_bundle = FileManager.read(relation_map_path)
        train_bundle = FileManager.read(train_path, streaming=True)
        valid_bundle = FileManager.read(valid_path, streaming=True)

        entity_map = entity_bundle.lazyframe().collect(engine="streaming")
        relation_map = relation_bundle.lazyframe().collect(engine="streaming")
        train_df = train_bundle.lazyframe().collect(engine="streaming")
        valid_df = valid_bundle.lazyframe().collect(engine="streaming")

        # Create lookup dicts efficiently
        entity_to_id = dict(zip(entity_map["label"], entity_map["id"]))
        relation_to_id = dict(zip(relation_map["label"], relation_map["id"]))
        relation_names = list(relation_map["label"])

        # Convert to numpy arrays
        def convert_df(df: pl.DataFrame) -> np.ndarray:
            mapped = df.select(
                pl.col("s").replace_strict(entity_to_id, default=0),
                pl.col("p").replace_strict(relation_to_id, default=0),
                pl.col("o").replace_strict(entity_to_id, default=0),
            )
            return np.asarray(mapped.to_numpy(), dtype=np.int64)

        self.check_interruption()

        train_triples = convert_df(train_df)
        valid_triples = convert_df(valid_df)

        num_entities = len(entity_map)
        num_relations = len(relation_map)

        logger.info(
            f"Dataset carregado: {len(train_triples):,} train, "
            f"{len(valid_triples):,} valid, {num_entities:,} entidades, "
            f"{num_relations} relações"
        )

        self.check_interruption()

        # Train with BERT
        stats = train_dslfm_kgc(
            train_triples=train_triples,
            valid_triples=valid_triples,
            num_entities=num_entities,
            num_relations=num_relations,
            relation_names=relation_names,
        )

        logger.success(
            f"DSLFM-KGC concluído: MRR={stats['best_val_mrr']:.4f} (epoch {stats['best_epoch']})"
        )

    async def _ensure_preprocessed_data(
        self,
        kg_config,
        train_path: Path,
        valid_path: Path,
        entity_map_path: Path,
        relation_map_path: Path,
    ) -> None:
        """Ensure preprocessed splits are in PostgreSQL and materialized locally."""
        from pff.domain.kg.preprocessing import (
            PreprocessingConfig,
            filter_attribute_relations,
        )  # noqa: PLC0415
        from pff.shared.core.file_manager import FileManager  # noqa: PLC0415

        if self.splits_repo is None:
            logger.info("splits_repo not available. Executando preprocess completo...")
            kg_pipeline = KGComponentFactory().create_pipeline(kg_config)
            await kg_pipeline.run_build_and_preprocess()
            return

        preprocessing_config = PreprocessingConfig.from_yaml()

        # If preprocessed missing in PostgreSQL, run preprocessing end-to-end
        preprocessed_exists = False
        try:
            preprocessed_exists = await self.splits_repo.preprocessed_exists()
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"PostgreSQL preprocessed check failed: {exc}")

        if not preprocessed_exists:
            logger.info(
                "Splits preprocessados nao encontrados no PostgreSQL. Executando preprocess..."
            )
            kg_pipeline = KGComponentFactory().create_pipeline(
                kg_config,
                checkpoints_repo=self.checkpoints_repo,
                splits_repo=self.splits_repo,
            )
            await kg_pipeline.run_build_and_preprocess()
        else:
            logger.info(
                "Splits preprocessados encontrados no PostgreSQL. Materializando para parquet..."
            )
            try:
                train_df, valid_df, test_df, _ = (
                    await self.splits_repo.load_preprocessed_splits(
                        fallback_to_raw=False
                    )
                )
                if train_df is None or valid_df is None:
                    raise RuntimeError("Preprocessed splits incompletos no PostgreSQL")
                train_df, valid_df, test_df, _ = filter_attribute_relations(
                    train_df, valid_df, test_df, preprocessing_config
                )
                FileManager.save(train_df, train_path)
                FileManager.save(valid_df, valid_path)
                logger.info(
                    f"Parquets materializados: train={len(train_df):,}, valid={len(valid_df):,}"
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Failed to materialize preprocessed splits: {exc}")
                logger.info("Executando preprocess para recomputar splits...")
                kg_pipeline = KGComponentFactory().create_pipeline(
                    kg_config,
                    checkpoints_repo=self.checkpoints_repo,
                    splits_repo=self.splits_repo,
                )
                await kg_pipeline.run_build_and_preprocess()

        # Validate required parquet files
        if not all(
            FileManager.exists(p)
            for p in [entity_map_path, relation_map_path, train_path, valid_path]
        ):
            logger.error("Preprocess failed - KG data still not found.")
            raise PreprocessedDataMissingError(
                "Preprocess failed - KG data still not found."
            )


@get_strategy_registry().register("all")
@get_strategy_registry().register("both")
@get_strategy_registry().register("")
class FullPipelineStrategy(TrainingStrategy):
    """
    Strategy for full pipeline (KG preprocess + DSLFM + PC).
    """

    async def _execute(self) -> None:
        """Execute full training pipeline with DSLFM-KGC + PC."""
        from pff.domain.kg.config import KGConfig  # noqa: PLC0415

        logger.info("Executando pipeline completa (KG preprocess + DSLFM-KGC + PC)")

        # Step 1/2: KG Pipeline
        logger.info("1/2: Executando pipeline do Knowledge Graph (preprocess)...")
        self.check_interruption()

        kg_pipeline = KGComponentFactory().create_pipeline(
            KGConfig(self.config_path),
            checkpoints_repo=self.checkpoints_repo,
            splits_repo=self.splits_repo,
        )
        await kg_pipeline.run_build_and_preprocess()
        self.check_interruption()

        logger.info(" Extração de regras externas desabilitada (modo DSLFM-KGC+PC).")

        # Step 2/2: DSLFM-KGC Pipeline (delegating to KGCTrainingStrategy)
        logger.info("2/2: Executando pipeline DSLFM-KGC + PC...")
        kgc_strategy = KGCTrainingStrategy(
            self.config_path,
            checkpoints_repo=self.checkpoints_repo,
            splits_repo=self.splits_repo,
        )
        await kgc_strategy.execute()
        self.check_interruption()

        logger.success(" Pipeline completa DSLFM-KGC+PC concluída.")


class LearnUseCase:
    """Coordinate the learn pipeline without exposing driver details."""

    def __init__(
        self,
        config_path: Path | None = None,
        strategy_registry=None,
        checkpoints_repo: PipelineCheckpointsPort | None = None,
        splits_repo: KGSplitsPort | None = None,
    ) -> None:
        """Initialize the use case.

        Args:
            config_path: Optional config path override.
            strategy_registry: Optional registry for strategies.
            checkpoints_repo: Optional repository for checkpoints (injected).
            splits_repo: Optional repository for splits (injected).
        """
        self._config_path = config_path or KG_PIPELINE_CONFIG_PATH
        self._strategy_registry = strategy_registry or get_strategy_registry()
        self._checkpoints_repo = checkpoints_repo
        self._splits_repo = splits_repo

    async def execute(self, model: str) -> None:
        """Execute training for a given model type.

        Args:
            model: Training strategy identifier.
        """
        strategy = self._resolve_strategy(model)
        await strategy.execute()

    def _resolve_strategy(self, model: str) -> TrainingStrategy:
        """Resolve training strategy for the requested model.

        Args:
            model: Training strategy identifier.

        Returns:
            Concrete TrainingStrategy implementation.
        """
        try:
            return self._strategy_registry.create(
                model.lower(),
                self._config_path,
                checkpoints_repo=self._checkpoints_repo,
                splits_repo=self._splits_repo,
            )
        except StrategyResolutionError as exc:
            logger.error(str(exc))
            raise
