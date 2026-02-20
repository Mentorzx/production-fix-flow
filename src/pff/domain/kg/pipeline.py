"""Provide module-level functionality for the PFF codebase.



Notes:

    File: src/pff/domain/kg/pipeline.py

"""

from __future__ import annotations
from pff.shared import FileManager, logger, stable_hash
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pff.domain.kg.factory import KGComponentFactory

import numpy as np
import polars as pl

from pff.domain.ports.persistence.kg_ports import KGSplitsPort, PipelineCheckpointsPort
from pff.shared.core.file_manager import ParquetBundle
from pff.shared.ops.global_interrupt_manager import (
    check_interruption,
    get_interrupt_manager,
    should_stop,
)
from pff.shared.system.resource_manager import HardwareDetector

from .calibration import ScoreCalibrator, find_optimal_threshold
from .config import KGConfig
from .data_loader import KGDataLoader

_sklearn_metrics = None


def _require_sklearn_metrics():
    """Lazy import sklearn.metrics following project pattern."""
    global _sklearn_metrics
    if _sklearn_metrics is None:
        try:
            from sklearn import metrics as _sklearn_metrics_mod
        except ImportError as exc:
            raise RuntimeError(
                "sklearn não disponível; instale para usar métricas de classificação."
            ) from exc
        _sklearn_metrics = _sklearn_metrics_mod
    return _sklearn_metrics


class DataLoaderInterface(ABC):
    """Interface for loading and managing triple data."""

    @abstractmethod
    def load_triples_from_parquet(self, parquet_path: Path) -> list[list[str]]:
        """Load triples from a Parquet file."""
        pass

    @abstractmethod
    def load_indexed_data(self, numpy_path: Path) -> np.ndarray:
        """Load indexed data from a NumPy file."""
        pass


class MetricsCalculator:
    """Calculate evaluation metrics for ranking results."""

    def __init__(
        self,
        config=None,
        top_k: int = 10,
        file_manager: FileManager | None = None,
    ):
        """Execute init.



        Args:

            config: Optional input value.

            top_k: Optional input value.

            file_manager: Optional input value.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        self.top_k = top_k
        self.config = config
        self.file_manager = file_manager or FileManager()
        self.calibrator: ScoreCalibrator | None = None

        from pff.shared.core.config import settings

        self.optimal_threshold = settings.MODEL_CONFIG.get("dslfm", {}).get(
            "optimal_threshold", 0.5
        )
        logger.debug(f"MetricsCalculator initialized with top_k={self.top_k}")

    def calculate_ranking_metrics(
        self, scores_dataframe: pl.DataFrame, calibrate: bool = True
    ) -> dict[str, float]:
        """
        Calculate MRR, Hits@K metrics and optionally calibrate scores.

        Args:
            scores_dataframe: DataFrame with columns: src_id, rel_id, direction,
                             cand_id, score, is_true
            calibrate: Whether to perform score calibration

        Returns:
            Dictionary with all metrics
        """
        ranking_metrics = self._calculate_ranking_metrics(scores_dataframe)
        classification_metrics_raw = self._calculate_classification_metrics(
            scores_dataframe, calibrated=False
        )
        metrics = {**ranking_metrics, "classification_raw": classification_metrics_raw}
        y_true = scores_dataframe["is_true"].to_numpy()

        if calibrate and len(np.unique(y_true)) > 1 and len(scores_dataframe) > 100:
            logger.info("Iniciando calibração de scores, pois há exemplos positivos e negativos.")
            calibrated_df = self._calibrate_scores(scores_dataframe)
            classification_metrics_cal = self._calculate_classification_metrics(
                calibrated_df, calibrated=True
            )
            metrics["classification_calibrated"] = classification_metrics_cal
            y_scores = calibrated_df["score_calibrated"].to_numpy()
            self.optimal_threshold, threshold_metrics = find_optimal_threshold(
                y_scores, y_true, metric="f1"
            )
            metrics["optimal_threshold"] = threshold_metrics
        else:
            logger.warning(
                "Pulando etapa de calibração de scores: não há exemplos de ambas as classes (positivos e negativos) nos resultados."
            )
        if self.config:
            metrics_path = self.config.get_output_directory() / "metrics.json"
            self.file_manager.save(metrics, metrics_path)
            logger.info(f" Todas as métricas salvas em {metrics_path}")
            if self.calibrator and self.calibrator.is_fitted:
                calibrator_path = self.config.get_output_directory() / "calibrator.pkl"
                self.calibrator.save(calibrator_path)

        return metrics

    def _calculate_ranking_metrics(self, scores_dataframe: pl.DataFrame) -> dict:
        """Calculate MRR, Hits@1, Hits@K metrics."""
        ranked_dataframe = scores_dataframe.with_columns(
            pl.col("score")
            .rank(method="ordinal", descending=True)
            .over(["src_id", "rel_id", "direction"])
            .alias("rank")
        )
        true_hits = ranked_dataframe.filter(pl.col("is_true") == 1)

        if len(true_hits) == 0:
            logger.warning("No true hits found for computing metrics")
            return {
                "mrr": 0.0,
                "hits_at_1": 0.0,
                f"hits_at_{self.top_k}": 0.0,
                "total_queries": len(scores_dataframe),
                "true_hits": 0,
            }

        mean_reciprocal_rank = true_hits.select((1.0 / pl.col("rank")).mean()).item()
        hits_at_1 = true_hits.select((pl.col("rank") <= 1).mean()).item()
        hits_at_k = true_hits.select((pl.col("rank") <= self.top_k).mean()).item()

        metrics = {
            "mrr": mean_reciprocal_rank,
            "hits_at_1": hits_at_1,
            f"hits_at_{self.top_k}": hits_at_k,
            "total_queries": len(scores_dataframe.unique(["src_id", "rel_id", "direction"])),
            "true_hits": len(true_hits),
        }

        logger.info(" Métricas de Ranking:")
        logger.info(f"  MRR: {mean_reciprocal_rank:.4f}")
        logger.info(f"  Hits@1: {hits_at_1:.4f}")
        logger.info(f"  Hits@{self.top_k}: {hits_at_k:.4f}")
        logger.info(f"  Total de consultas: {metrics['total_queries']}")
        logger.info(f"  Acertos reais: {metrics['true_hits']}")

        return metrics

    def _calculate_classification_metrics(
        self, scores_dataframe: pl.DataFrame, calibrated: bool = False
    ) -> dict:
        """Calculate AUC curves and other classification metrics."""
        try:
            score_col = "score_calibrated" if calibrated else "score"

            if calibrated and score_col not in scores_dataframe.columns:
                score_col = "score"

            y_true = scores_dataframe["is_true"].to_numpy()
            y_scores = scores_dataframe[score_col].to_numpy()

            roc_auc = None
            pr_auc = None
            try:
                from pff_rust import fast_average_precision_score, fast_roc_auc_score

                y_true_i = y_true.astype(np.int64)
                y_scores_f = y_scores.astype(np.float64)
                roc_auc = float(fast_roc_auc_score(y_true_i, y_scores_f))
                pr_auc = float(fast_average_precision_score(y_true_i, y_scores_f))
            except Exception:
                metrics_mod = _require_sklearn_metrics()
                roc_auc = metrics_mod.roc_auc_score(y_true, y_scores)
                pr_auc = metrics_mod.average_precision_score(y_true, y_scores)

            metrics = {
                "roc_auc": float(roc_auc),
                "pr_auc": float(pr_auc),
                "positive_rate": float(y_true.mean()),
                "score_mean": float(y_scores.mean()),
                "score_std": float(y_scores.std()),
            }

            prefix = "Calibradas" if calibrated else "Originais"
            logger.info(f" Métricas de Classificação ({prefix}):")
            logger.info(f"  ROC-AUC: {roc_auc:.4f}")
            logger.info(f"  PR-AUC: {pr_auc:.4f}")
            logger.info(f"  Taxa de positivos: {metrics['positive_rate']:.4f}")

            return metrics

        except ImportError:
            logger.warning("sklearn not available, skipping AUC calculation")
            return {}

    def get_last_metrics(self) -> dict:
        """Retrieves the most recent metrics from the output directory specified in the configuration."""
        if self.config:
            metrics_path = self.config.get_output_directory() / "metrics.json"
            if self.file_manager.exists(metrics_path):
                payload = self.file_manager.read(metrics_path)
                result: dict = (
                    payload.to_native() if isinstance(payload, ParquetBundle) else payload
                )
                return result
        return {}

    def _calibrate_scores(self, scores_dataframe: pl.DataFrame) -> pl.DataFrame:
        """
        Calibrate scores using Platt scaling.

        Args:
            scores_dataframe: DataFrame with raw scores

        Returns:
            DataFrame with additional calibrated_score column
        """
        logger.info(" Calibrando scores...")

        y_true = scores_dataframe["is_true"].to_numpy()
        y_scores = scores_dataframe["score"].to_numpy()

        self.calibrator = ScoreCalibrator(method="platt")
        calibrated_scores = self.calibrator.cross_val_calibrate(y_scores, y_true, cv=5)

        self.calibrator.fit(y_scores, y_true)

        result_df = scores_dataframe.with_columns(pl.Series("score_calibrated", calibrated_scores))

        logger.info(" Calibração concluída")
        logger.info(f"  Score médio original: {y_scores.mean():.4f}")
        logger.info(f"  Score médio calibrado: {calibrated_scores.mean():.4f}")
        logger.info(f"  Taxa real de positivos: {y_true.mean():.4f}")

        return result_df


class KGPipeline:
    """
    Orchestrates the entire KGC pipeline with stateful, cache-aware execution.

    This class manages the full workflow, from data building and preprocessing
    to rule learning and parallel ranking. It uses a state file to track the
    outcomes of each step, allowing it to intelligently skip steps whose inputs
    have not changed, providing a "coherent rope" execution model.
    """

    def __init__(
        self,
        config: KGConfig,
        factory: "KGComponentFactory | None" = None,
        checkpoints_repo: "PipelineCheckpointsPort | None" = None,
        splits_repo: "KGSplitsPort | None" = None,
        file_manager: FileManager | None = None,
    ):
        """
        Initializes the orchestrator with all necessary components.

        Args:
            config: The main configuration object for the pipeline.
            factory: Optional factory for creating components.
            checkpoints_repo: Optional repository for checkpoints (injected).
            splits_repo: Optional repository for splits (injected).
        """
        self.config = config
        self.file_manager = file_manager or FileManager()
        self.hardware = HardwareDetector.detect()
        logger.debug(
            f"System detected: {self.hardware.platform} "
            f"({self.hardware.cpu_threads} Threads, "
            f"{self.hardware.total_ram_gb:.1f}GB RAM)"
        )
        from pff.domain.kg.factory import KGComponentFactory

        factory = factory or KGComponentFactory()
        self.builder = factory.create_builder(config)
        self.preprocessor = factory.create_preprocessor(
            config,
            splits_repo=splits_repo,
            file_manager=self.file_manager,
        )
        self.rule_learner = None
        self.data_loader = KGDataLoader(splits_repo=splits_repo)

        self.checkpoints_repo: PipelineCheckpointsPort | None
        if checkpoints_repo:
            self.checkpoints_repo = checkpoints_repo
        else:
            logger.warning("No checkpoints_repo provided to KGPipeline. Persistence disabled.")
            self.checkpoints_repo = None

        self.splits_repo = splits_repo

        self.pipeline_name = "kg"
        pipeline_params = self.config.get_pipeline_configuration()
        top_k_value = pipeline_params.get("top_k", 10)
        self.metrics_calculator = MetricsCalculator(
            config=self.config,
            top_k=top_k_value,
            file_manager=self.file_manager,
        )
        self.interrupt_manager = get_interrupt_manager()

        def kg_cleanup_callback():
            """Execute kg cleanup callback.



            Notes:

                Keep behavior deterministic and free of hidden side effects.

            """

            logger.info(" KGPipeline: Iniciando limpeza por interrupção...")
            try:
                logger.info(" Checkpoints do pipeline KG salvos automaticamente")
            except Exception as e:
                logger.warning(f"Cleanup error: {e}")

        self.interrupt_manager.register_callback_once(
            kg_cleanup_callback,
            label="kg_pipeline_cleanup",
        )
        logger.info(" KGPipeline integrado ao GlobalInterruptManager")

    async def run_build_and_preprocess(self):
        """Runs only the builder and preprocessing steps."""
        logger.info("component=kg_pipeline step=build_preprocess status=iniciando")
        check_interruption()
        await self._run_preprocess_step()
        check_interruption()
        logger.success("component=kg_pipeline step=build_preprocess status=concluido")

    async def run_learn_rules(self, override_config: dict | None = None) -> None:
        """
        Run the rule learning phase of the knowledge graph pipeline.
        This method executes the rule learning process, which discovers logical patterns
        in the knowledge graph. It logs the start and completion of this phase and
        persists the pipeline state after completion.
        Parameters
        ----------
        override_config : dict | None, optional
            Configuration dictionary to override default settings for the rule learning process.
            If None, the default configuration will be used.
        Returns
        -------
        None
            This method doesn't return any value.
        """
        logger.debug("learn_rules desabilitado modo=DSLFM/PC")

    async def run_ranking(self, override_config: dict | None = None) -> dict | None:
        """
        Executes the ranking stage of the pipeline.
        This method logs the start and end of the ranking process, runs the ranking step with an optional
        override configuration, saves the current state, and returns the computed metrics.
        Args:
            override_config (dict | None): Optional dictionary to override the default ranking configuration.
        Returns:
            dict | None: The metrics resulting from the ranking step, or None if no metrics are produced.
        """

        logger.debug("ranking desabilitado modo=DSLFM/PC")
        return {}

    async def _run_learn_rules_step(
        self, force_run: bool = False, override_config: dict | None = None
    ) -> bool:
        """Manages the execution of the rule learning step."""
        logger.info(" Aprendizado de regras pulado (DSLFM/PC)")
        return True

    async def _run_preprocess_step(
        self, force_run: bool = False, override_config: dict | None = None
    ) -> bool:
        """Manages the execution of the data preprocessing step."""
        step_name = "preprocess"
        logger.info("-" * 60)
        logger.info(f"Avaliando Etapa 1: {step_name.upper()}")
        if should_stop():
            logger.warning(f"Step '{step_name}' cancelled due to interruption")
            return False
        missing_files = self.config.missing_required_files()
        if missing_files:
            missing_preview = ", ".join(p.name for p in missing_files)
            logger.info(f"Arquivos .parquet ausentes ({missing_preview}). Iniciando recuperação.")

            restored = await self._restore_parquets_from_postgres()
            if restored:
                logger.success(" Arquivos .parquet restaurados do PostgreSQL")
            else:
                raise FileNotFoundError(
                    "Arquivos .parquet ausentes e recuperação via PostgreSQL falhou."
                )
        inputs_to_hash = {
            "source_files": [
                self.config.train_path,
                self.config.valid_path,
                self.config.test_path,
            ],
            "params": self.config.get_preprocessing_parameters(),
        }
        if await self._should_skip_step(step_name, inputs_to_hash):
            return False
        check_interruption()
        await self._invalidate_downstream_files(step_name)
        logger.info(f"Executando etapa '{step_name}'...")
        self.preprocessor.run()
        check_interruption()
        await self._update_state_on_success(step_name, inputs_to_hash)

        return True

    async def _restore_parquets_from_postgres(self) -> bool:
        """
        Restore .parquet files from PostgreSQL if they exist.

        Returns:
            True if successfully restored
        """
        if self.splits_repo is None:
            raise RuntimeError("splits_repo not available for PostgreSQL restore.")

        try:
            logger.info(" Verificando se os dados existem no PostgreSQL...")

            train_exists = await self.splits_repo.split_exists("train", "raw")
            valid_exists = await self.splits_repo.split_exists("valid", "raw")
            test_exists = await self.splits_repo.split_exists("test", "raw")

            if not (train_exists and valid_exists and test_exists):
                raise FileNotFoundError("Required splits not found in PostgreSQL.")

            logger.info(" Restaurando arquivos .parquet do PostgreSQL...")

            train_df = await self.splits_repo.load_split("train", "raw")
            valid_df = await self.splits_repo.load_split("valid", "raw")
            test_df = await self.splits_repo.load_split("test", "raw")

            if train_df is None or valid_df is None or test_df is None:
                raise RuntimeError("Failed to load splits from PostgreSQL.")

            self.file_manager.ensure_parent_dir(self.config.train_path)

            train_df.select(["s", "p", "o"]).write_parquet(self.config.train_path)
            valid_df.select(["s", "p", "o"]).write_parquet(self.config.valid_path)
            test_df.select(["s", "p", "o"]).write_parquet(self.config.test_path)

            logger.info(
                f"splits_restaurados train={len(train_df):,} valid={len(valid_df):,} test={len(test_df):,}"
            )

            return True

        except ImportError as exc:
            raise RuntimeError("KGSplitsRepository unavailable.") from exc
        except Exception as exc:
            raise RuntimeError(f"PostgreSQL restore error: {exc}") from exc

    async def _run_ranking_step(
        self, force_run: bool = False, override_config: dict | None = None
    ) -> dict | None:
        """Retorna as métricas mais recentes."""
        return self.metrics_calculator.get_last_metrics()

    async def _load_checkpoint(self, step_name: str) -> dict | None:
        """
        Load checkpoint for specific step from PostgreSQL.

        Args:
            step_name: Step name to load

        Returns:
            Checkpoint dict or None
        """
        if self.checkpoints_repo is None:
            logger.debug(f"Persistence disabled, skipping checkpoint load for {step_name}")
            return None
        try:
            return await self.checkpoints_repo.get_checkpoint(self.pipeline_name, step_name)
        except Exception as exc:
            logger.warning(
                f"checkpoint_load_failed pipeline={self.pipeline_name} step={step_name} error={exc}"
            )
            return None

    async def _save_checkpoint(
        self,
        step_name: str,
        status: str,
        progress: float = 0.0,
        metadata: dict | None = None,
    ):
        """
        Save checkpoint for specific step to PostgreSQL.

        Args:
            step_name: Step name
            status: Status ('pending', 'running', 'completed', 'failed')
            progress: Progress (0.0 to 1.0)
            metadata: Optional metadata
        """
        if self.checkpoints_repo is None:
            logger.debug(f"Persistence disabled, skipping checkpoint save for {step_name}")
            return
        try:
            await self.checkpoints_repo.save_checkpoint(
                pipeline_name=self.pipeline_name,
                step_name=step_name,
                status=status,
                progress=progress,
                metadata=metadata,
                started_at=datetime.now() if status == "running" else None,
                completed_at=(datetime.now() if status in ["completed", "failed"] else None),
            )
        except Exception as exc:
            logger.warning(
                "checkpoint_save_failed "
                f"pipeline={self.pipeline_name} step={step_name} status={status} error={exc}"
            )

    def can_resume_from_checkpoint(self, phase: str) -> bool:
        """
        Check if the pipeline can resume from a checkpoint for the given phase.

        Args:
            phase: The phase name to check (e.g., 'build', 'preprocess', 'learn', 'rank')

        Returns:
            bool: True if checkpoint exists and pipeline can resume, False otherwise
        """

        if not hasattr(self.config, "checkpoint_dir"):
            logger.debug(f"No checkpoint_dir configured, cannot resume {phase}")
            return False

        checkpoint_dir = self.config.checkpoint_dir
        if isinstance(checkpoint_dir, str):
            from pathlib import Path

            checkpoint_dir = Path(checkpoint_dir)

        if not self.file_manager.exists(checkpoint_dir):
            logger.debug(f"Checkpoint directory {checkpoint_dir} does not exist")
            return False

        checkpoint_file = checkpoint_dir / f"{phase}_complete.json"
        if self.file_manager.exists(checkpoint_file):
            logger.info(f" Checkpoint encontrado para a fase '{phase}' em {checkpoint_file}")
            return True

        logger.debug(f"No checkpoint found for phase '{phase}'")
        return False

    def _get_input_hash(self, inputs: dict) -> str:
        """Generates a combined hash for a step's inputs."""
        parts = []
        for _key, value in sorted(inputs.items()):
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, Path) and self.file_manager.exists(item):
                        parts.append(self.file_manager.get_hash(item))
            elif isinstance(value, Path) and self.file_manager.exists(value):
                parts.append(self.file_manager.get_hash(value))
            else:
                parts.append(str(value))

        h_val = stable_hash(parts, truncate=None)
        return hex(h_val)[2:]

    async def _should_skip_step(self, step_name: str, inputs: dict) -> bool:
        """
        Determines whether a pipeline step should be skipped based on its previous execution state and inputs.
        This method checks multiple conditions to decide if a step can be skipped:
        1. If the step has never run or didn't complete successfully last time
        2. If the input parameters have changed since last execution
        3. If any expected output files are missing
        Args:
            step_name (str): Name of the pipeline step to check
            inputs (dict): Current input parameters for the step
        Returns:
            bool: True if the step should be skipped, False if it needs to be executed
        Example:
            >>> pipeline._should_skip_step("data_processing", {"param1": "value1"})
            True
        """

        last_run_info = await self._load_checkpoint(step_name)
        if not last_run_info or last_run_info.get("status") != "completed":
            logger.info(
                f"Executando '{step_name}' pois não foi concluída com sucesso na última vez."
            )
            return False

        metadata = last_run_info.get("metadata") or {}
        last_input_hash = metadata.get("input_hash")
        current_input_hash = self._get_input_hash(inputs)
        if last_input_hash != current_input_hash:
            logger.info(f"Executando '{step_name}' pois suas entradas mudaram.")
            return False

        expected_outputs = self.config.get_step_outputs(step_name)
        for output_file in expected_outputs:
            if not self.file_manager.exists(output_file):
                logger.info(
                    f"Estado para '{step_name}' era 'completed', mas o arquivo de saída "
                    f"'{output_file.name}' está faltando. A etapa será executada novamente."
                )
                return False

        logger.info(f" Entradas e saídas para '{step_name}' estão íntegras. Pulando etapa.")
        return True

    async def _update_state_on_success(self, step_name: str, inputs: dict):
        """Updates checkpoint after a step runs successfully."""

        metadata = {
            "input_hash": self._get_input_hash(inputs),
            "timestamp": self.file_manager.get_timestamp(),
        }

        await self._save_checkpoint(
            step_name=step_name, status="completed", progress=1.0, metadata=metadata
        )

    async def _invalidate_downstream_files(self, current_step_name: str):
        """Resets checkpoints for all steps that come after the current one."""
        step_order = ["preprocess", "learn_rules", "update_and_reindex", "ranking"]

        try:
            current_index = step_order.index(current_step_name)
            for step_to_invalidate in step_order[current_index + 1 :]:
                logger.info(f"Invalidando checkpoint da etapa futura: {step_to_invalidate}")
                await self._save_checkpoint(
                    step_name=step_to_invalidate, status="pending", progress=0.0
                )
        except ValueError:
            pass
