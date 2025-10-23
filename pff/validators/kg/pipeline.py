import gc
import hashlib
import os
import platform
import sys
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TypedDict

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, roc_auc_score

from pff.utils import ConcurrencyManager, FileManager, logger
from pff.utils.global_interrupt_manager import (
    check_interruption,
    get_interrupt_manager,
    should_stop,
)

from .anyburl import AnyBURLLearner, RuleParser
from .builder import KGBuilder
from .calibration import ScoreCalibrator, find_optimal_threshold
from .config import KGConfig
from .preprocess import KGPreprocessor
from .ranking import create_test_data_chunks, parallel_ranking_worker

os.environ["RAY_LOGGING_CONFIG_ENCODING"] = "TEXT"

if sys.platform != "win32":
    try:
        import ray  # noqa: F401
        from ray.exceptions import RayTaskError

        HAS_RAY = True
    except ImportError:
        HAS_RAY = False
        warnings.warn("Ray não disponível, usando fallback")
else:
    HAS_RAY = False

    class RayTaskError(Exception):
        pass


fm = FileManager()


class SystemInfo:
    """
    SystemInfo provides static methods to retrieve system information and make decisions about optimal computation backends and safe parallelism.
    Methods
    -------
    get_system_info() -> dict
        Gathers and returns a dictionary with details about the current system, including OS, CPU count, memory, and Python version.
    get_optimal_backend() -> list
        Determines and returns a list of optimal computation backends based on the current system and available libraries.
    get_memory_safe_workers(chunk_size: int = 1000) -> int
        Calculates and returns the maximum number of parallel workers that can safely run based on available system memory and the specified chunk size.
    """

    @staticmethod
    def get_system_info():
        """
        Gathers and returns basic system information.

        Returns:
            dict: A dictionary containing the following system information:
                - os (str): The name of the operating system.
                - is_windows (bool): True if the system is Windows, False otherwise.
                - cpu_count (int): The number of CPU cores available.
                - memory_gb (float): Total system memory in gigabytes.
                - available_memory_gb (float): Available system memory in gigabytes.
                - python_version (str): The version of Python currently running.
        """
        import psutil
        return {
            "os": platform.system(),
            "is_windows": sys.platform == "win32",
            "cpu_count": os.cpu_count(),
            "memory_gb": psutil.virtual_memory().total / (1024**3),
            "available_memory_gb": psutil.virtual_memory().available / (1024**3),
            "python_version": sys.version,
        }

    @staticmethod
    def get_optimal_backend():
        """
        Determines and returns a prioritized list of optimal backends for task execution based on the current system's operating system and available libraries.
        Returns:
            list of str: A list of backend names in order of preference. On Windows, returns ["dask", "thread", "sequential"]. On other systems, returns ["ray", "dask", "thread", "sequential"] if Ray is available, otherwise ["dask", "thread", "sequential"].
        Dependencies:
            - SystemInfo.get_system_info(): Should return a dictionary with at least the key "is_windows" (bool).
            - HAS_RAY: Boolean indicating if the Ray library is available.
        """
        info = SystemInfo.get_system_info()

        if info["is_windows"]:
            return ["dask", "thread", "sequential"]
        else:
            if HAS_RAY:
                return ["ray", "dask", "thread", "sequential"]
            else:
                return ["dask", "thread", "sequential"]

    @staticmethod
    def get_memory_safe_workers(chunk_size: int = 1000):
        """
        Calculates the optimal number of worker processes that can be safely spawned based on available system memory and CPU count.
        Args:
            chunk_size (int, optional): The size of data chunks each worker will process. Defaults to 1000.
        Returns:
            int: The recommended number of worker processes, constrained by both available memory and CPU cores.
        Notes:
            - Assumes each worker requires approximately 0.5 GB of memory per 1000 chunk size.
            - Limits the total memory usage to 70% of available system memory.
            - Ensures at least one worker is returned, and does not exceed the number of CPU cores.
        """
        import psutil

        available_gb = psutil.virtual_memory().available / (1024**3)
        memory_per_worker_gb = 0.5 * (chunk_size / 1000)
        safe_workers = int((available_gb * 0.7) / memory_per_worker_gb)
        cpu_count = os.cpu_count() or 4
        
        return max(1, min(safe_workers, cpu_count))


class WorkerResult(TypedDict):
    worker_id: int
    status: str
    error: str
    ranking_lines: list[str]
    detailed_scores: list[dict]


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


class StandardDataLoader(DataLoaderInterface):
    """Standard implementation for loading triple data."""

    def load_triples_from_parquet(self, parquet_path: Path) -> list[list[str]]:
        """
        Load triples from a Parquet file containing subject, predicate, object columns.

        Args:
            parquet_path: Path to the Parquet file

        Returns:
            List of triples as [subject, predicate, object]

        Raises:
            ValueError: If required columns are missing
        """
        logger.info(f"Carregando triplas de {parquet_path}...")

        dataframe = fm.read(parquet_path)
        required_columns = ["s", "p", "o"]

        if not all(column in dataframe.columns for column in required_columns):
            raise ValueError(
                f"Arquivo deve conter colunas {required_columns}, "
                f"encontradas: {dataframe.columns}"
            )

        triples = [list(row) for row in dataframe.select(required_columns).iter_rows()]

        logger.info(f"Carregadas {len(triples)} triplas")
        return triples

    def load_indexed_data(self, numpy_path: Path) -> np.ndarray:
        """
        Load indexed data from a NumPy file.

        Args:
            numpy_path: Path to the NumPy file

        Returns:
            NumPy array with indexed data

        Raises:
            FileNotFoundError: If file does not exist
        """
        if not numpy_path.exists():
            raise FileNotFoundError(f"Arquivo NumPy não encontrado: {numpy_path}")

        data = np.load(numpy_path)
        logger.info(f"Carregados {len(data)} índices de {numpy_path}")
        return data


class MetricsCalculator:
    """Calculate evaluation metrics for ranking results."""

    def __init__(self, config=None, top_k: int = 10):
        self.top_k = top_k
        self.config = config
        self.calibrator = None
        self.optimal_threshold = 0.5
        logger.info(f"MetricsCalculator inicializada com top_k={self.top_k}")

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
            logger.info(
                "Iniciando calibração de scores, pois há exemplos positivos e negativos."
            )
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
            fm.save(metrics, metrics_path)
            logger.info(f"✅ Todas as métricas salvas em {metrics_path}")
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
            logger.warning("Nenhum hit verdadeiro encontrado para calcular métricas")
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
            "total_queries": len(
                scores_dataframe.unique(["src_id", "rel_id", "direction"])
            ),
            "true_hits": len(true_hits),
        }

        logger.info("📊 Métricas de Ranking:")
        logger.info(f"  MRR: {mean_reciprocal_rank:.4f}")
        logger.info(f"  Hits@1: {hits_at_1:.4f}")
        logger.info(f"  Hits@{self.top_k}: {hits_at_k:.4f}")
        logger.info(f"  Total queries: {metrics['total_queries']}")
        logger.info(f"  True hits: {metrics['true_hits']}")

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

            roc_auc = roc_auc_score(y_true, y_scores)
            pr_auc = average_precision_score(y_true, y_scores)

            metrics = {
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
                "positive_rate": float(y_true.mean()),
                "score_mean": float(y_scores.mean()),
                "score_std": float(y_scores.std()),
            }

            prefix = "Calibradas" if calibrated else "Originais"
            logger.info(f"📈 Métricas de Classificação ({prefix}):")
            logger.info(f"  ROC-AUC: {roc_auc:.4f}")
            logger.info(f"  PR-AUC: {pr_auc:.4f}")
            logger.info(f"  Taxa de positivos: {metrics['positive_rate']:.4f}")

            return metrics

        except ImportError:
            logger.warning("sklearn não disponível, pulando cálculo de AUC")
            return {}

    def get_last_metrics(self) -> dict:
        """Retrieves the most recent metrics from the output directory specified in the configuration."""
        if self.config:
            metrics_path = self.config.get_output_directory() / "metrics.json"
            if metrics_path.exists():
                return fm.read(metrics_path)
        return {}

    def _calibrate_scores(self, scores_dataframe: pl.DataFrame) -> pl.DataFrame:
        """
        Calibrate scores using Platt scaling.

        Args:
            scores_dataframe: DataFrame with raw scores

        Returns:
            DataFrame with additional calibrated_score column
        """
        logger.info("🔧 Calibrando scores...")

        y_true = scores_dataframe["is_true"].to_numpy()
        y_scores = scores_dataframe["score"].to_numpy()

        self.calibrator = ScoreCalibrator(method="platt")
        calibrated_scores = self.calibrator.cross_val_calibrate(y_scores, y_true, cv=5)

        self.calibrator.fit(y_scores, y_true)

        result_df = scores_dataframe.with_columns(
            pl.Series("score_calibrated", calibrated_scores)
        )

        logger.info("✅ Calibração concluída")
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

    def __init__(self, config: KGConfig):
        """
        Initializes the orchestrator with all necessary components.

        Args:
            config: The main configuration object for the pipeline.
        """
        self.config = config
        self.system_info = SystemInfo.get_system_info()
        logger.info(
            f"🖥️ Sistema detectado: {self.system_info['os']} "
            f"({self.system_info['cpu_count']} CPUs, "
            f"{self.system_info['memory_gb']:.1f}GB RAM)"
        )
        builder_params = self.config.get_builder_config()
        self.builder = KGBuilder(
            source_path=builder_params["source_path"],
            output_dir=self.config.graph_directory,
            max_members=builder_params.get("max_members"),
            parallel=builder_params.get("parallel", True),
            disk_cache=builder_params.get("disk_cache", False),
            workers=builder_params.get("workers"),
        )
        self.preprocessor = KGPreprocessor(config)
        self.rule_learner = AnyBURLLearner()
        self.data_loader = StandardDataLoader()
        self.rule_parser = RuleParser()
        self.state_file_path = (
            self.config.get_pyclause_directory() / "pipeline_state.json"
        )
        self.state = self._load_state()
        pipeline_params = self.config.get_pipeline_configuration()
        top_k_value = pipeline_params.get("top_k", 10)
        self.metrics_calculator = MetricsCalculator(
            config=self.config, top_k=top_k_value
        )
        self.interrupt_manager = get_interrupt_manager()

        def kg_cleanup_callback():
            logger.info("🧹 KGPipeline: Iniciando limpeza por interrupção...")
            try:
                self._save_state()
                logger.info("✅ Estado da pipeline KG salvo durante interrupção")
            except Exception as e:
                logger.warning(f"⚠️ Erro ao salvar estado durante interrupção: {e}")

        self.interrupt_manager.register_callback(kg_cleanup_callback)
        logger.info("✅ KGPipeline integrado ao GlobalInterruptManager")

    async def run_build_and_preprocess(self):
        """Runs only the builder and preprocessing steps."""
        logger.info("=" * 60)
        logger.info("INICIANDO ETAPA DE BUILD E PREPROCESS")
        logger.info("=" * 60)
        check_interruption()
        await self._run_preprocess_step()
        check_interruption()
        self._save_state()
        logger.success("✅ Etapa de Build e Preprocess concluída.")

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
        logger.info("=" * 60)
        logger.info("INICIANDO ETAPA DE APRENDIZADO DE REGRAS")
        logger.info("=" * 60)
        check_interruption()
        await self._run_learn_rules_step(override_config=override_config)
        check_interruption()
        self._save_state()
        logger.success("✅ Etapa de Aprendizado de Regras concluída.")

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

        logger.info("=" * 60)
        logger.info("INICIANDO ETAPA DE RANKING")
        logger.info("=" * 60)
        check_interruption()
        metrics = await self._run_ranking_step(override_config=override_config)
        check_interruption()
        self._save_state()
        logger.success("✅ Etapa de Ranking concluída.")

        return metrics

    async def _run_learn_rules_step(
        self, force_run: bool = False, override_config: dict | None = None
    ) -> bool:
        """Manages the execution of the rule learning step."""
        step_name = "learn_rules"
        logger.info("-" * 60)
        logger.info(f"Avaliando Etapa 2: {step_name.upper()}")
        if should_stop():
            logger.warning(f"🛑 Etapa '{step_name}' cancelada por interrupção")
            return False
        anyburl_params = self.config.get_anyburl_parameters()
        inputs_to_hash = {
            "indexed_data": [
                self.config.train_numpy_path,
                self.config.valid_numpy_path,
                self.config.test_numpy_path,
            ],
            "params": anyburl_params,
        }
        if self._should_skip_step(step_name, inputs_to_hash) and not force_run:
            return False
        if should_stop():
            return False
        self._invalidate_downstream_files(step_name)
        logger.warning(f"▶️ Executando etapa '{step_name}'...")
        await self.rule_learner.learn_rules(self.config)
        if should_stop():
            return False
        self._update_state_on_success(step_name, inputs_to_hash)
        return True

    async def _run_preprocess_step(
        self, force_run: bool = False, override_config: dict | None = None
    ) -> bool:
        """Manages the execution of the data preprocessing step."""
        step_name = "preprocess"
        logger.info("-" * 60)
        logger.info(f"Avaliando Etapa 1: {step_name.upper()}")
        if should_stop():
            logger.warning(f"🛑 Etapa '{step_name}' cancelada por interrupção")
            return False
        if not self.config.validate():
            logger.warning(
                "Arquivos .parquet não encontrados no diretório configurado. "
                "Acionando KGBuilder..."
            )
            check_interruption()
            await self.builder.run()
            check_interruption()
            if not self.config.validate():
                logger.error(
                    "KGBuilder falhou em criar os arquivos necessários. Abortando."
                )
                raise FileNotFoundError(
                    "Arquivos de entrada .parquet não puderam ser construídos."
                )
        inputs_to_hash = {
            "source_files": [
                self.config.train_path,
                self.config.valid_path,
                self.config.test_path,
            ],
            "params": self.config.get_preprocessing_parameters(),
        }
        if self._should_skip_step(step_name, inputs_to_hash):
            return False
        check_interruption()
        self._invalidate_downstream_files(step_name)
        logger.warning(f"▶️ Executando etapa '{step_name}'...")
        self.preprocessor.run()
        check_interruption()
        self._update_state_on_success(step_name, inputs_to_hash)

        return True

    async def _run_ranking_step(
        self, force_run: bool = False, override_config: dict | None = None
    ) -> dict | None:
        """Manages the execution of the parallel ranking step."""
        step_name = "ranking"
        logger.info("-" * 60)
        logger.info(f"Avaliando Etapa 3: {step_name.upper()}")
        check_interruption()
        inputs_to_hash = {
            "rules": self.config.rules_path,
            "indexed_data": [
                self.config.train_numpy_path,
                self.config.valid_numpy_path,
                self.config.test_numpy_path,
            ],
            "params": self.config.get_pipeline_configuration(),
        }
        if self._should_skip_step(step_name, inputs_to_hash) and not force_run:
            return self.metrics_calculator.get_last_metrics()
        check_interruption()
        logger.warning(f"▶️ Executando etapa '{step_name}'...")
        results = await self._execute_parallel_ranking(override_config=override_config)
        check_interruption()
        metrics = self._save_results(results)

        self._update_state_on_success(step_name, inputs_to_hash)
        return metrics

    async def _execute_parallel_ranking(
        self, override_config: dict | None = None
    ) -> dict:
        """Execute parallel ranking by passing raw data directly to workers."""
        check_interruption()
        logger.info("Etapa 4: Preparando dados para o ranking...")
        check_interruption()

        logger.info("Carregando dados indexados (.npy)...")
        train_array = self.data_loader.load_indexed_data(self.config.train_numpy_path)
        valid_array = self.data_loader.load_indexed_data(self.config.valid_numpy_path)
        test_array = self.data_loader.load_indexed_data(self.config.test_numpy_path)

        rules, _ = self.rule_parser.parse_rules_file(self.config.rules_path)
        logger.info(f"Carregadas {len(rules)} regras")

        logger.info("Construindo conjunto de filtro a partir de dados indexados...")
        filter_set = set()
        for split_data in [train_array, valid_array]:
            for row in split_data:
                filter_set.add((row[0], row[1], row[2]))
        filter_array = np.vstack([train_array, valid_array])
        logger.info(f"Filter set contém {len(filter_set)} triplas")
        test_in_filter = 0
        for triple in test_array:
            if tuple(triple) in filter_set:
                test_in_filter += 1
        logger.info(
            f"Triplas de teste presentes no filter_set: {test_in_filter}/{len(test_array)}"
        )
        filter_array = np.unique(filter_array, axis=0)
        logger.info(f"Conjunto de filtro preparado: {len(filter_array)} triplas únicas")
        logger.info("Etapa 6: Preparando distribuição de trabalho...")
        pipeline_config = self.config.get_pipeline_configuration()
        max_chunk_size = self.config.get_max_chunk_size()
        if self.system_info["available_memory_gb"] < 4:
            max_chunk_size = min(max_chunk_size, 300)
            logger.warning(
                f"⚠️ Memória limitada ({self.system_info['available_memory_gb']:.1f}GB). "
                f"Reduzindo chunk_size para {max_chunk_size}"
            )
        chunks = create_test_data_chunks(
            test_array,
            pipeline_config["chunk_size"],
            pipeline_config["num_workers"],
            max_chunk_size,
        )
        logger.info("Etapa 7: Executando ranking paralelo...")
        if len(chunks) == 1 and len(chunks[0]) > max_chunk_size:
            original_chunk = chunks[0]
            chunks = [
                original_chunk[i : i + max_chunk_size]
                for i in range(0, len(original_chunk), max_chunk_size)
            ]
            logger.info(
                f"🧠 Redividindo chunks grandes: {len(chunks)} chunks de até {max_chunk_size} itens"
            )
        check_interruption()
        results = await self._launch_ranking_workers(
            chunks,
            rules,
            filter_array,
            train_array,
            self.config.get_entity_map_path(),
            self.config.get_relation_map_path(),
            override_config=override_config,
        )

        return results

    async def _launch_ranking_workers(
        self,
        chunks: list[np.ndarray],
        rules: list[str],
        filter_triples: np.ndarray,
        train_triples: np.ndarray,
        entity_map_path: Path,
        relation_map_path: Path,
        override_config: dict | None = None,
    ) -> dict:
        """Launch parallel ranking workers using ConcurrencyManager with smart fallback."""
        logger.info(
            "Etapa 7.1: Preparando dados compartilhados para os workers paralelos..."
        )
        cm = ConcurrencyManager()
        pyclause_config = self.config.get_pyclause_options_dictionary()
        ranking_handler_config = pyclause_config["ranking_handler"]
        if override_config and "pyclause" in override_config:
            ranking_handler_config.update(
                override_config["pyclause"]["ranking_handler"]
            )
        rules_path = self.config.get_rules_path()
        shared_data = {
            "rules_path": str(rules_path),
            "filter_triples": str(self.config.train_numpy_path),
            "train_triples": str(self.config.train_numpy_path),
            "entity_map_path": str(entity_map_path),
            "relation_map_path": str(relation_map_path),
            "aggregation_function": ranking_handler_config["aggregation_function"],
            "tie_handling": ranking_handler_config["tie_handling"],
            "filter_w_data": ranking_handler_config["filter_w_data"],
            "num_threads": ranking_handler_config["num_threads"],
            "verbose": bool(pyclause_config.get("verbose", False)),
        }
        worker_args = [(i, chunk) for i, chunk in enumerate(chunks)]
        backends = SystemInfo.get_optimal_backend()
        logger.info(f"🔧 Backends disponíveis em ordem de preferência: {backends}")
        for backend_idx, task_type in enumerate(backends):
            try:
                logger.info(f"🚀 Tentando executar com backend: {task_type}")
                if task_type == "dask":
                    dask_config = self.config.get_dask_configuration()
                    safe_workers = SystemInfo.get_memory_safe_workers(
                        chunk_size=max(len(chunk) for chunk in chunks)
                    )
                    backend_kwargs = {
                        "n_workers": min(safe_workers, dask_config.get("n_workers", 4)),
                        "threads_per_worker": 1,
                        "memory_limit": "2GB",
                        "processes": True,
                        "silence_logs": 30,
                    }
                    logger.info(
                        f"📊 Dask configurado com {backend_kwargs['n_workers']} workers, "
                        f"{backend_kwargs['memory_limit']} por worker"
                    )
                elif task_type == "ray":
                    if not HAS_RAY:
                        logger.warning("Ray não disponível, pulando...")
                        continue
                    backend_kwargs = {}
                elif task_type == "thread":
                    backend_kwargs = {"max_workers": min(2, os.cpu_count() or 2)}
                else:  # sequential
                    backend_kwargs = {}
                def ranking_worker_adapter(worker_id, chunk):
                    gc.collect()
                    result = parallel_ranking_worker(shared_data, worker_id, chunk)
                    gc.collect()
                    
                    return result

                results = await cm.execute(
                    ranking_worker_adapter,
                    worker_args,
                    task_type=task_type,
                    backend_kwargs=backend_kwargs,
                    desc=f"Ranking paralelo ({task_type})",
                )

                logger.success(f"✅ Ranking executado com sucesso usando {task_type}")
                return self._aggregate_results(results)

            except Exception as e:
                logger.error(f"❌ Falha com backend {task_type}: {str(e)}")
                if backend_idx == len(backends) - 1:
                    logger.error("Todos os backends falharam!")
                    raise
                logger.info("Tentando próximo backend...")
                gc.collect()
                continue

    def _aggregate_results(self, results: list[dict]) -> dict:
        """Aggregate results from all workers."""
        all_ranking_lines = []
        all_detailed_scores = []
        errors = []

        for result in results:
            if result.get("status") == "success":
                all_ranking_lines.extend(result.get("ranking_lines", []))
                all_detailed_scores.extend(result.get("detailed_scores", []))
            else:
                errors.append(result)
                worker_id = result.get("worker_id", "desconhecido")
                error_msg = result.get("error", "Erro desconhecido")
                logger.error(f"Worker {worker_id}: {error_msg}")

        logger.info(
            f"🎯 TOTAL coletado: {len(all_ranking_lines)} ranking lines, "
            f"{len(all_detailed_scores)} scores"
        )
        if errors:
            logger.warning(f"Encontrados {len(errors)} erros durante processamento")

        return {
            "ranking_lines": all_ranking_lines,
            "detailed_scores": all_detailed_scores,
            "errors": errors,
        }

    def _save_results(self, results: dict) -> dict[str, float] | None:
        """Save ranking results and calculate metrics with calibration."""
        logger.info("Etapa 9: Salvando resultados...")
        ranking_path = self.config.get_ranking_path()
        scores_path = self.config.get_scores_path()
        fm.save("\n".join(results["ranking_lines"]), ranking_path)
        logger.info(
            f"✅ Ranking salvo: {len(results['ranking_lines'])} linhas em {ranking_path}"
        )
        if results["detailed_scores"]:
            scores_dataframe = pl.DataFrame(results["detailed_scores"])
            fm.save(scores_dataframe, scores_path)
            logger.info(
                f"✅ Scores salvos: {len(scores_dataframe)} registros em {scores_path}"
            )
            calibration_config = self.config.get_calibration_config()
            metrics = self.metrics_calculator.calculate_ranking_metrics(
                scores_dataframe,
                calibrate=calibration_config["enabled"],
            )
            self._save_execution_metadata(results)

            return metrics

        self._save_execution_metadata(results)
        return None

    def _save_execution_metadata(self, results: dict) -> None:
        """Save execution metadata for analysis."""
        metadata = {
            "total_rules": len(results.get("rules", [])),
            "total_test_triples": len(results.get("test_triples", [])),
            "chunks_processed": len(results.get("chunks", [])),
            "errors": len(results.get("errors", [])),
            "system_info": self.system_info,
            "ranking_lines_generated": len(results.get("ranking_lines", [])),
            "detailed_scores_generated": len(results.get("detailed_scores", [])),
        }

        metadata_path = self.config.get_output_directory() / "execution_metadata.json"
        fm.save(metadata, metadata_path)
        logger.info(f"✅ Metadata salva em {metadata_path}")

    # STATE MANAGER

    def _load_state(self) -> dict:
        """Loads the last execution state from the state file."""
        if self.state_file_path.exists():
            logger.debug(f"Carregando estado de {self.state_file_path}")
            return fm.read(self.state_file_path)
        logger.debug("Nenhum arquivo de estado encontrado, iniciando do zero.")
        return {}

    def _save_state(self):
        """Saves the current pipeline state to the state file."""
        logger.debug(f"Salvando estado em {self.state_file_path}")
        fm.save(self.state, self.state_file_path)

    def can_resume_from_checkpoint(self, phase: str) -> bool:
        """
        Check if the pipeline can resume from a checkpoint for the given phase.

        Args:
            phase: The phase name to check (e.g., 'build', 'preprocess', 'learn', 'rank')

        Returns:
            bool: True if checkpoint exists and pipeline can resume, False otherwise
        """
        # Check if checkpoint directory exists
        if not hasattr(self.config, 'checkpoint_dir'):
            logger.debug(f"No checkpoint_dir configured, cannot resume {phase}")
            return False

        # Convert to Path if string
        checkpoint_dir = self.config.checkpoint_dir
        if isinstance(checkpoint_dir, str):
            from pathlib import Path
            checkpoint_dir = Path(checkpoint_dir)

        if not checkpoint_dir.exists():
            logger.debug(f"Checkpoint directory {checkpoint_dir} does not exist")
            return False

        # Check for phase-specific checkpoint file
        checkpoint_file = checkpoint_dir / f"{phase}_complete.json"
        if checkpoint_file.exists():
            logger.info(f"✅ Checkpoint found for phase '{phase}' at {checkpoint_file}")
            return True

        logger.debug(f"No checkpoint found for phase '{phase}'")
        return False

    def _get_input_hash(self, inputs: dict) -> str:
        """Generates a combined hash for a step's inputs."""
        hasher = hashlib.md5()
        for key, value in sorted(inputs.items()):
            if isinstance(value, list):  # Handle lists of paths
                for item in value:
                    if isinstance(item, Path) and item.exists():
                        hasher.update(fm.get_hash(item).encode())
            elif isinstance(value, Path) and value.exists():
                hasher.update(fm.get_hash(value).encode())
            else:
                hasher.update(str(value).encode())
        return hasher.hexdigest()

    def _should_skip_step(self, step_name: str, inputs: dict) -> bool:
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
            True  # Returns True if step can be skipped, False otherwise
        """

        last_run_info = self.state.get(step_name)
        if not last_run_info or last_run_info.get("status") != "success":
            logger.info(
                f"Executando '{step_name}' pois não foi concluída com sucesso na última vez."
            )
            return False

        last_input_hash = last_run_info.get("input_hash")
        current_input_hash = self._get_input_hash(inputs)
        if last_input_hash != current_input_hash:
            logger.info(f"Executando '{step_name}' pois suas entradas mudaram.")
            return False

        expected_outputs = self.config.get_step_outputs(step_name)
        for output_file in expected_outputs:
            if not output_file.exists():
                logger.warning(
                    f"Estado para '{step_name}' era 'success', mas o arquivo de saída "
                    f"'{output_file.name}' está faltando. A etapa será executada novamente."
                )
                return False

        logger.info(
            f"✅ Entradas e saídas para '{step_name}' estão íntegras. Pulando etapa."
        )
        return True

    def _update_state_on_success(self, step_name: str, inputs: dict):
        """Updates the state file after a step runs successfully."""
        self.state[step_name] = {
            "input_hash": self._get_input_hash(inputs),
            "status": "success",
            "timestamp": fm.get_timestamp(),
        }
        self._save_state()

    def _invalidate_downstream_files(self, current_step_name: str):
        """Deletes the state of all steps that come after the current one."""
        step_order = ["preprocess", "learn_rules", "update_and_reindex", "ranking"]

        try:
            current_index = step_order.index(current_step_name)
            for step_to_invalidate in step_order[current_index + 1 :]:
                if step_to_invalidate in self.state:
                    logger.warning(
                        f"Invalidando estado da etapa futura: {step_to_invalidate}"
                    )
                    del self.state[step_to_invalidate]
        except ValueError:
            pass
