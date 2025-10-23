import argparse
import asyncio
import os
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import optuna
import polars as pl
import psutil
import yaml
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from pff.utils import CacheManager, logger
from pff.validators.kg.config import KGConfig
from pff.validators.kg.pipeline import KGPipeline

# Cache utility
CACHE = CacheManager()


@dataclass
class SystemProfile:
    cpu_count: int
    cpu_frequency_mhz: float
    memory_gb: float
    disk_type: str
    disk_read_speed_mb_per_second: float
    has_gpu: bool


@dataclass
class DataProfile:
    total_triples: int
    number_of_entities: int
    number_of_relations: int
    density: float
    file_size_mb: float


@dataclass
class OptimizedConfiguration:
    chunk_size: int
    number_of_workers: int
    java_heap_gb: int
    anyburl_threads: int
    snapshots: list[int]
    ray_object_store_gb: float
    expected_runtime_minutes: float
    expected_memory_peak_gb: float
    homogeneity_level: float
    minimum_support: int


class StandardSystemProfiler:
    """Profiles hardware capabilities."""

    def profile_system(self) -> SystemProfile:
        cpu_count = psutil.cpu_count(logical=True) or 1
        freq = psutil.cpu_freq()
        cpu_freq = freq.current if freq else 2000.0
        mem = psutil.virtual_memory().total / (1024**3)
        disk_type, disk_speed = self._profile_disk()
        has_gpu = self._detect_gpu()
        return SystemProfile(
            cpu_count=cpu_count,
            cpu_frequency_mhz=cpu_freq,
            memory_gb=mem,
            disk_type=disk_type,
            disk_read_speed_mb_per_second=disk_speed,
            has_gpu=has_gpu,
        )

    def _profile_disk(self) -> tuple[str, float]:
        test_file = Path("disk_benchmark_temp.bin")
        size_mb = 100
        try:
            start = time.time()
            with open(test_file, "wb") as f:
                f.write(os.urandom(size_mb * 1024**2))
            _ = time.time() - start
            start = time.time()
            with open(test_file, "rb") as f:
                f.read()
            read_sec = time.time() - start
            speed = size_mb / read_sec
            return ("ssd" if speed > 100 else "hdd", speed)
        finally:
            if test_file.exists():
                test_file.unlink()

    def _detect_gpu(self) -> bool:
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False


class StandardDataProfiler:
    """Profiles a Knowledge Graph dataset."""

    def profile_data(self, config: KGConfig) -> DataProfile:
        train_path = config.get_split_path("train")
        if not train_path.exists():
            raise FileNotFoundError(
                f"Train file not found: {train_path}. Run build first."
            )
        df = pl.read_parquet(train_path)
        total = len(df)
        entities = self._count_entities(df)
        relations = df["p"].n_unique()
        max_trip = entities * entities * relations
        density = total / max_trip if max_trip else 0.0
        size_mb = train_path.stat().st_size / (1024**2)
        return DataProfile(
            total_triples=total,
            number_of_entities=entities,
            number_of_relations=relations,
            density=density,
            file_size_mb=size_mb,
        )

    def _count_entities(self, df: pl.DataFrame) -> int:
        subs = set(df["s"].unique())
        objs = set(df["o"].unique())
        return len(subs | objs)


class StandardOptimizationStrategy:
    """Generates configuration heuristically."""

    def optimize(
        self, system: SystemProfile, data: DataProfile, target: str = "balanced"
    ) -> OptimizedConfiguration:
        workers = max(1, system.cpu_count)
        chunk = max(10000, int(data.total_triples / workers / 10) * 1000)
        heap = min(int(system.memory_gb * 0.4), 2 + int(data.total_triples / 1e6 * 2))
        threads = min(system.cpu_count, 16)
        snapshots = {
            "speed": [30, 60],
            "quality": [60, 120, 300, 600],
            "balanced": [30, 60, 120],
        }.get(target, [30, 60, 120])
        ray_mem = min(
            (data.file_size_mb / 1024 * 3 + 0.5) * 1.5, system.memory_gb * 0.3
        )
        runtime = max(snapshots) / 60 + data.total_triples / (10000 * workers) / 60 + 5
        peak = (2 + heap + workers * 2 + ray_mem) * 1.2
        return OptimizedConfiguration(
            chunk_size=chunk,
            number_of_workers=workers,
            java_heap_gb=heap,
            anyburl_threads=threads,
            snapshots=snapshots,
            ray_object_store_gb=ray_mem,
            expected_runtime_minutes=runtime,
            expected_memory_peak_gb=peak,
            homogeneity_level=0.5,
            minimum_support=3 if data.total_triples < 1e7 else 5,
        )


class PerformanceOptimizer:
    """Runs profiling, optimization, and reporting."""

    def __init__(self, config: KGConfig):
        self.config = config
        self.sys_profiler = StandardSystemProfiler()
        self.data_profiler = StandardDataProfiler()
        self.strategy = StandardOptimizationStrategy()
        self.system_profile = self.sys_profiler.profile_system()

    def optimize_configuration(
        self, target: str = "balanced"
    ) -> OptimizedConfiguration:
        data_profile = self.data_profiler.profile_data(self.config)
        return self.strategy.optimize(self.system_profile, data_profile, target)

    def generate_configuration_file(
        self, cfg: OptimizedConfiguration, path: Path
    ) -> None:
        content = {
            "anyburl": {
                "SNAPSHOTS_AT": cfg.snapshots,
                "WORKER_THREADS": cfg.anyburl_threads,
                "JAVA_HEAP": f"{cfg.java_heap_gb}G",
            },
            "ray": {
                "num_cpus": cfg.number_of_workers,
                "object_store_memory_gb": cfg.ray_object_store_gb,
            },
            "pipeline": {
                "chunk_size": cfg.chunk_size,
                "num_workers": cfg.number_of_workers,
                "preprocess": {
                    "enabled": True,
                    "homogeneity_level": cfg.homogeneity_level,
                    "min_support": cfg.minimum_support,
                },
            },
        }
        with open(path, "w") as f:
            yaml.dump(content, f)
        print(f"Configuração otimizada salva em {path}")

    def print_optimization_report(
        self, data_profile: DataProfile, cfg: OptimizedConfiguration
    ) -> None:
        print("\n" + "=" * 40)
        print("RELATÓRIO DE OTIMIZAÇÃO")
        print("=" * 40)
        print(
            f"CPUs: {self.system_profile.cpu_count} @ {self.system_profile.cpu_frequency_mhz:.0f} MHz"
        )
        print(f"RAM: {self.system_profile.memory_gb:.1f} GB")
        print(f"Triplas: {data_profile.total_triples:,}")
        print(f"Densidade: {data_profile.density:.6f}")
        print(f"Workers: {cfg.number_of_workers}")
        print(f"Chunk size: {cfg.chunk_size:,}")
        print(f"Java heap: {cfg.java_heap_gb} GB")
        print(f"Snapshots: {cfg.snapshots}")
        print(f"Tempo estimado: {cfg.expected_runtime_minutes:.0f} min")
        print(f"Pico memória: {cfg.expected_memory_peak_gb:.1f} GB")
        print("=" * 40)


@CACHE.disk_cache(ttl=None)
async def objective(
    trial: optuna.Trial, config: KGConfig, pipeline: KGPipeline
) -> dict:
    """
    Objective function for Optuna hyperparameter optimization of knowledge graph models.
    This function defines the objective to be maximized during hyperparameter optimization.
    It configures a trial with different hyperparameters for AnyBURL and PyClause components,
    executes the pipeline with these parameters, and returns the Mean Reciprocal Rank (MRR)
    as the optimization target.
    Parameters
    ----------
    trial : optuna.Trial
        The trial object that suggests hyperparameter values.
    config : KGConfig
        Configuration object containing baseline settings for the knowledge graph pipeline.
    pipeline : KGPipeline
        Pipeline object that implements the knowledge graph learning and ranking steps.
    Returns
    -------
    float
        The Mean Reciprocal Rank (MRR) value achieved with the trial's parameters.
        Returns -1.0 if an error occurs or no metrics are returned.
    Notes
    -----
    The function logs all parameters and metrics to MLflow for experiment tracking.
    It only executes the 'learn_rules' and 'ranking' steps of the pipeline with the new parameters.
    """

    with mlflow.start_run(run_name=f"Trial-{trial.number}"):
        logger.info(f"--- Iniciando Trial {trial.number} ---")

        override_config = {
            "anyburl": {
                "MAX_LENGTH_ACYCLIC": trial.suggest_int("MAX_LENGTH_ACYCLIC", 1, 3),
                "THRESHOLD_CORRECT_PREDICTIONS": trial.suggest_int(
                    "THRESHOLD_CORRECT_PREDICTIONS", 2, 50
                ),
                "THRESHOLD_CONFIDENCE": trial.suggest_float(
                    "THRESHOLD_CONFIDENCE", 0.001, 0.1, log=True
                ),
                "RULE_ZERO_WEIGHT": trial.suggest_float(
                    "RULE_ZERO_WEIGHT", 0.001, 0.1, log=True
                ),
                "RULE_AC2_WEIGHT": trial.suggest_float(
                    "RULE_AC2_WEIGHT", 0.01, 0.2, log=True
                ),
            },
            "pyclause": {
                "ranking_handler": {
                    "aggregation_function": trial.suggest_categorical(
                        "aggregation_function", ["noisyor", "maxplus"]
                    ),
                    "tie_handling": trial.suggest_categorical(
                        "tie_handling", ["random", "first", "frequency"]
                    ),
                }
            },
        }
        params_to_log = {
            **override_config["anyburl"],
            **override_config["pyclause"]["ranking_handler"],
        }
        mlflow.log_params(params_to_log)
        logger.info(f"Parâmetros do Trial: {params_to_log}")

        try:
            await pipeline.run_learn_rules(override_config=override_config)
            metrics = await pipeline.run_ranking(override_config=override_config)

            if not metrics:
                return {"mrr": -1.0, "pr_auc_raw": 0.0, "hits_at_10": 0.0}

            mrr = metrics.get("mrr", 0.0)
            hits_at_10 = metrics.get("hits_at_10", 0.0)
            pr_auc = metrics.get("classification_raw", {}).get("pr_auc", 0.0)

            mlflow.log_metric("mrr", mrr)
            mlflow.log_metric("hits_at_10", hits_at_10)
            if pr_auc:
                mlflow.log_metric("pr_auc_raw", pr_auc)

            if "classification_raw" in metrics and metrics["classification_raw"]:
                pr_auc = metrics["classification_raw"].get("pr_auc", 0.0)
                roc_auc = metrics["classification_raw"].get("roc_auc", 0.0)
                if roc_auc:
                    mlflow.log_metric("roc_auc_raw", roc_auc)
                if pr_auc:
                    mlflow.log_metric("pr_auc_raw", pr_auc)
            mlflow.log_metric("mrr", mrr)
            mlflow.log_metric("hits_at_10", hits_at_10)

            if (
                "classification_calibrated" in metrics
                and metrics["classification_calibrated"]
            ):
                pr_auc_cal = metrics["classification_calibrated"].get("pr_auc", 0.0)
                roc_auc_cal = metrics["classification_calibrated"].get("roc_auc", 0.0)
                if roc_auc_cal:
                    mlflow.log_metric("roc_auc_calibrated", roc_auc_cal)
                if pr_auc_cal:
                    mlflow.log_metric("pr_auc_calibrated", pr_auc_cal)

            logger.info(
                f"Resultado do Trial {trial.number}: MRR = {mrr:.4f}, Hits@10 = {hits_at_10:.4f}, PR-AUC = {pr_auc:.4f}"
            )
            return {"mrr": mrr, "pr_auc_raw": pr_auc, "hits_at_10": hits_at_10}

        except Exception as e:
            logger.exception(f"Trial {trial.number} falhou: {e}", exc_info=True)
            return {"mrr": -1.0, "pr_auc_raw": 0.0, "hits_at_10": 0.0}


async def run_experimental_optimization(
    config_path: str, n_trials: int, sample_frac: float | None
):
    """
    Executes an experimental hyperparameter optimization loop for a knowledge graph completion pipeline.
    This function prepares the environment, optionally samples the training data, runs preprocessing,
    and performs an interactive Optuna optimization loop with real-time plotting of metrics (MRR and Hits@10).
    Results are tracked with MLflow, and the best hyperparameters are reported at the end.
    Args:
        config_path (str): Path to the configuration file for the pipeline.
        n_trials (int): Number of optimization trials to run.
        sample_frac (float | None): Fraction of the training data to sample for faster experimentation.
            If None, the full training set is used.
    Side Effects:
        - May delete and recreate the output directory specified in the config.
        - May create and delete a temporary sampled training file.
        - Displays a real-time matplotlib plot of optimization progress.
        - Logs results and progress using the configured logger.
        - Tracks experiments using MLflow.
    Raises:
        Any exceptions raised during pipeline setup, preprocessing, or optimization will propagate.
    Notes:
        - To visualize experiment details, run `mlflow ui` in the terminal after execution.
        - Temporary files created for sampled data are cleaned up automatically.
    """

    temp_train_file = None
    try:
        logger.info("Preparando a configuração e o ambiente para a otimização...")

        config = KGConfig(config_path)
        output_dir = config.get_output_directory()
        if output_dir.exists():
            logger.info(f"Limpando diretório de saída existente: {output_dir}")
            shutil.rmtree(output_dir)
        config = KGConfig(config_path)
        pipeline_for_build = KGPipeline(config)
        await pipeline_for_build.run_build_and_preprocess()

        if sample_frac:
            train_path = config.get_split_path("train")
            logger.info(f"Criando amostra de {sample_frac:.0%} do arquivo de treino...")
            train_df = pl.read_parquet(train_path)
            sampled_df = train_df.sample(fraction=sample_frac, shuffle=True)

            with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
                sampled_df.write_parquet(tmp.name)
                temp_train_file = Path(tmp.name)
                config.train_path = temp_train_file
                logger.info(
                    f"Otimização usará amostra de {len(sampled_df)} triplas em {temp_train_file.name}"
                )

        pipeline_for_trials = KGPipeline(config)
        await pipeline_for_trials.run_build_and_preprocess()

        mlflow.set_experiment("Otimizacao KGC AnyBURL")
        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(n_startup_trials=10),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=3),
        )

        plt.ion()
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax2 = ax1.twinx()

        ax1.set_ylabel("MRR / Hits@10", color="tab:blue", fontsize=12)
        ax2.set_ylabel("PR-AUC (Média Móvel)", color="tab:green", fontsize=12)
        ax1.tick_params(axis="y", labelcolor="tab:blue")
        ax2.tick_params(axis="y", labelcolor="tab:green")

        (line_mrr,) = ax1.plot([], [], "o-", color="tab:blue", label="MRR")
        (line_hits,) = ax1.plot([], [], "s-", color="tab:orange", label="Hits@10")

        (line_mrr_avg,) = ax1.plot(
            [], [], "-", color="navy", alpha=0.5, label="MRR (Média Móvel)"
        )
        (line_pr_auc_avg,) = ax2.plot(
            [], [], "-", color="tab:green", alpha=0.8, label="PR-AUC (Média Móvel)"
        )
        (line_pr_auc,) = ax2.plot(
            [], [], "^", color="darkgreen", alpha=0.6, label="PR-AUC (Trial)"
        )

        (best_mrr_dot,) = ax1.plot(
            [], [], "*", color="gold", markersize=15, label="Melhor MRR"
        )

        ax1.grid(True, linestyle="--", alpha=0.6)

        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc="upper left")

        trial_numbers, mrr_values, hits_values, pr_auc_values = [], [], [], []

        best_mrr_so_far = -1.0
        best_trial_num = 0

        logger.info(f"Iniciando loop de otimização para {n_trials} trials...")
        with ThreadPoolExecutor(max_workers=1) as executor:
            for i in range(n_trials):
                import asyncio

                trial = study.ask()
                loop = asyncio.get_event_loop()
                coro = objective(trial, config, pipeline_for_trials)
                future = executor.submit(loop.run_until_complete, coro)

                while not future.done():
                    fig.canvas.flush_events()
                    plt.pause(0.5)

                metrics_result = future.result()
                study.tell(trial, metrics_result.get("mrr", -1.0))
                mrr = metrics_result.get("mrr", 0.0)
                pr_auc = metrics_result.get("pr_auc_raw", 0.0)
                hits = metrics_result.get("hits_at_10", 0.0)

                trial_numbers.append(i)
                mrr_values.append(mrr)
                pr_auc_values.append(pr_auc)
                hits_values.append(hits)

                if mrr > best_mrr_so_far:
                    best_mrr_so_far = mrr
                    best_trial_num = i

                # Polars rolling window (5x-54x mais rápido que pandas)
                mrr_avg = pl.Series(mrr_values).rolling_mean(window_size=5, min_periods=1).to_list()
                pr_auc_avg = pl.Series(pr_auc_values).rolling_mean(window_size=5, min_periods=1).to_list()
                line_mrr.set_data(trial_numbers, mrr_values)
                line_hits.set_data(trial_numbers, hits_values)
                line_pr_auc.set_data(trial_numbers, pr_auc_values)
                line_mrr_avg.set_data(trial_numbers, mrr_avg)
                line_pr_auc_avg.set_data(trial_numbers, pr_auc_avg)
                best_mrr_dot.set_data([best_trial_num], [best_mrr_so_far])
                ax1.relim()
                ax1.autoscale_view()
                ax2.relim()
                ax2.autoscale_view()
                fig.canvas.draw()
                fig.canvas.flush_events()
        print("\n" + "=" * 60)
        print("--- Otimização Concluída ---")
        print(f"Melhor Trial: {study.best_trial.number}")
        print(f"Melhor MRR: {study.best_value:.4f}")
        print("Melhores Hiperparâmetros encontrados:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")
        print("=" * 60)
        print(
            "\nPara visualizar detalhes e gráficos interativos, execute no terminal: mlflow ui"
        )

        plt.ioff()
        plt.show()

    finally:
        if temp_train_file:
            temp_train_file.unlink(missing_ok=True)
            logger.info(
                f"Arquivo de amostra temporário {temp_train_file.name} removido."
            )


async def main():
    """
    Main entry point for the Knowledge Graph pipeline optimizer.
    This function parses command-line arguments and executes the requested optimization strategy:
    - 'generate': Creates an optimized configuration file using heuristic approaches.
      The optimization can target speed, quality, or a balanced approach.
    - 'tune': Performs experimental optimization using Optuna and MLflow tracking.
      This approach runs multiple trials to find optimal parameters.
    Command line arguments:
    generate:
        --config: Path to base configuration file (default: config/kg.yaml)
        --target: Optimization target (choices: speed, quality, balanced; default: quality)
        --output: Output path for the generated config file (default: optimized_config.yaml)
    tune:
        --config: Path to base configuration file (default: config/kg.yaml)
        --trials: Number of optimization trials to run (default: 50)
        --sample-frac: Fraction of training data to use for faster optimization (e.g., 0.1 for 10%)
    Returns:
        None
    """
    parser = argparse.ArgumentParser(
        description="Otimizador de performance para pipeline KGC"
    )
    subparsers = parser.add_subparsers(
        dest="command", required=True, help="Ação a executar"
    )

    parser_gen = subparsers.add_parser(
        "generate", help="Gera um config otimizado heuristicamente"
    )
    parser_gen.add_argument(
        "--config", type=Path, default="config/kg.yaml", help="Config base"
    )
    parser_gen.add_argument(
        "--target", choices=["speed", "quality", "balanced"], default="quality"
    )
    parser_gen.add_argument("--output", type=Path, default="optimized_config.yaml")

    parser_tune = subparsers.add_parser(
        "tune", help="Executa otimização experimental com Optuna"
    )
    parser_tune.add_argument(
        "--config", type=Path, default="config/kg.yaml", help="Config base"
    )
    parser_tune.add_argument(
        "--trials", type=int, default=50, help="Número de experimentos para executar"
    )
    parser_tune.add_argument(
        "--sample-frac",
        type=float,
        default=None,
        help="Use uma fração do dataset de treino para otimização mais rápida (ex: 0.1 para 10%)",
    )

    args = parser.parse_args()

    try:
        config = KGConfig(args.config)
    except FileNotFoundError as e:
        logger.error(f"Arquivo de configuração principal não encontrado: {e}")
        return

    if args.command == "generate":
        print("--- Modo: Geração de Configuração Heurística ---")
        optimizer = PerformanceOptimizer(config)
        optimized_config = optimizer.optimize_configuration(target=args.target)
        # O data_profile é recalculado aqui, mas é um custo aceitável para este modo.
        data_profile = optimizer.data_profiler.profile_data(config)
        optimizer.print_optimization_report(data_profile, optimized_config)
        optimizer.generate_configuration_file(optimized_config, args.output)

    elif args.command == "tune":
        print("--- Modo: Otimização Experimental com Optuna & MLflow ---")
        await run_experimental_optimization(str(args.config), args.trials, args.sample_frac)


if __name__ == "__main__":
    asyncio.run(main())
