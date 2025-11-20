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

from pff.utils import CacheManager, FileManager, logger
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
            test_file.write_bytes(os.urandom(size_mb * 1024**2))
            _ = time.time() - start
            start = time.time()
            test_file.read_bytes()
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
        df = FileManager().read(train_path)
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
