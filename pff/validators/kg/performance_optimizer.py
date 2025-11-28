"""
SOTA Performance Optimizer for AnyBURL & PyClause

Reuses existing PFF utilities:
- pff.utils.system.hardware_detector.HardwareDetector
- pff.utils.core.cache.CacheManager
- pff.utils.system.resource_manager.ResourceManager

Author: PFF Team
Date: 2025-11-04
Version: 2.0.0
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from pff import settings
from pff.utils import logger
from pff.utils.system.hardware_detector import HardwareDetector, HardwareProfile
from pff.utils.core.cache import CacheManager
from pff.utils.system.resource_manager import ResourceManager


class DataAnalyzer:
    """Analyze data characteristics for optimization."""

    @staticmethod
    def estimate_triple_count(file_path: Path) -> int:
        """
        Estimate number of triples in dataset.

        Args:
            file_path: Path to data file

        Returns:
            Estimated triple count
        """
        try:
            if file_path.suffix == '.npy':
                import numpy as np
                return np.load(file_path).shape[0]
            elif file_path.suffix in ['.parquet', '.pq']:
                import polars as pl
                df = pl.scan_parquet(file_path)
                return df.select(pl.len()).collect().item()
        except Exception:
            pass

        size_mb = file_path.stat().st_size / (1024 * 1024)
        triples_per_mb = 1000
        return int(size_mb * triples_per_mb)

    @staticmethod
    def get_data_density_category(triple_count: int, entity_count: int) -> str:
        """
        Categorize data density.

        Args:
            triple_count: Number of triples
            entity_count: Number of entities

        Returns:
            Density category: 'sparse', 'moderate', 'dense'
        """
        triple_per_entity = triple_count / max(1, entity_count)

        if triple_per_entity < 5:
            return 'sparse'
        elif triple_per_entity < 20:
            return 'moderate'
        else:
            return 'dense'


class AnyBURLPerformanceOptimizer:
    """Optimize AnyBURL parameters based on hardware and data."""

    def __init__(self):
        self.profile = HardwareDetector.detect()
        self.resource_mgr = ResourceManager()
        self.analyzer = DataAnalyzer()

    def optimize_parameters(
        self,
        current_config: dict[str, Any],
        train_data_path: Path | None = None,
    ) -> dict[str, Any]:
        """
        Optimize AnyBURL parameters using HardwareDetector.

        Args:
            current_config: Current AnyBURL configuration
            train_data_path: Path to training data

        Returns:
            Optimized configuration
        """
        import copy
        optimized = copy.deepcopy(current_config)

        logger.debug(f"Optimizing AnyBURL for profile: {self.profile.profile_name}")
        logger.debug(f"Hardware: RAM={self.profile.total_ram_gb:.1f}GB, cores={self.profile.cpu_cores}, GPU={'Yes' if self.profile.has_gpu else 'No'}")

        if self.profile.profile_name == "high_spec":
            if 'WORKER_THREADS' in current_config:
                current_threads = current_config['WORKER_THREADS']
                # 90% of threads, reserving 10% for safety
                optimal_threads = min(20, int(self.profile.cpu_threads * 0.9))
                optimized['WORKER_THREADS'] = optimal_threads
                logger.info(f"   Worker threads: {current_threads} → {optimal_threads}")

            if 'JAVA_HEAP' in current_config:
                # Use 50% of total RAM, reserving 10% for system
                heap_gb = min(28, int(self.profile.total_ram_gb * 0.5))
                optimized['JAVA_HEAP'] = f"{heap_gb}G"
                logger.info(f"   Java heap: {current_config.get('JAVA_HEAP', 'N/A')} → {optimized['JAVA_HEAP']}")

            if 'SAMPLE_SIZE' in current_config:
                current_sample = current_config['SAMPLE_SIZE']
                optimized['SAMPLE_SIZE'] = min(1200, int(current_sample * 2.0))
                logger.info(f"   Sample size: {current_sample} → {optimized['SAMPLE_SIZE']}")

            optimized['MAX_LENGTH_CYCLIC'] = optimized.get('MAX_LENGTH_CYCLIC', 4)
            optimized['MAX_LENGTH_ACYCLIC'] = optimized.get('MAX_LENGTH_ACYCLIC', 3)

        elif self.profile.profile_name == "mid_spec":
            if 'WORKER_THREADS' in current_config:
                current_threads = current_config['WORKER_THREADS']
                optimal_threads = min(16, int(self.profile.cpu_threads * 0.75))
                optimized['WORKER_THREADS'] = optimal_threads
                logger.info(f"   Worker threads: {current_threads} → {optimal_threads}")

            if 'JAVA_HEAP' in current_config:
                heap_gb = min(32, int(self.profile.total_ram_gb * 0.6))
                optimized['JAVA_HEAP'] = f"{heap_gb}G"
                logger.info(f"   Java heap: {current_config.get('JAVA_HEAP', 'N/A')} → {optimized['JAVA_HEAP']}")

            if 'SAMPLE_SIZE' in current_config:
                current_sample = current_config['SAMPLE_SIZE']
                optimized['SAMPLE_SIZE'] = min(600, current_sample * 1.2)
                logger.info(f"   Sample size: {current_sample} → {optimal_threads}")

        else:
            if 'WORKER_THREADS' in current_config:
                current_threads = current_config['WORKER_THREADS']
                optimal_threads = max(2, int(self.profile.cpu_threads * 0.5))
                optimized['WORKER_THREADS'] = optimal_threads
                logger.info(f"   Worker threads: {current_threads} → {optimal_threads}")

            if 'JAVA_HEAP' in current_config:
                heap_gb = min(16, int(self.profile.total_ram_gb * 0.5))
                optimized['JAVA_HEAP'] = f"{heap_gb}G"
                logger.info(f"   Java heap: {current_config.get('JAVA_HEAP', 'N/A')} → {optimized['JAVA_HEAP']}")

        return optimized


class PyClausePerformanceOptimizer:
    """Optimize PyClause parameters for better ranking performance."""

    def __init__(self):
        self.profile = HardwareDetector.detect()
        self.resource_mgr = ResourceManager()
        self.analyzer = DataAnalyzer()

    def optimize_parameters(
        self,
        current_config: dict[str, Any],
        test_data_path: Path | None = None,
    ) -> dict[str, Any]:
        """
        Optimize PyClause ranking parameters.

        Args:
            current_config: Current PyClause configuration
            test_data_path: Path to test data

        Returns:
            Optimized configuration
        """
        import copy
        optimized = copy.deepcopy(current_config)

        logger.debug("Optimizing PyClause parameters")
        logger.debug(f"Profile: {self.profile.profile_name}")

        ranking_config = optimized.get('ranking_handler', {})

        if self.profile.profile_name == "high_spec":
            current_threads = ranking_config.get('num_threads', 1)
            optimal_threads = min(8, max(4, self.profile.cpu_threads // 2))
            ranking_config['num_threads'] = optimal_threads
            logger.debug(f"Ranking threads: {current_threads} -> {optimal_threads}")

            ranking_config['aggregation_function'] = ranking_config.get('aggregation_function', 'maxplus')

        elif self.profile.profile_name == "mid_spec":
            current_threads = ranking_config.get('num_threads', 1)
            optimal_threads = min(4, max(2, self.profile.cpu_threads // 4))
            ranking_config['num_threads'] = optimal_threads
            logger.debug(f"Ranking threads: {current_threads} -> {optimal_threads}")

            ranking_config['aggregation_function'] = ranking_config.get('aggregation_function', 'noisyor')

        else:
            current_threads = ranking_config.get('num_threads', 1)
            ranking_config['num_threads'] = max(1, min(2, self.profile.cpu_threads // 4))
            logger.debug(f"Ranking threads: {current_threads} -> {ranking_config['num_threads']}")

            ranking_config['aggregation_function'] = 'noisyor'

        optimized['ranking_handler'] = ranking_config
        return optimized


class IntelligentChunking:
    """Intelligent data chunking for better performance."""

    @staticmethod
    def calculate_optimal_chunk_size(
        total_triples: int,
        profile: HardwareProfile,
    ) -> int:
        """
        Calculate optimal chunk size for parallel processing.

        Args:
            total_triples: Total number of triples
            profile: Hardware profile

        Returns:
            Optimal chunk size
        """
        if profile.profile_name == "high_spec":
            base_chunk = 1000
        elif profile.profile_name == "mid_spec":
            base_chunk = 500
        else:
            base_chunk = 200

        chunk_size = min(base_chunk, total_triples // max(1, profile.cpu_threads))
        chunk_size = max(100, chunk_size)

        logger.info(f"   Optimal chunk size: {chunk_size} triples")
        return chunk_size


class UnifiedPerformanceOptimizer:
    """Unified performance optimizer for entire KG pipeline."""

    def __init__(self):
        self.anyburl_opt = AnyBURLPerformanceOptimizer()
        self.pyclause_opt = PyClausePerformanceOptimizer()
        self.chunker = IntelligentChunking()
        self.analyzer = DataAnalyzer()

        self.cache_mgr = CacheManager(
            cache_dir=settings.CACHE_DIR / "kg_optimization",
            max_memory_items=1000
        )

    def optimize_pipeline(
        self,
        pipeline_config: dict[str, Any],
        train_data_path: Path | None = None,
        test_data_path: Path | None = None,
    ) -> dict[str, Any]:
        """
        Optimize entire KG pipeline using existing PFF utilities.

        Args:
            pipeline_config: Full pipeline configuration
            train_data_path: Path to training data
            test_data_path: Path to test data

        Returns:
            Optimized pipeline configuration
        """
        optimized = pipeline_config.copy()

        if 'anyburl' in pipeline_config:
            optimized['anyburl'] = self.anyburl_opt.optimize_parameters(
                pipeline_config['anyburl'],
                train_data_path,
            )

        if 'pyclause' in pipeline_config:
            optimized['pyclause'] = self.pyclause_opt.optimize_parameters(
                pipeline_config['pyclause'],
                test_data_path,
            )

        if train_data_path and test_data_path:
            train_count = self.analyzer.estimate_triple_count(train_data_path)
            test_count = self.analyzer.estimate_triple_count(test_data_path)

            profile = HardwareDetector.detect()

            if 'pipeline' not in optimized:
                optimized['pipeline'] = {}

            optimized['pipeline']['chunk_size'] = self.chunker.calculate_optimal_chunk_size(
                test_count,
                profile,
            )

            if train_count > 1000000:
                optimized['pipeline']['preprocess'] = optimized['pipeline'].get('preprocess', {})
                optimized['pipeline']['preprocess']['homogeneity_level'] = 0.7

            optimized['pipeline']['num_workers'] = min(
                profile.cpu_threads,
                int(profile.cpu_threads * 0.8),
            )

        logger.success(" Pipeline optimization complete!")
        return optimized


if __name__ == "__main__":
    print("SOTA Performance Optimizer for AnyBURL & PyClause v2.0")
    print("Reuses existing PFF utilities:")
    print("  - HardwareDetector (pff.utils.system.hardware_detector)")
    print("  - CacheManager (pff.utils.core.cache)")
    print("  - ResourceManager (pff.utils.system.resource_manager)")
    print("=" * 60)
