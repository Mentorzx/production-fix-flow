#!/usr/bin/env python3
"""Benchmark script for vectorized masking functions.

Measures performance of:
- _mask_known_tails()
- _build_inbatch_known_positive_mask()

The key optimization is eliminating per-row .item() calls that cause GPU->CPU syncs.
On CPU, the benefit is eliminating Python loop overhead.

Usage:
    python scripts/bench_vectorization.py
"""

from __future__ import annotations

import statistics
import time
import warnings
from unittest.mock import MagicMock

import numpy as np
import torch

warnings.filterwarnings("ignore")


def create_manager():
    """Create a DSLFMKGCManager with mock dependencies."""
    from pff.domain.learning.dslfm.dslfm_kgc import DSLFMKGCConfig
    from pff.domain.learning.dslfm.kgc_manager import (
        DSLFMKGCManager,
        KGCTrainingConfig,
    )

    mock_persistence = MagicMock()
    mock_persistence.save_checkpoint = MagicMock()
    mock_persistence.load_checkpoint = MagicMock(return_value=None)

    manager = DSLFMKGCManager(
        model_config=DSLFMKGCConfig(num_entities=10000, num_relations=50),
        training_config=KGCTrainingConfig(use_compile=False),
        persistence_port=mock_persistence,
        device=torch.device("cpu"),
    )
    return manager


def setup_filter_arrays(manager, num_keys: int, avg_tails_per_key: int):
    """Populate filter arrays with synthetic data."""
    rng = np.random.default_rng(42)
    manager._filter_arrays = {}
    manager._filter_tensors = {}

    for _ in range(num_keys):
        h = rng.integers(0, 100)
        r = rng.integers(0, 50)
        num_tails = rng.integers(1, avg_tails_per_key * 2)
        tails = rng.choice(10000, size=num_tails, replace=False).astype(np.int64)
        manager._filter_arrays[(h, r)] = np.sort(tails)


def bench_mask_known_tails(manager, batch_size: int, num_candidates: int, iterations: int):
    """Benchmark _mask_known_tails."""
    rng = np.random.default_rng(123)

    keys = list(manager._filter_arrays.keys())
    if not keys:
        return float("nan"), float("nan")

    times = []
    for _ in range(iterations):
        key_indices = rng.choice(len(keys), size=batch_size, replace=True)
        h_list, r_list = [], []
        for ki in key_indices:
            h_list.append(keys[ki][0])
            r_list.append(keys[ki][1])

        h = torch.tensor(h_list, dtype=torch.long)
        r = torch.tensor(r_list, dtype=torch.long)
        t = torch.randint(0, num_candidates, (batch_size,), dtype=torch.long)
        candidates = torch.arange(0, num_candidates, dtype=torch.long)
        scores = torch.zeros((batch_size, num_candidates), dtype=torch.float32)

        start = time.perf_counter()
        manager._mask_known_tails(scores, h, r, candidates, t)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    times_ms = [t * 1000 for t in times[2:]]
    return statistics.mean(times_ms), statistics.stdev(times_ms) if len(times_ms) > 1 else 0


def bench_build_inbatch_mask(manager, batch_size: int, iterations: int):
    """Benchmark _build_inbatch_known_positive_mask."""
    rng = np.random.default_rng(456)

    keys = list(manager._filter_arrays.keys())
    if not keys:
        return float("nan"), float("nan")

    times = []
    for _ in range(iterations):
        key_indices = rng.choice(len(keys), size=batch_size, replace=True)
        h_list, r_list = [], []
        for ki in key_indices:
            h_list.append(keys[ki][0])
            r_list.append(keys[ki][1])

        h = torch.tensor(h_list, dtype=torch.long)
        r = torch.tensor(r_list, dtype=torch.long)
        t = torch.randint(0, 10000, (batch_size,), dtype=torch.long)

        start = time.perf_counter()
        manager._build_inbatch_known_positive_mask(h, r, t)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    times_ms = [t * 1000 for t in times[2:]]
    return statistics.mean(times_ms), statistics.stdev(times_ms) if len(times_ms) > 1 else 0


def main():
    print("=" * 70)
    print("Vectorization Benchmark (CPU baseline - GPU benefits are larger)")
    print("=" * 70)

    manager = create_manager()
    setup_filter_arrays(manager, num_keys=500, avg_tails_per_key=50)

    configs = [
        {"batch_size": 64, "num_candidates": 1000, "iterations": 100},
        {"batch_size": 256, "num_candidates": 5000, "iterations": 50},
        {"batch_size": 512, "num_candidates": 10000, "iterations": 30},
    ]

    print("\n_mask_known_tails benchmark:")
    print("-" * 70)
    print(f"{'Config':<35} {'Mean (ms)':>12} {'Std (ms)':>12}")
    print("-" * 70)
    for cfg in configs:
        mean_ms, std_ms = bench_mask_known_tails(
            manager,
            cfg["batch_size"],
            cfg["num_candidates"],
            cfg["iterations"],
        )
        config_str = f"batch={cfg['batch_size']}, candidates={cfg['num_candidates']}"
        print(f"{config_str:<35} {mean_ms:>12.3f} {std_ms:>12.3f}")

    print("\n_build_inbatch_known_positive_mask benchmark:")
    print("-" * 70)
    print(f"{'Config':<35} {'Mean (ms)':>12} {'Std (ms)':>12}")
    print("-" * 70)
    for cfg in configs:
        mean_ms, std_ms = bench_build_inbatch_mask(
            manager,
            cfg["batch_size"],
            cfg["iterations"],
        )
        config_str = f"batch={cfg['batch_size']}"
        print(f"{config_str:<35} {mean_ms:>12.3f} {std_ms:>12.3f}")

    print("\n" + "=" * 70)
    print("Note: True benefit is on GPU where .item() syncs are eliminated")
    print("=" * 70)


if __name__ == "__main__":
    main()
