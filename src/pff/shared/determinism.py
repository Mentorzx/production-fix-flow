"""
Determinism utilities for reproducible training.
"""

import os
import random

import numpy as np


def configure_torch_determinism(*, enforce: bool = True) -> None:
    """Configure PyTorch determinism knobs without changing RNG state."""
    try:
        import torch
    except ImportError:
        return

    os.environ.setdefault(
        "CUBLAS_WORKSPACE_CONFIG",
        os.getenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8"),
    )

    try:
        torch.use_deterministic_algorithms(enforce)
    except (RuntimeError, AttributeError):
        pass

    if hasattr(torch, "set_float32_matmul_precision"):
        try:
            torch.set_float32_matmul_precision("highest")
        except (RuntimeError, ValueError):
            pass

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def configure_numba_threads() -> int:
    """Configure Numba threads based on physical cores."""
    try:
        from pff.shared.system.resource_manager import (
            configure_numba_threads as _configure,
        )
    except (ImportError, OSError):
        return 0
    return _configure()


def set_global_seed(seed: int = 42) -> None:
    """
    Set random seeds for all libraries to ensure reproducibility.

    Args:
        seed: Random seed value (default: 42)
    Note:
        PYTHONHASHSEED must be set before interpreter start to affect this process.
        Setting it here only impacts subprocesses spawned after this call.
    """
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    configure_torch_determinism(enforce=True)

    os.environ["PYTHONHASHSEED"] = str(seed)

    disable_warnings = os.getenv("NUMBA_DISABLE_PERFORMANCE_WARNINGS")
    if disable_warnings is None:
        disable_warnings = "1"
    os.environ.setdefault("NUMBA_DISABLE_PERFORMANCE_WARNINGS", disable_warnings)


def validate_determinism(func, *args, n_runs: int = 3, tolerance: float = 1e-6, **kwargs):
    """
    Validate that a function produces deterministic results.

    Args:
        func: Function to test
        args: Positional arguments for func
        n_runs: Number of runs to validate
        tolerance: Maximum allowed difference between runs
        kwargs: Keyword arguments for func

    Returns:
        bool: True if deterministic, False otherwise

    Raises:
        AssertionError: If results are not deterministic
    """
    results = []
    for i in range(n_runs):
        set_global_seed(42)
        result = func(*args, **kwargs)
        results.append(result)

    for i in range(1, n_runs):
        if isinstance(results[0], (int, float)):
            diff = abs(results[i] - results[0])
            assert diff < tolerance, f"Run {i} differs from run 0: {diff} > {tolerance}"
        elif isinstance(results[0], np.ndarray):
            diff = np.max(np.abs(results[i] - results[0]))
            assert diff < tolerance, f"Run {i} differs from run 0: {diff} > {tolerance}"
        else:
            assert results[i] == results[0], f"Run {i} differs from run 0"

    return True
