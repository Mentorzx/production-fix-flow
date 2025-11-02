"""
Determinism utilities for reproducible training.

Sprint 31: Fix non-deterministic behavior across runs.
"""

import os
import random

import numpy as np


def set_global_seed(seed: int = 42) -> None:
    """
    Set random seeds for all libraries to ensure reproducibility.
    
    Args:
        seed: Random seed value (default: 42)
    """
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch (if available)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    
    # XGBoost (set via parameter, not global)
    # Will be set in XGBClassifier(random_state=seed)
    
    # Scikit-learn (set via parameter)
    # Will be set in train_test_split(random_state=seed)
    
    # Environment variables for additional determinism
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # PyTorch determinism
    
    # Numba (parallel execution order)
    # Note: Numba parallel may still have non-determinism in reduction operations
    # Best to use parallel=False for reproducibility or ensure vocabulary is pre-built
    os.environ['NUMBA_DISABLE_PERFORMANCE_WARNINGS'] = '1'


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
        set_global_seed(42)  # Reset seed before each run
        result = func(*args, **kwargs)
        results.append(result)
    
    # Check all results are equal (within tolerance)
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


# Set global seed on module import
set_global_seed(42)
