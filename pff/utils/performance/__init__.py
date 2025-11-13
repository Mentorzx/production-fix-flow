"""
Performance SOTA utilities.

PyTorch 2.5.1+ optimizations, CUDA memory management, and distributed computing.
"""

from pff.utils.performance.performance import apply_sota_optimizations
from pff.utils.performance.observability import ObservabilityManager

__all__ = [
    "PerformanceOptimizer",
    "apply_sota_optimizations",
    "ObservabilityManager",
    "setup_observability",
]
