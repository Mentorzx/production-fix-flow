"""
Symbolic Feature Processing Module

This module contains specialized processors for symbolic feature extraction,
implementing various design patterns for better organization and maintainability.

Patterns Used:
- Strategy: Different processing strategies (Numba, Parallel, Indexed)
- Factory: Creates appropriate processors based on configuration
- Command: Encapsulates debugging and validation operations
"""

from .base import (
    FeatureProcessor,
    ProcessingResult,
    ProcessingStrategy,
    DebugCommand,
    ValidationCommand,
)

from .strategies import (
    NumbaProcessingStrategy,
    ParallelProcessingStrategy,
    IndexedProcessingStrategy,
    SequentialProcessingStrategy,
)

from .factory import ProcessorFactory
from .config import ProcessorConfig, ProcessorConfigBuilder
from .debug import DebugManager
from .validation import ValidationManager
from .symbolic_processor import SymbolicFeatureProcessorV2

__all__ = [
    "FeatureProcessor",
    "ProcessingResult",
    "ProcessingStrategy",
    "DebugCommand",
    "ValidationCommand",
    "NumbaProcessingStrategy",
    "ParallelProcessingStrategy",
    "IndexedProcessingStrategy",
    "SequentialProcessingStrategy",
    "ProcessorFactory",
    "ProcessorConfig",
    "ProcessorConfigBuilder",
    "DebugManager",
    "ValidationManager",
    "SymbolicFeatureProcessorV2",
]