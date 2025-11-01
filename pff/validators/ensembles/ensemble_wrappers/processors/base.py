"""
Base classes and interfaces for symbolic feature processing.

This module defines the abstract base classes and common interfaces
used throughout the processing system.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray


@dataclass
class ProcessingResult:
    """Result of a processing operation with metadata."""

    data: NDArray[np.int8] | list[NDArray[np.int8]]
    success: bool
    metadata: dict[str, Any]
    processing_time: float = 0.0
    error_message: str | None = None

    def is_valid(self) -> bool:
        """Check if the result is valid and usable."""
        return self.success and self.error_message is None


class FeatureProcessor(ABC):
    """Abstract base class for feature processors."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self._is_fitted = False

    @abstractmethod
    def fit(self, X: list[Any], y: Any = None) -> FeatureProcessor:
        """Fit the processor to the data."""
        pass

    @abstractmethod
    def transform(self, X: list[Any]) -> NDArray[np.int8]:
        """Transform the input data."""
        pass

    @abstractmethod
    def get_stats(self) -> dict[str, Any]:
        """Get processing statistics."""
        pass

    def fit_transform(self, X: list[Any], y: Any = None) -> NDArray[np.int8]:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)

    def is_fitted(self) -> bool:
        """Check if the processor has been fitted."""
        return self._is_fitted


class ProcessingStrategy(ABC):
    """Abstract base class for processing strategies."""

    @abstractmethod
    def can_process(self, data: list[Any], config: dict[str, Any]) -> bool:
        """Check if this strategy can process the given data."""
        pass

    @abstractmethod
    def process(self, data: list[Any], config: dict[str, Any]) -> ProcessingResult:
        """Process the data using this strategy."""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get the name of this strategy."""
        pass


class Command(ABC):
    """Abstract base class for command objects."""

    @abstractmethod
    def execute(self, context: dict[str, Any]) -> Any:
        """Execute the command."""
        pass


class DebugCommand(Command):
    """Command for debugging operations."""

    @abstractmethod
    def save_debug_info(self, data: Any, error: Exception, context: dict[str, Any]) -> None:
        """Save debug information."""
        pass


class ValidationCommand(Command):
    """Command for validation operations."""

    @abstractmethod
    def validate(self, data: Any, context: dict[str, Any]) -> bool:
        """Validate the data."""
        pass


class ProcessorRegistry(Protocol):
    """Protocol for processor registry."""

    def register_strategy(self, name: str, strategy: ProcessingStrategy) -> None:
        """Register a processing strategy."""
        pass

    def get_strategy(self, name: str) -> ProcessingStrategy | None:
        """Get a registered strategy."""
        pass

    def list_strategies(self) -> list[str]:
        """List all registered strategies."""
        pass


class Configurable(Protocol):
    """Protocol for configurable objects."""

    def configure(self, **kwargs) -> None:
        """Configure the object."""
        pass

    def get_config(self) -> dict[str, Any]:
        """Get current configuration."""
        pass