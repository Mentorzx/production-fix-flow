"""
Configuration classes for symbolic feature processors.

This module provides configuration classes using the Builder pattern
for flexible and type-safe configuration management.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ProcessorConfig:
    """Configuration for symbolic feature processors."""

    # Basic configuration
    rules_path: str
    min_confidence_threshold: float = 0.0

    # Feature grouping configuration
    enable_grouping: bool = False
    n_groups: int = 50
    boost_factor: float = 10.0

    # Performance configuration
    enable_rule_indexing: bool = True
    enable_numba: bool = True
    parallel_threshold: int = 100

    # Debug configuration
    enable_debug: bool = False
    debug_output_dir: str = "debug"

    # Advanced configuration
    batch_size: int = 1000
    max_workers: int | None = None
    cache_enabled: bool = True

    # Context variables
    use_context_violations: bool = True

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.min_confidence_threshold < 0 or self.min_confidence_threshold > 1:
            raise ValueError("min_confidence_threshold must be between 0 and 1")

        if self.n_groups <= 0:
            raise ValueError("n_groups must be positive")

        if self.boost_factor <= 0:
            raise ValueError("boost_factor must be positive")


class ProcessorConfigBuilder:
    """Builder for ProcessorConfig with fluent interface."""

    def __init__(self, rules_path: str):
        self._config = {
            "rules_path": rules_path,
            "min_confidence_threshold": 0.0,
            "enable_grouping": False,
            "n_groups": 50,
            "boost_factor": 10.0,
            "enable_rule_indexing": True,
            "enable_numba": True,
            "parallel_threshold": 100,
            "enable_debug": False,
            "debug_output_dir": "debug",
            "batch_size": 1000,
            "max_workers": None,
            "cache_enabled": True,
            "use_context_violations": True,
        }

    def with_confidence_threshold(self, threshold: float) -> ProcessorConfigBuilder:
        """Set minimum confidence threshold."""
        self._config["min_confidence_threshold"] = threshold
        return self

    def with_feature_grouping(
        self,
        enabled: bool = True,
        n_groups: int = 50,
        boost_factor: float = 10.0
    ) -> ProcessorConfigBuilder:
        """Configure feature grouping."""
        self._config["enable_grouping"] = enabled
        self._config["n_groups"] = n_groups
        self._config["boost_factor"] = boost_factor
        return self

    def with_rule_indexing(self, enabled: bool = True) -> ProcessorConfigBuilder:
        """Enable or disable rule indexing."""
        self._config["enable_rule_indexing"] = enabled
        return self

    def with_numba(self, enabled: bool = True) -> ProcessorConfigBuilder:
        """Enable or disable Numba acceleration."""
        self._config["enable_numba"] = enabled
        return self

    def with_performance(
        self,
        parallel_threshold: int = 100,
        batch_size: int = 1000,
        max_workers: int | None = None,
        cache_enabled: bool = True,
    ) -> ProcessorConfigBuilder:
        """Configure performance settings."""
        self._config["parallel_threshold"] = parallel_threshold
        self._config["batch_size"] = batch_size
        self._config["max_workers"] = max_workers
        self._config["cache_enabled"] = cache_enabled
        return self

    def with_debug(
        self,
        enabled: bool = True,
        output_dir: str = "debug",
    ) -> ProcessorConfigBuilder:
        """Configure debug settings."""
        self._config["enable_debug"] = enabled
        self._config["debug_output_dir"] = output_dir
        return self

    def with_context_violations(self, enabled: bool = True) -> ProcessorConfigBuilder:
        """Enable or disable context violations."""
        self._config["use_context_violations"] = enabled
        return self

    def build(self) -> ProcessorConfig:
        """Build the ProcessorConfig."""
        return ProcessorConfig(**self._config)


def create_default_config(rules_path: str) -> ProcessorConfig:
    """Create a default configuration for the given rules path."""
    return ProcessorConfigBuilder(rules_path).build()


def create_high_performance_config(rules_path: str) -> ProcessorConfig:
    """Create a high-performance configuration."""
    return (
        ProcessorConfigBuilder(rules_path)
        .with_feature_grouping(enabled=True, n_groups=100)
        .with_numba(enabled=True)
        .with_performance(
            parallel_threshold=50,
            batch_size=2000,
            cache_enabled=True,
        )
        .build()
    )


def create_debug_config(rules_path: str, debug_dir: str = "debug") -> ProcessorConfig:
    """Create a configuration optimized for debugging."""
    return (
        ProcessorConfigBuilder(rules_path)
        .with_debug(enabled=True, output_dir=debug_dir)
        .with_performance(parallel_threshold=1000)  # Less parallel for easier debugging
        .build()
    )