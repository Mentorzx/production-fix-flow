"""
Hierarchical Ensemble Configuration Loader.

This module provides configuration loading for the hierarchical ensemble
architecture, including defaults for backward compatibility and deep merging.

Design Patterns Applied:
    - **Factory Pattern:** Creates HierarchicalConfig from YAML or defaults.
    - **Builder Pattern:** Deep merge of user config with defaults.
    - **Null Object Pattern:** Returns safe defaults when config is missing.

Configuration Hierarchy:
    1. Built-in defaults (this module)
    2. config/models/hierarchical_ensemble.yaml (user overrides)
    3. Runtime overrides (HPO trials)

Usage:
    from pff.validators.ensembles.hierarchical import load_hierarchical_config

    config = load_hierarchical_config()
    if config.is_hierarchical:
        # Use hierarchical pathway
        ...
    else:
        # Use flat ensemble (backward compatible)
        ...
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pff.config import HIERARCHICAL_ENSEMBLE_CONFIG_PATH
from pff.utils.core.file_manager import FileManager
from pff.utils.logger import logger


@dataclass
class AggregatorConfig:
    """Configuration for an aggregator (symbolic or neural).

    Attributes:
        strategy: Aggregation strategy name (noisy_or, max_confidence, etc.)
        params: Strategy-specific parameters.
    """

    strategy: str = "noisy_or"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class DecisionRouterConfig:
    """Configuration for the decision router.

    Attributes:
        symbolic_confidence_threshold: Min confidence for SYMBOLIC_DECIDES.
        symbolic_low_threshold: Max symbolic for NEURAL_FALLBACK consideration.
        neural_confidence_threshold: Min neural score for NEURAL_FALLBACK.
        blend_weight_symbolic: Symbolic weight in BLEND mode.
        blend_weight_neural: Neural weight in BLEND mode.
    """

    symbolic_confidence_threshold: float = 0.70
    symbolic_low_threshold: float = 0.30
    neural_confidence_threshold: float = 0.50
    blend_weight_symbolic: float = 0.6
    blend_weight_neural: float = 0.4

    @property
    def symbolic_high_threshold(self) -> float:
        """Alias for symbolic_confidence_threshold (backward compatibility)."""
        return self.symbolic_confidence_threshold

    @property
    def neural_fallback_threshold(self) -> float:
        """Alias for neural_confidence_threshold (backward compatibility)."""
        return self.neural_confidence_threshold


@dataclass
class NeuralAggregatorConfig:
    """Extended configuration for neural aggregator.

    Attributes:
        strategy: Aggregation strategy name.
        params: Strategy-specific parameters.
        entropy_based_confidence: Use entropy for confidence calculation.
    """

    strategy: str = "weighted_average"
    params: dict[str, Any] = field(default_factory=dict)
    entropy_based_confidence: bool = True


@dataclass
class PenaltiesConfig:
    """Configuration for penalty application.

    Attributes:
        symbolic_dominance_enabled_in_flat: Apply penalty in flat mode.
        symbolic_dominance_enabled_in_hierarchical: Apply penalty in hierarchical mode.
        symbolic_dominance_threshold: Symbolic ratio triggering penalty.
        symbolic_dominance_penalty_factor: Multiplier when triggered.
    """

    symbolic_dominance_enabled_in_flat: bool = True
    symbolic_dominance_enabled_in_hierarchical: bool = False
    symbolic_dominance_threshold: float = 0.95
    symbolic_dominance_penalty_factor: float = 0.2


@dataclass
class HierarchicalConfig:
    """Complete hierarchical ensemble configuration.

    Attributes:
        architecture_type: 'flat' (default, backward compatible) or 'hierarchical'.
        symbolic_aggregator: Configuration for symbolic aggregation.
        neural_aggregator: Configuration for neural aggregation.
        decision_router: Configuration for routing decisions.
        penalties: Penalty configuration.
        raw_config: Original YAML dict for passthrough access.
    """

    architecture_type: str = "flat"
    symbolic_aggregator: AggregatorConfig = field(default_factory=AggregatorConfig)
    neural_aggregator: NeuralAggregatorConfig = field(
        default_factory=NeuralAggregatorConfig
    )
    decision_router: DecisionRouterConfig = field(default_factory=DecisionRouterConfig)
    penalties: PenaltiesConfig = field(default_factory=PenaltiesConfig)
    raw_config: dict[str, Any] = field(default_factory=dict)

    @property
    def is_hierarchical(self) -> bool:
        """Check if hierarchical mode is enabled."""
        return self.architecture_type == "hierarchical"

    @property
    def is_flat(self) -> bool:
        """Check if flat mode is enabled (default)."""
        return self.architecture_type == "flat"

    @property
    def should_apply_symbolic_dominance_penalty(self) -> bool:
        """Determine if symbolic dominance penalty should be applied.

        In flat mode, penalty is applied to counteract modality collapse.
        In hierarchical mode, penalty is disabled as the architecture
        already handles modality separation.

        Returns:
            bool: True if penalty should be applied.
        """
        if self.is_hierarchical:
            return self.penalties.symbolic_dominance_enabled_in_hierarchical
        return self.penalties.symbolic_dominance_enabled_in_flat


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge override dict into base dict.

    Args:
        base: Base dictionary with defaults.
        override: Override dictionary with user values.

    Returns:
        Merged dictionary with override values taking precedence.
    """
    result = base.copy()
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _get_defaults() -> dict[str, Any]:
    """Return built-in defaults for hierarchical ensemble config.

    These defaults ensure backward compatibility when config file
    is missing or incomplete.

    Returns:
        dict: Complete default configuration.
    """
    return {
        "architecture": {
            "type": "flat",
            "description": "flat=backward compatible, hierarchical=modality separation",
        },
        "decision_router": {
            "thresholds": {
                "symbolic_confidence": 0.70,
                "neural_confidence": 0.50,
            },
            "blend_weights": {
                "symbolic": 0.6,
                "neural": 0.4,
            },
            "symbolic_low_threshold": 0.30,
        },
        "aggregators": {
            "symbolic": {
                "strategy": "noisy_or",
                "params": {
                    "base_confidence": 0.01,
                    "max_rules": 1500,
                    "min_confidence": 0.30,
                },
            },
            "neural": {
                "strategy": "weighted_average",
                "params": {
                    "normalize_weights": True,
                },
                "entropy_based_confidence": True,
            },
        },
        "penalties": {
            "symbolic_dominance": {
                "enabled_in_flat": True,
                "enabled_in_hierarchical": False,
                "threshold": 0.95,
                "penalty_factor": 0.2,
            },
        },
        "calibration": {
            "enabled": False,
            "method": "isotonic",
        },
        "logging": {
            "log_routing_decisions": False,
            "log_aggregation_details": False,
        },
    }


def _parse_config(raw: dict[str, Any]) -> HierarchicalConfig:
    """Parse raw config dict into HierarchicalConfig dataclass.

    Args:
        raw: Raw configuration dictionary.

    Returns:
        HierarchicalConfig: Parsed configuration object.
    """
    arch_type = raw.get("architecture", {}).get("type", "flat")

    symbolic_agg_raw = raw.get("aggregators", {}).get("symbolic", {})
    symbolic_agg = AggregatorConfig(
        strategy=symbolic_agg_raw.get("strategy", "noisy_or"),
        params=symbolic_agg_raw.get("params", {}),
    )

    neural_agg_raw = raw.get("aggregators", {}).get("neural", {})
    neural_agg = NeuralAggregatorConfig(
        strategy=neural_agg_raw.get("strategy", "weighted_average"),
        params=neural_agg_raw.get("params", {}),
        entropy_based_confidence=neural_agg_raw.get("entropy_based_confidence", True),
    )

    router_raw = raw.get("decision_router", {})
    thresholds = router_raw.get("thresholds", {})
    blend_weights = router_raw.get("blend_weights", {})
    decision_router = DecisionRouterConfig(
        symbolic_confidence_threshold=thresholds.get("symbolic_confidence", 0.70),
        symbolic_low_threshold=router_raw.get("symbolic_low_threshold", 0.30),
        neural_confidence_threshold=thresholds.get("neural_confidence", 0.50),
        blend_weight_symbolic=blend_weights.get("symbolic", 0.6),
        blend_weight_neural=blend_weights.get("neural", 0.4),
    )

    penalties_raw = raw.get("penalties", {}).get("symbolic_dominance", {})
    penalties = PenaltiesConfig(
        symbolic_dominance_enabled_in_flat=penalties_raw.get("enabled_in_flat", True),
        symbolic_dominance_enabled_in_hierarchical=penalties_raw.get(
            "enabled_in_hierarchical", False
        ),
        symbolic_dominance_threshold=penalties_raw.get("threshold", 0.95),
        symbolic_dominance_penalty_factor=penalties_raw.get("penalty_factor", 0.2),
    )

    return HierarchicalConfig(
        architecture_type=arch_type,
        symbolic_aggregator=symbolic_agg,
        neural_aggregator=neural_agg,
        decision_router=decision_router,
        penalties=penalties,
        raw_config=raw,
    )


def load_hierarchical_config(
    config_path: Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> HierarchicalConfig:
    """Load hierarchical ensemble configuration.

    Loads configuration from YAML file with fallback to defaults.
    Supports runtime overrides for HPO trials.

    Args:
        config_path: Path to config file. Defaults to HIERARCHICAL_ENSEMBLE_CONFIG_PATH.
        overrides: Runtime overrides to apply on top of loaded config.

    Returns:
        HierarchicalConfig: Parsed and validated configuration.

    Example:
        >>> config = load_hierarchical_config()
        >>> if config.is_hierarchical:
        ...     # Use hierarchical pathway
        ...     pass
    """
    path = config_path or HIERARCHICAL_ENSEMBLE_CONFIG_PATH
    defaults = _get_defaults()

    user_config: dict[str, Any] = {}
    if path.exists():
        try:
            user_config = FileManager.read(path) or {}
            logger.debug(f"Hierarchical config loaded from {path}")
        except Exception as exc:
            logger.warning(
                f"Failed to load hierarchical config from {path}: {exc}; using defaults"
            )
    else:
        logger.debug(
            f"Hierarchical config not found at {path}; using defaults (flat mode)"
        )

    merged = _deep_merge(defaults, user_config)

    if overrides:
        merged = _deep_merge(merged, overrides)
        logger.debug("Applied runtime overrides to hierarchical config")

    return _parse_config(merged)


def get_architecture_type(
    config_path: Path | None = None,
) -> str:
    """Quick check for architecture type without full config parsing.

    Useful for early branching decisions in pipeline code.

    Args:
        config_path: Path to config file.

    Returns:
        str: 'flat' or 'hierarchical'.
    """
    config = load_hierarchical_config(config_path)
    return config.architecture_type
