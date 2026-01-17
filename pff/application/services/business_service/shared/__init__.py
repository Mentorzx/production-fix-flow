"""
Shared Utilities for Business Service.

This subpackage contains shared utilities and patterns used by the
business service modules.

Modules:
    - rule_builder: Builder/Factory patterns for Rule construction (AnyBURL deprecated)
    - validation_observer: Observer pattern for validation events
    - violation_penalty: Penalty calculator for rule violations
"""

from .rule_builder import (
    Rule,
    RuleBuilder,
    RuleSource,
    ManualRuleSource,
    RuleSourceFactory,
)
from .validation_observer import (
    ValidationEventType,
    ValidationEvent,
    ValidationObserver,
    LoggingValidationObserver,
    MetricsValidationObserver,
    CompositeValidationObserver,
)
from .violation_penalty import (
    PenaltyConfig,
    ViolationPenaltyCalculator,
)

__all__ = [
    # Rule Builder
    "Rule",
    "RuleBuilder",
    "RuleSource",
    "ManualRuleSource",
    "RuleSourceFactory",
    # Validation Observer
    "ValidationEventType",
    "ValidationEvent",
    "ValidationObserver",
    "LoggingValidationObserver",
    "MetricsValidationObserver",
    "CompositeValidationObserver",
    # Violation Penalty
    "PenaltyConfig",
    "ViolationPenaltyCalculator",
]
