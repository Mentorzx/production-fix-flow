"""
Shared Utilities for Business Service.

This subpackage contains shared utilities and patterns used by the
business service modules.

Modules:
    - rule_builder: Builder/Factory patterns for Rule construction
    - validation_observer: Observer pattern for validation events
    - violation_penalty: Penalty calculator for rule violations
"""

from .rule_builder import (
    ManualRuleSource,
    Rule,
    RuleBuilder,
    RuleSource,
    RuleSourceFactory,
)
from .validation_observer import (
    CompositeValidationObserver,
    LoggingValidationObserver,
    MetricsValidationObserver,
    ValidationEvent,
    ValidationEventType,
    ValidationObserver,
)
from .violation_penalty import (
    PenaltyConfig,
    ViolationPenaltyCalculator,
)

__all__ = [
    "Rule",
    "RuleBuilder",
    "RuleSource",
    "ManualRuleSource",
    "RuleSourceFactory",
    "ValidationEventType",
    "ValidationEvent",
    "ValidationObserver",
    "LoggingValidationObserver",
    "MetricsValidationObserver",
    "CompositeValidationObserver",
    "PenaltyConfig",
    "ViolationPenaltyCalculator",
]
