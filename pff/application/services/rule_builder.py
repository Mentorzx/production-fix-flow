"""
Rule Builder and Factory - DEPRECATED LOCATION.

This module re-exports from the new location for backward compatibility.
Please update imports to use:
    from pff.application.services.business_service.shared import RuleBuilder, RuleSourceFactory
"""

from pff.application.services.business_service.shared.rule_builder import (
    ManualRuleSource,
    Rule,
    RuleBuilder,
    RuleSource,
    RuleSourceFactory,
    _parse_pattern,
)

__all__ = [
    "Rule",
    "RuleBuilder",
    "RuleSource",
    "ManualRuleSource",
    "RuleSourceFactory",
    "_parse_pattern",
]
