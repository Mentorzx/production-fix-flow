"""
Rule Builder and Factory - DEPRECATED LOCATION.

This module re-exports from the new location for backward compatibility.
Please update imports to use:
    from pff.services.business_service.shared import RuleBuilder, RuleSourceFactory
"""

from pff.services.business_service.shared.rule_builder import (
    Rule,
    RuleBuilder,
    RuleSource,
    ManualRuleSource,
    AnyBURLRuleSource,
    RuleSourceFactory,
    _parse_pattern,
)

__all__ = [
    "Rule",
    "RuleBuilder",
    "RuleSource",
    "ManualRuleSource",
    "AnyBURLRuleSource",
    "RuleSourceFactory",
    "_parse_pattern",
]
