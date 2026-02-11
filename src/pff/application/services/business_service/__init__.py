"""
Business Service Package - Core Rule Validation and KG Operations.

This package provides the core business logic for rule-based validation,
knowledge graph operations, and MSISDN processing in the PFF platform.

Design Patterns Applied:
    - **Strategy Pattern:** `_TripleIndexStrategy` encapsulates different triple
      indexing and lookup strategies for optimized KG queries.
    - **Factory Pattern:** `RuleEngine` creates Rule instances from various
      sources (YAML, manual).
    - **Decorator Pattern:** `_collect` decorator wraps methods to automatically
      collect results into a ResultCollector.
    - **Template Method:** `BusinessService.validate()` defines the validation
      skeleton with customizable rule matching steps.
    - **Adapter Pattern:** `RuleViolation.to_dict()` adapts internal violation
      objects to external JSON-compatible format.
    - **Dependency Injection:** Services receive FileManager, ConcurrencyManager,
      and DiskCache instances for testability.

Performance:
    - Uses Numba-accelerated triple matching via `find_matching_triples_accelerated`
    - Leverages `VocabularyEncoder` for vectorized entity encoding
    - Caches rule compilations and triple indices via `DiskCache`

Example:
    >>> from pff.application.services.business_service import BusinessService
    >>> service = BusinessService()
    >>> result = service.validate("data/test.json")
    >>> print(result["is_valid"])
    True
"""

from pff.shared.acceleration.numba_kernels import (  # noqa: E402
    NUMBA_AVAILABLE,
    VocabularyEncoder,
)

from .core import BusinessService  # noqa: E402
from .model_integration import ModelIntegration  # noqa: E402
from .models import Rule, RuleViolation  # noqa: E402
from .rule_engine import RuleEngine, aggregate_duplicate_rules  # noqa: E402
from .rule_validator import (  # noqa: E402
    RuleValidator,
    run_rule_check_indexed,
    run_rule_check_shared,
)
from .triple_index import TripleIndex  # noqa: E402

_aggregate_duplicate_rules = aggregate_duplicate_rules
_run_rule_check_indexed = run_rule_check_indexed
_run_rule_check_shared = run_rule_check_shared


def _run_rule_check(rule: Rule, triples: list[tuple]) -> list[RuleViolation]:
    """
    Executes a rule check on a list of triples using the provided Rule.

    **DEPRECATED:** Use _run_rule_check_shared instead to avoid memory explosion.

    Args:
        rule (Rule): The rule to be checked.
        triples (list[tuple]): A list of triples to validate against the rule.
    Returns:
        list[RuleViolation]: A list of rule violations found during validation.
    """
    temp_validator = RuleValidator()
    return temp_validator._check_single_rule(rule, triples)


from .rule_validator import (  # noqa: E402
    bind_or_check_standalone,
    check_head_satisfied_indexed,
    check_head_satisfied_standalone,
    find_rule_violations_indexed,
    find_rule_violations_standalone,
    substitute_vars_standalone,
    try_unify_standalone,
)

_bind_or_check_standalone = bind_or_check_standalone
_substitute_vars_standalone = substitute_vars_standalone
_try_unify_standalone = try_unify_standalone
_check_head_satisfied_standalone = check_head_satisfied_standalone
_check_head_satisfied_indexed = check_head_satisfied_indexed
_find_rule_violations_standalone = find_rule_violations_standalone
_find_rule_violations_indexed = find_rule_violations_indexed


from .rule_validator import (  # noqa: E402
    IndexedViolationStrategy,
    StandaloneViolationStrategy,
    ViolationFindingStrategy,
    ViolationStrategyFactory,
)
from .shared import (  # noqa: E402
    CompositeValidationObserver,
    LoggingValidationObserver,
    ManualRuleSource,
    MetricsValidationObserver,
    PenaltyConfig,
    RuleBuilder,
    RuleSource,
    RuleSourceFactory,
    ValidationEvent,
    ValidationEventType,
    ValidationObserver,
    ViolationPenaltyCalculator,
)

__all__ = [
    "BusinessService",
    "ModelIntegration",
    "Rule",
    "RuleViolation",
    "RuleEngine",
    "RuleValidator",
    "TripleIndex",
    "ViolationFindingStrategy",
    "IndexedViolationStrategy",
    "StandaloneViolationStrategy",
    "ViolationStrategyFactory",
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
    "NUMBA_AVAILABLE",
    "VocabularyEncoder",
    "aggregate_duplicate_rules",
    "run_rule_check_indexed",
    "run_rule_check_shared",
    "bind_or_check_standalone",
    "substitute_vars_standalone",
    "try_unify_standalone",
    "check_head_satisfied_standalone",
    "check_head_satisfied_indexed",
    "find_rule_violations_standalone",
    "find_rule_violations_indexed",
    "_aggregate_duplicate_rules",
    "_run_rule_check",
    "_run_rule_check_indexed",
    "_run_rule_check_shared",
    "_bind_or_check_standalone",
    "_substitute_vars_standalone",
    "_try_unify_standalone",
    "_check_head_satisfied_standalone",
    "_check_head_satisfied_indexed",
    "_find_rule_violations_standalone",
    "_find_rule_violations_indexed",
]
