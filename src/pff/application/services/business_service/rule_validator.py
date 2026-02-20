"""
Rule Validator - Core Rule Validation Logic.

This module implements the core validation logic for checking rules against
triples using unification and pattern matching.

Design Patterns Applied:
    - **Strategy Pattern:** ViolationFindingStrategy for indexed vs standalone.
    - **Factory Pattern:** ViolationStrategyFactory for strategy selection.
    - **Template Method:** `validate_rules` defines the validation skeleton.

Performance:
    - Uses TripleIndex for O(1) head satisfaction checks
    - Supports Rust-accelerated triple matching
    - Parallel validation via ConcurrencyManager
"""

from __future__ import annotations

import functools
import sys
import time
from typing import Any, Protocol

from pff.application.services.business_service.shared.validation_observer import (
    ValidationEvent,
    ValidationEventType,
    ValidationObserver,
)
from pff.shared import ConcurrencyManager, load_config, logger
from pff_rust import VocabularyEncoder
from pff.shared.core.config import VALIDATOR_CONFIG_PATH
from pff.shared.research import _TripleIndexStrategy

from .models import Rule, RuleViolation
from .rule_engine import aggregate_duplicate_rules
from .triple_index import TripleIndex

MIN_ARGS_PATTERN = 2
MIN_UNIQUE_LABELS = 2
ACCELERATION_MIN_TRIPLES = 10


class ViolationFindingStrategy(Protocol):
    """
    Protocol for violation finding strategies.

    Design Pattern: Strategy
    - Defines a family of algorithms (indexed vs standalone violation finding)
    - Encapsulates each algorithm
    - Makes them interchangeable
    """

    def find_violations(
        self,
        rule: Rule,
        triples: list[tuple],
        triple_index: TripleIndex | None,
        max_depth: int | None,
    ) -> list[RuleViolation]:
        """
        Find rule violations using the strategy's algorithm.

        Args:
            rule: Rule to validate
            triples: List of triples to match against
            triple_index: Optional pre-built triple index
            max_depth: Maximum recursion depth

        Returns:
            List of rule violations found
        """
        ...


class IndexedViolationStrategy:
    """
    Violation finding strategy using TripleIndex for O(1) lookups.

    This is the preferred strategy when TripleIndex is available.
    """

    def find_violations(
        self,
        rule: Rule,
        triples: list[tuple],
        triple_index: TripleIndex | None,
        max_depth: int | None,
    ) -> list[RuleViolation]:
        """Find violations using indexed O(1) lookups."""
        if triple_index is None:
            triple_index = TripleIndex(triples)

        violations: list[RuleViolation] = []
        find_rule_violations_indexed(
            rule.body,
            triples,
            triple_index,
            0,
            {},
            violations,
            rule,
            max_depth=max_depth,
        )
        return violations


class StandaloneViolationStrategy:
    """
    Violation finding strategy using linear search (no index).

    Falls back to this strategy when TripleIndex is not available
    or for small datasets where index construction overhead is not worth it.
    """

    def __init__(self, encoder: VocabularyEncoder | None = None):
        """Initialize with optional Rust encoder."""
        self.encoder = encoder

    def find_violations(
        self,
        rule: Rule,
        triples: list[tuple],
        triple_index: TripleIndex | None,
        max_depth: int | None,
    ) -> list[RuleViolation]:
        """Find violations using linear search."""
        violations: list[RuleViolation] = []
        find_rule_violations_standalone(rule.body, triples, 0, {}, violations, rule, self.encoder)
        return violations


class ViolationStrategyFactory:
    """
    Factory for creating violation finding strategies.

    Design Pattern: Factory
    - Encapsulates strategy creation logic
    - Allows runtime strategy selection
    """

    @staticmethod
    def create(
        use_index: bool = True, encoder: VocabularyEncoder | None = None
    ) -> ViolationFindingStrategy:
        """
        Create a violation finding strategy.

        Args:
            use_index: Whether to use indexed strategy (default: True)
            encoder: Optional Rust encoder for standalone strategy

        Returns:
            ViolationFindingStrategy instance
        """
        if use_index:
            return IndexedViolationStrategy()
        return StandaloneViolationStrategy(encoder)


class RuleValidator:
    """
    Validates data against rules using unification and pattern matching.

    This class implements the core validation logic, checking if rules
    are satisfied by the given triples through variable unification.

    Design Patterns:
        - **Strategy Pattern:** Uses ViolationFindingStrategy for algorithm selection.
        - **Factory Pattern:** Uses ViolationStrategyFactory for strategy creation.
        - **Template Method:** `validate_rules` defines the validation skeleton.
        - **Observer Pattern:** Uses ValidationObserver for event emission.
    """

    def __init__(
        self,
        strategy: ViolationFindingStrategy | None = None,
        observer: ValidationObserver | None = None,
    ):
        """
        Initialize the rule validator.

        Args:
            strategy: Optional violation finding strategy (DI pattern).
                     Defaults to IndexedViolationStrategy.
            observer: Optional validation observer for event emission (Observer pattern).
        """
        self.triple_index = _TripleIndexStrategy()
        self._strategy = strategy or ViolationStrategyFactory.create(use_index=True)
        self._observer = observer

    def _emit_event(self, event: ValidationEvent) -> None:
        """Emit validation event to observer if configured."""
        if self._observer is not None:
            self._observer.on_event(event)

    def validate_rules(
        self, rules: list[Rule], triples: list[tuple[Any, str, Any]]
    ) -> tuple[list[RuleViolation], list[Rule]]:
        """
        Validate all rules against the given triples in parallel.

        Emits ValidationEvents to observer if configured:
        - VALIDATION_STARTED at the beginning
        - VALIDATION_COMPLETED at the end with summary

        Args:
            rules: List of rules to validate (128K+ supported)
            triples: List of (subject, predicate, object) triples (1-10K typical)

        Returns:
            Tuple of (violations list, satisfied rules list)
        """
        if not rules:
            return [], []

        self._emit_event(
            ValidationEvent(
                event_type=ValidationEventType.VALIDATION_STARTED,
                metadata={"rule_count": len(rules), "triple_count": len(triples)},
            )
        )

        original_rule_count = len(rules)

        valid_rules = [r for r in rules[:10] if r is not None]
        already_aggregated = any(
            r.occurrences > 1 or r.aggregated_confidence > 0 for r in valid_rules
        )

        if not already_aggregated:
            t_agg_start = time.time()
            rules = aggregate_duplicate_rules(rules)
            t_agg_end = time.time()
            logger.debug(f"Rule aggregation completed in {t_agg_end - t_agg_start:.2f}s")
        else:
            logger.debug("Rules already aggregated (loaded from cache), skipping")

        from pff.shared.system.resource_manager import (
            get_resource_manager,
        )

        resource_manager = get_resource_manager()

        first_valid_rule = next((r for r in rules if r is not None), None)
        estimated_task_size = sys.getsizeof(first_valid_rule) if first_valid_rule else 5000
        shared_data_size = sum(sys.getsizeof(t) for t in triples[:10]) * len(triples) // 10
        limits = resource_manager.calculate_limits(
            task_count=len(rules),
            estimated_task_size=estimated_task_size,
            shared_data_size=shared_data_size,
        )

        logger.info(
            f"Alocacao adaptativa: {limits.optimal_workers} workers, "
            f"{limits.max_pending_futures} max pendentes, "
            f"{limits.safe_memory_limit / 1024**3:.1f} GB limite seguro"
        )

        t0 = time.time()
        triple_index = TripleIndex(triples)
        index_build_time = time.time() - t0
        logger.info(
            f"Indice de triplas construido: {len(triples)} triplas em {index_build_time:.2f}s "
            f"(speedup esperado: 5-10x)"
        )

        shared_data = (triples, triple_index)
        fn_with_index = functools.partial(run_rule_check_indexed, shared_data)
        args_list = [(rule,) for rule in rules]
        cm = ConcurrencyManager()

        perf_cfg = load_config(VALIDATOR_CONFIG_PATH).get("performance", {})
        ray_threshold = perf_cfg.get("ray_threshold_rules", 10000)
        thread_threshold = perf_cfg.get("thread_threshold_rules", 200)
        if original_rule_count <= thread_threshold:
            task_type = "thread"
        else:
            task_type = "ray" if original_rule_count > ray_threshold else "process"

        try:
            results: list[list[RuleViolation]] = cm.execute_sync(
                fn=fn_with_index,
                args_list=args_list,
                task_type=task_type,
                max_workers=limits.optimal_workers,
                desc=f" Validating {len(rules):,} rules (indexed, backend={task_type})",
            )
        except PermissionError as exc:
            logger.warning(f"Process backend unavailable ({exc}); retrying with thread executor")
            results = cm.execute_sync(
                fn=fn_with_index,
                args_list=args_list,
                task_type="thread",
                max_workers=limits.optimal_workers,
                desc=f" Validating {len(rules):,} rules (indexed, backend=thread)",
            )

        violations = []
        satisfied_rules = []
        for rule, rule_violations in zip(rules, results):
            if rule_violations is None:
                continue
            elif rule_violations:
                violations.extend(rule_violations)
            else:
                satisfied_rules.append(rule)

        self._emit_event(
            ValidationEvent(
                event_type=ValidationEventType.VALIDATION_COMPLETED,
                metadata={
                    "total_rules": original_rule_count,
                    "violations_count": len(violations),
                    "satisfied_count": len(satisfied_rules),
                    "backend": task_type,
                },
            )
        )

        return violations, satisfied_rules

    def _check_single_rule(
        self, rule: Rule, triples: list[tuple[Any, str, Any]]
    ) -> list[RuleViolation]:
        """
        Check if a single rule is satisfied by the triples.

        Args:
            rule: Rule to check
            triples: List of triples

        Returns:
            List of violations (empty if rule is satisfied)
        """
        violations: list[RuleViolation] = []
        self._find_rule_violations(rule.body, triples, 0, {}, violations, rule)
        return violations

    def _find_rule_violations(
        self,
        body_predicates: list[dict],
        triples: list[tuple],
        pred_idx: int,
        bindings: dict[str, Any],
        violations: list[RuleViolation],
        rule: Rule,
    ) -> None:
        """Recursively find violations by checking if body predicates are satisfied."""
        if pred_idx >= len(body_predicates):
            if not self._check_head_satisfied(rule.head, triples, bindings):
                substituted_head = self._substitute_vars(rule.head["args"], bindings)
                head_str = f"{rule.head['predicate']}({', '.join(map(str, substituted_head))})"
                bindings_str = ", ".join(f"{k}='{v}'" for k, v in bindings.items())
                description = (
                    f"Conclusão esperada '{head_str}' não encontrada. "
                    f"A violação ocorreu porque as condições da regra foram "
                    f"satisfeitas com as variáveis: [{bindings_str}]"
                )
                violation = RuleViolation(
                    rule_id=rule.id,
                    confidence=rule.confidence,
                    description=description,
                    bindings=bindings.copy(),
                )
                violations.append(violation)
            return

        pattern = body_predicates[pred_idx]

        for triple in triples:
            new_bindings = self._try_unify(pattern, triple, bindings)
            if new_bindings is not None:
                self._find_rule_violations(
                    body_predicates,
                    triples,
                    pred_idx + 1,
                    new_bindings,
                    violations,
                    rule,
                )

    def _try_unify(
        self,
        pattern: dict[str, Any],
        triple: tuple[Any, str, Any],
        bindings: dict[str, Any],
    ) -> dict[str, Any] | None:
        """
        Try to unify a pattern with a triple given current bindings.

        Args:
            pattern: Pattern to match
            triple: Triple to match against
            bindings: Current variable bindings

        Returns:
            Updated bindings if unification succeeds, None otherwise
        """
        subject, predicate, obj = triple

        if pattern["predicate"] != predicate and pattern["predicate"] != "*":
            return None
        args = pattern.get("args", [])
        if len(args) < MIN_ARGS_PATTERN:
            return None
        new_bindings = bindings.copy()
        if not self._bind_or_check(args[0], subject, new_bindings):
            return None
        if not self._bind_or_check(args[1], obj, new_bindings):
            return None

        return new_bindings

    def _bind_or_check(self, var: str, value: Any, bindings: dict[str, Any]) -> bool:
        """
        Bind variable to value or check consistency with existing binding.

        Variables are purely alphabetic uppercase strings (e.g., A, B, VAR).
        Literals may contain quotes (e.g., 'BARRED', 'active').

        Args:
            var: Variable name (uppercase alpha) or literal
            value: Value to bind/check
            bindings: Current bindings

        Returns:
            True if successful, False otherwise
        """

        if var.isalpha() and var.isupper():
            if var in bindings:
                return str(bindings[var]) == str(value)
            else:
                bindings[var] = value
                return True
        else:
            literal = var.strip("'\"")
            return str(literal) == str(value)

    def _check_head_satisfied(
        self,
        head_pattern: dict[str, Any],
        triples: list[tuple],
        bindings: dict[str, Any],
    ) -> bool:
        """
        Check if head pattern exists in triples with current bindings.

        Handles unbound variables as wildcards.

        Args:
            head_pattern: Head predicate pattern
            triples: List of triples
            bindings: Current variable bindings

        Returns:
            True if head is satisfied
        """
        args = head_pattern["args"]
        predicate = head_pattern["predicate"]

        if len(args) < MIN_ARGS_PATTERN:
            return False

        subject_arg, obj_arg = args[0], args[1]

        subject_is_var = (
            isinstance(subject_arg, str) and subject_arg.isalpha() and subject_arg.isupper()
        )
        obj_is_var = isinstance(obj_arg, str) and obj_arg.isalpha() and obj_arg.isupper()

        subject = bindings.get(subject_arg, subject_arg) if subject_is_var else subject_arg
        obj = bindings.get(obj_arg, obj_arg) if obj_is_var else obj_arg

        if not subject_is_var and isinstance(subject, str):
            subject = subject.strip("'\"")
        if not obj_is_var and isinstance(obj, str):
            obj = obj.strip("'\"")

        subject_bound = not (subject_is_var and subject_arg not in bindings)
        obj_bound = not (obj_is_var and obj_arg not in bindings)

        for triple in triples:
            if triple[1] != predicate:
                continue
            if subject_bound and str(triple[0]) != str(subject):
                continue
            if obj_bound and str(triple[2]) != str(obj):
                continue
            return True

        return False

    def _substitute_vars(self, args: list[str], bindings: dict[str, Any]) -> list[str]:
        """
        Substitute variables with their bound values.

        Args:
            args: List of variables/literals
            bindings: Variable bindings

        Returns:
            List with variables replaced by values
        """
        result = []
        for arg in args:
            if arg.isupper() and arg in bindings:
                result.append(str(bindings[arg]))
            else:
                result.append(arg)
        return result


def _is_variable(arg: Any) -> bool:
    """
    Check if an argument is a Datalog variable.

    Variables are uppercase alphabetic strings without quotes or special chars.
    Examples:
        - "A", "B", "X" -> True (single letter vars)
        - "VAR", "FOO" -> True (multi-letter vars)
        - "'BARRED'", "'value'" -> False (literals with quotes)
        - "123", "a" -> False (not uppercase alpha)

    Args:
        arg: Argument to check

    Returns:
        True if arg is a variable, False if literal
    """
    if not isinstance(arg, str):
        return False

    return arg.isalpha() and arg.isupper()


def bind_or_check_standalone(var: Any, value: Any, bindings: dict[str, Any]) -> bool:
    """Standalone version of _bind_or_check without instance dependencies."""
    if _is_variable(var):
        if var in bindings:
            return bool(bindings[var] == value)
        bindings[var] = value
        return True
    literal = var.strip("'\"") if isinstance(var, str) else var
    return str(literal) == str(value)


def substitute_vars_standalone(args: list[Any], bindings: dict[str, Any]) -> list[Any]:
    """Standalone version of _substitute_vars without instance dependencies."""
    return [
        bindings.get(arg, arg) if isinstance(arg, str) and arg.isupper() else arg for arg in args
    ]


def try_unify_standalone(
    pattern: dict[str, Any],
    triple: tuple[Any, str, Any],
    bindings: dict[str, Any],
) -> dict[str, Any] | None:
    """Standalone version of _try_unify without instance dependencies."""
    subject, predicate, obj = triple

    if pattern["predicate"] != predicate and pattern["predicate"] != "*":
        return None
    args = pattern.get("args", [])
    if len(args) < MIN_ARGS_PATTERN:
        return None

    new_bindings = bindings.copy()

    if not bind_or_check_standalone(args[0], subject, new_bindings):
        return None
    if not bind_or_check_standalone(args[1], obj, new_bindings):
        return None

    return new_bindings


def check_head_satisfied_standalone(
    head: dict, triples: list[tuple], bindings: dict[str, Any]
) -> bool:
    """
    Standalone version of _check_head_satisfied without instance dependencies.

    Handles unbound variables as wildcards - checks if any matching triple exists.

    **DEPRECATED:** Use check_head_satisfied_indexed for O(1) lookup.
    """
    args = head["args"]
    predicate = head["predicate"]

    if len(args) < MIN_ARGS_PATTERN:
        return False

    subject_arg, obj_arg = args[0], args[1]

    subject_is_var = _is_variable(subject_arg)
    obj_is_var = _is_variable(obj_arg)

    subject = bindings.get(subject_arg, subject_arg) if subject_is_var else subject_arg
    obj = bindings.get(obj_arg, obj_arg) if obj_is_var else obj_arg

    if not subject_is_var and isinstance(subject, str):
        subject = subject.strip("'\"")
    if not obj_is_var and isinstance(obj, str):
        obj = obj.strip("'\"")

    subject_bound = not (subject_is_var and subject_arg not in bindings)
    obj_bound = not (obj_is_var and obj_arg not in bindings)

    for triple in triples:
        if triple[1] != predicate:
            continue
        if subject_bound and str(triple[0]) != str(subject):
            continue
        if obj_bound and str(triple[2]) != str(obj):
            continue
        return True

    return False


def check_head_satisfied_indexed(
    head: dict, triple_index: TripleIndex, bindings: dict[str, Any]
) -> bool:
    """
    OPTIMIZED: Check if head is satisfied using indexed lookup.

    Handles unbound variables as wildcards:
    - If subject is unbound: check if any subject exists with (predicate, object)
    - If object is unbound: check if any object exists with (subject, predicate)
    - If both bound: exact O(1) lookup

    Complexity: O(1) average case (vs O(n) for linear search)
    Expected speedup: 5-10x

    Args:
        head: Head predicate to check
        triple_index: Pre-built triple index for fast lookups
        bindings: Variable bindings

    Returns:
        True if head is satisfied, False otherwise
    """
    args = head["args"]
    predicate = head["predicate"]

    if len(args) < MIN_ARGS_PATTERN:
        return False

    subject_arg, obj_arg = args[0], args[1]

    subject_is_var = _is_variable(subject_arg)
    subject = bindings.get(subject_arg, subject_arg) if subject_is_var else subject_arg

    obj_is_var = _is_variable(obj_arg)
    obj = bindings.get(obj_arg, obj_arg) if obj_is_var else obj_arg

    if not subject_is_var and isinstance(subject, str):
        subject = subject.strip("'\"")
    if not obj_is_var and isinstance(obj, str):
        obj = obj.strip("'\"")

    subject_bound = not (subject_is_var and subject_arg not in bindings)
    obj_bound = not (obj_is_var and obj_arg not in bindings)

    if subject_bound and obj_bound:
        return triple_index.exists(subject, predicate, obj)
    elif subject_bound and not obj_bound:
        objects = triple_index.get_objects(subject, predicate)
        return len(objects) > 0
    elif not subject_bound and obj_bound:
        subjects = triple_index.get_subjects(predicate, obj)
        return len(subjects) > 0
    else:
        return predicate in triple_index.pos


def _find_matching_triples(
    pattern: dict[str, Any],
    triples: list[tuple],
) -> list[int]:
    """Find indices of triples matching a pattern via string comparison."""
    predicate = pattern["predicate"]
    args = pattern.get("args", [])

    if len(args) < MIN_ARGS_PATTERN:
        return []

    arg0, arg1 = args[0], args[1]
    arg0_is_var = isinstance(arg0, str) and arg0.isupper()
    arg1_is_var = isinstance(arg1, str) and arg1.isupper()

    matching: list[int] = []
    for i, (s, p, o) in enumerate(triples):
        s, p, o = str(s), str(p), str(o)
        if predicate != "*" and predicate != p:
            continue
        if not arg0_is_var and arg0 != s:
            continue
        if not arg1_is_var and arg1 != o:
            continue
        matching.append(i)
    return matching


def find_rule_violations_standalone(
    body_predicates: list[dict],
    triples: list[tuple],
    pred_idx: int,
    bindings: dict[str, Any],
    violations: list[RuleViolation],
    rule: Rule,
    encoder: VocabularyEncoder | None = None,
) -> None:
    """
    Standalone version of _find_rule_violations without instance dependencies.

    **DEPRECATED:** Use find_rule_violations_indexed for O(1) triple lookups.

    Args:
        encoder: Optional VocabularyEncoder for Rust acceleration
    """
    if pred_idx >= len(body_predicates):
        if not check_head_satisfied_standalone(rule.head, triples, bindings):
            substituted_head = substitute_vars_standalone(rule.head["args"], bindings)
            head_str = f"{rule.head['predicate']}({', '.join(map(str, substituted_head))})"
            bindings_str = ", ".join(f"{k}='{v}'" for k, v in bindings.items())
            description = (
                f"Conclusão esperada '{head_str}' não encontrada. "
                f"A violação ocorreu porque as condições da regra foram "
                f"satisfeitas com as variáveis: [{bindings_str}]"
            )
            violation = RuleViolation(
                rule_id=rule.id,
                confidence=rule.confidence,
                description=description,
                bindings=bindings.copy(),
            )
            violations.append(violation)
        return

    pattern = body_predicates[pred_idx]

    if encoder is not None and len(triples) > ACCELERATION_MIN_TRIPLES:
        matching_indices = _find_matching_triples(pattern, triples)
        for idx in matching_indices:
            triple = triples[idx]
            new_bindings = try_unify_standalone(pattern, triple, bindings)
            if new_bindings is not None:
                find_rule_violations_standalone(
                    body_predicates,
                    triples,
                    pred_idx + 1,
                    new_bindings,
                    violations,
                    rule,
                    encoder,
                )
    else:
        for triple in triples:
            new_bindings = try_unify_standalone(pattern, triple, bindings)
            if new_bindings is not None:
                find_rule_violations_standalone(
                    body_predicates,
                    triples,
                    pred_idx + 1,
                    new_bindings,
                    violations,
                    rule,
                    encoder,
                )


def find_rule_violations_indexed(
    body_predicates: list[dict],
    triples: list[tuple],
    triple_index: TripleIndex,
    pred_idx: int,
    bindings: dict[str, Any],
    violations: list[RuleViolation],
    rule: Rule,
    max_depth: int | None = None,
    _current_depth: int = 0,
) -> None:
    """Execute find rule violations indexed.



    Args:

        body_predicates: Input value used by this callable.

        triples: Input value used by this callable.

        triple_index: Input value used by this callable.

        pred_idx: Input value used by this callable.

        bindings: Input value used by this callable.

        violations: Input value used by this callable.

        rule: Input value used by this callable.

        max_depth: Optional input value.

        _current_depth: Optional input value.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    if max_depth is not None and _current_depth > max_depth:
        logger.warning(
            f"Recursion depth limit ({max_depth}) reached for rule {rule.id}, truncating"
        )
        return

    if pred_idx >= len(body_predicates):
        if not check_head_satisfied_indexed(rule.head, triple_index, bindings):
            substituted_head = substitute_vars_standalone(rule.head["args"], bindings)
            head_str = f"{rule.head['predicate']}({', '.join(map(str, substituted_head))})"
            bindings_str = ", ".join(f"{k}='{v}'" for k, v in bindings.items())
            description = (
                f"Conclusão esperada '{head_str}' não encontrada. "
                f"A violação ocorreu porque as condições da regra foram "
                f"satisfeitas com as variáveis: [{bindings_str}]"
            )
            violation = RuleViolation(
                rule_id=rule.id,
                confidence=rule.confidence,
                description=description,
                bindings=bindings.copy(),
            )
            violations.append(violation)
        return

    pattern = body_predicates[pred_idx]
    predicate = pattern.get("predicate", "*")
    args = pattern.get("args", [])

    s_val = (
        bindings.get(args[0])
        if len(args) > 0 and isinstance(args[0], str) and args[0].isupper()
        else None
    )
    o_val = (
        bindings.get(args[1])
        if len(args) > 1 and isinstance(args[1], str) and args[1].isupper()
        else None
    )

    if s_val is None and len(args) > 0:
        if not (isinstance(args[0], str) and args[0].isupper()):
            s_val = args[0]

    if o_val is None and len(args) > 1:
        if not (isinstance(args[1], str) and args[1].isupper()):
            o_val = args[1]

    candidate_triples = triple_index.get_triples(subject=s_val, predicate=predicate, obj=o_val)

    for triple in candidate_triples:
        new_bindings = try_unify_standalone(pattern, triple, bindings)
        if new_bindings is not None:
            find_rule_violations_indexed(
                body_predicates,
                triples,
                triple_index,
                pred_idx + 1,
                new_bindings,
                violations,
                rule,
                max_depth,
                _current_depth + 1,
            )


def run_rule_check_indexed(
    shared_data: tuple[list[tuple], TripleIndex], rule: Rule
) -> list[RuleViolation] | None:
    """
    Uses pre-built TripleIndex for O(1) head satisfaction checks.

    Args:
        shared_data: Tuple of (triples_list, triple_index) shared across workers
        rule: Rule to validate

    Returns:
        List of rule violations found, or None if body couldn't be checked
    """
    shared_triples, triple_index = shared_data

    if not shared_triples and rule.body:
        return None

    validation_cfg = load_config(VALIDATOR_CONFIG_PATH).get("validation", {})
    max_depth = validation_cfg.get("max_recursion_depth", 20)

    violations: list[RuleViolation] = []
    find_rule_violations_indexed(
        rule.body,
        shared_triples,
        triple_index,
        0,
        {},
        violations,
        rule,
        max_depth=max_depth,
    )
    return violations


def run_rule_check_shared(triples: list[tuple], rule: Rule) -> list[RuleViolation]:
    """
    Wrapper for run_rule_check_indexed with argument order compatible with tests.

    Args:
        triples: List of (subject, predicate, object) triples
        rule: Rule to validate

    Returns:
        List of rule violations found (empty list if body couldn't be checked)
    """
    triple_index = TripleIndex(triples)
    shared_data = (triples, triple_index)
    result = run_rule_check_indexed(shared_data, rule)
    return result if result is not None else []


_bind_or_check_standalone = bind_or_check_standalone
_substitute_vars_standalone = substitute_vars_standalone
_try_unify_standalone = try_unify_standalone
_check_head_satisfied_standalone = check_head_satisfied_standalone
_check_head_satisfied_indexed = check_head_satisfied_indexed
_find_rule_violations_standalone = find_rule_violations_standalone
_find_rule_violations_indexed = find_rule_violations_indexed
_run_rule_check_indexed = run_rule_check_indexed
_run_rule_check_shared = run_rule_check_shared
