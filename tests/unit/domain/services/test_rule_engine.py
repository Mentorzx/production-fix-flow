"""Provide module-level functionality for the PFF codebase.



Notes:

    File: tests/unit/domain/services/test_rule_engine.py

"""

from __future__ import annotations

from pff.application.services.business_service.models import Rule
from pff.application.services.business_service.rule_engine import (
    aggregate_duplicate_rules,
)


def test_aggregate_duplicate_rules_groups_by_head_and_body() -> None:
    """Execute test aggregate duplicate rules groups by head and body.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    rules = [
        Rule(
            id="r1",
            confidence=0.8,
            head={"predicate": "h", "args": ["A", "B"]},
            body=[{"predicate": "p", "args": ["A", "C"]}],
            source="manual",
        ),
        Rule(
            id="r1_dup",
            confidence=0.6,
            head={"predicate": "h", "args": ["A", "B"]},
            body=[{"predicate": "p", "args": ["A", "C"]}],
            source="manual",
        ),
        Rule(
            id="r2",
            confidence=0.4,
            head={"predicate": "h2", "args": ["X", "Y"]},
            body=[{"predicate": "q", "args": ["X", "Z"]}],
            source="manual",
        ),
    ]

    aggregated = aggregate_duplicate_rules(rules)

    assert len(aggregated) == 2
    by_head = {r.head["predicate"]: r for r in aggregated}

    first = by_head["h"]
    assert first.occurrences == 2
    assert first.aggregated_confidence == 1.4
    assert first.confidence == 0.8

    second = by_head["h2"]
    assert second.occurrences == 1
    assert second.aggregated_confidence == 0.4
