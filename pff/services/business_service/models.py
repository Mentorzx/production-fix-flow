"""
Business Service Models - Core Data Classes for Rule Validation.

This module contains the core data classes used throughout the business service
for representing rules and rule violations.

Design Patterns Applied:
    - **Adapter Pattern:** `RuleViolation.to_dict()` adapts internal violation
      objects to external JSON-compatible format.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Rule:
    """
    Represents a validation rule with metadata.

    Attributes:
        id: Unique identifier for the rule
        confidence: Confidence score for the rule (0-1)
        head: Head predicate of the rule
        body: List of body predicates
        source: Source of the rule (manual, anyburl)
        total_predictions: Total predictions for AnyBURL rules
        correct_predictions: Correct predictions for AnyBURL rules
        occurrences: Number of times this exact rule pattern appears (v10.8.0)
        aggregated_confidence: Sum of confidences from all occurrences (v10.8.0)
    """

    id: str
    confidence: float
    head: dict[str, Any]
    body: list[dict[str, Any]]
    source: str
    total_predictions: int = 0
    correct_predictions: int = 0
    occurrences: int = 1  # v10.8.0: Track rule frequency
    aggregated_confidence: float = 0.0  # v10.8.0: Sum of confidence from duplicates


@dataclass
class RuleViolation:
    """
    Represents a rule violation found during validation.

    Attributes:
        rule_id: ID of the violated rule
        confidence: Confidence of the violated rule
        description: Human-readable description of the violation
        bindings: Variable bindings when violation was detected

    Design Patterns:
        - **Adapter Pattern:** `to_dict()` adapts internal structure to JSON format.
    """

    rule_id: str
    confidence: float
    description: str
    bindings: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert violation to dictionary format.

        Returns:
            JSON-compatible dictionary representation
        """
        return {
            "rule_id": self.rule_id,
            "confidence": self.confidence,
            "description": self.description,
            "bindings": self.bindings,
        }
