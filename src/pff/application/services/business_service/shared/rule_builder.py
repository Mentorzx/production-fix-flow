"""
Rule Builder and Factory - Design Patterns for Rule Construction.

This module provides Builder and Factory patterns for constructing Rule objects
from manual JSON/YAML sources.

Design Patterns Applied:
    - **Builder Pattern:** `RuleBuilder` provides fluent interface for Rule construction.
    - **Factory Pattern:** `RuleSourceFactory` creates rules from different file formats.
    - **Strategy Pattern:** Different parsing strategies for each source type.

Example:
    rule = (RuleBuilder()
        .with_id("rule_001")
        .with_confidence(0.85)
        .with_head("knows", ["A", "B"])
        .with_body_clause("friend", ["A", "C"])
        .with_body_clause("friend", ["C", "B"])
        .from_source("manual")
        .build())

    rules = RuleSourceFactory.load_rules(Path("rules.json"), source_type="manual")
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pff.shared import FileManager, logger


def split_rule_body_clauses(body_str: str) -> list[str]:
    """Split a body string into clause fragments preserving trailing ')' semantics."""
    body_clauses_parts = [c.strip() for c in body_str.strip().split("),") if c.strip()]
    clauses: list[str] = []
    for i, clause_part in enumerate(body_clauses_parts):
        clauses.append(clause_part + ")" if i < len(body_clauses_parts) - 1 else clause_part)
    return clauses


@dataclass
class Rule:
    """
    Represents a validation rule with metadata.

    Attributes:
        id: Unique identifier for the rule
        confidence: Confidence score for the rule (0-1)
        head: Head predicate of the rule
        body: List of body predicates
        source: Source of the rule (manual, etc.)
        total_predictions: Optional metadata for rule provenance
        correct_predictions: Optional metadata for rule provenance
        occurrences: Number of times this exact rule pattern appears
        aggregated_confidence: Sum of confidences from all occurrences
    """

    id: str
    confidence: float
    head: dict[str, Any]
    body: list[dict[str, Any]]
    source: str
    total_predictions: int = 0
    correct_predictions: int = 0
    occurrences: int = 1
    aggregated_confidence: float = 0.0


class RuleBuilder:
    """
    Builder for constructing Rule objects with fluent interface.

    Provides a readable, step-by-step way to construct complex Rule objects
    from various input formats.

    Example:
        >>> rule = (RuleBuilder()
        ...     .with_id("rule_001")
        ...     .with_confidence(0.85)
        ...     .with_head("knows", ["A", "B"])
        ...     .with_body_clause("friend", ["A", "C"])
        ...     .from_source("manual")
        ...     .build())
    """

    def __init__(self) -> None:
        """Initialize builder with default values."""
        self._id: str = ""
        self._confidence: float = 0.0
        self._head: dict[str, Any] = {}
        self._body: list[dict[str, Any]] = []
        self._source: str = "unknown"
        self._total_predictions: int = 0
        self._correct_predictions: int = 0
        self._occurrences: int = 1
        self._aggregated_confidence: float = 0.0

    def with_id(self, rule_id: str) -> RuleBuilder:
        """Set the rule ID."""
        self._id = rule_id
        return self

    def with_confidence(self, confidence: float) -> RuleBuilder:
        """Set the confidence score (0-1)."""
        self._confidence = max(0.0, min(1.0, confidence))
        return self

    def with_head(self, predicate: str, args: list[str]) -> RuleBuilder:
        """
        Set the head predicate.

        Args:
            predicate: Predicate name (e.g., "knows")
            args: List of arguments (e.g., ["A", "B"])
        """
        self._head = {"predicate": predicate, "args": args}
        return self

    def with_head_dict(self, head: dict[str, Any]) -> RuleBuilder:
        """Set the head predicate from a dictionary."""
        self._head = head
        return self

    def with_body_clause(self, predicate: str, args: list[str]) -> RuleBuilder:
        """
        Add a body clause.

        Args:
            predicate: Predicate name
            args: List of arguments
        """
        self._body.append({"predicate": predicate, "args": args})
        return self

    def with_body(self, body: list[dict[str, Any]]) -> RuleBuilder:
        """Set the entire body from a list of clause dicts."""
        self._body = body
        return self

    def from_source(self, source: str) -> RuleBuilder:
        """Set the rule source (manual, etc.)."""
        self._source = source
        return self

    def with_occurrences(self, count: int, aggregated_conf: float = 0.0) -> RuleBuilder:
        """Set occurrence tracking for aggregated rules."""
        self._occurrences = count
        self._aggregated_confidence = aggregated_conf
        return self

    def from_pattern_string(self, pattern: str) -> RuleBuilder:
        """
        Parse a Datalog-like pattern string.

        Args:
            pattern: Pattern like "head(A,B) <= body1(A,C), body2(C,B)"

        Returns:
            Self for chaining
        """
        head, body = _parse_pattern(pattern)
        self._head = head
        self._body = body
        return self

    def build(self) -> Rule:
        """
        Build and return the Rule object.

        Returns:
            Constructed Rule instance

        Raises:
            ValueError: If required fields are missing
        """
        if not self._id:
            raise ValueError("Rule ID is required")
        if not self._head:
            raise ValueError("Rule head is required")

        return Rule(
            id=self._id,
            confidence=self._confidence,
            head=self._head,
            body=self._body,
            source=self._source,
            total_predictions=self._total_predictions,
            correct_predictions=self._correct_predictions,
            occurrences=self._occurrences,
            aggregated_confidence=self._aggregated_confidence,
        )


class RuleSource(ABC):
    """
    Abstract base class for rule sources.

    Strategy pattern: each source type implements its own parsing logic.
    """

    @abstractmethod
    def load(self, filepath: Path) -> list[Rule]:
        """
        Load rules from the source file.

        Args:
            filepath: Path to the source file

        Returns:
            List of parsed Rule objects
        """
        pass


class ManualRuleSource(RuleSource):
    """
    Loads rules from manual JSON files.

    Expected format:
        {
            "category_name": [
                {"id": "rule_1", "confidence": 0.9, "pattern": "head(A,B) <= body(A,B)"},
                ...
            ]
        }
    """

    def __init__(self) -> None:
        """Execute init.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        self.file_manager = FileManager()

    def load(self, filepath: Path) -> list[Rule]:
        """Load rules from JSON file."""
        rules: list[Rule] = []
        try:
            data = self.file_manager.read(filepath, return_native=True)
            if not isinstance(data, dict):
                logger.warning(f"Invalid manual rules format in {filepath}")
                return []

            for category, rules_list in data.items():
                if not isinstance(rules_list, list):
                    continue

                for idx, rule_data in enumerate(rules_list):
                    try:
                        rule = (
                            RuleBuilder()
                            .with_id(str(rule_data.get("id", f"{category}_{idx}")))
                            .with_confidence(float(rule_data.get("confidence", 0.0)))
                            .from_pattern_string(str(rule_data.get("pattern", "")))
                            .from_source("manual")
                            .build()
                        )
                        rules.append(rule)
                    except (ValueError, KeyError) as e:
                        logger.debug(f"Error parsing manual rule {idx} in {category}: {e}")

            logger.info(f"{len(rules)} regras manuais carregadas de {filepath.name}")

        except FileNotFoundError:
            logger.warning(f"Manual rules file not found: {filepath}")
        except Exception as e:
            logger.error(f"Failed to load manual rules from {filepath}: {e}")

        return rules


class RuleSourceFactory:
    """
    Factory for creating rule sources based on file type.

    Provides a unified interface for loading rules from various formats.

    Example:
        >>> rules = RuleSourceFactory.load_rules(Path("manual.json"), "manual")
    """

    _sources: dict[str, type[RuleSource]] = {
        "manual": ManualRuleSource,
        "json": ManualRuleSource,
    }

    @classmethod
    def register_source(cls, source_type: str, source_class: type[RuleSource]) -> None:
        """
        Register a new rule source type.

        Args:
            source_type: Type identifier (e.g., "yaml")
            source_class: RuleSource subclass to handle this type
        """
        cls._sources[source_type] = source_class

    @classmethod
    def load_rules(cls, filepath: Path, source_type: str | None = None) -> list[Rule]:
        """
        Load rules from a file using the appropriate source.

        Args:
            filepath: Path to the rules file
            source_type: Source type (auto-detected from extension if None)

        Returns:
            List of parsed Rule objects

        Raises:
            ValueError: If source type is unknown
        """
        if source_type is None:
            ext = filepath.suffix.lower()
            if ext == ".json":
                source_type = "json"
            else:
                raise ValueError(f"Cannot auto-detect source type for extension: {ext}")

        source_class = cls._sources.get(source_type)
        if source_class is None:
            raise ValueError(f"Unknown rule source type: {source_type}")

        source = source_class()
        return source.load(filepath)

    @classmethod
    def get_available_sources(cls) -> list[str]:
        """Get list of registered source types."""
        return list(cls._sources.keys())


def _parse_pattern(pattern_str: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """
    Parse a Datalog-like pattern string into head and body structures.

    Args:
        pattern_str: Pattern string like "head(A,B) <= body1(A,C), body2(C,B)"

    Returns:
        Tuple of (head_dict, body_list)
    """

    def parse_single_clause(clause_str: str) -> dict[str, Any]:
        """Parse a single predicate clause."""
        clause_str = clause_str.strip()
        match = re.match(r"(\w+)\(([^)]*)\)", clause_str)
        if not match:
            return {"predicate": clause_str, "args": []}
        predicate = match.group(1)
        args_str = match.group(2)
        args = [arg.strip() for arg in args_str.split(",") if arg.strip()]
        return {"predicate": predicate, "args": args}

    if "<=" in pattern_str:
        head_str, body_str = pattern_str.split("<=", 1)
    elif "<-" in pattern_str:
        head_str, body_str = pattern_str.split("<-", 1)
    else:
        return parse_single_clause(pattern_str), []

    head = parse_single_clause(head_str)

    body = [parse_single_clause(clause) for clause in split_rule_body_clauses(body_str)]

    return head, body
