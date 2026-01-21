"""
Validation Event Observer - Observer Pattern for Rule Validation Events.

This module provides Observer pattern implementation for tracking and reacting
to validation events in the rule engine.

Design Patterns Applied:
    - **Observer Pattern:** `ValidationObserver` ABC and concrete implementations.
    - **Composite Pattern:** `CompositeValidationObserver` for multi-observer dispatch.
    - **Strategy Pattern:** Different event handlers for various validation types.

Example:
    # Create composite observer with multiple handlers
    observer = CompositeValidationObserver([
        LoggingValidationObserver(),
        MetricsValidationObserver(metrics_dir),
    ])

    # Emit events during validation
    observer.on_event(ValidationEvent(
        event_type=ValidationEventType.RULE_MATCHED,
        rule_id="rule_001",
        triple=("A", "knows", "B"),
        confidence=0.85,
    ))
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, cast

from pff.shared import logger
from pff.shared.observer import CompositeObserver as SharedCompositeObserver


class ValidationEventType(Enum):
    """Types of validation events."""

    RULE_LOADED = "rule_loaded"
    RULE_MATCHED = "rule_matched"
    RULE_VIOLATION = "rule_violation"
    TRIPLE_VALIDATED = "triple_validated"
    TRIPLE_REJECTED = "triple_rejected"
    VALIDATION_STARTED = "validation_started"
    VALIDATION_COMPLETED = "validation_completed"
    BATCH_COMPLETED = "batch_completed"


@dataclass
class ValidationEvent:
    """
    Represents a validation event.

    Attributes:
        event_type: Type of the event
        rule_id: ID of the rule involved (if applicable)
        triple: Triple being validated (subject, predicate, object)
        confidence: Confidence score of the match/validation
        metadata: Additional event-specific data
        timestamp: When the event occurred
    """

    event_type: ValidationEventType
    rule_id: str | None = None
    triple: tuple[str, str, str] | None = None
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert event to dictionary for serialization."""
        return {
            "event_type": self.event_type.value,
            "rule_id": self.rule_id,
            "triple": list(self.triple) if self.triple else None,
            "confidence": self.confidence,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }


class ValidationObserver(ABC):
    """
    Abstract base class for validation observers.

    Implementations receive notifications about validation events and can
    react accordingly (logging, metrics collection, alerting, etc.).
    """

    @abstractmethod
    def on_event(self, event: ValidationEvent) -> None:
        """
        Handle a validation event.

        Args:
            event: The validation event to process
        """
        pass

    def on_batch_events(self, events: list[ValidationEvent]) -> None:
        """
        Handle a batch of events (default: process individually).

        Args:
            events: List of validation events
        """
        for event in events:
            self.on_event(event)


class LoggingValidationObserver(ValidationObserver):
    """
    Observer that logs validation events.

    Logs are formatted according to the language contract:
    - Info/success: PT-BR
    - Warning/error: EN
    """

    def __init__(self, log_level: str = "info"):
        """
        Initialize logging observer.

        Args:
            log_level: Default logging level (info, debug)
        """
        self.log_level = log_level
        self._event_counts: dict[ValidationEventType, int] = {}

    def on_event(self, event: ValidationEvent) -> None:
        """Log the validation event."""
        event_type = event.event_type
        self._event_counts[event_type] = self._event_counts.get(event_type, 0) + 1

        if event_type == ValidationEventType.VALIDATION_STARTED:
            logger.info(f"Validacao iniciada: {event.metadata.get('total_triples', 'N/A')} triplas")

        elif event_type == ValidationEventType.VALIDATION_COMPLETED:
            validated = event.metadata.get("validated", 0)
            rejected = event.metadata.get("rejected", 0)
            duration = event.metadata.get("duration_seconds", 0)
            logger.info(
                f"Validacao concluida: {validated} triplas validadas, "
                f"{rejected} rejeitadas em {duration:.2f}s"
            )

        elif event_type == ValidationEventType.RULE_LOADED:
            source = event.metadata.get("source", "unknown")
            count = event.metadata.get("count", 0)
            logger.info(f"{count} regras carregadas de {source}")

        elif event_type == ValidationEventType.RULE_VIOLATION:
            if self.log_level == "debug":
                logger.debug(
                    f"Rule violation: rule={event.rule_id}, "
                    f"triple={event.triple}, conf={event.confidence:.3f}"
                )

        elif event_type == ValidationEventType.BATCH_COMPLETED:
            batch_num = event.metadata.get("batch_num", 0)
            batch_size = event.metadata.get("batch_size", 0)
            logger.debug(f"Batch {batch_num} completed: {batch_size} items")

    def get_event_counts(self) -> dict[str, int]:
        """Get counts of each event type."""
        return {k.value: v for k, v in self._event_counts.items()}


class MetricsValidationObserver(ValidationObserver):
    """
    Observer that collects validation metrics.

    Aggregates statistics for later reporting and analysis.
    """

    def __init__(self, output_dir: Path | None = None):
        """
        Initialize metrics observer.

        Args:
            output_dir: Directory for metrics output (optional)
        """
        self.output_dir = output_dir
        self._metrics: dict[str, Any] = {
            "total_events": 0,
            "rules_loaded": 0,
            "rules_matched": 0,
            "violations": 0,
            "triples_validated": 0,
            "triples_rejected": 0,
            "confidences": [],
            "start_time": None,
            "end_time": None,
        }

    def on_event(self, event: ValidationEvent) -> None:
        """Collect metrics from the event."""
        self._metrics["total_events"] += 1

        if event.event_type == ValidationEventType.VALIDATION_STARTED:
            self._metrics["start_time"] = event.timestamp

        elif event.event_type == ValidationEventType.VALIDATION_COMPLETED:
            self._metrics["end_time"] = event.timestamp

        elif event.event_type == ValidationEventType.RULE_LOADED:
            self._metrics["rules_loaded"] += event.metadata.get("count", 0)

        elif event.event_type == ValidationEventType.RULE_MATCHED:
            self._metrics["rules_matched"] += 1
            if event.confidence > 0:
                self._metrics["confidences"].append(event.confidence)

        elif event.event_type == ValidationEventType.RULE_VIOLATION:
            self._metrics["violations"] += 1

        elif event.event_type == ValidationEventType.TRIPLE_VALIDATED:
            self._metrics["triples_validated"] += 1

        elif event.event_type == ValidationEventType.TRIPLE_REJECTED:
            self._metrics["triples_rejected"] += 1

    def get_summary(self) -> dict[str, Any]:
        """
        Get summary of collected metrics.

        Returns:
            Dictionary with aggregated metrics
        """
        confidences = self._metrics["confidences"]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        duration = None
        if self._metrics["start_time"] and self._metrics["end_time"]:
            duration = (self._metrics["end_time"] - self._metrics["start_time"]).total_seconds()

        return {
            "total_events": self._metrics["total_events"],
            "rules_loaded": self._metrics["rules_loaded"],
            "rules_matched": self._metrics["rules_matched"],
            "violations": self._metrics["violations"],
            "triples_validated": self._metrics["triples_validated"],
            "triples_rejected": self._metrics["triples_rejected"],
            "average_confidence": avg_confidence,
            "duration_seconds": duration,
        }

    def reset(self) -> None:
        """Reset all metrics."""
        self._metrics = {
            "total_events": 0,
            "rules_loaded": 0,
            "rules_matched": 0,
            "violations": 0,
            "triples_validated": 0,
            "triples_rejected": 0,
            "confidences": [],
            "start_time": None,
            "end_time": None,
        }


class CompositeValidationObserver(SharedCompositeObserver, ValidationObserver):
    """Composite observer that dispatches events to multiple observers."""

    def __init__(self, observers: Sequence[ValidationObserver] | None = None):
        super().__init__(observers or [])

    def on_batch_events(self, events: list[ValidationEvent]) -> None:
        """Dispatch batch to all observers."""
        for observer in self._observers:
            try:
                if hasattr(observer, "on_batch_events"):
                    cast(ValidationObserver, observer).on_batch_events(events)
                else:
                    for event in events:
                        observer.on_event(event)
            except Exception as e:
                logger.error(f"Observer {observer.__class__.__name__} batch failed: {e}")

    def get_observer(self, observer_type: type) -> ValidationObserver | None:
        """
        Get an observer by type.

        Args:
            observer_type: Class of the observer to find

        Returns:
            Observer instance or None
        """
        for observer in self._observers:
            if isinstance(observer, observer_type):
                return observer
        return None
