"""
Tests for ValidationObserver pattern implementation.

Tests cover:
    - ValidationEvent creation and serialization
    - LoggingValidationObserver event handling
    - MetricsValidationObserver aggregation
    - CompositeValidationObserver dispatch
"""

from datetime import datetime

import pytest

from pff.services.validation_observer import (
    CompositeValidationObserver,
    LoggingValidationObserver,
    MetricsValidationObserver,
    ValidationEvent,
    ValidationEventType,
    ValidationObserver,
)


class TestValidationEvent:
    """Tests for ValidationEvent dataclass."""

    def test_create_minimal_event(self):
        """Test creating event with minimal fields."""
        event = ValidationEvent(event_type=ValidationEventType.RULE_LOADED)
        assert event.event_type == ValidationEventType.RULE_LOADED
        assert event.rule_id is None
        assert event.triple is None
        assert event.confidence == 0.0
        assert event.metadata == {}
        assert isinstance(event.timestamp, datetime)

    def test_create_full_event(self):
        """Test creating event with all fields."""
        event = ValidationEvent(
            event_type=ValidationEventType.RULE_MATCHED,
            rule_id="rule_001",
            triple=("A", "knows", "B"),
            confidence=0.85,
            metadata={"source": "anyburl"},
        )
        assert event.rule_id == "rule_001"
        assert event.triple == ("A", "knows", "B")
        assert event.confidence == 0.85
        assert event.metadata["source"] == "anyburl"

    def test_to_dict(self):
        """Test serialization to dictionary."""
        event = ValidationEvent(
            event_type=ValidationEventType.RULE_VIOLATION,
            rule_id="rule_002",
            triple=("X", "hates", "Y"),
            confidence=0.3,
        )
        data = event.to_dict()

        assert data["event_type"] == "rule_violation"
        assert data["rule_id"] == "rule_002"
        assert data["triple"] == ["X", "hates", "Y"]
        assert data["confidence"] == 0.3
        assert "timestamp" in data


class TestLoggingValidationObserver:
    """Tests for LoggingValidationObserver."""

    def test_count_events(self):
        """Test event counting."""
        observer = LoggingValidationObserver()

        observer.on_event(ValidationEvent(event_type=ValidationEventType.RULE_LOADED))
        observer.on_event(ValidationEvent(event_type=ValidationEventType.RULE_LOADED))
        observer.on_event(ValidationEvent(event_type=ValidationEventType.RULE_MATCHED))

        counts = observer.get_event_counts()
        assert counts["rule_loaded"] == 2
        assert counts["rule_matched"] == 1

    def test_log_level_setting(self):
        """Test log level configuration."""
        observer = LoggingValidationObserver(log_level="debug")
        assert observer.log_level == "debug"

    def test_validation_started_event(self):
        """Test handling validation started event."""
        observer = LoggingValidationObserver()
        event = ValidationEvent(
            event_type=ValidationEventType.VALIDATION_STARTED,
            metadata={"total_triples": 1000},
        )
        observer.on_event(event)
        assert observer.get_event_counts()["validation_started"] == 1

    def test_validation_completed_event(self):
        """Test handling validation completed event."""
        observer = LoggingValidationObserver()
        event = ValidationEvent(
            event_type=ValidationEventType.VALIDATION_COMPLETED,
            metadata={"validated": 900, "rejected": 100, "duration_seconds": 5.5},
        )
        observer.on_event(event)
        assert observer.get_event_counts()["validation_completed"] == 1


class TestMetricsValidationObserver:
    """Tests for MetricsValidationObserver."""

    def test_aggregate_metrics(self):
        """Test metrics aggregation."""
        observer = MetricsValidationObserver()

        observer.on_event(ValidationEvent(
            event_type=ValidationEventType.VALIDATION_STARTED,
        ))
        observer.on_event(ValidationEvent(
            event_type=ValidationEventType.RULE_LOADED,
            metadata={"count": 10},
        ))
        observer.on_event(ValidationEvent(
            event_type=ValidationEventType.RULE_MATCHED,
            confidence=0.8,
        ))
        observer.on_event(ValidationEvent(
            event_type=ValidationEventType.RULE_MATCHED,
            confidence=0.9,
        ))
        observer.on_event(ValidationEvent(
            event_type=ValidationEventType.TRIPLE_VALIDATED,
        ))
        observer.on_event(ValidationEvent(
            event_type=ValidationEventType.TRIPLE_REJECTED,
        ))
        observer.on_event(ValidationEvent(
            event_type=ValidationEventType.RULE_VIOLATION,
        ))
        observer.on_event(ValidationEvent(
            event_type=ValidationEventType.VALIDATION_COMPLETED,
        ))

        summary = observer.get_summary()

        assert summary["total_events"] == 8
        assert summary["rules_loaded"] == 10
        assert summary["rules_matched"] == 2
        assert summary["violations"] == 1
        assert summary["triples_validated"] == 1
        assert summary["triples_rejected"] == 1
        assert summary["average_confidence"] == pytest.approx(0.85)

    def test_reset_metrics(self):
        """Test metrics reset."""
        observer = MetricsValidationObserver()
        observer.on_event(ValidationEvent(event_type=ValidationEventType.RULE_MATCHED, confidence=0.9))
        observer.reset()

        summary = observer.get_summary()
        assert summary["total_events"] == 0
        assert summary["rules_matched"] == 0

    def test_empty_average_confidence(self):
        """Test average confidence with no matched rules."""
        observer = MetricsValidationObserver()
        summary = observer.get_summary()
        assert summary["average_confidence"] == 0.0


class TestCompositeValidationObserver:
    """Tests for CompositeValidationObserver."""

    def test_dispatch_to_multiple_observers(self):
        """Test dispatching events to all observers."""
        logging_obs = LoggingValidationObserver()
        metrics_obs = MetricsValidationObserver()

        composite = CompositeValidationObserver([logging_obs, metrics_obs])

        event = ValidationEvent(
            event_type=ValidationEventType.RULE_MATCHED,
            confidence=0.85,
        )
        composite.on_event(event)

        assert logging_obs.get_event_counts()["rule_matched"] == 1
        assert metrics_obs.get_summary()["rules_matched"] == 1

    def test_add_remove_observer(self):
        """Test adding and removing observers."""
        composite = CompositeValidationObserver()
        logging_obs = LoggingValidationObserver()

        composite.add_observer(logging_obs)
        assert len(composite._observers) == 1

        composite.remove_observer(logging_obs)
        assert len(composite._observers) == 0

    def test_get_observer_by_type(self):
        """Test retrieving observer by type."""
        logging_obs = LoggingValidationObserver()
        metrics_obs = MetricsValidationObserver()
        composite = CompositeValidationObserver([logging_obs, metrics_obs])

        found = composite.get_observer(MetricsValidationObserver)
        assert found is metrics_obs

        not_found = composite.get_observer(str)  # Type not in composite
        assert not_found is None

    def test_batch_events(self):
        """Test batch event processing."""
        metrics_obs = MetricsValidationObserver()
        composite = CompositeValidationObserver([metrics_obs])

        events = [
            ValidationEvent(event_type=ValidationEventType.RULE_MATCHED, confidence=0.8),
            ValidationEvent(event_type=ValidationEventType.RULE_MATCHED, confidence=0.9),
            ValidationEvent(event_type=ValidationEventType.TRIPLE_VALIDATED),
        ]
        composite.on_batch_events(events)

        summary = metrics_obs.get_summary()
        assert summary["rules_matched"] == 2
        assert summary["triples_validated"] == 1

    def test_observer_error_handling(self):
        """Test that errors in one observer don't break others."""

        class FailingObserver(ValidationObserver):
            def on_event(self, event):
                raise RuntimeError("Test error")

        metrics_obs = MetricsValidationObserver()
        failing_obs = FailingObserver()

        composite = CompositeValidationObserver([failing_obs, metrics_obs])

        # Should not raise, and metrics should still be updated
        event = ValidationEvent(event_type=ValidationEventType.RULE_MATCHED, confidence=0.8)
        composite.on_event(event)

        assert metrics_obs.get_summary()["rules_matched"] == 1


class TestValidationEventType:
    """Tests for ValidationEventType enum."""

    def test_all_event_types(self):
        """Test that all expected event types exist."""
        expected_types = [
            "RULE_LOADED",
            "RULE_MATCHED",
            "RULE_VIOLATION",
            "TRIPLE_VALIDATED",
            "TRIPLE_REJECTED",
            "VALIDATION_STARTED",
            "VALIDATION_COMPLETED",
            "BATCH_COMPLETED",
        ]
        for type_name in expected_types:
            assert hasattr(ValidationEventType, type_name)

    def test_event_type_values(self):
        """Test event type string values."""
        assert ValidationEventType.RULE_LOADED.value == "rule_loaded"
        assert ValidationEventType.RULE_VIOLATION.value == "rule_violation"
