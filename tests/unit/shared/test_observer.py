"""Tests for pff/shared/observer.py - observer pattern utilities."""

from pff.shared.observer import (
    CompositeObserver,
    NullObserver,
)


class TestNullObserver:
    """Tests for NullObserver no-op implementation."""

    def test_null_observer_on_start_returns_none(self):
        """Verify on_start returns None."""
        observer = NullObserver()
        result = observer.on_start()
        assert result is None

    def test_null_observer_on_step_returns_none(self):
        """Verify on_step returns None."""
        observer = NullObserver()
        result = observer.on_step()
        assert result is None

    def test_null_observer_on_complete_returns_none(self):
        """Verify on_complete returns None."""
        observer = NullObserver()
        result = observer.on_complete()
        assert result is None

    def test_null_observer_on_error_returns_none(self):
        """Verify on_error returns None."""
        observer = NullObserver()
        result = observer.on_error()
        assert result is None

    def test_null_observer_accepts_context(self):
        """Verify context parameter is accepted."""
        observer = NullObserver()
        context = {"step": 1, "total": 100}
        # Should not raise
        observer.on_start(context)
        observer.on_step(context)
        observer.on_complete(context)
        observer.on_error(context)


class MockEventObserver:
    """Mock implementation of EventObserver for testing."""

    def __init__(self):
        """Execute init."""

        self.events = []

    def on_event(self, event):
        """Execute on event.



        Args:

            event: Input value used by this callable.

        """

        self.events.append(event)


class FailingObserver:
    """Observer that always raises an exception."""

    def on_event(self, event):
        """Execute on event.



        Args:

            event: Input value used by this callable.

        """

        raise ValueError("Intentional failure")


class TestCompositeObserver:
    """Tests for CompositeObserver pattern."""

    def test_composite_observer_empty_on_event(self):
        """Verify empty composite handles events without error."""
        composite = CompositeObserver()
        composite.on_event({"type": "test"})

    def test_composite_observer_add_single(self):
        """Verify single observer receives events."""
        mock = MockEventObserver()
        composite = CompositeObserver()
        composite.add(mock)
        composite.on_event({"type": "test"})
        assert len(mock.events) == 1
        assert mock.events[0] == {"type": "test"}

    def test_composite_observer_add_multiple(self):
        """Verify multiple observers receive same event."""
        mock1 = MockEventObserver()
        mock2 = MockEventObserver()
        composite = CompositeObserver([mock1, mock2])
        composite.on_event({"type": "broadcast"})
        assert len(mock1.events) == 1
        assert len(mock2.events) == 1
        assert mock1.events[0] == mock2.events[0]

    def test_composite_observer_fluent_add(self):
        """Verify fluent API for adding observers."""
        mock = MockEventObserver()
        composite = CompositeObserver().add(mock)
        assert composite is not None
        composite.on_event("test")
        assert "test" in mock.events

    def test_composite_observer_remove(self):
        """Verify observer can be removed."""
        mock = MockEventObserver()
        composite = CompositeObserver([mock])
        composite.remove(mock)
        composite.on_event("test")
        assert len(mock.events) == 0

    def test_composite_observer_remove_nonexistent(self):
        """Verify removing nonexistent observer doesn't raise."""
        mock = MockEventObserver()
        composite = CompositeObserver()
        composite.remove(mock)

    def test_composite_observer_add_observer_method(self):
        """Verify add_observer method works."""
        mock = MockEventObserver()
        composite = CompositeObserver()
        composite.add_observer(mock)
        composite.on_event("event1")
        assert "event1" in mock.events

    def test_composite_observer_remove_observer_method(self):
        """Verify remove_observer method works."""
        mock = MockEventObserver()
        composite = CompositeObserver([mock])
        composite.remove_observer(mock)
        composite.on_event("event1")
        assert len(mock.events) == 0

    def test_composite_observer_handles_failing_observer(self):
        """Verify failing observer doesn't break others."""
        mock = MockEventObserver()
        failing = FailingObserver()
        composite = CompositeObserver([failing, mock])
        # Should not raise, and mock should still receive event
        composite.on_event("test")
        assert "test" in mock.events

    def test_composite_observer_multiple_events(self):
        """Verify multiple events are tracked separately."""
        mock = MockEventObserver()
        composite = CompositeObserver([mock])
        composite.on_event("event1")
        composite.on_event("event2")
        composite.on_event("event3")
        assert mock.events == ["event1", "event2", "event3"]

    def test_composite_observer_event_ordering(self):
        """Verify events are dispatched in order of observer addition."""
        order = []

        class OrderTracker:
            """Represent OrderTracker."""

            def __init__(self, name):
                """Execute init.



                Args:

                    name: Input value used by this callable.

                """

                self.name = name

            def on_event(self, event):
                """Execute on event.



                Args:

                    event: Input value used by this callable.

                """

                order.append(self.name)

        composite = CompositeObserver([OrderTracker("first"), OrderTracker("second")])
        composite.on_event("test")
        assert order == ["first", "second"]
