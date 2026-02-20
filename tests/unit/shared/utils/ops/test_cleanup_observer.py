"""Provide module-level functionality for the PFF codebase.



Notes:

    File: tests/unit/shared/utils/ops/test_cleanup_observer.py

"""

from pff.infrastructure.cleanup.observer import (
    CompositeCleanupObserver,
    LoggingCleanupObserver,
)


class DummyCommand:
    """Represent DummyCommand."""

    def __init__(self, label: str):
        """Execute init.



        Args:

            label: Input value used by this callable.

        """

        self.label = label

    def execute(self) -> None:
        """Execute execute."""

        return


def test_composite_observer_dispatches():
    """Execute test composite observer dispatches.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    events = []

    class Recorder(LoggingCleanupObserver):
        """Represent Recorder.



        Notes:

            Encapsulates behavior while preserving architecture boundaries.

        """

        def on_command_start(self, cmd):
            """Execute on command start.



            Args:

                cmd: Input value used by this callable.

            """

            events.append(("start", cmd.label))

        def on_command_complete(self, cmd, duration_ms):
            """Execute on command complete.



            Args:

                cmd: Input value used by this callable.

                duration_ms: Input value used by this callable.

            """

            events.append(("complete", cmd.label))

        def on_cleanup_complete(self, total_freed_bytes):
            """Execute on cleanup complete.



            Args:

                total_freed_bytes: Input value used by this callable.

            """

            events.append(("done", total_freed_bytes))

    obs = CompositeCleanupObserver([Recorder()])
    cmd = DummyCommand("x")

    obs.on_command_start(cmd)
    obs.on_command_complete(cmd, 1.0)
    obs.on_cleanup_complete(10)

    assert ("start", "x") in events
    assert ("complete", "x") in events
    assert ("done", 10) in events
