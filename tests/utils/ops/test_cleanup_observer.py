from pff.utils.ops.cleanup.observer import CompositeCleanupObserver, LoggingCleanupObserver


class DummyCommand:
    def __init__(self, label: str):
        self.label = label

    def execute(self) -> None:
        return


def test_composite_observer_dispatches():
    events = []

    class Recorder(LoggingCleanupObserver):
        def on_command_start(self, cmd):
            events.append(("start", cmd.label))

        def on_command_complete(self, cmd, duration_ms):
            events.append(("complete", cmd.label))

        def on_cleanup_complete(self, total_freed_bytes):
            events.append(("done", total_freed_bytes))

    obs = CompositeCleanupObserver([Recorder()])
    cmd = DummyCommand("x")

    obs.on_command_start(cmd)
    obs.on_command_complete(cmd, 1.0)
    obs.on_cleanup_complete(10)

    assert ("start", "x") in events
    assert ("complete", "x") in events
    assert ("done", 10) in events
