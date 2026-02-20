"""Provide module-level functionality for the PFF codebase.



Notes:

    File: tests/unit/shared/utils/test_global_interrupt_manager.py

"""

import signal

import pytest

from pff.shared.ops import global_interrupt_manager as gim


def test_global_interrupt_manager_logs_warnings_in_english(monkeypatch):
    """Execute test global interrupt manager logs warnings in english.



    Args:

        monkeypatch: Input value used by this callable.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    messages: dict[str, list[str]] = {"warning": [], "error": [], "info": []}

    class DummyLogger:
        """Represent DummyLogger."""

        def warning(self, msg, *args, **kwargs):
            """Execute warning.



            Args:

                msg: Input value used by this callable.

                *args: Additional positional arguments.

                **kwargs: Additional keyword arguments.

            """

            messages["warning"].append(str(msg))

        def error(self, msg, *args, **kwargs):
            """Execute error.



            Args:

                msg: Input value used by this callable.

                *args: Additional positional arguments.

                **kwargs: Additional keyword arguments.

            """

            messages["error"].append(str(msg))

        def info(self, msg, *args, **kwargs):
            """Execute info.



            Args:

                msg: Input value used by this callable.

                *args: Additional positional arguments.

                **kwargs: Additional keyword arguments.

            """

            messages["info"].append(str(msg))

        def debug(self, msg, *args, **kwargs):
            """Execute debug.



            Args:

                msg: Input value used by this callable.

                *args: Additional positional arguments.

                **kwargs: Additional keyword arguments.

            """

            pass

    monkeypatch.setattr(gim, "logger", DummyLogger())
    manager = gim.get_interrupt_manager()
    manager.reset()


def test_global_interrupt_manager_signal_is_idempotent(monkeypatch):
    """Execute test global interrupt manager signal is idempotent.



    Args:

        monkeypatch: Input value used by this callable.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    messages: dict[str, list[str]] = {"warning": [], "error": [], "info": []}

    class DummyLogger:
        """Represent DummyLogger."""

        def warning(self, msg, *args, **kwargs):
            """Execute warning.



            Args:

                msg: Input value used by this callable.

                *args: Additional positional arguments.

                **kwargs: Additional keyword arguments.

            """

            messages["warning"].append(str(msg))

        def error(self, msg, *args, **kwargs):
            """Execute error.



            Args:

                msg: Input value used by this callable.

                *args: Additional positional arguments.

                **kwargs: Additional keyword arguments.

            """

            messages["error"].append(str(msg))

        def info(self, msg, *args, **kwargs):
            """Execute info.



            Args:

                msg: Input value used by this callable.

                *args: Additional positional arguments.

                **kwargs: Additional keyword arguments.

            """

            messages["info"].append(str(msg))

        def debug(self, msg, *args, **kwargs):
            """Execute debug.



            Args:

                msg: Input value used by this callable.

                *args: Additional positional arguments.

                **kwargs: Additional keyword arguments.

            """

            pass

    monkeypatch.setattr(gim, "logger", DummyLogger())
    manager = gim.get_interrupt_manager()
    manager.reset()

    calls = {"count": 0}

    def _callback():
        calls["count"] += 1

    manager.register_callback(_callback, label="test_callback")

    with pytest.raises(KeyboardInterrupt):
        manager._handle_signal(signal.SIGINT)

    assert calls["count"] == 1
    assert sum("SIGINT received" in msg for msg in messages["warning"]) == 1

    with pytest.raises(SystemExit):
        manager._handle_signal(signal.SIGINT)

    manager.reset()

    manager.force_stop("test-run")
    assert any("Forced stop requested" in msg for msg in messages["warning"])

    manager.reset()
    manager._stop_event.set()

    with pytest.raises(KeyboardInterrupt):
        gim.check_interruption()

    assert any("Operation interrupted" in msg for msg in messages["warning"])

    manager.reset()
