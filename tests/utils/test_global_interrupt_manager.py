import pytest

from pff.utils.ops import global_interrupt_manager as gim


def test_global_interrupt_manager_logs_warnings_in_english(monkeypatch):
    messages: dict[str, list[str]] = {"warning": [], "error": [], "info": []}

    class DummyLogger:
        def warning(self, msg, *args, **kwargs):
            messages["warning"].append(str(msg))

        def error(self, msg, *args, **kwargs):
            messages["error"].append(str(msg))

        def info(self, msg, *args, **kwargs):
            messages["info"].append(str(msg))

        def debug(self, msg, *args, **kwargs):
            pass

    monkeypatch.setattr(gim, "logger", DummyLogger())
    manager = gim.get_interrupt_manager()
    manager.reset()

    manager.force_stop("test-run")
    assert any("Forced stop requested" in msg for msg in messages["warning"])

    manager.reset()
    manager._should_stop = True

    with pytest.raises(KeyboardInterrupt):
        gim.check_interruption()

    assert any("Operation interrupted" in msg for msg in messages["warning"])

    manager.reset()
