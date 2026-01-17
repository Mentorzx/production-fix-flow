import importlib
import sys


def _clear_logger_modules() -> None:
    for name in [
        "pff.shared.core.logger",
        "pff.shared.core.logger",
        "pff.shared.db.events",
        "pff.shared.db",
        "pff.shared",
        "pff",
    ]:
        sys.modules.pop(name, None)


def test_logger_alias_available_during_utils_init() -> None:
    _clear_logger_modules()

    utils = importlib.import_module("pff.shared")

    assert "pff.shared.core.logger" in sys.modules
    assert (
        sys.modules["pff.shared.core.logger"] is sys.modules["pff.shared.core.logger"]
    )

    events = sys.modules["pff.shared.db.events"]
    assert events.logger is utils.logger
