"""Provide module-level functionality for the PFF codebase.



Notes:

    File: tests/unit/infrastructure/hpo/test_background_process.py

"""

from pff.infrastructure.hpo.background_process import BackgroundProcess


def test_stop_is_noop_without_process() -> None:
    """Execute test stop is noop without process.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    proc = BackgroundProcess(["echo", "noop"])
    proc.stop()
