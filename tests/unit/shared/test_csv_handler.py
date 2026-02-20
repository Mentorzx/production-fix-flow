"""Provide module-level functionality for the PFF codebase.



Notes:

    File: tests/unit/shared/test_csv_handler.py

"""

from __future__ import annotations

import asyncio
import threading

import polars as pl

from pff.shared.core.file_manager.handlers.csv import CSVHandler


def test_csv_handler_async_save_invokes_once(tmp_path, monkeypatch) -> None:
    """Execute test csv handler async save invokes once.



    Args:

        tmp_path: Input value used by this callable.

        monkeypatch: Input value used by this callable.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    handler = CSVHandler()
    calls = {"count": 0}
    lock = threading.Lock()

    def _save(obj, path, **kwargs):
        """Execute save.



        Args:

            obj: Input value used by this callable.

            path: Input value used by this callable.

            **kwargs: Additional keyword arguments.

        """

        with lock:
            calls["count"] += 1

    monkeypatch.setattr(handler, "save", _save)

    df = pl.DataFrame({"a": [1], "b": ["x"]})
    asyncio.run(handler.async_save(df, tmp_path / "out.csv"))

    assert calls["count"] == 1
