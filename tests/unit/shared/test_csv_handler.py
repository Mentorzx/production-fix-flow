from __future__ import annotations

import asyncio
import threading

import polars as pl

from pff.shared.core.file_manager.handlers.csv import CSVHandler


def test_csv_handler_async_save_invokes_once(tmp_path, monkeypatch) -> None:
    handler = CSVHandler()
    calls = {"count": 0}
    lock = threading.Lock()

    def _save(obj, path, **kwargs):
        with lock:
            calls["count"] += 1

    monkeypatch.setattr(handler, "save", _save)

    df = pl.DataFrame({"a": [1], "b": ["x"]})
    asyncio.run(handler.async_save(df, tmp_path / "out.csv"))

    assert calls["count"] == 1
