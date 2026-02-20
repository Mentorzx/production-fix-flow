"""Provide module-level functionality for the PFF codebase.



Notes:

    File: tests/unit/shared/test_async_io.py

"""

from __future__ import annotations

import asyncio

from pff.shared.core.file_manager import FileManager
from pff.shared.core.file_manager.async_io import read_async_content


def test_read_async_content_chunked(tmp_path) -> None:
    """Execute test read async content chunked.



    Args:

        tmp_path: Input value used by this callable.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    payload = b"abc" * 1024
    path = tmp_path / "payload.bin"
    FileManager.write_bytes(payload, path)

    content = asyncio.run(read_async_content(path, chunk_size=128))

    assert content == payload
