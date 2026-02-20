"""Provide module-level functionality for the PFF codebase.



Notes:

    File: tests/unit/shared/utils/test_file_manager_read_text.py

"""

from __future__ import annotations

import shutil
from pathlib import Path

from pff.shared import FileManager


def test_file_manager_read_text_reads_roundtrip() -> None:
    """Execute test file manager read text reads roundtrip.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    fm = FileManager()
    tmp_dir = Path("outputs") / "temp_tests" / "file_manager_read_text"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    path = tmp_dir / "sample.txt"
    fm.save("hello", path)
    assert FileManager.read_text(path) == "hello"

    shutil.rmtree(tmp_dir, ignore_errors=True)
