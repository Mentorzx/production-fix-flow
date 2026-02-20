"""Provide module-level functionality for the PFF codebase.



Notes:

    File: tests/unit/shared/utils/test_file_ops.py

"""

import pytest

from pff.infrastructure.cleanup.file_ops import FileOps


def test_rmtree_sync_respects_interrupt(monkeypatch, tmp_path):
    """Execute test rmtree sync respects interrupt.



    Args:

        monkeypatch: Input value used by this callable.

        tmp_path: Input value used by this callable.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    target = tmp_path / "dir"
    target.mkdir()
    (target / "file.txt").write_text("data")

    monkeypatch.setattr("pff.infrastructure.cleanup.file_ops.should_stop", lambda: True)

    removed = FileOps.rmtree_sync(target)

    assert removed is False
    assert target.exists()


@pytest.mark.asyncio
async def test_rmtree_async_removes_dir(monkeypatch, tmp_path):
    """Execute test rmtree async removes dir.



    Args:

        monkeypatch: Input value used by this callable.

        tmp_path: Input value used by this callable.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    target = tmp_path / "dir_async"
    target.mkdir()
    (target / "file.txt").write_text("async data")

    monkeypatch.setattr("pff.infrastructure.cleanup.file_ops.should_stop", lambda: False)

    removed = await FileOps.rmtree_async(target)

    assert removed is True
    assert target.exists() is False


def test_calculate_size_counts_files(tmp_path):
    """Execute test calculate size counts files.



    Args:

        tmp_path: Input value used by this callable.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    target = tmp_path / "measure"
    target.mkdir()
    (target / "a.bin").write_bytes(b"a" * 10)
    sub = target / "sub"
    sub.mkdir()
    (sub / "b.bin").write_bytes(b"b" * 5)

    size = FileOps.calculate_size(target)

    assert size == 15
