from pathlib import Path

import pytest

from pff.utils.core.file_ops import FileOps


def test_rmtree_sync_respects_interrupt(monkeypatch, tmp_path):
    target = tmp_path / "dir"
    target.mkdir()
    (target / "file.txt").write_text("data")

    monkeypatch.setattr("pff.utils.core.file_ops.should_stop", lambda: True)

    removed = FileOps.rmtree_sync(target)

    assert removed is False
    assert target.exists()


@pytest.mark.asyncio
async def test_rmtree_async_removes_dir(monkeypatch, tmp_path):
    target = tmp_path / "dir_async"
    target.mkdir()
    (target / "file.txt").write_text("async data")

    monkeypatch.setattr("pff.utils.core.file_ops.should_stop", lambda: False)

    removed = await FileOps.rmtree_async(target)

    assert removed is True
    assert target.exists() is False


def test_calculate_size_counts_files(tmp_path):
    target = tmp_path / "measure"
    target.mkdir()
    (target / "a.bin").write_bytes(b"a" * 10)
    sub = target / "sub"
    sub.mkdir()
    (sub / "b.bin").write_bytes(b"b" * 5)

    size = FileOps.calculate_size(target)

    assert size == 15
