"""Provide module-level functionality for the PFF codebase.



Notes:

    File: tests/unit/infrastructure/cleanup/test_dir_clean_command.py

"""

from pathlib import Path

from pff.infrastructure.cleanup.commands.filesystem import DirCleanCommand


def _write_file(path: Path, content: str = "data") -> None:
    path.write_text(content, encoding="utf-8")


def test_dir_clean_command_removes_items(tmp_path: Path) -> None:
    """Execute test dir clean command removes items.



    Args:

        tmp_path: Input value used by this callable.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    root = tmp_path / "root"
    root.mkdir()
    _write_file(root / "file.txt")
    subdir = root / "nested"
    subdir.mkdir()
    _write_file(subdir / "nested.txt")

    cmd = DirCleanCommand(label="test", directory=root)
    cmd.execute()

    assert not (root / "file.txt").exists()
    assert not subdir.exists()


def test_dir_clean_command_excludes_dirs(tmp_path: Path) -> None:
    """Execute test dir clean command excludes dirs.



    Args:

        tmp_path: Input value used by this callable.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    root = tmp_path / "root"
    root.mkdir()
    subdir = root / "keep"
    subdir.mkdir()
    _write_file(subdir / "keep.txt")

    cmd = DirCleanCommand(label="test", directory=root, exclude_dirs=[subdir])
    cmd.execute()

    assert subdir.exists()
    assert (subdir / "keep.txt").exists()
