"""Tests for pff/shared/core/file_manager/manager.py - FileManager facade.

Tests the FileManager static methods for file I/O operations without
requiring actual file system operations where possible.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from pff.shared.core.file_manager import FileManager


# ─────────────────────────── Extension Support Tests ───────────────────────────


class TestFileManagerExtensions:
    """Tests for extension support methods."""

    def test_supported_extensions_returns_set(self) -> None:
        """supported_extensions should return a set."""
        result = FileManager.supported_extensions()
        assert isinstance(result, set)
        assert len(result) > 0

    def test_supported_extensions_includes_common_formats(self) -> None:
        """Common file formats should be supported."""
        supported = FileManager.supported_extensions()
        for ext in [".csv", ".json", ".parquet", ".yaml", ".zip"]:
            assert ext in supported, f"Expected {ext} to be supported"

    def test_supports_extension_csv(self) -> None:
        """CSV extension should be supported."""
        assert FileManager.supports_extension(".csv") is True
        assert FileManager.supports_extension(".CSV") is True

    def test_supports_extension_parquet(self) -> None:
        """Parquet extension should be supported."""
        assert FileManager.supports_extension(".parquet") is True
        assert FileManager.supports_extension(".pq") is True

    def test_supports_extension_unsupported(self) -> None:
        """Unsupported extension should return False."""
        assert FileManager.supports_extension(".xyz") is False
        assert FileManager.supports_extension(".unsupported") is False

    def test_assert_supported_path_valid(self) -> None:
        """assert_supported_path should return extension for valid paths."""
        ext = FileManager.assert_supported_path("data/file.csv")
        assert ext == ".csv"

    def test_assert_supported_path_invalid_raises(self) -> None:
        """assert_supported_path should raise for unsupported extensions."""
        with pytest.raises(ValueError, match="Unsupported extension"):
            FileManager.assert_supported_path("data/file.xyz")

    def test_assert_supported_path_with_allowed_exts(self) -> None:
        """assert_supported_path should respect allowed_exts filter."""
        ext = FileManager.assert_supported_path(
            "data/file.csv", allowed_exts=[".csv", ".json"]
        )
        assert ext == ".csv"

        with pytest.raises(ValueError, match="Unsupported extension"):
            FileManager.assert_supported_path("data/file.csv", allowed_exts=[".json"])

    def test_same_extension_true(self) -> None:
        """same_extension should return True for matching extensions."""
        assert FileManager.same_extension("a.csv", "b.csv") is True
        assert FileManager.same_extension("a.CSV", "b.csv") is True

    def test_same_extension_false(self) -> None:
        """same_extension should return False for different extensions."""
        assert FileManager.same_extension("a.csv", "b.json") is False


# ─────────────────────────── Directory Operations Tests ───────────────────────


class TestFileManagerDirectoryOps:
    """Tests for directory operations."""

    def test_exists_file(self, tmp_path: Path) -> None:
        """exists should return True for existing file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")
        assert FileManager.exists(test_file) is True

    def test_exists_nonexistent(self, tmp_path: Path) -> None:
        """exists should return False for non-existent path."""
        assert FileManager.exists(tmp_path / "nonexistent.txt") is False

    def test_ensure_dir_creates_directory(self, tmp_path: Path) -> None:
        """ensure_dir should create directory if it doesn't exist."""
        new_dir = tmp_path / "new_dir"
        result = FileManager.ensure_dir(new_dir)
        assert result == new_dir
        assert new_dir.exists()

    def test_ensure_dir_nested(self, tmp_path: Path) -> None:
        """ensure_dir should create nested directories."""
        nested_dir = tmp_path / "a" / "b" / "c"
        result = FileManager.ensure_dir(nested_dir)
        assert result == nested_dir
        assert nested_dir.exists()

    def test_ensure_parent_dir(self, tmp_path: Path) -> None:
        """ensure_parent_dir should create parent directory."""
        file_path = tmp_path / "subdir" / "file.txt"
        result = FileManager.ensure_parent_dir(file_path)
        assert result == file_path.parent
        assert file_path.parent.exists()

    def test_glob_finds_files(self, tmp_path: Path) -> None:
        """glob should find files matching pattern."""
        (tmp_path / "a.csv").write_text("a")
        (tmp_path / "b.csv").write_text("b")
        (tmp_path / "c.json").write_text("{}")

        result = FileManager.glob(tmp_path, "*.csv")
        assert len(result) == 2
        assert all(p.suffix == ".csv" for p in result)

    def test_glob_no_matches(self, tmp_path: Path) -> None:
        """glob should return empty list when no matches."""
        result = FileManager.glob(tmp_path, "*.xyz")
        assert result == []


# ─────────────────────────── Bytes/Text I/O Tests ───────────────────────────


class TestFileManagerBytesTextIO:
    """Tests for bytes and text I/O."""

    def test_read_bytes(self, tmp_path: Path) -> None:
        """read_bytes should return file contents as bytes."""
        test_file = tmp_path / "test.bin"
        test_file.write_bytes(b"binary content")
        result = FileManager.read_bytes(test_file)
        assert result == b"binary content"

    def test_write_bytes(self, tmp_path: Path) -> None:
        """write_bytes should write bytes to file."""
        test_file = tmp_path / "test.bin"
        FileManager.write_bytes(b"new content", test_file)
        assert test_file.read_bytes() == b"new content"

    def test_write_bytes_creates_parent_dirs(self, tmp_path: Path) -> None:
        """write_bytes should create parent directories."""
        test_file = tmp_path / "subdir" / "test.bin"
        FileManager.write_bytes(b"content", test_file)
        assert test_file.exists()

    def test_read_text(self, tmp_path: Path) -> None:
        """read_text should return file contents as string."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world", encoding="utf-8")
        result = FileManager.read_text(test_file)
        assert result == "hello world"

    def test_write_text(self, tmp_path: Path) -> None:
        """write_text should write string to file."""
        test_file = tmp_path / "test.txt"
        FileManager.write_text("hello", test_file)
        assert test_file.read_text() == "hello"

    def test_write_text_creates_parent_dirs(self, tmp_path: Path) -> None:
        """write_text should create parent directories."""
        test_file = tmp_path / "subdir" / "test.txt"
        FileManager.write_text("content", test_file)
        assert test_file.exists()


# ─────────────────────────── JSON Operations Tests ───────────────────────────


class TestFileManagerJSON:
    """Tests for JSON operations."""

    def test_json_dumps_basic(self) -> None:
        """json_dumps should serialize object to JSON string."""
        obj = {"key": "value", "number": 42}
        result = FileManager.json_dumps(obj)
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert parsed == obj

    def test_json_dumps_sort_keys(self) -> None:
        """json_dumps with sort_keys should sort keys."""
        obj = {"z": 1, "a": 2, "m": 3}
        result = FileManager.json_dumps(obj, sort_keys=True)
        # First key should be 'a' when sorted
        assert result.index('"a"') < result.index('"m"') < result.index('"z"')

    def test_json_loads_string(self) -> None:
        """json_loads should parse JSON string."""
        json_str = '{"key": "value", "number": 42}'
        result = FileManager.json_loads(json_str)
        assert result == {"key": "value", "number": 42}

    def test_json_loads_bytes(self) -> None:
        """json_loads should parse JSON bytes."""
        json_bytes = b'{"key": "value"}'
        result = FileManager.json_loads(json_bytes)
        assert result == {"key": "value"}


# ─────────────────────────── Polars Scan Tests ───────────────────────────


class TestFileManagerPolarsScans:
    """Tests for Polars scan operations."""

    def test_scan_csv_returns_lazyframe(self, tmp_path: Path) -> None:
        """scan_csv should return a LazyFrame."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("a,b\n1,2\n3,4")
        result = FileManager.scan_csv(str(csv_file))
        assert isinstance(result, pl.LazyFrame)

    def test_scan_parquet_returns_lazyframe(self, tmp_path: Path) -> None:
        """scan_parquet should return a LazyFrame."""
        df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
        parquet_file = tmp_path / "test.parquet"
        df.write_parquet(parquet_file)
        result = FileManager.scan_parquet(str(parquet_file))
        assert isinstance(result, pl.LazyFrame)

    def test_scan_ndjson_returns_lazyframe(self, tmp_path: Path) -> None:
        """scan_ndjson should return a LazyFrame."""
        ndjson_file = tmp_path / "test.ndjson"
        ndjson_file.write_text('{"a": 1, "b": 2}\n{"a": 3, "b": 4}')
        result = FileManager.scan_ndjson(str(ndjson_file))
        assert isinstance(result, pl.LazyFrame)


# ─────────────────────────── File Operations Tests ───────────────────────────


class TestFileManagerFileOps:
    """Tests for file manipulation operations."""

    def test_get_hash_returns_md5(self, tmp_path: Path) -> None:
        """get_hash should return MD5 hash of file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")
        result = FileManager.get_hash(test_file)
        assert isinstance(result, str)
        assert len(result) == 32  # MD5 hex digest length
        # Same content should produce same hash
        assert FileManager.get_hash(test_file) == result

    def test_get_hash_nonexistent_file(self, tmp_path: Path) -> None:
        """get_hash should return empty string for non-existent file."""
        result = FileManager.get_hash(tmp_path / "nonexistent.txt")
        assert result == ""

    def test_delete_file(self, tmp_path: Path) -> None:
        """delete_file should remove file and return True."""
        test_file = tmp_path / "to_delete.txt"
        test_file.write_text("content")
        assert test_file.exists()
        result = FileManager.delete_file(test_file)
        assert result is True
        assert not test_file.exists()

    def test_delete_file_nonexistent(self, tmp_path: Path) -> None:
        """delete_file should return False for non-existent file."""
        result = FileManager.delete_file(tmp_path / "nonexistent.txt")
        assert result is False

    def test_delete_directory(self, tmp_path: Path) -> None:
        """delete_directory should remove directory and contents."""
        test_dir = tmp_path / "to_delete"
        test_dir.mkdir()
        (test_dir / "file.txt").write_text("content")
        assert test_dir.exists()
        result = FileManager.delete_directory(test_dir)
        assert result is True
        assert not test_dir.exists()

    def test_delete_directory_nonexistent(self, tmp_path: Path) -> None:
        """delete_directory should return False for non-existent directory."""
        result = FileManager.delete_directory(tmp_path / "nonexistent")
        assert result is False

    def test_copy_file(self, tmp_path: Path) -> None:
        """copy_file should copy file to destination."""
        src = tmp_path / "source.txt"
        src.write_text("content")
        dest = tmp_path / "dest.txt"
        result = FileManager.copy_file(src, dest)
        assert result == dest
        assert dest.read_text() == "content"

    def test_copy_file_creates_parent_dirs(self, tmp_path: Path) -> None:
        """copy_file should create parent directories."""
        src = tmp_path / "source.txt"
        src.write_text("content")
        dest = tmp_path / "subdir" / "dest.txt"
        FileManager.copy_file(src, dest)
        assert dest.exists()

    def test_copy_directory(self, tmp_path: Path) -> None:
        """copy_directory should copy directory recursively."""
        src_dir = tmp_path / "source_dir"
        src_dir.mkdir()
        (src_dir / "file.txt").write_text("content")
        dest_dir = tmp_path / "dest_dir"
        result = FileManager.copy_directory(src_dir, dest_dir)
        assert result == dest_dir
        assert (dest_dir / "file.txt").read_text() == "content"


# ─────────────────────────── Timestamp Tests ───────────────────────────


class TestFileManagerTimestamp:
    """Tests for timestamp utility."""

    def test_get_timestamp_format(self) -> None:
        """get_timestamp should return ISO format string."""
        result = FileManager.get_timestamp()
        assert isinstance(result, str)
        # Should be ISO format like "2024-01-15T12:00:00+00:00"
        assert "T" in result
        assert len(result) >= 19  # Minimum ISO format length


# ─────────────────────────── Read/Save Integration Tests ───────────────────────


class TestFileManagerReadSave:
    """Integration tests for read/save operations."""

    def test_read_csv_native(self, tmp_path: Path) -> None:
        """read with return_native should return DataFrame for CSV."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("a,b\n1,2\n3,4")
        result = FileManager.read(csv_file, return_native=True)
        assert isinstance(result, pl.DataFrame)
        assert result.shape == (2, 2)

    def test_save_polars_dataframe_csv(self, tmp_path: Path) -> None:
        """save should write DataFrame to CSV."""
        df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
        csv_file = tmp_path / "output.csv"
        FileManager.save(df, csv_file)
        assert csv_file.exists()

    def test_save_polars_dataframe_parquet(self, tmp_path: Path) -> None:
        """save should write DataFrame to Parquet."""
        df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
        parquet_file = tmp_path / "output.parquet"
        FileManager.save(df, parquet_file)
        assert parquet_file.exists()
        # Verify content
        read_df = pl.read_parquet(parquet_file)
        assert read_df.shape == (2, 2)

    def test_save_dict_json(self, tmp_path: Path) -> None:
        """save should write dict to JSON."""
        data = {"key": "value", "number": 42}
        json_file = tmp_path / "output.json"
        FileManager.save(data, json_file)
        assert json_file.exists()
        content = json.loads(json_file.read_text())
        assert content == data

    def test_read_json_native(self, tmp_path: Path) -> None:
        """read with return_native should return dict for JSON."""
        json_file = tmp_path / "test.json"
        json_file.write_text('{"key": "value"}')
        result = FileManager.read(json_file, return_native=True)
        assert result == {"key": "value"}

    def test_read_yaml_native(self, tmp_path: Path) -> None:
        """read with return_native should return dict for YAML."""
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text("key: value\nnumber: 42")
        result = FileManager.read(yaml_file, return_native=True)
        assert result == {"key": "value", "number": 42}

    def test_save_dict_yaml(self, tmp_path: Path) -> None:
        """save should write dict to YAML."""
        data = {"key": "value", "number": 42}
        yaml_file = tmp_path / "output.yaml"
        FileManager.save(data, yaml_file)
        assert yaml_file.exists()
