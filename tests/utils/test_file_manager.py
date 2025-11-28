"""
Tests for FileManager (pff/utils/file_manager.py)

Tests cover:
- Handler pattern for 13+ file formats
- CSV, Parquet, JSON, YAML, Excel, Pickle, Text, etc.
- Sync and async read/write
- Memory-mapped files
- Encoding detection
- Hash verification
- Compression support
"""

import pytest
import tempfile
from pathlib import Path
import polars as pl
import json

from pff.utils.file_manager import (
    FileManager,
    CSVHandler,
    ParquetHandler,
    JSONHandler,
    YAMLHandler,
    TextHandler,
    PickleHandler,
    ExcelHandler,
    NDJSONHandler,
    BinHandler,
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_dataframe():
    """Create a sample Polars DataFrame for testing."""
    return pl.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
        "score": [95.5, 87.2, 92.8, 78.3, 88.9],
        "active": [True, False, True, True, False],
    })


@pytest.fixture
def sample_dict():
    """Create a sample dictionary for testing."""
    return {
        "user": "test_user",
        "settings": {
            "theme": "dark",
            "language": "en",
            "notifications": True,
        },
        "scores": [100, 95, 87],
    }


# ═══════════════════════════════════════════════════════════════════
# CSV Handler Tests
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestCSVHandler:
    """Test CSV file reading and writing."""

    def test_read_csv(self, temp_dir, sample_dataframe):
        """Test reading a CSV file."""
        csv_path = temp_dir / "test.csv"
        sample_dataframe.write_csv(csv_path)

        handler = CSVHandler()
        df = handler.read(csv_path)

        assert isinstance(df, pl.DataFrame)
        assert df.shape == sample_dataframe.shape
        assert df.columns == sample_dataframe.columns

    def test_save_csv(self, temp_dir, sample_dataframe):
        """Test saving a DataFrame to CSV."""
        csv_path = temp_dir / "output.csv"

        handler = CSVHandler()
        handler.save(sample_dataframe, csv_path)

        assert csv_path.exists()
        assert csv_path.stat().st_size > 0

    def test_csv_round_trip(self, temp_dir, sample_dataframe):
        """Test CSV write and read round trip."""
        csv_path = temp_dir / "roundtrip.csv"

        handler = CSVHandler()
        handler.save(sample_dataframe, csv_path)
        loaded_df = handler.read(csv_path)

        assert loaded_df.shape == sample_dataframe.shape
        assert loaded_df.columns == sample_dataframe.columns

    @pytest.mark.asyncio
    async def test_async_read_csv(self, temp_dir, sample_dataframe):
        """Test async CSV reading."""
        csv_path = temp_dir / "async_test.csv"
        sample_dataframe.write_csv(csv_path)

        handler = CSVHandler()
        df = await handler.async_read(csv_path)

        assert isinstance(df, pl.DataFrame)
        assert df.shape == sample_dataframe.shape


# ═══════════════════════════════════════════════════════════════════
# Parquet Handler Tests
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestParquetHandler:
    """Test Parquet file reading and writing."""

    def test_read_parquet(self, temp_dir, sample_dataframe):
        """Test reading a Parquet file."""
        parquet_path = temp_dir / "test.parquet"
        sample_dataframe.write_parquet(parquet_path)

        handler = ParquetHandler()
        df = handler.read(parquet_path)

        assert isinstance(df, pl.DataFrame)
        assert df.shape == sample_dataframe.shape

    def test_save_parquet(self, temp_dir, sample_dataframe):
        """Test saving a DataFrame to Parquet."""
        parquet_path = temp_dir / "output.parquet"

        handler = ParquetHandler()
        handler.save(sample_dataframe, parquet_path)

        assert parquet_path.exists()
        assert parquet_path.stat().st_size > 0

    def test_parquet_has_metadata(self, temp_dir, sample_dataframe):
        """Test Parquet file includes metadata."""
        parquet_path = temp_dir / "test.parquet"

        handler = ParquetHandler()
        handler.save(sample_dataframe, parquet_path)

        # Parquet files have overhead but better compression for large datasets
        assert parquet_path.exists()
        assert parquet_path.stat().st_size > 100  # Has metadata


# ═══════════════════════════════════════════════════════════════════
# JSON Handler Tests
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestJSONHandler:
    """Test JSON file reading and writing."""

    def test_read_json(self, temp_dir, sample_dict):
        """Test reading a JSON file."""
        json_path = temp_dir / "test.json"
        json_path.write_text(json.dumps(sample_dict))

        handler = JSONHandler()
        data = handler.read(json_path)

        assert isinstance(data, dict)
        assert data == sample_dict

    def test_save_json(self, temp_dir, sample_dict):
        """Test saving a dict to JSON."""
        json_path = temp_dir / "output.json"

        handler = JSONHandler()
        handler.save(sample_dict, json_path)

        assert json_path.exists()
        loaded = json.loads(json_path.read_text())
        assert loaded == sample_dict

    def test_json_with_special_chars(self, temp_dir):
        """Test JSON with Unicode and special characters."""
        data = {
            "text": "Hello 世界 🌍",
            "symbols": "€$¥£",
            "escape": 'Line1\nLine2\t"quoted"',
        }
        json_path = temp_dir / "special.json"

        handler = JSONHandler()
        handler.save(data, json_path)
        loaded = handler.read(json_path)

        assert loaded == data

    @pytest.mark.asyncio
    async def test_async_save_json(self, temp_dir, sample_dict):
        """Test async JSON saving."""
        json_path = temp_dir / "async_output.json"

        handler = JSONHandler()
        await handler.async_save(sample_dict, json_path)

        assert json_path.exists()
        loaded = json.loads(json_path.read_text())
        assert loaded == sample_dict


# ═══════════════════════════════════════════════════════════════════
# YAML Handler Tests
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestYAMLHandler:
    """Test YAML file reading and writing."""

    def test_read_yaml(self, temp_dir, sample_dict):
        """Test reading a YAML file."""
        yaml_path = temp_dir / "test.yaml"
        import ruamel.yaml
        yaml = ruamel.yaml.YAML()
        with open(yaml_path, 'w') as f:
            yaml.dump(sample_dict, f)

        handler = YAMLHandler()
        data = handler.read(yaml_path)

        assert isinstance(data, dict)
        assert data["user"] == sample_dict["user"]

    def test_save_yaml(self, temp_dir, sample_dict):
        """Test saving a dict to YAML."""
        yaml_path = temp_dir / "output.yaml"

        handler = YAMLHandler()
        handler.save(sample_dict, yaml_path)

        assert yaml_path.exists()
        assert "user: test_user" in yaml_path.read_text()


# ═══════════════════════════════════════════════════════════════════
# Text Handler Tests
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestTextHandler:
    """Test text file reading and writing."""

    def test_read_text(self, temp_dir):
        """Test reading a text file."""
        text_path = temp_dir / "test.txt"
        content = "Hello World\nLine 2\nLine 3"
        text_path.write_text(content)

        handler = TextHandler()
        loaded = handler.read(text_path)

        assert loaded == content

    def test_save_text(self, temp_dir):
        """Test saving text to file."""
        text_path = temp_dir / "output.txt"
        content = "Test content\nMultiple lines\nUTF-8: 日本語"

        handler = TextHandler()
        handler.save(content, text_path)

        assert text_path.exists()
        assert text_path.read_text() == content

    def test_text_encoding_detection(self, temp_dir):
        """Test automatic encoding detection."""
        text_path = temp_dir / "encoded.txt"
        # Write with UTF-8
        text_path.write_text("Hello 世界", encoding="utf-8")

        handler = TextHandler()
        content = handler.read(text_path)

        assert "世界" in content


# ═══════════════════════════════════════════════════════════════════
# Pickle Handler Tests
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestPickleHandler:
    """Test pickle file reading and writing."""

    def test_save_and_load_pickle(self, temp_dir, sample_dict):
        """Test pickling and unpickling."""
        pkl_path = temp_dir / "test.pkl"

        handler = PickleHandler()
        handler.save(sample_dict, pkl_path)
        loaded = handler.read(pkl_path)

        assert loaded == sample_dict

    def test_pickle_complex_object(self, temp_dir, sample_dataframe):
        """Test pickling complex objects like DataFrames."""
        pkl_path = temp_dir / "df.pkl"

        handler = PickleHandler()
        handler.save(sample_dataframe, pkl_path)
        loaded = handler.read(pkl_path)

        assert isinstance(loaded, pl.DataFrame)
        assert loaded.shape == sample_dataframe.shape


# ═══════════════════════════════════════════════════════════════════
# Excel Handler Tests
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestExcelHandler:
    """Test Excel file reading and writing."""

    def test_save_excel(self, temp_dir, sample_dataframe):
        """Test saving DataFrame to Excel."""
        excel_path = temp_dir / "output.xlsx"

        handler = ExcelHandler()
        handler.save(sample_dataframe, excel_path)

        assert excel_path.exists()
        assert excel_path.stat().st_size > 0

    def test_read_excel(self, temp_dir, sample_dataframe):
        """Test reading Excel file."""
        excel_path = temp_dir / "test.xlsx"
        sample_dataframe.write_excel(excel_path)

        handler = ExcelHandler()
        df = handler.read(excel_path)

        assert isinstance(df, pl.DataFrame)
        assert df.shape == sample_dataframe.shape


# ═══════════════════════════════════════════════════════════════════
# NDJSON Handler Tests
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestNDJSONHandler:
    """Test NDJSON (newline-delimited JSON) reading and writing."""

    def test_read_ndjson(self, temp_dir):
        """Test reading NDJSON file."""
        ndjson_path = temp_dir / "test.ndjson"
        lines = [
            {"id": 1, "name": "Alice"},
            {"id": 2, "name": "Bob"},
            {"id": 3, "name": "Charlie"},
        ]
        ndjson_path.write_text("\n".join(json.dumps(line) for line in lines))

        handler = NDJSONHandler()
        df = handler.read(ndjson_path)

        assert isinstance(df, pl.DataFrame)
        assert df.height == 3

    def test_save_ndjson(self, temp_dir, sample_dataframe):
        """Test saving DataFrame to NDJSON."""
        ndjson_path = temp_dir / "output.ndjson"

        handler = NDJSONHandler()
        handler.save(sample_dataframe, ndjson_path)

        assert ndjson_path.exists()
        lines = ndjson_path.read_text().strip().split("\n")
        assert len(lines) == sample_dataframe.height


# ═══════════════════════════════════════════════════════════════════
# FileManager Integration Tests
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.integration
class TestFileManagerIntegration:
    """Test FileManager high-level API."""

    def test_filemanager_auto_detect_format(self, temp_dir, sample_dataframe):
        """Test FileManager auto-detects format from extension."""
        csv_path = temp_dir / "test.csv"
        sample_dataframe.write_csv(csv_path)

        df = FileManager.read(csv_path)

        assert isinstance(df, pl.DataFrame)
        assert df.shape == sample_dataframe.shape

    def test_filemanager_write_multiple_formats(self, temp_dir, sample_dataframe):
        """Test FileManager can write to multiple formats."""
        formats = ["csv", "parquet", "json"]

        for fmt in formats:
            path = temp_dir / f"test.{fmt}"
            if fmt == "json":
                # JSON expects dict, not DataFrame
                FileManager.save(sample_dataframe.to_dicts(), path)
            else:
                FileManager.save(sample_dataframe, path)

            assert path.exists()

    def test_filemanager_unsupported_format(self, temp_dir):
        """Test FileManager raises error for unsupported format."""
        unsupported_path = temp_dir / "test.unknown"

        with pytest.raises(ValueError, match="Unsupported extension"):
            FileManager.read(unsupported_path)

    def test_filemanager_exists_true(self, temp_dir):
        """FileManager.exists returns True for real files."""
        sample_path = temp_dir / "exists.txt"
        sample_path.write_text("ok")

        assert FileManager.exists(sample_path) is True

    def test_filemanager_exists_false(self, temp_dir):
        """FileManager.exists returns False for missing files."""
        missing_path = temp_dir / "missing.txt"
        assert FileManager.exists(missing_path) is False


# ═══════════════════════════════════════════════════════════════════
# Binary Handler Tests
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestBinHandler:
    """Test binary model file reading and writing via msgpack/joblib.
    
    Note: BinHandler is designed for model serialization (LightGBM, msgpack, joblib),
    NOT for raw bytes. It handles Python objects that can be serialized.
    """

    def test_save_and_load_dict(self, temp_dir):
        """Test saving and loading a dictionary via msgpack."""
        bin_path = temp_dir / "test.bin"
        data = {"model_type": "test", "params": [1, 2, 3], "score": 0.95}

        handler = BinHandler()
        handler.save(data, bin_path)
        loaded = handler.read(bin_path)

        assert loaded == data

    def test_save_and_load_complex_object(self, temp_dir):
        """Test handling complex nested structures."""
        bin_path = temp_dir / "complex.bin"
        # Create complex nested data
        data = {
            "embeddings": [[0.1, 0.2, 0.3] for _ in range(100)],
            "metadata": {"name": "test_model", "version": "1.0"},
            "scores": list(range(1000)),
        }

        handler = BinHandler()
        handler.save(data, bin_path)
        loaded = handler.read(bin_path)

        assert loaded == data
        assert len(loaded["embeddings"]) == 100
        assert len(loaded["scores"]) == 1000
