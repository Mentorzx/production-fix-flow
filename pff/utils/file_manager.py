import hashlib
import io
import mmap
import os
import pickle
import tempfile
import zipfile
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiofile
import joblib
import lightgbm as lgb
import msgspec
import orjson
import polars as pl
import ruamel.yaml
from charset_normalizer import detect

from pff.utils import ConcurrencyManager, logger

SUPPORTED_EXTS = {
    ".csv",
    ".tsv",
    ".parquet",
    ".pq",
    ".parq",
    ".ndjson",
    ".jsonl",
    ".json",
    ".yaml",
    ".yml",
    ".txt",
    ".pkl",
    ".xlsx",
    ".bin",
}

cm = ConcurrencyManager()


class FileHandler(ABC):
    @abstractmethod
    def read(self, path: Path | io.BytesIO, **kwargs) -> Any:
        """Read data from a file path or in-memory buffer and return the corresponding Python object.

        Args:
            path (Path | io.BytesIO): Filesystem path or byte buffer to read from.
            **kwargs: Additional format-specific read options.

        Returns:
            Any: Parsed object (e.g., DataFrame, dict, str).
        """
        pass

    @abstractmethod
    def save(self, obj: Any, path: Path, **kwargs) -> None:
        """Serialize and save an object to the given filesystem path.

        Args:
            obj (Any): Object to serialize and save.
            path (Path): Filesystem path where the object will be written.
            **kwargs: Additional format-specific save options.
        """
        pass

    @abstractmethod
    async def async_read(self, path: Path, **kwargs) -> Any:
        """Asynchronously read data from a file path and return the corresponding Python object.

        Args:
            path (Path): Filesystem path to read from.
            **kwargs: Additional format-specific read options.

        Returns:
            Any: Parsed object (e.g., DataFrame, dict, str).
        """
        pass

    @abstractmethod
    async def async_save(self, obj: Any, path: Path, **kwargs) -> None:
        """Asynchronously serialize and save an object to the given filesystem path.

        Args:
            obj (Any): Object to serialize and save.
            path (Path): Filesystem path where the object will be written.
            **kwargs: Additional format-specific save options.
        """
        pass

    def load_bytes(self, raw: bytes, **kwargs) -> Any:
        """Load and parse raw bytes via the handler's read method.

        Args:
            raw (bytes): Raw byte content to parse.
            **kwargs: Additional read options to pass through.

        Returns:
            Any: Parsed object.
        """
        buf = io.BytesIO(raw)
        return self.read(buf, **kwargs)


class BinHandler(FileHandler):
    """Handler for binary model files."""

    def read(self, path: Path | io.BytesIO, **kw) -> Any:
        if isinstance(path, io.BytesIO):
            with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
                tmp.write(path.getbuffer())
                tmp_path = Path(tmp.name)
            try:
                return lgb.Booster(model_file=str(tmp_path))
            except Exception:
                tmp_path.unlink()
                return msgspec.msgpack.decode(path.getbuffer())
        else:
            p = Path(path)
            try:
                return lgb.Booster(model_file=str(p))
            except Exception as e:
                logger.debug(f"LGBM load falhou: {e!s}")
            try:
                with (
                    p.open("rb") as f,
                    mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm,
                ):
                    return msgspec.msgpack.decode(mm)
            except Exception as e:
                logger.debug(f"msgspec decode falhou: {e!s}")

            return joblib.load(p)

    def save(self, obj: Any, path: Path, **kw) -> None:
        from lightgbm.basic import Booster  # lazy-import

        _ensure_dir(path)

        if isinstance(obj, Booster):
            obj.save_model(str(path))
            return
        try:
            encoded = msgspec.msgpack.encode(obj)
            path.write_bytes(encoded)
        except (TypeError, msgspec.EncodeError):
            logger.warning("Objeto não MessagePack-safe, caindo para pickle")
            joblib.dump(obj, path, protocol=pickle.HIGHEST_PROTOCOL)

    async def async_read(self, path: Path, **kw) -> Any:
        return self.read(path, **kw)

    async def async_save(self, obj: Any, path: Path, **kw) -> None:
        await _async_ensure_dir(path)
        self.save(obj, path, **kw)


class CSVHandler(FileHandler):
    def read(self, path: Path | io.BytesIO, **kwargs) -> pl.DataFrame:
        """Read a CSV file or buffer into a Polars DataFrame, with dialect fallback."""

        if isinstance(path, Path) and path.suffix.lower() == ".tsv":
            kwargs["separator"] = "\t"
            kwargs.setdefault("has_header", False)
            kwargs.setdefault("truncate_ragged_lines", True)
            kwargs.setdefault("ignore_errors", True)
        if isinstance(path, io.BytesIO):
            raw = path.read()
            sep, encoding = _detect_csv_dialect(raw)
            path.seek(0)
            if "separator" not in kwargs:
                kwargs["separator"] = sep
            kwargs.setdefault("encoding", encoding)
        else:
            kwargs.setdefault("encoding", "utf-8")

        try:
            return pl.read_csv(path, **kwargs)
        except (pl.ComputeError, pl.exceptions.PolarsError) as e:
            logger.warning(f"Erro inicial ao ler CSV/TSV: {e}")
            kwargs["truncate_ragged_lines"] = True
            kwargs["ignore_errors"] = True
            return pl.read_csv(path, **kwargs)

    def save(self, obj: Any, path: Path, **kwargs) -> None:
        """Save a Polars DataFrame or LazyFrame as CSV, creating parent dirs if needed.

        Args:
            obj (pl.DataFrame | pl.LazyFrame): Table to save.
            path (Path): Destination file path.
            **kwargs: Arguments forwarded to `write_csv` or `sink_csv`.
        """
        _ensure_dir(path)
        if isinstance(obj, pl.LazyFrame):
            obj.sink_csv(path, **kwargs)
        else:
            obj.write_csv(path, **kwargs)

    async def async_read(self, path: Path, **kwargs) -> pl.DataFrame:
        """Asynchronously read a CSV file into a Polars DataFrame using real async I/O.

        Args:
            path (Path): CSV file path.
            **kwargs: Arguments forwarded to the synchronous read method.

        Returns:
            pl.DataFrame: Loaded table.
        """
        async with aiofile.async_open(path, "rb") as f:
            content = await f.read()
        buffer = io.BytesIO(content)
        return self.read(buffer, **kwargs)

    async def async_save(self, obj: Any, path: Path, **kwargs) -> None:
        """Asynchronously save a Polars DataFrame as CSV using real async I/O.

        Args:
            obj (pl.DataFrame | pl.LazyFrame): Table to save.
            path (Path): Destination file path.
            **kwargs: Arguments forwarded to CSV serialization.
        """
        await _async_ensure_dir(path)

        # Serialize to in-memory buffer first
        buffer = io.StringIO()
        if isinstance(obj, pl.LazyFrame):
            obj.collect().write_csv(buffer, **kwargs)
        else:
            obj.write_csv(buffer, **kwargs)

        # Write buffer content to disk asynchronously with real async I/O
        async with aiofile.async_open(path, "w", encoding="utf-8") as f:
            await f.write(buffer.getvalue())


class ParquetHandler(FileHandler):
    def read(self, path: Path | io.BytesIO, **kwargs) -> pl.DataFrame:
        """Read a Parquet file or buffer into a Polars DataFrame.

        Uses Polars' optimized Parquet reader with Arrow backend for maximum performance.

        Args:
            path (Path | io.BytesIO): Parquet file path or in-memory buffer.
            **kwargs: Arguments forwarded to `pl.read_parquet`.

        Returns:
            pl.DataFrame: Loaded table.
        """
        return pl.read_parquet(path, **kwargs)

    def save(
        self,
        obj: Any,
        path: Path,
        compression: str = "zstd",
        statistics: bool = True,
        row_group_size: int = 512000,
        **kwargs,
    ) -> None:
        """Save a Polars DataFrame or LazyFrame as Parquet with optimized settings.

        Args:
            obj (pl.DataFrame | pl.LazyFrame): Table to save.
            path (Path): Destination file path.
            compression (str): Compression codec (default 'zstd').
            statistics (bool): Write column statistics (default True).
            row_group_size (int): Number of rows per row group.
            **kwargs: Additional arguments forwarded to `write_parquet` or `sink_parquet`.
        """
        _ensure_dir(path)
        kwargs.setdefault("compression", compression)
        kwargs.setdefault("statistics", statistics)
        if isinstance(obj, pl.LazyFrame):
            obj.sink_parquet(path, **kwargs)
        else:
            obj.write_parquet(path, row_group_size=row_group_size, **kwargs)

    async def async_read(self, path: Path, **kwargs) -> pl.DataFrame:
        """Asynchronously read a Parquet file into a Polars DataFrame using real async I/O.

        Args:
            path (Path): Parquet file path.
            **kwargs: Arguments forwarded to the synchronous read method.

        Returns:
            pl.DataFrame: Loaded table.
        """
        async with aiofile.async_open(path, "rb") as f:
            content = await f.read()
        buffer = io.BytesIO(content)
        return self.read(buffer, **kwargs)

    async def async_save(
        self,
        obj: Any,
        path: Path,
        compression: str = "zstd",
        statistics: bool = True,
        row_group_size: int = 512000,
        **kwargs,
    ) -> None:
        """Asynchronously save a Polars DataFrame as Parquet using real async I/O.

        Args:
            obj (pl.DataFrame | pl.LazyFrame): Table to save.
            path (Path): Destination file path.
            compression (str): Compression codec (default 'zstd').
            statistics (bool): Write column statistics (default True).
            row_group_size (int): Number of rows per row group.
            **kwargs: Additional arguments forwarded to Parquet serialization.
        """
        await _async_ensure_dir(path)

        # Serialize to in-memory buffer first
        buffer = io.BytesIO()
        kwargs.setdefault("compression", compression)
        kwargs.setdefault("statistics", statistics)

        if isinstance(obj, pl.LazyFrame):
            obj.collect().write_parquet(buffer, row_group_size=row_group_size, **kwargs)
        else:
            obj.write_parquet(buffer, row_group_size=row_group_size, **kwargs)

        # Write buffer content to disk asynchronously with real async I/O
        async with aiofile.async_open(path, "wb") as f:
            await f.write(buffer.getvalue())


class ExcelHandler(FileHandler):
    def read(self, path: Path | io.BytesIO, **kwargs) -> pl.DataFrame:
        """
        Reads an Excel file using Polars' optimized Excel reader.

        Args:
            path (Path | io.BytesIO): The file system path or a BytesIO object pointing to the Excel file.
            **kwargs: Additional keyword arguments to pass to `pl.read_excel`.

        Returns:
            pl.DataFrame: The contents of the Excel file as a Polars DataFrame.
        """
        return pl.read_excel(source=path, **kwargs)

    def save(self, obj: Any, path: Path, **kwargs) -> None:
        """
        Saves an object as an Excel file using Polars' optimized Excel writer.

        Args:
            obj (Any): The object to be saved. Can be a Polars DataFrame or any object convertible to one.
            path (Path): The file path where the Excel file will be saved.
            **kwargs: Additional keyword arguments passed to Polars' write_excel method.
        """
        _ensure_dir(path)
        df = obj if isinstance(obj, pl.DataFrame) else pl.DataFrame(obj)
        df.write_excel(path, **kwargs)

    async def async_read(self, path: Path, **kwargs) -> pl.DataFrame:
        """Asynchronously read an Excel file using real async I/O.

        Args:
            path (Path): Excel file path.
            **kwargs: Arguments forwarded to the synchronous read method.

        Returns:
            pl.DataFrame: Loaded table.
        """
        async with aiofile.async_open(path, "rb") as f:
            content = await f.read()
        buffer = io.BytesIO(content)
        return self.read(buffer, **kwargs)

    async def async_save(self, obj: Any, path: Path, **kwargs) -> None:
        """Asynchronously save an object as an Excel file using real async I/O.

        Args:
            obj (Any): The object to be saved.
            path (Path): The file path where the Excel file will be saved.
            **kwargs: Additional keyword arguments passed to Excel serialization.
        """
        await _async_ensure_dir(path)

        # Serialize to in-memory buffer first
        buffer = io.BytesIO()
        df = obj if isinstance(obj, pl.DataFrame) else pl.DataFrame(obj)
        df.write_excel(buffer, **kwargs)

        # Write buffer content to disk asynchronously with real async I/O
        async with aiofile.async_open(path, "wb") as f:
            await f.write(buffer.getvalue())


class NDJSONHandler(FileHandler):
    def read(self, path: Path | io.BytesIO, **kwargs) -> pl.DataFrame:
        """Read a newline-delimited JSON file using Polars' optimized NDJSON reader.

        Args:
            path (Path | io.BytesIO): NDJSON file path or in-memory buffer.
            **kwargs: Arguments forwarded to `pl.read_ndjson`.

        Returns:
            pl.DataFrame: Loaded table.
        """
        return pl.read_ndjson(path, **kwargs)

    def save(self, obj: Any, path: Path, **kwargs) -> None:
        """Save a Polars DataFrame or LazyFrame as NDJSON, creating parent dirs if needed.

        Args:
            obj (pl.DataFrame | pl.LazyFrame): Table to save.
            path (Path): Destination file path.
            **kwargs: Arguments forwarded to `write_ndjson` or `sink_ndjson`.
        """
        _ensure_dir(path)
        if isinstance(obj, pl.LazyFrame):
            obj.sink_ndjson(path, **kwargs)
        else:
            obj.write_ndjson(path, **kwargs)

    async def async_read(self, path: Path, **kwargs) -> pl.DataFrame:
        """Asynchronously read an NDJSON file using real async I/O.

        Args:
            path (Path): NDJSON file path.
            **kwargs: Arguments forwarded to the synchronous read method.

        Returns:
            pl.DataFrame: Loaded table.
        """
        async with aiofile.async_open(path, "rb") as f:
            content = await f.read()
        buffer = io.BytesIO(content)
        return self.read(buffer, **kwargs)

    async def async_save(self, obj: Any, path: Path, **kwargs) -> None:
        """Asynchronously save a Polars DataFrame as NDJSON using real async I/O.

        Args:
            obj (pl.DataFrame | pl.LazyFrame): Table to save.
            path (Path): Destination file path.
            **kwargs: Arguments forwarded to NDJSON serialization.
        """
        await _async_ensure_dir(path)

        # Serialize to in-memory buffer first
        buffer = io.StringIO()
        if isinstance(obj, pl.LazyFrame):
            obj.collect().write_ndjson(buffer, **kwargs)
        else:
            obj.write_ndjson(buffer, **kwargs)

        # Write buffer content to disk asynchronously with real async I/O
        async with aiofile.async_open(path, "w", encoding="utf-8") as f:
            await f.write(buffer.getvalue())


class JSONHandler(FileHandler):
    """
    Estado da arte JSON handler usando msgspec para máxima performance.

    Performance gains:
    - msgspec leitura: 2-3x mais rápido que pysimdjson + conversão
    - msgspec escrita: 1.5-2x mais rápido que orjson para JSON compacto
    - Uso de memória: 6-9x menor que orjson/simdjson
    - Caching automático de chaves repetidas

    Maintains 100% API compatibility with previous implementation.
    """

    def __init__(self):
        # Create reusable encoder/decoder instances for better performance
        self._encoder = msgspec.json.Encoder()
        self._decoder = msgspec.json.Decoder()

    def read(self, path: Path | io.BytesIO, **kwargs) -> Any:
        """
        Deserialize JSON content using msgspec for maximum performance.

        Returns native Python types (dict, list, int, float, str, bool, None) exactly like
        the previous pysimdjson implementation, ensuring 100% compatibility.

        Args:
            path (Path | io.BytesIO): JSON file path or in-memory buffer.
            **kwargs: (unused) Reserved for future options.

        Returns:
            Any: Parsed JSON data as native Python types.
        """
        try:
            if isinstance(path, io.BytesIO):
                content = path.read()
            else:
                content = Path(path).read_bytes()

            if not content:
                return {}

            # msgspec.json.decode returns native Python types directly
            # No conversion needed unlike pysimdjson
            return self._decoder.decode(content)
        except Exception as e:
            logger.error(f"Failed to parse JSON with msgspec on path {path}: {e}")
            raise

    def save(self, obj: Any, path: Path, **kwargs) -> None:
        """
        Serialize an object to JSON using msgspec for optimal performance.

        Uses msgspec for compact JSON (faster than orjson) and falls back to orjson
        for indented JSON when needed, maintaining 100% API compatibility.

        Args:
            obj (Any): Object to serialize.
            path (Path): Destination file path.
            **kwargs: Additional options. 'indent' triggers orjson fallback.
        """
        _ensure_dir(path)

        # Check if indentation is requested (compatibility with existing code)
        needs_indent = kwargs.get("indent", False) or any(
            k in kwargs for k in ["indent", "pretty", "format"]
        )

        if needs_indent:
            # Fallback to orjson for indented JSON
            json_bytes = orjson.dumps(obj, option=orjson.OPT_INDENT_2)
        else:
            # Use msgspec for compact JSON (faster than orjson)
            json_bytes = self._encoder.encode(obj)

        path.write_bytes(json_bytes)

    async def async_read(self, path: Path, **kwargs) -> Any:
        """Asynchronously deserialize JSON content using msgspec and real async I/O.

        Args:
            path (Path): JSON file path.
            **kwargs: (unused) Reserved for future options.

        Returns:
            Any: Parsed JSON data as native Python types.
        """
        try:
            async with aiofile.async_open(path, "rb") as f:
                content = await f.read()

            if not content:
                return {}

            # msgspec.json.decode returns native Python types directly
            return self._decoder.decode(content)
        except Exception as e:
            logger.error(f"Failed to parse JSON with msgspec on path {path}: {e}")
            raise

    async def async_save(self, obj: Any, path: Path, **kwargs) -> None:
        """Asynchronously serialize an object to JSON using msgspec and real async I/O.

        Args:
            obj (Any): Object to serialize.
            path (Path): Destination file path.
            **kwargs: Additional options. 'indent' triggers orjson fallback.
        """
        await _async_ensure_dir(path)

        # Check if indentation is requested (compatibility with existing code)
        needs_indent = kwargs.get("indent", False) or any(
            k in kwargs for k in ["indent", "pretty", "format"]
        )

        if needs_indent:
            # Fallback to orjson for indented JSON
            json_bytes = orjson.dumps(obj, option=orjson.OPT_INDENT_2)
        else:
            # Use msgspec for compact JSON (faster than orjson)
            json_bytes = self._encoder.encode(obj)

        # Write to disk asynchronously with real async I/O
        async with aiofile.async_open(path, "wb") as f:
            await f.write(json_bytes)


class YAMLHandler(FileHandler):
    """
    Estado da arte YAML handler usando ruamel.yaml para máximas funcionalidades.

    Features:
    - Suporta YAML 1.2 (vs PyYAML's YAML 1.1)
    - Preserva comentários e formatação
    - Roundtrip parsing
    - Mais seguro que PyYAML
    """

    def __init__(self):
        # Create reusable YAML instance for better performance
        self._yaml = ruamel.yaml.YAML()
        self._yaml.preserve_quotes = True
        self._yaml.width = 4096  # Avoid line wrapping

    def read(
        self, path: Path | io.BytesIO, custom_tags: dict | None = None, **kwargs
    ) -> Any:
        """Deserialize YAML content using ruamel.yaml for YAML 1.2 support and better features.

        Args:
            path (Path | io.BytesIO): YAML file path or in-memory buffer.
            custom_tags (dict | None): Custom YAML tags and their constructors.
            **kwargs: (unused) Reserved for future options.

        Returns:
            Any: Parsed YAML data.
        """
        if isinstance(path, io.BytesIO):
            content = path.read().decode("utf-8")
        else:
            content = Path(path).read_text("utf-8")

        if custom_tags:
            yaml_instance = ruamel.yaml.YAML()
            for tag, constructor in custom_tags.items():
                yaml_instance.constructor.add_constructor(tag, constructor)
            return yaml_instance.load(content)
        else:
            return self._yaml.load(content)

    def save(self, obj: Any, path: Path, **kwargs) -> None:
        """Serialize an object to YAML using ruamel.yaml for YAML 1.2 compliance.

        Args:
            obj (Any): Object to serialize.
            path (Path): Destination file path.
            **kwargs: (unused) Reserved for future options.
        """
        _ensure_dir(path)
        with path.open("w", encoding="utf-8") as f:
            self._yaml.dump(obj, f)

    async def async_read(
        self, path: Path, custom_tags: dict | None = None, **kwargs
    ) -> Any:
        """Asynchronously deserialize YAML content using real async I/O.

        Args:
            path (Path): YAML file path.
            custom_tags (dict | None): Custom YAML tags and their constructors.
            **kwargs: (unused) Reserved for future options.

        Returns:
            Any: Parsed YAML data.
        """
        async with aiofile.async_open(path, "r", encoding="utf-8") as f:
            content = await f.read()

        if custom_tags:
            yaml_instance = ruamel.yaml.YAML()
            for tag, constructor in custom_tags.items():
                yaml_instance.constructor.add_constructor(tag, constructor)
            return yaml_instance.load(content)
        else:
            return self._yaml.load(content)

    async def async_save(self, obj: Any, path: Path, **kwargs) -> None:
        """Asynchronously serialize an object to YAML using real async I/O.

        Args:
            obj (Any): Object to serialize.
            path (Path): Destination file path.
            **kwargs: (unused) Reserved for future options.
        """
        await _async_ensure_dir(path)

        # Serialize to string in memory first
        from io import StringIO

        buffer = StringIO()
        self._yaml.dump(obj, buffer)

        # Write to disk asynchronously with real async I/O
        async with aiofile.async_open(path, "w", encoding="utf-8") as f:
            await f.write(buffer.getvalue())


class TextHandler(FileHandler):
    """
    Estado da arte text handler com detecção de encoding inteligente.

    Uses charset-normalizer (2x faster than chardet) for encoding detection.
    """

    def read(self, path: Path | io.BytesIO, **kwargs) -> str:
        """Read text content with intelligent encoding detection using charset-normalizer.

        Args:
            path (Path | io.BytesIO): Text file path or in-memory buffer.
            **kwargs: (unused) Reserved for future options.

        Returns:
            str: Decoded text.
        """
        if isinstance(path, io.BytesIO):
            raw = path.read()
        else:
            raw = Path(path).read_bytes()

        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            # Use charset-normalizer for intelligent encoding detection
            detection = detect(raw)
            encoding = detection.get("encoding") or "latin-1"
            return raw.decode(encoding, errors="ignore")

    def save(self, obj: Any, path: Path, **kwargs) -> None:
        """Save text content to a file, creating dirs if needed.

        Args:
            obj (Any): Object to convert to string and save.
            path (Path): Destination file path.
            **kwargs: (unused) Reserved for future options.
        """
        _ensure_dir(path)
        path.write_text(str(obj), encoding="utf-8")

    async def async_read(self, path: Path, **kwargs) -> str:
        """Asynchronously read text content with intelligent encoding detection using real async I/O.

        Args:
            path (Path): Text file path.
            **kwargs: (unused) Reserved for future options.

        Returns:
            str: Decoded text.
        """
        async with aiofile.async_open(path, "rb") as f:
            raw = await f.read()

        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            # Use charset-normalizer for intelligent encoding detection
            detection = detect(raw)
            encoding = detection.get("encoding", "latin-1")
            return raw.decode(encoding, errors="ignore")

    async def async_save(self, obj: Any, path: Path, **kwargs) -> None:
        """Asynchronously save text content using real async I/O.

        Args:
            obj (Any): Object to convert to string and save.
            path (Path): Destination file path.
            **kwargs: (unused) Reserved for future options.
        """
        await _async_ensure_dir(path)

        async with aiofile.async_open(path, "w", encoding="utf-8") as f:
            await f.write(str(obj))


class PickleHandler(FileHandler):
    """
    Pickle handler - não há alternativa melhor para objetos Python.

    Maintains security warnings and best practices.
    """

    def read(self, path: Path | io.BytesIO, **kwargs) -> Any:
        """Deserialize a Python object from a pickle file or buffer.

        WARNING: Unpickling data from untrusted sources is unsafe.
        Only use this method with trusted, validated data.

        Args:
            path (Path | io.BytesIO): Pickle file path or in-memory buffer.
            **kwargs: (unused) Reserved for future options.

        Returns:
            Any: Unpickled Python object.
        """
        if isinstance(path, io.BytesIO):
            return pickle.load(path)
        with Path(path).open("rb") as f:
            return pickle.load(f)

    def save(self, obj: Any, path: Path, **kwargs) -> None:
        """Serialize a Python object with pickle, creating dirs if needed.

        Args:
            obj (Any): Python object to pickle.
            path (Path): Destination file path.
            **kwargs: (unused) Reserved for future options.
        """
        _ensure_dir(path)
        with path.open("wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

    async def async_read(self, path: Path, **kwargs) -> Any:
        """Asynchronously deserialize a Python object using real async I/O.

        WARNING: Unpickling data from untrusted sources is unsafe.
        Only use this method with trusted, validated data.

        Args:
            path (Path): Pickle file path.
            **kwargs: (unused) Reserved for future options.

        Returns:
            Any: Unpickled Python object.
        """
        async with aiofile.async_open(path, "rb") as f:
            content = await f.read()
        buffer = io.BytesIO(content)
        return pickle.load(buffer)

    async def async_save(self, obj: Any, path: Path, **kwargs) -> None:
        """Asynchronously serialize a Python object using real async I/O.

        Args:
            obj (Any): Python object to pickle.
            path (Path): Destination file path.
            **kwargs: (unused) Reserved for future options.
        """
        await _async_ensure_dir(path)

        # Serialize to bytes in memory first
        buffer = io.BytesIO()
        pickle.dump(obj, buffer, protocol=pickle.HIGHEST_PROTOCOL)

        # Write to disk asynchronously with real async I/O
        async with aiofile.async_open(path, "wb") as f:
            await f.write(buffer.getvalue())


_HANDLER_REGISTRY = {
    ".csv": CSVHandler(),
    ".tsv": CSVHandler(),
    ".parquet": ParquetHandler(),
    ".pq": ParquetHandler(),
    ".parq": ParquetHandler(),
    ".ndjson": NDJSONHandler(),
    ".jsonl": NDJSONHandler(),
    ".json": JSONHandler(),
    ".yaml": YAMLHandler(),
    ".yml": YAMLHandler(),
    ".txt": TextHandler(),
    ".pkl": PickleHandler(),
    ".xlsx": ExcelHandler(),
    ".bin": BinHandler(),
}


def _zip_entry_process(item, handler_kwargs):
    """Process a single entry from a ZIP archive using the appropriate handler."""
    name, raw = item
    handler = _HANDLER_REGISTRY.get(Path(name).suffix.lower())
    if handler:
        try:
            return name, handler.load_bytes(raw, **handler_kwargs)
        except Exception:
            return name, None
    return name, raw


def _ensure_dir(path: Path) -> None:
    """Ensure the parent directory of a given path exists, creating if necessary."""
    path.parent.mkdir(parents=True, exist_ok=True)


async def _async_ensure_dir(path: Path) -> None:
    """Asynchronously ensure the parent directory of a given path exists."""
    # mkdir is not async in pathlib, but it's a fast operation
    path.parent.mkdir(parents=True, exist_ok=True)


def _detect_csv_dialect(raw: bytes) -> tuple[str, str]:
    """Detect delimiter and encoding of a CSV sample using charset-normalizer.

    Args:
        raw (bytes): Sample of CSV file content.

    Returns:
        tuple[str, str]: (delimiter, encoding) inferred from the sample.
    """
    # Use charset-normalizer for better encoding detection
    detection = detect(raw)
    encoding = detection.get("encoding") or "utf-8"

    try:
        text = raw.decode(encoding)
    except (UnicodeDecodeError, TypeError):
        text = raw.decode("utf-8", errors="ignore")
        encoding = "utf-8"

    # Detect delimiter
    if text.count(";") > text.count(","):
        sep = ";"
    elif text.count("\t") > text.count(","):
        sep = "\t"
    else:
        sep = ","

    return sep, encoding


class FileManager:
    """
    State-of-the-art FileManager with all libraries optimized for maximum performance.

    Performance improvements:
    - Polars: 10-100x faster than pandas
    - msgspec: 2-3x faster than orjson+pysimdjson
    - aiofile: True async I/O on Linux via libaio
    - ruamel.yaml: YAML 1.2, preserves comments
    - charset-normalizer: 2x faster than chardet

    Maintains 100% compatibility with previous API.
    """

    @staticmethod
    def read(path: str | Path, **kwargs) -> Any:
        """Read a path (file, directory or ZIP) and dispatch to the appropriate handler.

        Args:
            path (str | Path): Filesystem path or pattern.
            **kwargs: Options forwarded to handlers.

        Returns:
            Any: Handler-specific return (DataFrame, dict, bytes, etc.).
        """
        p = Path(path)
        if p.is_dir():
            return FileManager.load_directory(p, **kwargs)
        if p.suffix.lower() == ".zip":
            return FileManager.load_zip(p, **kwargs)
        handler = _HANDLER_REGISTRY.get(p.suffix.lower())
        if handler:
            return handler.read(p, **kwargs)
        raise ValueError(f"Unsupported extension: {p.suffix}")

    @staticmethod
    def save(obj: Any, path: str | Path, **kwargs) -> None:
        """Save an object to a given path by selecting the correct handler.

        Args:
            obj (Any): Object to save.
            path (str | Path): Destination file path.
            **kwargs: Options forwarded to handlers.
        """
        p = Path(path)
        handler = _HANDLER_REGISTRY.get(p.suffix.lower())
        if handler:
            handler.save(obj, p, **kwargs)
            return
        raise ValueError(f"Unsupported extension: {p.suffix}")

    @staticmethod
    async def async_read(path: str | Path, **kwargs) -> Any:
        """Asynchronously read a file using real async I/O when available.

        Args:
            path (str | Path): Filesystem path.
            **kwargs: Options forwarded to handlers.

        Returns:
            Any: Handler-specific return (DataFrame, dict, bytes, etc.).
        """
        p = Path(path)
        if p.is_dir():
            raise ValueError("Async directory reading not supported yet")
        if p.suffix.lower() == ".zip":
            raise ValueError("Async ZIP reading not supported yet")
        handler = _HANDLER_REGISTRY.get(p.suffix.lower())
        if handler:
            return await handler.async_read(p, **kwargs)
        raise ValueError(f"Unsupported extension: {p.suffix}")

    @staticmethod
    async def async_save(obj: Any, path: str | Path, **kwargs) -> None:
        """Asynchronously save an object using real async I/O when available.

        Args:
            obj (Any): Object to save.
            path (str | Path): Destination file path.
            **kwargs: Options forwarded to handlers.
        """
        p = Path(path)
        handler = _HANDLER_REGISTRY.get(p.suffix.lower())
        if handler:
            await handler.async_save(obj, p, **kwargs)
            return
        raise ValueError(f"Unsupported extension: {p.suffix}")

    @staticmethod
    def scan_csv(pattern: str, **kwargs) -> pl.LazyFrame:
        """Return a Polars LazyFrame scanning CSV files matching a glob pattern.

        Args:
            pattern (str): Glob pattern for CSV files.
            **kwargs: Arguments forwarded to `pl.scan_csv`.

        Returns:
            pl.LazyFrame: Lazy query over all matching CSVs.
        """
        return pl.scan_csv(pattern, **kwargs)

    @staticmethod
    def scan_parquet(pattern: str, **kwargs) -> pl.LazyFrame:
        """Return a Polars LazyFrame scanning Parquet files matching a glob pattern.

        Args:
            pattern (str): Glob pattern for Parquet files.
            **kwargs: Arguments forwarded to `pl.scan_parquet`.

        Returns:
            pl.LazyFrame: Lazy query over all matching Parquets.
        """
        return pl.scan_parquet(pattern, **kwargs)

    @staticmethod
    def scan_ndjson(pattern: str, **kwargs) -> pl.LazyFrame:
        """Return a Polars LazyFrame scanning NDJSON files matching a glob pattern.

        Args:
            pattern (str): Glob pattern for NDJSON files.
            **kwargs: Arguments forwarded to `pl.scan_ndjson`.

        Returns:
            pl.LazyFrame: Lazy query over all matching NDJSONs.
        """
        return pl.scan_ndjson(pattern, **kwargs)

    @staticmethod
    def scan_directory(dir_path: Path, **kwargs) -> pl.LazyFrame:
        """Scan a homogeneous directory using Polars' optimized lazy evaluation.

        Args:
            dir_path (str | Path): Directory containing only one supported file type.
            **kwargs: Arguments forwarded to the underlying `pl.scan_*`.

        Returns:
            pl.LazyFrame: Lazy query over all files of that type.
        """
        if next(dir_path.glob("*.csv"), None):
            return pl.scan_csv(str(dir_path / "*.csv"), **kwargs)
        if next(dir_path.glob("*.parquet"), None):
            return pl.scan_parquet(str(dir_path / "*.parquet"), **kwargs)
        if next(dir_path.glob("*.ndjson"), None):
            return pl.scan_ndjson(str(dir_path / "*.ndjson"), **kwargs)
        raise ValueError(
            f"Directory '{dir_path}' contains no single, scannable file type."
        )

    @staticmethod
    async def load_directory(dir_path: Path, **kwargs) -> pl.DataFrame | dict[str, Any]:
        """Load an entire directory using Polars' optimized processing.

        Args:
            dir_path (str | Path): Directory containing homogeneous files.
            **kwargs: Arguments forwarded to `scan_directory`.

        Returns:
            pl.DataFrame: Collected DataFrame of all files.
        """
        try:
            return FileManager.scan_directory(dir_path, **kwargs).collect()
        except ValueError:
            files = [
                p
                for p in dir_path.rglob("*")
                if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
            ]

            def worker(p: Path) -> tuple[str, Any]:
                return str(p.relative_to(dir_path)), FileManager.read(p, **kwargs)

            pairs = await cm.execute(
                worker,
                [(p,) for p in files],
                task_type="process",
                max_workers=os.cpu_count(),
                desc="Reading mixed directory",
            )
            return {k: v for k, v in pairs if v is not None}

    @staticmethod
    async def load_zip(zip_path: str | Path, **kwargs) -> Any:
        """Extract and load supported files from a ZIP archive in parallel.

        Args:
            zip_path (str | Path): Path to ZIP file.
            **kwargs: Arguments forwarded to handlers.

        Returns:
            dict[str, Any]: Mapping of member names to loaded objects or raw bytes.
        """
        p = Path(zip_path)
        with zipfile.ZipFile(p, "r") as zf:
            members = [
                m
                for m in zf.namelist()
                if not m.endswith("/") and Path(m).suffix.lower() in SUPPORTED_EXTS
            ]
            raw_data = [(name, zf.read(name)) for name in members]
        args_list = [(item, kwargs) for item in raw_data]
        result = await cm.execute(
            _zip_entry_process,
            args_list,
            task_type="process",
            max_workers=os.cpu_count(),
            desc="Loading ZIP",
        )
        return dict(result)

    @staticmethod
    def get_hash(path: Path, block_size: int = 65536) -> str:
        """Generate MD5 hash of a file."""
        hasher = hashlib.md5()
        try:
            with open(path, "rb") as f:
                buf = f.read(block_size)
                while len(buf) > 0:
                    hasher.update(buf)
                    buf = f.read(block_size)
            return hasher.hexdigest()
        except FileNotFoundError:
            return ""

    @staticmethod
    def get_timestamp() -> str:
        """Generate UTC timestamp."""
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    @staticmethod
    def json_dumps(obj: Any, *, sort_keys: bool = False) -> str:
        """
        Serialize Python object to JSON string using msgspec (2-3x faster than stdlib json).

        Sprint 16.5: Centralized JSON serialization using msgspec for performance.

        Args:
            obj: Python object to serialize (dict, list, etc.)
            sort_keys: Sort dictionary keys (for deterministic output)

        Returns:
            JSON string

        Example:
            >>> fm = FileManager()
            >>> fm.json_dumps({"b": 2, "a": 1}, sort_keys=True)
            '{"a":1,"b":2}'
        """
        encoder = msgspec.json.Encoder()
        json_bytes = encoder.encode(obj)
        json_str = json_bytes.decode('utf-8')

        # msgspec doesn't have built-in sort_keys, so we need to handle it
        if sort_keys and isinstance(obj, dict):
            import json as stdlib_json
            # Use stdlib json only for sorting (rare case)
            return stdlib_json.dumps(obj, sort_keys=True, separators=(',', ':'))

        return json_str

    @staticmethod
    def json_loads(s: str | bytes) -> Any:
        """
        Deserialize JSON string to Python object using msgspec (2-3x faster than stdlib json).

        Sprint 16.5: Centralized JSON deserialization using msgspec for performance.

        Args:
            s: JSON string or bytes to deserialize

        Returns:
            Python object (dict, list, int, float, str, bool, None)

        Example:
            >>> fm = FileManager()
            >>> fm.json_loads('{"a": 1, "b": 2}')
            {'a': 1, 'b': 2}
        """
        decoder = msgspec.json.Decoder()
        if isinstance(s, str):
            s = s.encode('utf-8')
        return decoder.decode(s)

    @staticmethod
    async def count_lines(path: Path, parallel_threshold_mb: int = 100) -> int:
        """
        Count lines in a file using parallel processing for large files.

        Args:
            path: Path to the text file.
            parallel_threshold_mb: File size in MB to trigger parallel execution.

        Returns:
            The total number of lines in the file.
        """
        try:
            file_size = path.stat().st_size
        except FileNotFoundError:
            logger.warning(f"File not found when trying to count lines: {path}")
            return 0

        if file_size < parallel_threshold_mb * 1024 * 1024:
            logger.debug(
                f"Small file ({file_size / 1e6:.1f}MB), using single-thread line counting."
            )
            with path.open("r", encoding="utf-8", errors="ignore") as f:
                return sum(1 for _ in f)

        num_workers = cm.hardware.physical_cores
        chunk_size = file_size // num_workers
        logger.debug(
            f"Large file ({file_size / 1e6:.1f}MB), using {num_workers} parallel "
            f"processes to count lines."
        )

        args_list = []
        for i in range(num_workers):
            start = i * chunk_size
            end = file_size if i == num_workers - 1 else (i + 1) * chunk_size
            args_list.append((str(path), start, end))
        results = await cm.execute(
            _count_lines_in_chunk,
            args_list,
            task_type="process",
            desc="Counting lines in parallel",
        )

        return sum(results)


def _count_lines_in_chunk(filepath: str, start: int, end: int) -> int:
    """Count newlines in a specific byte chunk of a file."""
    count = 0
    with open(filepath, "rb") as f:
        if start != 0:
            f.seek(start - 1)
            f.readline()

        while f.tell() < end:
            buffer = f.read(1024 * 1024)
            if not buffer:
                break
            count += buffer.count(b"\n")
    return count
