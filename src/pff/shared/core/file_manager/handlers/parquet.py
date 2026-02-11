"""Parquet file handler."""

from __future__ import annotations

import asyncio
import io
from pathlib import Path
from typing import Any, Iterator, Literal, cast

import msgspec
import orjson
import polars as pl
import pyarrow.parquet as pq

from ..async_io import async_ensure_dir, read_async_content
from ...cache import CacheManager
from ..utils import ensure_dir
from .base import FileHandler
from .tabular_utils import read_tabular

_STRUCT_COLUMNS = [
    "id",
    "externalId",
    "status",
    "account",
    "contract",
    "contactMediumAssociation",
    "characteristic",
    "relatedPartyId",
    "relatedPartyExternalId",
    "homeTimeZone",
]

_json_encoder = msgspec.json.Encoder()
_SCHEMA_CACHE = CacheManager(max_memory_items=256)


def _cached_parquet_schema_names(
    path_str: str,
    mtime_ns: int,
    size_bytes: int,
) -> tuple[str, ...]:
    key = f"parquet_schema::{path_str}::{mtime_ns}::{size_bytes}"
    cached = _SCHEMA_CACHE.get(key)
    if cached is not None:
        return cast(tuple[str, ...], cached)
    schema_names = tuple(pl.read_parquet_schema(path_str).keys())
    _SCHEMA_CACHE.set(key, schema_names, ttl=3600, tags=["parquet_schema"])
    return schema_names


def iter_parquet_as_json(
    parquet_path: Path,
    *,
    batch_size: int = 1024,
    include_source_name: bool = True,
    prefer_struct: bool = True,
) -> Iterator[tuple[str | None, str | None, str]]:
    """Iterate over parquet rows yielding reconstructed JSON.

    This function supports both legacy parquets with _raw_json column
    and optimized parquets with struct columns only.

    Args:
        parquet_path: Path to parquet file.
        batch_size: Number of rows per batch.
        include_source_name: Whether to include source name in output.
        prefer_struct: If True and struct columns exist, use them instead of _raw_json
                      for better performance (default True).

    Yields:
        Tuples of (source_name, external_id, json_string)
    """
    parquet_file = pq.ParquetFile(parquet_path)
    schema_names = set(parquet_file.schema_arrow.names)

    has_raw_json = "_raw_json" in schema_names
    has_struct = any(col in schema_names for col in _STRUCT_COLUMNS)

    use_struct = has_struct and (prefer_struct or not has_raw_json)

    if use_struct:
        columns = [c for c in _STRUCT_COLUMNS if c in schema_names]
        if "_source_name" in schema_names:
            columns.append("_source_name")

        for batch in parquet_file.iter_batches(columns=columns, batch_size=batch_size):
            col_data = {name: batch.column(name).to_pylist() for name in columns}
            num_rows = len(batch)

            source_col = col_data.get("_source_name", [None] * num_rows)
            ext_id_col = col_data.get("externalId", [None] * num_rows)

            data_cols = [c for c in columns if c != "_source_name"]
            data_values = [col_data[c] for c in data_cols]

            for i in range(num_rows):
                source = source_col[i]
                ext_id = ext_id_col[i]

                row_clean = {
                    col: val
                    for col, val_list in zip(data_cols, data_values)
                    if (val := val_list[i]) is not None
                }

                json_str = orjson.dumps(row_clean).decode("utf-8")
                yield (source, ext_id, json_str)

    elif has_raw_json:
        columns = ["_raw_json"]
        if "_source_name" in schema_names:
            columns.append("_source_name")
        if "externalId" in schema_names:
            columns.append("externalId")
        if "_parse_error" in schema_names:
            columns.append("_parse_error")

        for batch in parquet_file.iter_batches(columns=columns, batch_size=batch_size):
            raw_list = batch.column(
                batch.schema.get_field_index("_raw_json")
            ).to_pylist()
            source_list = (
                batch.column(batch.schema.get_field_index("_source_name")).to_pylist()
                if "_source_name" in columns
                else [None] * len(raw_list)
            )
            ext_list = (
                batch.column(batch.schema.get_field_index("externalId")).to_pylist()
                if "externalId" in columns
                else [None] * len(raw_list)
            )
            error_list = (
                batch.column(batch.schema.get_field_index("_parse_error")).to_pylist()
                if "_parse_error" in columns
                else [None] * len(raw_list)
            )

            for raw_json, source, ext_id, error in zip(
                raw_list, source_list, ext_list, error_list
            ):
                if error or not raw_json:
                    continue
                yield (source, ext_id, raw_json)
    else:
        raise ValueError(
            f"Parquet at {parquet_path} has neither _raw_json nor struct columns"
        )


def iter_parquet_structs(
    parquet_path: Path,
    *,
    batch_size: int = 1024,
    prefer_struct: bool = True,
) -> Iterator[tuple[str | None, dict]]:
    """Iterate over parquet rows yielding dict structs directly.

    This is faster than iter_parquet_as_json when JSON is not needed.

    Args:
        parquet_path: Path to parquet file.
        batch_size: Number of rows per batch.
        prefer_struct: If True and struct columns exist, use them instead of _raw_json
                      for better performance (default True).

    Yields:
        Tuples of (source_name, row_dict)
    """
    parquet_file = pq.ParquetFile(parquet_path)
    schema_names = set(parquet_file.schema_arrow.names)

    has_raw_json = "_raw_json" in schema_names
    has_struct = any(col in schema_names for col in _STRUCT_COLUMNS)

    use_struct = has_struct and (prefer_struct or not has_raw_json)

    if use_struct:
        columns = [c for c in _STRUCT_COLUMNS if c in schema_names]
        if "_source_name" in schema_names:
            columns.append("_source_name")

        for batch in parquet_file.iter_batches(columns=columns, batch_size=batch_size):
            col_data = {name: batch.column(name).to_pylist() for name in columns}
            num_rows = len(batch)

            source_col = col_data.get("_source_name")

            data_cols = [c for c in columns if c != "_source_name"]
            data_values = [col_data[c] for c in data_cols]

            for i in range(num_rows):
                source = source_col[i] if source_col else None

                row_clean = {
                    col: val
                    for col, val_list in zip(data_cols, data_values)
                    if (val := val_list[i]) is not None
                }

                yield (source, row_clean)

    elif has_raw_json:
        decoder = msgspec.json.Decoder()
        columns = ["_raw_json"]
        if "_source_name" in schema_names:
            columns.append("_source_name")
        if "_parse_error" in schema_names:
            columns.append("_parse_error")

        for batch in parquet_file.iter_batches(columns=columns, batch_size=batch_size):
            raw_list = batch.column(
                batch.schema.get_field_index("_raw_json")
            ).to_pylist()
            source_list = (
                batch.column(batch.schema.get_field_index("_source_name")).to_pylist()
                if "_source_name" in columns
                else [None] * len(raw_list)
            )
            error_list = (
                batch.column(batch.schema.get_field_index("_parse_error")).to_pylist()
                if "_parse_error" in columns
                else [None] * len(raw_list)
            )

            for raw_json, source, error in zip(raw_list, source_list, error_list):
                if error or not raw_json:
                    continue
                try:
                    row_dict = decoder.decode(raw_json)
                    yield (source, row_dict)
                except Exception:
                    continue
    else:
        raise ValueError(
            f"Parquet at {parquet_path} has neither _raw_json nor struct columns"
        )


def optimize_parquet(
    source_path: Path,
    dest_path: Path | None = None,
    *,
    drop_raw_json: bool = True,
    compression: str = "lz4",
    row_group_size: int = 64000,
) -> dict[str, Any]:
    """Optimize a legacy parquet by removing redundant _raw_json column.

    This can reduce file size by 90%+ and improve read performance by 10x+
    for parquets that have both struct columns and _raw_json.

    Args:
        source_path: Path to source parquet file.
        dest_path: Path to write optimized parquet. If None, overwrites source.
        drop_raw_json: Whether to drop _raw_json column (default True).
        compression: Compression algorithm (default 'lz4').
        row_group_size: Row group size for read optimization (default 64000).

    Returns:
        Dict with optimization stats: original_size, optimized_size, reduction_percent
    """
    CompressionType = Literal[
        "lz4", "uncompressed", "snappy", "gzip", "lzo", "brotli", "zstd"
    ]

    if dest_path is None:
        dest_path = source_path

    original_size = source_path.stat().st_size

    df = pl.read_parquet(source_path)
    original_cols = df.columns

    if drop_raw_json and "_raw_json" in df.columns:
        df = df.drop(["_raw_json"])

    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    valid_compressions = {
        "lz4",
        "uncompressed",
        "snappy",
        "gzip",
        "lzo",
        "brotli",
        "zstd",
    }
    compression_typed = cast(
        CompressionType, compression if compression in valid_compressions else "lz4"
    )

    df.write_parquet(
        tmp_path,
        compression=compression_typed,  # type: ignore[arg-type]
        statistics=True,
        row_group_size=row_group_size,
    )

    optimized_size = tmp_path.stat().st_size

    import shutil

    shutil.move(str(tmp_path), str(dest_path))

    return {
        "original_size_mb": original_size / 1024 / 1024,
        "optimized_size_mb": optimized_size / 1024 / 1024,
        "reduction_percent": (1 - optimized_size / original_size) * 100,
        "columns_before": len(original_cols),
        "columns_after": len(df.columns),
        "dropped_columns": [c for c in original_cols if c not in df.columns],
    }


class ParquetHandler(FileHandler):
    """Handler for Parquet files using Polars/PyArrow."""

    def read(self, path: Path | io.BytesIO, **kwargs: Any) -> Any:
        """Read a Parquet file or buffer into a Polars DataFrame.

        Optimizations:
        - use_pyarrow: Faster for full scans
        - columns: Column pruning for I/O efficiency
        - memory_map: Zero-copy for large files
        - parallel: Multi-threaded decompression
        - exclude_raw_json: Skip _raw_json column if struct columns exist (default True)
        """
        lazy = bool(kwargs.pop("lazy", False))
        streaming = kwargs.pop("streaming", None)
        columns = kwargs.pop("columns", None)
        n_rows = kwargs.pop("n_rows", None)
        exclude_raw_json = kwargs.pop("exclude_raw_json", True)

        kwargs.setdefault("use_pyarrow", True)
        if isinstance(path, (Path, str)):
            kwargs.setdefault("memory_map", True)
        else:
            kwargs.setdefault("memory_map", False)
        kwargs.setdefault("parallel", "auto")

        if exclude_raw_json and columns is None and isinstance(path, (Path, str)):
            try:
                path_obj = Path(path)
                stat = path_obj.stat()
                schema_names = _cached_parquet_schema_names(
                    str(path_obj),
                    stat.st_mtime_ns,
                    stat.st_size,
                )
                schema_set = set(schema_names)
                has_raw_json = "_raw_json" in schema_set
                has_struct = any(col in schema_set for col in _STRUCT_COLUMNS)
                if has_raw_json and has_struct:
                    columns = [c for c in schema_names if c != "_raw_json"]
            except Exception:
                pass

        read_fn: Any = pl.scan_parquet if lazy or streaming else pl.read_parquet

        if columns is not None:
            kwargs["columns"] = columns
        if n_rows is not None:
            kwargs["n_rows"] = n_rows

        if lazy or streaming:
            kwargs.pop("use_pyarrow", None)
            kwargs.pop("memory_map", None)

        return read_tabular(
            path,
            lazy=lazy,
            streaming=streaming,
            scan_fn=pl.scan_parquet,
            read_fn=read_fn,
            **kwargs,
        )

    def save(
        self,
        obj: Any,
        path: Path,
        compression: Any = "lz4",
        statistics: bool = True,
        row_group_size: int = 64000,
        **kwargs: Any,
    ) -> None:
        """Save a Polars DataFrame or LazyFrame as Parquet.

        Optimizations:
        - statistics: Enable statistics for predicate pushdown
        - row_group_size: Tuned for read-heavy workloads (64k optimal per benchmark)
        """
        ensure_dir(path)
        kwargs.setdefault("compression", compression)
        kwargs.setdefault("statistics", statistics)

        if isinstance(obj, pl.LazyFrame):
            kwargs.setdefault("row_group_size", row_group_size)
            obj.sink_parquet(path, **kwargs)
        else:
            obj.write_parquet(
                path,
                row_group_size=row_group_size,
                compression=cast(Any, kwargs.get("compression", compression)),
                **{k: v for k, v in kwargs.items() if k != "compression"},
            )

    async def async_read(self, path: Path, **kwargs: Any) -> Any:
        """Asynchronously read a Parquet file into a Polars DataFrame."""
        chunk_size = kwargs.pop("chunk_size", None)

        if kwargs.get("lazy") or kwargs.get("streaming"):
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, lambda: self.read(path, **kwargs))

        if chunk_size is not None:
            try:
                chunk_size = int(chunk_size)
            except (TypeError, ValueError):
                chunk_size = None

        content = await read_async_content(path, chunk_size=chunk_size)
        buffer = io.BytesIO(content)
        return self.read(buffer, **kwargs)

    async def async_save(self, obj: Any, path: Path, **kwargs: Any) -> None:
        """Asynchronously save a Polars DataFrame as Parquet."""
        await async_ensure_dir(path)
        await asyncio.to_thread(self.save, obj, path, **kwargs)
