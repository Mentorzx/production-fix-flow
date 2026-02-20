"""Parquet I/O helpers for parquet-first ingestion."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from .config import (
    get_container_flush_rows,
    get_parquet_compression,
    get_parquet_row_group_size,
    get_streaming_threshold_bytes,
)


def raw_parquet_schema() -> pa.Schema:
    """Schema for RAW parquet chunks."""
    return pa.schema(
        [
            ("file_id", pa.string()),
            ("source_path", pa.string()),
            ("ext", pa.string()),
            ("mtime_ns", pa.int64()),
            ("size_bytes", pa.int64()),
            ("sha256", pa.string()),
            ("chunk_index", pa.int32()),
            ("chunk_bytes", pa.binary()),
            ("encoding", pa.string()),
            ("extra_metadata", pa.map_(pa.string(), pa.string())),
        ]
    )


def write_raw_parquet(
    path: Path,
    raw_parquet_path: Path,
    *,
    file_id: str,
    sha256: str,
    stat_sig: tuple[int, int],
    encoding: str | None,
    extra_metadata: dict[str, str] | None,
    chunk_size: int,
) -> None:
    """Write RAW parquet with chunked bytes, tuned for cold ingest speed."""
    raw_parquet_path.parent.mkdir(parents=True, exist_ok=True)
    schema = raw_parquet_schema()
    compression, level = get_parquet_compression()
    if stat_sig[1] <= get_streaming_threshold_bytes():
        compression = "uncompressed"
        level = None
    compression_for_writer = "NONE" if compression == "uncompressed" else compression
    streaming_limit = max(1, get_streaming_threshold_bytes())
    max_rows_by_mem = max(1, streaming_limit // max(1, chunk_size))
    flush_rows = min(get_container_flush_rows(), max_rows_by_mem)

    # Atomic write pattern: write to tmp, then rename
    tmp_fh, tmp_path = tempfile.mkstemp(
        dir=raw_parquet_path.parent, prefix=".tmp_raw_", suffix=".parquet"
    )
    os.close(tmp_fh)
    tmp_p = Path(tmp_path)

    writer = pq.ParquetWriter(
        tmp_p,
        schema=schema,
        compression=compression_for_writer,
        compression_level=level,
        use_dictionary=False,
        write_statistics=False,
    )

    source_path_str = str(path)
    ext = path.suffix.lower()
    mtime_ns, size_bytes = stat_sig

    buffer: dict[str, list[Any]] = {
        "file_id": [],
        "source_path": [],
        "ext": [],
        "mtime_ns": [],
        "size_bytes": [],
        "sha256": [],
        "chunk_index": [],
        "chunk_bytes": [],
        "encoding": [],
        "extra_metadata": [],
    }

    def _flush() -> None:
        """Execute flush."""

        if not buffer["chunk_index"]:
            return
        table = pa.Table.from_arrays(
            [
                pa.array(buffer["file_id"]),
                pa.array(buffer["source_path"]),
                pa.array(buffer["ext"]),
                pa.array(buffer["mtime_ns"]),
                pa.array(buffer["size_bytes"]),
                pa.array(buffer["sha256"]),
                pa.array(buffer["chunk_index"], type=pa.int32()),
                pa.array(buffer["chunk_bytes"], type=pa.binary()),
                pa.array(buffer["encoding"], type=pa.string()),
                pa.array(buffer["extra_metadata"], type=pa.map_(pa.string(), pa.string())),
            ],
            schema=schema,
        )
        writer.write_table(table, row_group_size=len(buffer["chunk_index"]))
        for key in buffer:
            buffer[key].clear()

    chunk_index = 0
    try:
        with path.open("rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                buffer["file_id"].append(file_id)
                buffer["source_path"].append(source_path_str)
                buffer["ext"].append(ext)
                buffer["mtime_ns"].append(mtime_ns)
                buffer["size_bytes"].append(size_bytes)
                buffer["sha256"].append(sha256)
                buffer["chunk_index"].append(chunk_index)
                buffer["chunk_bytes"].append(chunk)
                buffer["encoding"].append(encoding)
                buffer["extra_metadata"].append(extra_metadata or {})
                chunk_index += 1

                if len(buffer["chunk_index"]) >= flush_rows:
                    _flush()
        _flush()
    finally:
        writer.close()

    try:
        os.replace(tmp_p, raw_parquet_path)
    except Exception as e:
        if tmp_p.exists():
            tmp_p.unlink()
        raise OSError(f"Failed to atomically rename raw parquet: {e}") from e


def write_raw_parquet_from_bytes(
    data: bytes,
    raw_parquet_path: Path,
    *,
    source_path: Path,
    file_id: str,
    sha256: str,
    stat_sig: tuple[int, int],
    encoding: str | None,
    extra_metadata: dict[str, str] | None,
    chunk_size: int,
) -> None:
    """Write RAW parquet from in-memory bytes to avoid re-reading the source file."""
    raw_parquet_path.parent.mkdir(parents=True, exist_ok=True)
    schema = raw_parquet_schema()
    compression, level = get_parquet_compression()
    compression_for_writer = "NONE" if compression == "uncompressed" else compression
    streaming_limit = max(1, get_streaming_threshold_bytes())
    max_rows_by_mem = max(1, streaming_limit // max(1, chunk_size))
    flush_rows = min(get_container_flush_rows(), max_rows_by_mem)

    # Atomic write pattern
    tmp_fh, tmp_path = tempfile.mkstemp(
        dir=raw_parquet_path.parent, prefix=".tmp_raw_mem_", suffix=".parquet"
    )
    os.close(tmp_fh)
    tmp_p = Path(tmp_path)

    writer = pq.ParquetWriter(
        tmp_p,
        schema=schema,
        compression=compression_for_writer,
        compression_level=level,
        use_dictionary=False,
        write_statistics=False,
    )

    source_path_str = str(source_path)
    ext = source_path.suffix.lower()
    mtime_ns, size_bytes = stat_sig

    buffer: dict[str, list[Any]] = {
        "file_id": [],
        "source_path": [],
        "ext": [],
        "mtime_ns": [],
        "size_bytes": [],
        "sha256": [],
        "chunk_index": [],
        "chunk_bytes": [],
        "encoding": [],
        "extra_metadata": [],
    }

    def _flush() -> None:
        """Execute flush."""

        if not buffer["chunk_index"]:
            return
        table = pa.Table.from_arrays(
            [
                pa.array(buffer["file_id"]),
                pa.array(buffer["source_path"]),
                pa.array(buffer["ext"]),
                pa.array(buffer["mtime_ns"]),
                pa.array(buffer["size_bytes"]),
                pa.array(buffer["sha256"]),
                pa.array(buffer["chunk_index"], type=pa.int32()),
                pa.array(buffer["chunk_bytes"], type=pa.binary()),
                pa.array(buffer["encoding"], type=pa.string()),
                pa.array(buffer["extra_metadata"], type=pa.map_(pa.string(), pa.string())),
            ],
            schema=schema,
        )
        writer.write_table(table, row_group_size=len(buffer["chunk_index"]))
        for key in buffer:
            buffer[key].clear()

    chunk_index = 0
    try:
        view = memoryview(data)
        total = len(view)
        if total <= get_streaming_threshold_bytes():
            buffer["file_id"].append(file_id)
            buffer["source_path"].append(source_path_str)
            buffer["ext"].append(ext)
            buffer["mtime_ns"].append(mtime_ns)
            buffer["size_bytes"].append(size_bytes)
            buffer["sha256"].append(sha256)
            buffer["chunk_index"].append(chunk_index)
            buffer["chunk_bytes"].append(view)
            buffer["encoding"].append(encoding)
            buffer["extra_metadata"].append(extra_metadata or {})
            _flush()
        else:
            offset = 0
            while offset < total:
                end = min(offset + chunk_size, total)
                chunk = view[offset:end]
                buffer["file_id"].append(file_id)
                buffer["source_path"].append(source_path_str)
                buffer["ext"].append(ext)
                buffer["mtime_ns"].append(mtime_ns)
                buffer["size_bytes"].append(size_bytes)
                buffer["sha256"].append(sha256)
                buffer["chunk_index"].append(chunk_index)
                buffer["chunk_bytes"].append(chunk)
                buffer["encoding"].append(encoding)
                buffer["extra_metadata"].append(extra_metadata or {})
                chunk_index += 1
                offset = end

                if len(buffer["chunk_index"]) >= flush_rows:
                    _flush()
        _flush()
    finally:
        writer.close()

    try:
        os.replace(tmp_p, raw_parquet_path)
    except Exception as e:
        if tmp_p.exists():
            tmp_p.unlink()
        raise OSError(f"Failed to atomically rename raw mem parquet: {e}") from e


def stream_raw_parquet_to_path(raw_parquet_path: Path, dest_path: Path) -> str:
    """Stream RAW parquet bytes to a destination file and return SHA256."""
    import hashlib

    sha = hashlib.sha256()
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    table = pq.ParquetFile(raw_parquet_path)
    with dest_path.open("wb") as f:
        for batch in table.iter_batches(columns=["chunk_bytes"]):
            chunks = batch.column(0).to_pylist()
            for chunk in chunks:
                if chunk is None:
                    continue
                data = bytes(chunk)
                f.write(data)
                sha.update(data)
    return sha.hexdigest()


def write_parsed_payload_parquet(
    parsed_parquet_path: Path,
    *,
    file_id: str,
    payload_text: str | None,
    payload_msgpack: bytes | None,
    payload_bytes: bytes | None,
    parsed_kind: str,
    parse_metadata: dict[str, str] | None,
) -> None:
    """Write parsed payload parquet for JSON/YAML/text/bytes."""
    parsed_parquet_path.parent.mkdir(parents=True, exist_ok=True)
    schema = pa.schema(
        [
            ("file_id", pa.string()),
            ("parsed_kind", pa.string()),
            ("payload_text", pa.string()),
            ("payload_msgpack", pa.binary()),
            ("payload_bytes", pa.binary()),
            ("parse_metadata", pa.map_(pa.string(), pa.string())),
        ]
    )
    table = pa.Table.from_arrays(
        [
            pa.array([file_id]),
            pa.array([parsed_kind]),
            pa.array([payload_text]),
            pa.array([payload_msgpack], type=pa.binary()),
            pa.array([payload_bytes], type=pa.binary()),
            pa.array([parse_metadata or {}], type=pa.map_(pa.string(), pa.string())),
        ],
        schema=schema,
    )
    compression, level = get_parquet_compression()

    # Atomic write pattern
    tmp_fh, tmp_path = tempfile.mkstemp(
        dir=parsed_parquet_path.parent, prefix=".tmp_parsed_", suffix=".parquet"
    )
    os.close(tmp_fh)
    tmp_p = Path(tmp_path)

    try:
        pq.write_table(
            table,
            tmp_p,
            compression=compression,
            compression_level=level,
            use_dictionary=False,
            write_statistics=False,
        )
        os.replace(tmp_p, parsed_parquet_path)
    except Exception as e:
        if tmp_p.exists():
            tmp_p.unlink()
        raise OSError(f"Failed to atomically rename parsed payload parquet: {e}") from e


def _has_empty_struct(dtype: pl.DataType) -> bool:
    """Execute has empty struct.



    Args:

        dtype: Input value used by this callable.



    Returns:

        Return value produced by the callable.

    """

    if isinstance(dtype, pl.Struct):
        return not getattr(dtype, "fields", None)
    if isinstance(dtype, pl.List):
        inner = getattr(dtype, "inner", None)
        if inner is not None and isinstance(inner, pl.DataType):
            return _has_empty_struct(inner)
        return False
    if hasattr(pl, "Array") and isinstance(dtype, pl.Array):
        inner = getattr(dtype, "inner", None)
        if inner is not None and isinstance(inner, pl.DataType):
            return _has_empty_struct(inner)
        return False
    return False


def _sanitize_empty_structs(df: pl.DataFrame) -> pl.DataFrame:
    """Execute sanitize empty structs.



    Args:

        df: Input value used by this callable.



    Returns:

        Return value produced by the callable.

    """

    schema = df.schema
    replacements = []
    for name, dtype in schema.items():
        if _has_empty_struct(dtype):
            replacements.append(pl.lit(None, dtype=pl.Null).alias(name))
        else:
            replacements.append(pl.col(name))
    return df.select(replacements)


def write_tabular_parquet_from_path(
    path: Path,
    parsed_parquet_path: Path,
    *,
    ext: str,
    **kwargs: Any,
) -> None:
    """Write parsed parquet from tabular source using streaming where possible.

    Optimizations enabled:
    - statistics: Better predicate pushdown
    - use_dictionary: Better compression for repetitive values
    """
    parsed_parquet_path.parent.mkdir(parents=True, exist_ok=True)
    row_group_size = get_parquet_row_group_size()
    compression, level = get_parquet_compression()

    # Atomic write pattern
    tmp_fh, tmp_path = tempfile.mkstemp(
        dir=parsed_parquet_path.parent, prefix=".tmp_tab_", suffix=".parquet"
    )
    os.close(tmp_fh)
    tmp_p = Path(tmp_path)

    try:
        if ext in {".csv", ".tsv"}:
            scan = pl.scan_csv(str(path), **kwargs)
            scan.sink_parquet(
                tmp_p,
                compression=compression,
                compression_level=level,
                statistics=True,
                row_group_size=row_group_size,
                maintain_order=False,
            )
        elif ext in {".ndjson", ".jsonl"}:
            scan = pl.scan_ndjson(str(path), **kwargs)
            try:
                scan.sink_parquet(
                    tmp_p,
                    compression=compression,
                    compression_level=level,
                    statistics=True,
                    row_group_size=row_group_size,
                )
            except Exception:
                df = scan.collect()
                df.write_parquet(
                    tmp_p,
                    compression=compression,  # type: ignore[arg-type]
                    compression_level=level,
                    statistics=True,
                    row_group_size=row_group_size,
                )
        elif ext in {".parquet", ".pq", ".parq"}:
            shutil.copyfile(path, tmp_p)
        else:
            df = pl.read_parquet(str(path))
            df.write_parquet(
                tmp_p,
                compression=compression,  # type: ignore[arg-type]
                compression_level=level,
                statistics=False,
                row_group_size=row_group_size,
            )
        os.replace(tmp_p, parsed_parquet_path)
    except Exception as e:
        if tmp_p.exists():
            tmp_p.unlink()
        raise OSError(f"Failed to atomically write tabular parquet: {e}") from e
