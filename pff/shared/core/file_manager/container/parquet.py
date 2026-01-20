"""Container parquet writing utilities."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import msgspec
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from ....acceleration.concurrency import ConcurrencyManager
from ...logging import logger
from ..async_io import run_coroutine_sync
from ..config import (
    get_container_flush_rows,
    get_parquet_compression,
    get_parquet_row_group_size,
)
from ..handlers import FileHandler, get_handler
from ..utils import fast_suffix, make_json_safe
from .zip import ZipBytesSource, ZipPathSource, iter_zip_entries


def _container_parquet_schema() -> pa.Schema:
    return pa.schema(
        [
            ("file_id", pa.string()),
            ("entry_name", pa.string()),
            ("entry_ext", pa.string()),
            ("payload_kind", pa.string()),
            ("payload_msgpack", pa.binary()),
            ("payload_text", pa.string()),
            ("payload_bytes", pa.binary()),
            ("payload_parquet_path", pa.string()),
        ]
    )


def _path_key(p: Path) -> str:
    """Generate a unique key for an entry path."""
    return hashlib.sha256(str(p).encode()).hexdigest()


def write_container_parquet_from_entries(
    entries: Iterable[tuple[str, bytes]],
    parsed_parquet_path: Path,
    *,
    file_id: str,
    handler_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Write container parquet using an iterable of (name, raw_bytes)."""
    parsed_parquet_path.parent.mkdir(parents=True, exist_ok=True)
    entry_dir = parsed_parquet_path.parent / "entries"
    entry_dir.mkdir(parents=True, exist_ok=True)

    schema = _container_parquet_schema()
    compression, level = get_parquet_compression()

    writer = pq.ParquetWriter(
        parsed_parquet_path,
        schema=schema,
        compression=compression,
        compression_level=level,
        use_dictionary=False,
        write_statistics=False,
    )

    flush_rows = get_container_flush_rows()
    buffer: dict[str, list[Any]] = {
        "file_id": [],
        "entry_name": [],
        "entry_ext": [],
        "payload_kind": [],
        "payload_msgpack": [],
        "payload_text": [],
        "payload_bytes": [],
        "payload_parquet_path": [],
    }

    def _flush() -> None:
        if not buffer["entry_name"]:
            return
        table = pa.Table.from_arrays(
            [
                pa.array(buffer["file_id"]),
                pa.array(buffer["entry_name"]),
                pa.array(buffer["entry_ext"]),
                pa.array(buffer["payload_kind"]),
                pa.array(buffer["payload_msgpack"], type=pa.binary()),
                pa.array(buffer["payload_text"]),
                pa.array(buffer["payload_bytes"], type=pa.binary()),
                pa.array(buffer["payload_parquet_path"]),
            ],
            schema=schema,
        )
        writer.write_table(table, row_group_size=len(buffer["entry_name"]))
        for key in buffer:
            buffer[key].clear()

    entry_count = 0
    try:
        handler_cache: dict[str, FileHandler] = {}
        for name, raw in entries:
            entry_count += 1
            entry_ext = fast_suffix(name)
            handler = handler_cache.get(entry_ext)
            if handler is None:
                handler = get_handler(entry_ext)
                if handler is not None:
                    handler_cache[entry_ext] = handler

            payload_kind = "bytes"
            payload_msgpack: bytes | None = None
            payload_text: str | None = None
            payload_bytes: bytes | None = raw
            payload_parquet_path: str | None = None

            if entry_ext in {".json", ".yaml", ".yml", ".txt"}:
                if entry_ext == ".txt":
                    payload_kind = "text"
                else:
                    payload_kind = "json" if entry_ext == ".json" else "yaml"
                payload_bytes = raw
                payload_msgpack = None
                payload_text = None
            elif handler is not None:
                try:
                    obj = handler.load_bytes(raw, **handler_kwargs)
                    if isinstance(obj, pl.DataFrame):
                        entry_key = _path_key(Path(name))
                        entry_path = entry_dir / f"{entry_key}.parquet"
                        obj.write_parquet(
                            entry_path,
                            compression=compression,
                            compression_level=level,
                            statistics=False,
                            row_group_size=get_parquet_row_group_size(),
                        )
                        payload_kind = "tabular"
                        payload_parquet_path = str(entry_path)
                        payload_bytes = None
                    elif isinstance(obj, str):
                        payload_kind = "text"
                        payload_text = obj
                        payload_bytes = None
                    elif isinstance(obj, bytes):
                        payload_kind = "bytes"
                        payload_bytes = obj
                    else:
                        payload_kind = "json"
                        payload_msgpack = msgspec.msgpack.encode(make_json_safe(obj))
                        payload_bytes = None
                except Exception as exc:
                    logger.debug(
                        f"Failed to parse container entry name={name} ext={entry_ext}: {exc}"
                    )
                    payload_kind = "bytes"
                    payload_bytes = raw

            buffer["file_id"].append(file_id)
            buffer["entry_name"].append(name)
            buffer["entry_ext"].append(entry_ext)
            buffer["payload_kind"].append(payload_kind)
            buffer["payload_msgpack"].append(payload_msgpack)
            buffer["payload_text"].append(payload_text)
            buffer["payload_bytes"].append(payload_bytes)
            buffer["payload_parquet_path"].append(payload_parquet_path)

            if len(buffer["entry_name"]) >= flush_rows:
                _flush()
        _flush()
    finally:
        writer.close()

    return {
        "num_members": entry_count,
        "container_schema_version": "1.0",
    }


def write_container_parquet_index(
    members: Iterable[str],
    parsed_parquet_path: Path,
    *,
    file_id: str,
) -> dict[str, Any]:
    """Write container parquet metadata without reading entry payloads."""
    parsed_parquet_path.parent.mkdir(parents=True, exist_ok=True)
    schema = _container_parquet_schema()
    compression, level = get_parquet_compression()

    writer = pq.ParquetWriter(
        parsed_parquet_path,
        schema=schema,
        compression=compression,
        compression_level=level,
        use_dictionary=False,
        write_statistics=False,
    )

    flush_rows = get_container_flush_rows()
    buffer: dict[str, list[Any]] = {
        "file_id": [],
        "entry_name": [],
        "entry_ext": [],
        "payload_kind": [],
        "payload_msgpack": [],
        "payload_text": [],
        "payload_bytes": [],
        "payload_parquet_path": [],
    }

    def _flush() -> None:
        if not buffer["entry_name"]:
            return
        table = pa.Table.from_arrays(
            [
                pa.array(buffer["file_id"]),
                pa.array(buffer["entry_name"]),
                pa.array(buffer["entry_ext"]),
                pa.array(buffer["payload_kind"]),
                pa.array(buffer["payload_msgpack"], type=pa.binary()),
                pa.array(buffer["payload_text"]),
                pa.array(buffer["payload_bytes"], type=pa.binary()),
                pa.array(buffer["payload_parquet_path"]),
            ],
            schema=schema,
        )
        writer.write_table(table, row_group_size=len(buffer["entry_name"]))
        for key in buffer:
            buffer[key].clear()

    entry_count = 0
    try:
        for name in members:
            entry_count += 1
            entry_ext = fast_suffix(name)
            if entry_ext == ".txt":
                payload_kind = "text"
            elif entry_ext == ".json":
                payload_kind = "json"
            elif entry_ext in {".yaml", ".yml"}:
                payload_kind = "yaml"
            else:
                payload_kind = "bytes"

            buffer["file_id"].append(file_id)
            buffer["entry_name"].append(name)
            buffer["entry_ext"].append(entry_ext)
            buffer["payload_kind"].append(payload_kind)
            buffer["payload_msgpack"].append(None)
            buffer["payload_text"].append(None)
            buffer["payload_bytes"].append(None)
            buffer["payload_parquet_path"].append(None)

            if len(buffer["entry_name"]) >= flush_rows:
                _flush()
        _flush()
    finally:
        writer.close()

    return {
        "num_members": entry_count,
        "container_schema_version": "1.0",
        "container_payloads": "lazy",
    }


def _read_members_chunk(source: ZipPathSource, members: list[str]) -> list[tuple[str, bytes]]:
    return list(source.iter_members(members))


def _iter_parallel_entries(
    source: ZipPathSource,
    members: list[str],
    *,
    chunk_size: int,
    task_type: str,
) -> Iterable[tuple[str, bytes]]:
    cm = ConcurrencyManager()
    chunks = [members[i : i + chunk_size] for i in range(0, len(members), chunk_size)]
    if not chunks:
        return
    max_workers = max(1, min(cm.hardware.logical_cores, len(chunks)))
    if task_type != "thread":
        task_type = "thread"
    for idx in range(0, len(chunks), max_workers):
        batch = chunks[idx : idx + max_workers]
        read_args = [(source, chunk) for chunk in batch]
        chunk_results = run_coroutine_sync(
            cm.execute(
                _read_members_chunk,
                read_args,
                task_type=task_type,
                max_workers=max_workers,
                desc="Lendo entradas ZIP",
            )
        )
        for chunk in chunk_results:
            yield from chunk


def write_container_parquet_from_zip(
    zip_path: Path,
    parsed_parquet_path: Path,
    *,
    file_id: str,
    members: list[str],
    handler_kwargs: dict[str, Any],
    use_mmap: bool = True,
    read_parallel: bool = False,
    read_chunk_size: int | None = None,
    read_task_type: str = "thread",
) -> dict[str, Any]:
    """Write container parquet from a ZIP file path."""
    source = ZipPathSource(zip_path, use_mmap=use_mmap)
    if read_chunk_size is None:
        read_chunk_size = max(16, min(256, get_container_flush_rows() // 2))
    if read_parallel and len(members) > read_chunk_size:
        entries = _iter_parallel_entries(
            source,
            members,
            chunk_size=read_chunk_size,
            task_type=read_task_type,
        )
    else:
        entries = iter_zip_entries(source, members)
    return write_container_parquet_from_entries(
        entries,
        parsed_parquet_path,
        file_id=file_id,
        handler_kwargs=handler_kwargs,
    )


def write_container_parquet_from_zip_bytes(
    data: bytes,
    parsed_parquet_path: Path,
    *,
    file_id: str,
    members: list[str],
    handler_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Write container parquet from ZIP bytes."""
    source = ZipBytesSource(data)
    entries = iter_zip_entries(source, members)
    return write_container_parquet_from_entries(
        entries,
        parsed_parquet_path,
        file_id=file_id,
        handler_kwargs=handler_kwargs,
    )
