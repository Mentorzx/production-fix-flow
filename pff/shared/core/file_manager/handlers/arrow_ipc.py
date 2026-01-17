"""Arrow IPC (Feather V2) file handler.

Optimized for:
- Zero-copy reading via memory mapping (when uncompressed).
- Ultra-fast caching of intermediate dataframes.
- Interoperability between Polars and PyArrow.
"""

from __future__ import annotations

import asyncio
import io
from pathlib import Path
from typing import Any

import polars as pl
import pyarrow as pa
import pyarrow.ipc

from .base import FileHandler
from ..utils import ensure_dir
from ..async_io import async_ensure_dir


class ArrowIPCHandler(FileHandler):
    """Handler for Arrow IPC / Feather files (.arrow, .ipc, .feather).

    Best Practices:
    - Use compression='uncompressed' for local cache to enable memory mapping.
    - Use compression='lz4' for network transfer or long-term storage.
    - Reading is optimized using Polars native mmap support.
    """

    def read(self, path: Path | io.BytesIO, **kwargs: Any) -> Any:
        """Read Arrow IPC file.

        Args:
            path: File path or bytes buffer.
            **kwargs:
                memory_map: Force memory map on/off (default: True for paths).
                lazy: Return LazyFrame instead of DataFrame (default: False).
                use_pyarrow: Return pyarrow.Table instead of Polars (default: False).
                n_rows: Limit rows.
                columns: Select specific columns.

        Returns:
            Polars DataFrame, LazyFrame, or pyarrow.Table.
        """
        lazy = kwargs.pop("lazy", False)
        use_pyarrow = kwargs.pop("use_pyarrow", False)
        memory_map = kwargs.pop("memory_map", isinstance(path, (str, Path)))

        if use_pyarrow:
            if lazy:
                raise ValueError("Lazy reading not supported with use_pyarrow=True")

            # Resource safety: Ensure source is closed
            source = path
            should_close = False
            if isinstance(path, (str, Path)):
                if memory_map:
                    source = pa.memory_map(str(path), "r")
                else:
                    source = pa.OSFile(str(path), "rb")
                should_close = True

            try:
                # Use context manager for reader if possible, though open_file returns a reader object
                # that doesn't strictly require closing if the underlying source is managed.
                with pa.ipc.open_file(source) as reader:
                    columns = kwargs.get("columns")
                    if columns:
                        table = reader.read_all()
                        return table.select(columns)
                    return reader.read_all()
            finally:
                if should_close and hasattr(source, "close"):
                    source.close()

        # Scan (Lazy)
        if lazy:
            return pl.scan_ipc(path, memory_map=memory_map, **kwargs)

        # Read (Eager)
        return pl.read_ipc(path, memory_map=memory_map, **kwargs)

    def save(self, obj: Any, path: Path, **kwargs: Any) -> None:
        """Save DataFrame or Arrow Table to Arrow IPC.

        Uses atomic write (write to .tmp -> rename) to ensure safety,
        especially when the target file might be memory-mapped by readers.

        Args:
            obj: Polars DataFrame/LazyFrame, pyarrow.Table, or RecordBatchReader.
            path: Destination path.
            **kwargs:
                compression: 'uncompressed' (default), 'lz4', or 'zstd'.
        """
        # Fix 1: Ensure parent directory exists
        ensure_dir(path.parent)

        compression = kwargs.pop("compression", "uncompressed")
        # Fix 3: Robust compression validation
        if compression is None:
            compression = "uncompressed"

        if compression not in ("uncompressed", "lz4", "zstd"):
            compression = "uncompressed"

        # Atomic write strategy
        tmp_path = path.with_suffix(path.suffix + ".tmp")

        try:
            if isinstance(obj, (pl.DataFrame, pl.LazyFrame)):
                if isinstance(obj, pl.LazyFrame):
                    obj = obj.collect()
                obj.write_ipc(tmp_path, compression=compression, **kwargs)

            elif isinstance(obj, (pa.Table, pa.RecordBatchReader)):
                # Map compression string to PyArrow format
                pa_compression = None if compression == "uncompressed" else compression

                options = pa.ipc.IpcWriteOptions(compression=pa_compression)

                with pa.OSFile(str(tmp_path), "wb") as sink:
                    with pa.ipc.new_file(sink, obj.schema, options=options) as writer:
                        if isinstance(obj, pa.Table):
                            # Fix 4: Chunking to avoid OOM on large tables
                            writer.write_table(obj, max_chunksize=64 * 1024)
                        else:
                            for batch in obj:
                                writer.write_batch(batch)
            else:
                raise TypeError(
                    f"ArrowIPCHandler expects Polars DataFrame/LazyFrame or PyArrow Table/Reader, "
                    f"got {type(obj)}"
                )

            # Atomic rename to overwrite target
            tmp_path.replace(path)

        except Exception:
            # Cleanup temp file on failure
            if tmp_path.exists():
                tmp_path.unlink()
            raise

    async def async_read(self, path: Path, **kwargs: Any) -> Any:
        """Async read (delegates to thread pool)."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.read(path, **kwargs))

    async def async_save(self, obj: Any, path: Path, **kwargs: Any) -> None:
        """Async save."""
        # Fix 1: Ensure parent directory exists (async)
        await async_ensure_dir(path.parent)
        await asyncio.to_thread(self.save, obj, path, **kwargs)
