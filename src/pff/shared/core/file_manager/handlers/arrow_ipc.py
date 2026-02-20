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

from ..async_io import async_ensure_dir
from ..config import get_arrow_config
from ..utils import ensure_dir
from .base import FileHandler


class ArrowIPCHandler(FileHandler):
    """Handler for Arrow IPC / Feather files (.arrow, .ipc, .feather).

    Best Practices:
    - Use compression='uncompressed' for local cache to enable memory mapping.
    - Use compression='lz4' for network transfer or long-term storage.
    - Reading is optimized using Polars native mmap support.
    """

    def read(self, path: Path | io.BytesIO, **kwargs: Any) -> Any:
        """Execute read.



        Args:

            path: Input value used by this callable.

            **kwargs: Additional keyword arguments.



        Returns:

            Return value produced by the callable.



        Raises:

            Exception: Propagates domain-specific failures with context.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        cfg = get_arrow_config()

        lazy = kwargs.pop("lazy", False)
        use_pyarrow = kwargs.pop("use_pyarrow", cfg.get("read_engine") == "pyarrow")

        default_mmap = cfg.get("mmap_enabled", True) and isinstance(path, (str, Path))
        memory_map = kwargs.pop("memory_map", default_mmap)

        rechunk = kwargs.pop("rechunk", cfg.get("rechunk", False))

        if use_pyarrow:
            if lazy:
                raise ValueError("Lazy reading not supported with use_pyarrow=True")

            should_close = False
            source: Any = path
            if isinstance(path, (str, Path)):
                if memory_map:
                    source = pa.memory_map(str(path), "r")
                else:
                    source = pa.OSFile(str(path), "rb")
                should_close = True

            try:
                with pa.ipc.open_file(source) as reader:
                    columns = kwargs.get("columns")
                    if columns:
                        table = reader.read_all()
                        return table.select(columns)
                    return reader.read_all()
            finally:
                if should_close and hasattr(source, "close"):
                    source.close()

        if lazy:
            return pl.scan_ipc(path, memory_map=memory_map, rechunk=rechunk, **kwargs)

        return pl.read_ipc(path, memory_map=memory_map, rechunk=rechunk, **kwargs)

    def save(self, obj: Any, path: Path, **kwargs: Any) -> None:
        """Save DataFrame or Arrow Table to Arrow IPC.

        Uses atomic write (write to .tmp -> rename) to ensure safety.
        """
        ensure_dir(path.parent)
        cfg = get_arrow_config()

        compression = kwargs.pop("compression", "uncompressed")
        if compression is None or compression not in ("uncompressed", "lz4", "zstd"):
            compression = "uncompressed"

        tmp_path = path.with_suffix(path.suffix + ".tmp")

        try:
            if isinstance(obj, (pl.DataFrame, pl.LazyFrame)):
                if isinstance(obj, pl.LazyFrame):
                    obj = obj.collect()

                obj.write_ipc(tmp_path, compression=compression, **kwargs)

            elif isinstance(obj, (pa.Table, pa.ipc.RecordBatchReader)):
                pa_compression = None if compression == "uncompressed" else compression

                if pa_compression and not pa.Codec.is_available(pa_compression):
                    pa_compression = None

                use_threads = cfg.get("use_threads", True)
                unify_dictionaries = cfg.get("unify_dictionaries", False)

                options = pa.ipc.IpcWriteOptions(
                    compression=pa_compression,
                    use_threads=use_threads,
                    unify_dictionaries=unify_dictionaries,
                )

                with pa.OSFile(str(tmp_path), "wb") as sink:
                    with pa.ipc.new_file(sink, obj.schema, options=options) as writer:
                        if isinstance(obj, pa.Table):
                            writer.write_table(obj, max_chunksize=64 * 1024)
                        else:
                            for batch in obj:
                                writer.write_batch(batch)
            else:
                raise TypeError(
                    f"ArrowIPCHandler expects Polars DataFrame/LazyFrame or PyArrow Table/Reader, "
                    f"got {type(obj)}"
                )

            tmp_path.replace(path)

        except Exception:
            if tmp_path.exists():
                tmp_path.unlink()
            raise

    async def async_read(self, path: Path, **kwargs: Any) -> Any:
        """Async read (delegates to thread pool)."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.read(path, **kwargs))

    async def async_save(self, obj: Any, path: Path, **kwargs: Any) -> None:
        """Async save."""
        await async_ensure_dir(path.parent)
        await asyncio.to_thread(self.save, obj, path, **kwargs)
