"""Concrete materializer implementations."""

from __future__ import annotations

import io
import os
import zipfile
from typing import TYPE_CHECKING, Any

import msgspec
import polars as pl
import pyarrow.parquet as pq

from .base import Materializer

if TYPE_CHECKING:
    from ..bundles import ParquetBundle

from ...logging import logger
from ..utils import read_raw_bytes


class TabularMaterializer(Materializer):
    """Materializer for tabular parsed_kind bundles."""

    @property
    def parsed_kind(self) -> str:
        return "tabular"

    def materialize(self, bundle: ParquetBundle) -> pl.DataFrame:
        """Return the parsed parquet as a Polars DataFrame using streaming collect.

        Optimized with Arrow IPC sidecar caching for repeated reads.
        """
        if bundle.parsed_parquet_path is None:
            raise ValueError("Parsed parquet not available for tabular bundle")

        arrow_path = bundle.parsed_parquet_path.with_suffix(".arrow")
        if arrow_path.exists():
            try:
                return pl.read_ipc(arrow_path, memory_map=True)
            except Exception as e:
                logger.debug(
                    f"Failed to read Arrow sidecar {arrow_path}, falling back to parquet: {e}"
                )

        df = pl.scan_parquet(bundle.parsed_parquet_path).collect(engine="streaming")

        try:
            if os.access(arrow_path.parent, os.W_OK):
                df.write_ipc(arrow_path, compression="uncompressed")
                logger.debug(f"Created Arrow sidecar cache: {arrow_path.name}")
        except Exception as e:
            logger.debug(f"Failed to create Arrow sidecar: {e}")

        return df


class JsonYamlMaterializer(Materializer):
    """Materializer for json and yaml parsed_kind bundles."""

    @property
    def parsed_kind(self) -> str:
        return "json"

    def materialize(self, bundle: ParquetBundle) -> Any:
        """Decode JSON/YAML from parsed parquet or raw bytes."""
        if bundle.parsed_parquet_path and bundle.parsed_parquet_path.exists():
            if bundle.parsed_parquet_path.stat().st_size > 0:
                try:
                    table = pq.ParquetFile(bundle.parsed_parquet_path)
                    batch = next(table.iter_batches(columns=["payload_msgpack", "payload_bytes"]))
                    payload_msgpack = batch.column(batch.schema.get_field_index("payload_msgpack"))[
                        0
                    ].as_py()
                    payload_bytes = batch.column(batch.schema.get_field_index("payload_bytes"))[
                        0
                    ].as_py()
                    if payload_msgpack:
                        return msgspec.msgpack.decode(payload_msgpack)
                    if payload_bytes:
                        from ..handlers import get_handler

                        handler = get_handler(bundle.ext)
                        if handler:
                            return handler.load_bytes(bytes(payload_bytes))
                except Exception as exc:
                    logger.debug(
                        f"Parsed parquet cache invalid ({bundle.parsed_parquet_path}); "
                        f"autocleaning and falling back: {exc}"
                    )
                    try:
                        bundle.parsed_parquet_path.unlink(missing_ok=True)
                    except Exception:
                        pass

        raw = read_raw_bytes(bundle.raw_parquet_path)
        from ..handlers import get_handler

        handler = get_handler(bundle.ext)
        if handler:
            return handler.load_bytes(raw)
        return raw


class TextMaterializer(Materializer):
    """Materializer for text parsed_kind bundles."""

    @property
    def parsed_kind(self) -> str:
        return "text"

    def materialize(self, bundle: ParquetBundle) -> str:
        """Return text content from parsed parquet or raw bytes."""
        if bundle.parsed_parquet_path and bundle.parsed_parquet_path.exists():
            if bundle.parsed_parquet_path.stat().st_size > 0:
                try:
                    table = pq.ParquetFile(bundle.parsed_parquet_path)
                    batch = next(table.iter_batches(columns=["payload_text"]))
                    payload_text = batch.column(batch.schema.get_field_index("payload_text"))[
                        0
                    ].as_py()
                    if payload_text is not None:
                        return payload_text
                except Exception as exc:
                    logger.warning(
                        f"Parsed parquet cache invalid ({bundle.parsed_parquet_path}): {exc}"
                    )

        raw = read_raw_bytes(bundle.raw_parquet_path)
        from ..handlers import get_handler

        handler = get_handler(bundle.ext)
        if handler:
            return handler.load_bytes(raw)
        return raw.decode("utf-8", errors="ignore")


class BytesMaterializer(Materializer):
    """Materializer for bytes parsed_kind bundles."""

    @property
    def parsed_kind(self) -> str:
        return "bytes"

    def materialize(self, bundle: ParquetBundle) -> bytes:
        """Return raw bytes from parsed parquet or raw layer."""
        if bundle.parsed_parquet_path and bundle.parsed_parquet_path.exists():
            if bundle.parsed_parquet_path.stat().st_size > 0:
                try:
                    table = pq.ParquetFile(bundle.parsed_parquet_path)
                    batch = next(table.iter_batches(columns=["payload_bytes"]))
                    payload_bytes = batch.column(batch.schema.get_field_index("payload_bytes"))[
                        0
                    ].as_py()
                    if payload_bytes is not None:
                        return bytes(payload_bytes)
                except Exception as exc:
                    logger.debug(
                        f"Parsed parquet cache invalid ({bundle.parsed_parquet_path}); "
                        f"autocleaning and falling back: {exc}"
                    )
                    try:
                        bundle.parsed_parquet_path.unlink(missing_ok=True)
                    except Exception:
                        pass

        return read_raw_bytes(bundle.raw_parquet_path)


class ContainerMaterializer(Materializer):
    """Materializer for container parsed_kind bundles (ZIP contents)."""

    @property
    def parsed_kind(self) -> str:
        return "container"

    def materialize(self, bundle: ParquetBundle) -> dict[str, Any]:
        """Return dict of entry_name -> entry_value from container parquet."""
        if bundle.parsed_parquet_path is None:
            raise ValueError("Parsed parquet not available for container bundle")

        from ..handlers import get_handler
        from ..handlers.zstd import ZstdHandler

        result: dict[str, Any] = {}
        table = pq.ParquetFile(bundle.parsed_parquet_path)
        zip_bytes: bytes | None = None
        zip_reader: zipfile.ZipFile | None = None
        schema = table.schema_arrow
        field_indices = {
            "entry_name": schema.get_field_index("entry_name"),
            "entry_ext": schema.get_field_index("entry_ext"),
            "payload_kind": schema.get_field_index("payload_kind"),
            "payload_msgpack": schema.get_field_index("payload_msgpack"),
            "payload_text": schema.get_field_index("payload_text"),
            "payload_bytes": schema.get_field_index("payload_bytes"),
            "payload_parquet_path": schema.get_field_index("payload_parquet_path"),
        }
        handler_cache: dict[str, Any] = {}

        def _get_handler_cached(ext: str) -> Any:
            if ext in handler_cache:
                return handler_cache[ext]
            handler = get_handler(ext)
            handler_cache[ext] = handler
            return handler

        def _get_zip_reader() -> zipfile.ZipFile | None:
            nonlocal zip_bytes, zip_reader
            if zip_reader is not None:
                return zip_reader
            raw = read_raw_bytes(bundle.raw_parquet_path)
            if bundle.ext in {".zst", ".zstd"} and bundle.metadata.get("inner_ext") == ".zip":
                zip_bytes = ZstdHandler().load_bytes(raw)
            elif bundle.ext == ".zip":
                zip_bytes = raw
            else:
                return None
            if zip_bytes is None:
                return None
            zip_reader = zipfile.ZipFile(io.BytesIO(zip_bytes), "r")
            return zip_reader

        try:
            for batch in table.iter_batches():
                names = batch.column(field_indices["entry_name"]).to_pylist()
                exts = batch.column(field_indices["entry_ext"]).to_pylist()
                kinds = batch.column(field_indices["payload_kind"]).to_pylist()
                msgpacks = batch.column(field_indices["payload_msgpack"]).to_pylist()
                texts = batch.column(field_indices["payload_text"]).to_pylist()
                bytes_list = batch.column(field_indices["payload_bytes"]).to_pylist()
                parquet_paths = batch.column(field_indices["payload_parquet_path"]).to_pylist()

                for name, ext, kind, msgp, text, raw_bytes, parquet_path in zip(
                    names, exts, kinds, msgpacks, texts, bytes_list, parquet_paths
                ):
                    if kind == "tabular" and parquet_path:
                        result[name] = pl.read_parquet(parquet_path)
                        continue
                    if kind == "text":
                        if text is not None:
                            result[name] = text
                            continue
                        if raw_bytes is not None:
                            handler = _get_handler_cached(ext or ".txt")
                            if handler is not None:
                                try:
                                    result[name] = handler.load_bytes(bytes(raw_bytes))
                                    continue
                                except Exception as exc:
                                    logger.debug(f"Handler failed for text entry {name}: {exc}")
                            result[name] = bytes(raw_bytes)
                            continue
                    if kind == "json":
                        if msgp:
                            result[name] = msgspec.msgpack.decode(msgp)
                            continue
                        if raw_bytes is not None:
                            handler = _get_handler_cached(ext or ".json")
                            if handler is not None:
                                try:
                                    result[name] = handler.load_bytes(bytes(raw_bytes))
                                    continue
                                except Exception as exc:
                                    logger.debug(f"Handler failed for json entry {name}: {exc}")
                            result[name] = bytes(raw_bytes)
                            continue
                    if raw_bytes is None and parquet_path is None and text is None and msgp is None:
                        zf = _get_zip_reader()
                        if zf is not None:
                            try:
                                raw_bytes = zf.read(name)
                            except Exception as exc:
                                logger.debug(f"ZIP read failed for container entry {name}: {exc}")
                                raw_bytes = None
                    if raw_bytes is not None:
                        payload = bytes(raw_bytes)
                        handler = _get_handler_cached(ext or "")
                        if handler is not None:
                            try:
                                result[name] = handler.load_bytes(payload)
                                continue
                            except Exception as exc:
                                logger.debug(f"Handler failed for container entry {name}: {exc}")
                        result[name] = payload
        finally:
            if zip_reader is not None:
                zip_reader.close()

        return result


class ZipMaterializer(Materializer):
    """Materializer for ZIP files without parsed container parquet."""

    @property
    def parsed_kind(self) -> str:
        return "zip"

    def materialize(self, bundle: ParquetBundle) -> dict[str, Any]:
        """Load ZIP from raw bytes and return dict of contents."""
        from ..async_io import run_coroutine_sync
        from ..container.zip import load_zip_from_bytes

        raw = read_raw_bytes(bundle.raw_parquet_path)
        return run_coroutine_sync(
            load_zip_from_bytes(
                raw,
                parallel=False,
                task_type="thread",
                chunk_size=16,
                handler_kwargs={},
                fuse_processing=True,
                max_members=None,
            )
        )


class ZstdMaterializer(Materializer):
    """Materializer for Zstd files."""

    @property
    def parsed_kind(self) -> str:
        return "zstd"

    def materialize(self, bundle: ParquetBundle) -> Any:
        """Decompress and parse zstd content."""
        from ..handlers import get_handler

        raw = read_raw_bytes(bundle.raw_parquet_path)
        inner_ext = bundle.metadata.get("inner_ext")
        handler = get_handler(bundle.ext)
        if handler and inner_ext:
            return handler.read(io.BytesIO(raw), inner_suffix=inner_ext)
        return raw
