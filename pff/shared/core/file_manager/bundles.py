"""ParquetBundle and related dataclasses for file_manager package."""

from __future__ import annotations

import io
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import msgspec
import polars as pl
import pyarrow.parquet as pq

if TYPE_CHECKING:
    pass

from ..logging import logger


@dataclass
class ParquetBundle:
    """Parquet-first artifact with lossless RAW and optimized PARSED layers.

    Attributes:
        source_path: Original file path that was ingested.
        ext: File extension (lowercase with dot, e.g., ".csv").
        file_id: Unique identifier (typically SHA256 of content).
        raw_parquet_path: Path to RAW parquet containing chunked bytes.
        parsed_parquet_path: Path to PARSED parquet (tabular/json/text/container).
        parsed_kind: Type of parsed content.
        metadata: Additional metadata about the bundle.
        dirty: Whether the bundle has been modified since ingestion.
    """

    source_path: Path
    ext: str
    file_id: str
    raw_parquet_path: Path
    parsed_parquet_path: Path | None
    parsed_kind: Literal["tabular", "json", "yaml", "text", "bytes", "container", "none"]
    metadata: dict[str, Any] = field(default_factory=dict)
    dirty: bool = False

    def lazyframe(self) -> pl.LazyFrame:
        """Get a LazyFrame for tabular bundles."""
        if self.parsed_parquet_path is None:
            raise ValueError("No parsed parquet available")
        return pl.scan_parquet(self.parsed_parquet_path)

    def to_native(self, **kwargs: Any) -> Any:
        """Convert bundle to native Python object using materializers."""

        from .materializers import materialize_bundle

        return materialize_bundle(self, **kwargs)

    def mark_dirty(self) -> None:
        """Mark the bundle as modified."""
        self.dirty = True

    def iter_entries(self) -> Iterator[tuple[str, Any]]:
        """Iterate over container entries.

        Yields:
            Tuples of (entry_name, entry_value) for each file in the container.

        Raises:
            ValueError: If this is not a container bundle.
        """
        if self.parsed_kind != "container":
            raise ValueError("iter_entries is only available for container bundles")
        if not self.parsed_parquet_path:
            raise ValueError("Parsed parquet not available for container bundle")

        from .handlers import get_handler
        from .handlers.zstd import ZstdHandler
        from .utils import read_raw_bytes

        table = pq.ParquetFile(self.parsed_parquet_path)
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
            raw = read_raw_bytes(self.raw_parquet_path)
            if self.ext in {".zst", ".zstd"} and self.metadata.get("inner_ext") == ".zip":
                zip_bytes = ZstdHandler().load_bytes(raw)
            elif self.ext == ".zip":
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
                        yield name, pl.read_parquet(parquet_path)
                        continue
                    if kind == "text":
                        if text is not None:
                            yield name, text
                            continue
                        if raw_bytes is not None:
                            handler = _get_handler_cached(ext or ".txt")
                            if handler is not None:
                                try:
                                    yield name, handler.load_bytes(bytes(raw_bytes))
                                    continue
                                except Exception as exc:
                                    logger.debug(f"Handler failed for text entry {name}: {exc}")
                            yield name, bytes(raw_bytes)
                            continue
                    if kind == "json":
                        if msgp:
                            yield name, msgspec.msgpack.decode(msgp)
                            continue
                        if raw_bytes is not None:
                            handler = _get_handler_cached(ext or ".json")
                            if handler is not None:
                                try:
                                    yield name, handler.load_bytes(bytes(raw_bytes))
                                    continue
                                except Exception as exc:
                                    logger.debug(f"Handler failed for json entry {name}: {exc}")
                            yield name, bytes(raw_bytes)
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
                                yield name, handler.load_bytes(payload)
                                continue
                            except Exception as exc:
                                logger.debug(f"Handler failed for container entry {name}: {exc}")
                        yield name, payload
        finally:
            if zip_reader is not None:
                zip_reader.close()

    def iter_json_entries_as_dataframe(self, batch_size: int = 1000) -> Iterator[pl.DataFrame]:
        """Iterate over JSON container entries as DataFrames with parsed structs.

        This is an optimized alternative to iter_entries() for JSON-heavy containers.
        Instead of parsing JSON individually, it batches entries and returns DataFrames
        with a 'data' struct column containing the parsed JSON.

        Args:
            batch_size: Number of entries per DataFrame batch.

        Yields:
            DataFrames with columns: entry_name (str), data (struct)

        Raises:
            ValueError: If this is not a container bundle.
        """
        if self.parsed_kind != "container":
            raise ValueError(
                "iter_json_entries_as_dataframe is only available for container bundles"
            )
        if not self.parsed_parquet_path:
            raise ValueError("Parsed parquet not available for container bundle")

        from .handlers import get_handler
        from .handlers.zstd import ZstdHandler
        from .utils import read_raw_bytes

        table = pq.ParquetFile(self.parsed_parquet_path)
        zip_bytes: bytes | None = None
        zip_reader: zipfile.ZipFile | None = None

        def _get_zip_reader() -> zipfile.ZipFile | None:
            nonlocal zip_bytes, zip_reader
            if zip_reader is not None:
                return zip_reader
            raw = read_raw_bytes(self.raw_parquet_path)
            if self.ext in {".zst", ".zstd"} and self.metadata.get("inner_ext") == ".zip":
                zip_bytes = ZstdHandler().load_bytes(raw)
            elif self.ext == ".zip":
                zip_bytes = raw
            else:
                return None

            if zip_bytes is None:
                return None

            zip_reader = zipfile.ZipFile(io.BytesIO(zip_bytes), "r")
            return zip_reader

        decoder = msgspec.json.Decoder()
        batch_names: list[str] = []
        batch_data: list[dict] = []

        def _flush_batch() -> pl.DataFrame | None:
            nonlocal batch_names, batch_data
            if not batch_names:
                return None
            try:
                df = pl.DataFrame({"entry_name": batch_names, "data": batch_data})
                batch_names = []
                batch_data = []
                return df
            except Exception as exc:
                logger.debug(f"Failed to create DataFrame from JSON batch: {exc}")
                batch_names = []
                batch_data = []
                return None

        try:
            for batch in table.iter_batches():
                names = batch.column(batch.schema.get_field_index("entry_name")).to_pylist()
                exts = batch.column(batch.schema.get_field_index("entry_ext")).to_pylist()
                kinds = batch.column(batch.schema.get_field_index("payload_kind")).to_pylist()
                msgpacks = batch.column(batch.schema.get_field_index("payload_msgpack")).to_pylist()
                bytes_list = batch.column(batch.schema.get_field_index("payload_bytes")).to_pylist()

                for name, ext, kind, msgp, raw_bytes in zip(
                    names, exts, kinds, msgpacks, bytes_list
                ):
                    parsed_obj = None

                    if kind == "json":
                        if msgp:
                            try:
                                parsed_obj = msgspec.msgpack.decode(msgp)
                            except Exception:
                                pass
                        elif raw_bytes is not None:
                            try:
                                parsed_obj = decoder.decode(bytes(raw_bytes))
                            except Exception:
                                pass

                    if parsed_obj is None and raw_bytes is None:
                        zf = _get_zip_reader()
                        if zf is not None:
                            try:
                                raw_bytes = zf.read(name)
                                if ext in {".json", ".yaml", ".yml"}:
                                    handler = get_handler(ext)
                                    if handler:
                                        parsed_obj = handler.load_bytes(bytes(raw_bytes))
                            except Exception:
                                pass

                    if isinstance(parsed_obj, dict):
                        batch_names.append(name)
                        batch_data.append(parsed_obj)

                        if len(batch_names) >= batch_size:
                            df = _flush_batch()
                            if df is not None:
                                yield df

            df = _flush_batch()
            if df is not None:
                yield df

        finally:
            if zip_reader is not None:
                zip_reader.close()
