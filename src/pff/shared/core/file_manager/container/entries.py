"""Container entry iteration helpers for parquet-backed container bundles."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Any

import msgspec
import polars as pl
import pyarrow.parquet as pq

from ...logging import logger
from ..handlers import get_handler
from ..handlers.zstd import ZstdHandler
from ..utils import read_raw_bytes

_SKIP: object = object()


def iter_container_entries(
    *,
    parsed_parquet_path: Path,
    raw_parquet_path: Path,
    source_ext: str,
    metadata: dict[str, Any],
) -> Any:
    """Execute iter container entries.



    Args:

        parsed_parquet_path: Input value used by this callable.

        raw_parquet_path: Input value used by this callable.

        source_ext: Input value used by this callable.

        metadata: Input value used by this callable.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    table = pq.ParquetFile(parsed_parquet_path)
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
    resolver = _ContainerPayloadResolver(
        raw_parquet_path=raw_parquet_path,
        source_ext=source_ext,
        metadata=metadata,
    )
    try:
        for batch in table.iter_batches():
            rows = _iter_batch_rows(batch=batch, field_indices=field_indices)
            for row in rows:
                payload = resolver.resolve_payload(row)
                if payload is _SKIP:
                    continue
                yield row["entry_name"], payload
    finally:
        resolver.close()


class _ContainerPayloadResolver:
    def __init__(
        self,
        *,
        raw_parquet_path: Path,
        source_ext: str,
        metadata: dict[str, Any],
    ) -> None:
        """Execute init.



        Args:

            raw_parquet_path: Input value used by this callable.

            source_ext: Input value used by this callable.

            metadata: Input value used by this callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        self._raw_parquet_path = raw_parquet_path
        self._source_ext = source_ext
        self._metadata = metadata
        self._handler_cache: dict[str, Any] = {}
        self._zip_bytes: bytes | None = None
        self._zip_reader: zipfile.ZipFile | None = None

    def resolve_payload(self, row: dict[str, Any]) -> Any:
        """Execute resolve payload.



        Args:

            row: Input value used by this callable.



        Returns:

            Return value produced by the callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        tabular_payload = self._resolve_tabular_payload(row)
        if tabular_payload is not _SKIP:
            return tabular_payload
        text_payload = self._resolve_text_payload(row)
        if text_payload is not _SKIP:
            return text_payload
        json_payload = self._resolve_json_payload(row)
        if json_payload is not _SKIP:
            return json_payload
        raw_bytes = self._resolve_raw_bytes(row)
        if raw_bytes is None:
            return _SKIP
        return self._decode_generic_payload(ext=row["entry_ext"], payload=raw_bytes)

    def close(self) -> None:
        """Execute close.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        if self._zip_reader is not None:
            self._zip_reader.close()
            self._zip_reader = None

    def _resolve_tabular_payload(self, row: dict[str, Any]) -> Any:
        """Execute resolve tabular payload.



        Args:

            row: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        if row["payload_kind"] == "tabular" and row["payload_parquet_path"]:
            return pl.read_parquet(row["payload_parquet_path"])
        return _SKIP

    def _resolve_text_payload(self, row: dict[str, Any]) -> Any:
        """Execute resolve text payload.



        Args:

            row: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        if row["payload_kind"] != "text":
            return _SKIP
        if row["payload_text"] is not None:
            return row["payload_text"]
        if row["payload_bytes"] is None:
            return _SKIP
        payload = bytes(row["payload_bytes"])
        handler = self._get_handler(row["entry_ext"] or ".txt")
        if handler is not None:
            try:
                return handler.load_bytes(payload)
            except Exception as exc:
                logger.debug(f"Handler failed for text entry {row['entry_name']}: {exc}")
        return payload

    def _resolve_json_payload(self, row: dict[str, Any]) -> Any:
        """Execute resolve json payload.



        Args:

            row: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        if row["payload_kind"] != "json":
            return _SKIP
        if row["payload_msgpack"]:
            return msgspec.msgpack.decode(row["payload_msgpack"])
        if row["payload_bytes"] is None:
            return _SKIP
        payload = bytes(row["payload_bytes"])
        handler = self._get_handler(row["entry_ext"] or ".json")
        if handler is not None:
            try:
                return handler.load_bytes(payload)
            except Exception as exc:
                logger.debug(f"Handler failed for json entry {row['entry_name']}: {exc}")
        return payload

    def _resolve_raw_bytes(self, row: dict[str, Any]) -> bytes | None:
        """Execute resolve raw bytes.



        Args:

            row: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        raw_bytes = row["payload_bytes"]
        if raw_bytes is not None:
            return bytes(raw_bytes)
        if row["payload_parquet_path"] is not None:
            return None
        if row["payload_text"] is not None:
            return None
        if row["payload_msgpack"] is not None:
            return None
        zip_reader = self._get_zip_reader()
        if zip_reader is None:
            return None
        try:
            return zip_reader.read(row["entry_name"])
        except Exception as exc:
            logger.debug(f"ZIP read failed for container entry {row['entry_name']}: {exc}")
            return None

    def _decode_generic_payload(self, *, ext: str | None, payload: bytes) -> Any:
        """Execute decode generic payload.



        Args:

            ext: Input value used by this callable.

            payload: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        handler = self._get_handler(ext or "")
        if handler is not None:
            try:
                return handler.load_bytes(payload)
            except Exception as exc:
                logger.debug(f"Handler failed for container entry ext={ext}: {exc}")
        return payload

    def _get_handler(self, ext: str) -> Any:
        """Execute get handler.



        Args:

            ext: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        if ext in self._handler_cache:
            return self._handler_cache[ext]
        handler = get_handler(ext)
        self._handler_cache[ext] = handler
        return handler

    def _get_zip_reader(self) -> zipfile.ZipFile | None:
        """Execute get zip reader.



        Returns:

            Return value produced by the callable.

        """

        if self._zip_reader is not None:
            return self._zip_reader
        raw = read_raw_bytes(self._raw_parquet_path)
        if self._source_ext in {".zst", ".zstd"} and self._metadata.get("inner_ext") == ".zip":
            self._zip_bytes = ZstdHandler().load_bytes(raw)
        elif self._source_ext == ".zip":
            self._zip_bytes = raw
        else:
            return None
        if self._zip_bytes is None:
            return None
        self._zip_reader = zipfile.ZipFile(io.BytesIO(self._zip_bytes), "r")
        return self._zip_reader


def _iter_batch_rows(batch: Any, field_indices: dict[str, int]) -> Any:
    """Execute iter batch rows.



    Args:

        batch: Input value used by this callable.

        field_indices: Input value used by this callable.

    """

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
        yield {
            "entry_name": name,
            "entry_ext": ext,
            "payload_kind": kind,
            "payload_msgpack": msgp,
            "payload_text": text,
            "payload_bytes": raw_bytes,
            "payload_parquet_path": parquet_path,
        }
