"""Cache serialization with parquet-first routing."""

from __future__ import annotations

import importlib
import os
import pickle
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

try:
    import msgspec

    MSGSPEC_AVAILABLE = True
except Exception:
    msgspec = None  # type: ignore[assignment]
    MSGSPEC_AVAILABLE = False


class CacheSerializer:
    """Parquet-first serializer with msgspec payloads and isolated pickle fallback."""

    _MAGIC_MSGSPEC = b"MSP1"
    _MAGIC_PICKLE = b"PKL1"

    def _encode_wrapper(self, payload: dict[str, Any]) -> bytes:
        if MSGSPEC_AVAILABLE and msgspec is not None:
            return self._MAGIC_MSGSPEC + msgspec.msgpack.encode(payload)
        payload_bytes = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        return self._MAGIC_PICKLE + payload_bytes

    def _decode_wrapper(self, data: bytes) -> dict[str, Any] | None:
        if data.startswith(self._MAGIC_MSGSPEC) and msgspec is not None:
            decoded = msgspec.msgpack.decode(data[len(self._MAGIC_MSGSPEC) :])
            return decoded if isinstance(decoded, dict) else None
        if data.startswith(self._MAGIC_PICKLE):
            decoded = pickle.loads(data[len(self._MAGIC_PICKLE) :])
            return decoded if isinstance(decoded, dict) else None
        return None

    def serialize(
        self,
        obj: Any,
        *,
        cache_root: Path | None = None,
        cache_key: str | None = None,
    ) -> bytes:
        """Serialize an object with parquet-first routing."""
        try:
            from pff.shared.core.file_manager import ParquetBundle
        except Exception:
            ParquetBundle = None  # type: ignore[misc,assignment]

        obj_type = type(obj)

        if obj_type in (dict, list, str, int, float, bool) or obj is None:
            payload = {"_cache_kind": "msgpack", "value": obj}
            return self._encode_wrapper(payload)

        if obj_type in (bytes, bytearray, memoryview):
            payload = {"_cache_kind": "bytes", "value": bytes(obj)}
            return self._encode_wrapper(payload)

        if ParquetBundle is not None and isinstance(obj, ParquetBundle):
            payload = {
                "_cache_kind": "bundle_ref",
                "source_path": str(obj.source_path),
                "ext": obj.ext,
                "file_id": obj.file_id,
                "raw_parquet_path": str(obj.raw_parquet_path),
                "parsed_parquet_path": (
                    str(obj.parsed_parquet_path) if obj.parsed_parquet_path else None
                ),
                "parsed_kind": obj.parsed_kind,
                "metadata": obj.metadata,
                "dirty": obj.dirty,
            }
            return self._encode_wrapper(payload)

        if os.environ.get("PFF_CLEAN_MODE") == "1":
            payload_bytes = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
            digest = sha256(payload_bytes).hexdigest()
            payload = {
                "_cache_kind": "pickle",
                "sha256": digest,
                "payload": payload_bytes,
            }
            return self._encode_wrapper(payload)

        if isinstance(obj, pl.LazyFrame):
            if cache_root is None or cache_key is None:
                raise ValueError("LazyFrame cache requires cache_root and cache_key")
            else:
                parquet_path = cache_root / f"{cache_key}.parquet"
                obj_any: Any = obj
                obj_any = cast(Any, obj_any)
                obj_any.sink_parquet(
                    parquet_path,
                    compression="lz4",
                    row_group_size=100000,
                )
                payload = {
                    "_cache_kind": "parquet_ref",
                    "table_kind": "polars_lazy",
                    "path": str(parquet_path),
                }
                return self._encode_wrapper(payload)

        if isinstance(obj, pl.DataFrame):
            if cache_root is None or cache_key is None:
                raise ValueError("DataFrame cache requires cache_root and cache_key")
            else:
                parquet_path = cache_root / f"{cache_key}.parquet"
                obj_any = cast(Any, obj)
                obj_any.write_parquet(
                    parquet_path,
                    compression="lz4",
                    statistics=True,
                    row_group_size=100000,
                )
                payload = {
                    "_cache_kind": "parquet_ref",
                    "table_kind": "polars",
                    "path": str(parquet_path),
                }
                return self._encode_wrapper(payload)

        if isinstance(obj, pa.Table):
            if cache_root is None or cache_key is None:
                raise ValueError("Arrow Table cache requires cache_root and cache_key")
            else:
                parquet_path = cache_root / f"{cache_key}.parquet"
                pq_any = cast(Any, pq)
                pq_any.write_table(parquet_path, obj)
                payload = {
                    "_cache_kind": "parquet_ref",
                    "table_kind": "arrow",
                    "path": str(parquet_path),
                }
                return self._encode_wrapper(payload)

        if (
            not isinstance(obj, pl.LazyFrame)
            and hasattr(obj, "to_dict")
            and callable(getattr(obj, "to_dict"))
        ):
            obj_with_dict: Any = obj
            payload = {
                "_cache_kind": "object",
                "class_path": f"{obj.__class__.__module__}.{obj.__class__.__qualname__}",
                "data": obj_with_dict.to_dict(),
            }
            return self._encode_wrapper(payload)

        payload_bytes = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        digest = sha256(payload_bytes).hexdigest()
        payload = {
            "_cache_kind": "pickle",
            "sha256": digest,
            "payload": payload_bytes,
        }
        return self._encode_wrapper(payload)

    def deserialize(self, data: bytes, *, cache_root: Path | None = None) -> Any:
        """Deserialize bytes into cached objects."""
        try:
            from pff.shared.core.file_manager import ParquetBundle
        except Exception:
            ParquetBundle = None  # type: ignore[misc,assignment]

        wrapper = self._decode_wrapper(data)
        if wrapper and "_cache_kind" in wrapper:
            kind = wrapper.get("_cache_kind")
            if kind == "bundle_ref" and ParquetBundle is not None:
                parsed_path = wrapper.get("parsed_parquet_path")
                return ParquetBundle(
                    source_path=Path(wrapper.get("source_path", "")),
                    ext=wrapper.get("ext", ""),
                    file_id=wrapper.get("file_id", ""),
                    raw_parquet_path=Path(wrapper.get("raw_parquet_path", "")),
                    parsed_parquet_path=Path(parsed_path) if parsed_path else None,
                    parsed_kind=wrapper.get("parsed_kind", "none"),
                    metadata=wrapper.get("metadata", {}),
                    dirty=bool(wrapper.get("dirty", False)),
                )
            if kind == "parquet_ref":
                if os.environ.get("PFF_CLEAN_MODE") == "1":
                    return None
                path_str = wrapper.get("path")
                if not path_str:
                    return None
                path = Path(path_str)
                if not path.is_absolute() and cache_root is not None:
                    path = cache_root / path
                table_kind = wrapper.get("table_kind")
                if table_kind == "polars_lazy":
                    pl_any = cast(Any, pl)
                    return pl_any.scan_parquet(path)
                if table_kind == "arrow":
                    pq_any = cast(Any, pq)
                    return pq_any.read_table(path)
                pl_any = cast(Any, pl)
                return pl_any.read_parquet(path)
            if kind == "bytes":
                return wrapper.get("value", b"")
            if kind == "msgpack":
                return wrapper.get("value")
            if kind == "object":
                class_path = wrapper.get("class_path")
                data_payload = wrapper.get("data")
                if class_path:
                    try:
                        module_name, _, cls_name = class_path.rpartition(".")
                        module = importlib.import_module(module_name)
                        cls = getattr(module, cls_name, None)
                        if cls and hasattr(cls, "from_dict"):
                            return cls.from_dict(data_payload)
                    except Exception:
                        return data_payload
                return data_payload
            if kind == "pickle":
                payload = wrapper.get("payload", b"")
                digest = wrapper.get("sha256")
                if digest and sha256(payload).hexdigest() != digest:
                    raise ValueError("Cache pickle payload hash mismatch")
                return pickle.loads(payload)

        if data.startswith(self._MAGIC_PICKLE):
            return pickle.loads(data[len(self._MAGIC_PICKLE) :])
        if MSGSPEC_AVAILABLE and msgspec is not None:
            try:
                return msgspec.msgpack.decode(data)
            except Exception:
                pass
        return pickle.loads(data)
