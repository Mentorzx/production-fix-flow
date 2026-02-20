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
        """Execute encode wrapper.



        Args:

            payload: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        if MSGSPEC_AVAILABLE and msgspec is not None:
            return self._MAGIC_MSGSPEC + msgspec.msgpack.encode(payload)
        payload_bytes = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        return self._MAGIC_PICKLE + payload_bytes

    def _decode_wrapper(self, data: bytes) -> dict[str, Any] | None:
        """Execute decode wrapper.



        Args:

            data: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

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

        primitive_payload = self._serialize_primitive(obj, obj_type=obj_type)
        if primitive_payload is not None:
            return self._encode_wrapper(primitive_payload)

        bytes_payload = self._serialize_bytes(obj, obj_type=obj_type)
        if bytes_payload is not None:
            return self._encode_wrapper(bytes_payload)

        if ParquetBundle is not None and isinstance(obj, ParquetBundle):
            return self._encode_wrapper(self._serialize_bundle_ref(obj))

        if os.environ.get("PFF_CLEAN_MODE") == "1":
            return self._encode_wrapper(self._serialize_pickle_payload(obj))

        if isinstance(obj, pl.LazyFrame):
            parquet_path = self._build_cache_parquet_path(
                cache_root=cache_root,
                cache_key=cache_key,
                error_message="LazyFrame cache requires cache_root and cache_key",
            )
            obj_any = cast(Any, obj)
            obj_any.sink_parquet(
                parquet_path,
                compression="lz4",
                row_group_size=100000,
            )
            return self._encode_wrapper(self._serialize_parquet_ref("polars_lazy", parquet_path))

        if isinstance(obj, pl.DataFrame):
            parquet_path = self._build_cache_parquet_path(
                cache_root=cache_root,
                cache_key=cache_key,
                error_message="DataFrame cache requires cache_root and cache_key",
            )
            obj_any = cast(Any, obj)
            obj_any.write_parquet(
                parquet_path,
                compression="lz4",
                statistics=True,
                row_group_size=100000,
            )
            return self._encode_wrapper(self._serialize_parquet_ref("polars", parquet_path))

        if isinstance(obj, pa.Table):
            parquet_path = self._build_cache_parquet_path(
                cache_root=cache_root,
                cache_key=cache_key,
                error_message="Arrow Table cache requires cache_root and cache_key",
            )
            pq_any = cast(Any, pq)
            pq_any.write_table(parquet_path, obj)
            return self._encode_wrapper(self._serialize_parquet_ref("arrow", parquet_path))

        if (
            not isinstance(obj, pl.LazyFrame)
            and hasattr(obj, "to_dict")
            and callable(getattr(obj, "to_dict"))
        ):
            return self._encode_wrapper(self._serialize_object_with_dict(obj))

        return self._encode_wrapper(self._serialize_pickle_payload(obj))

    @staticmethod
    def _serialize_primitive(obj: Any, *, obj_type: type) -> dict[str, Any] | None:
        """Execute serialize primitive.



        Args:

            obj: Input value used by this callable.

            obj_type: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        if obj_type in (dict, list, str, int, float, bool) or obj is None:
            return {"_cache_kind": "msgpack", "value": obj}
        return None

    @staticmethod
    def _serialize_bytes(obj: Any, *, obj_type: type) -> dict[str, Any] | None:
        """Execute serialize bytes.



        Args:

            obj: Input value used by this callable.

            obj_type: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        if obj_type in (bytes, bytearray, memoryview):
            return {"_cache_kind": "bytes", "value": bytes(obj)}
        return None

    @staticmethod
    def _serialize_bundle_ref(obj: Any) -> dict[str, Any]:
        return {
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

    @staticmethod
    def _serialize_pickle_payload(obj: Any) -> dict[str, Any]:
        """Execute serialize pickle payload.



        Args:

            obj: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        payload_bytes = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        digest = sha256(payload_bytes).hexdigest()
        return {
            "_cache_kind": "pickle",
            "sha256": digest,
            "payload": payload_bytes,
        }

    @staticmethod
    def _build_cache_parquet_path(
        *,
        cache_root: Path | None,
        cache_key: str | None,
        error_message: str,
    ) -> Path:
        """Execute build cache parquet path.



        Args:

            cache_root: Input value used by this callable.

            cache_key: Input value used by this callable.

            error_message: Input value used by this callable.



        Returns:

            Return value produced by the callable.



        Raises:

            Exception: Propagates domain-specific failures with context.

        """

        if cache_root is None or cache_key is None:
            raise ValueError(error_message)
        return cache_root / f"{cache_key}.parquet"

    @staticmethod
    def _serialize_parquet_ref(table_kind: str, parquet_path: Path) -> dict[str, str]:
        return {
            "_cache_kind": "parquet_ref",
            "table_kind": table_kind,
            "path": str(parquet_path),
        }

    @staticmethod
    def _serialize_object_with_dict(obj: Any) -> dict[str, Any]:
        return {
            "_cache_kind": "object",
            "class_path": f"{obj.__class__.__module__}.{obj.__class__.__qualname__}",
            "data": obj.to_dict(),
        }

    def deserialize(self, data: bytes, *, cache_root: Path | None = None) -> Any:
        """Deserialize bytes into cached objects."""
        bundle_cls = self._resolve_bundle_cls()
        wrapper = self._decode_wrapper(data)
        if wrapper and "_cache_kind" in wrapper:
            return self._deserialize_wrapper(
                wrapper,
                cache_root=cache_root,
                bundle_cls=bundle_cls,
            )
        return self._deserialize_without_wrapper(data)

    @staticmethod
    def _resolve_bundle_cls() -> Any | None:
        """Execute resolve bundle cls.



        Returns:

            Return value produced by the callable.

        """

        try:
            from pff.shared.core.file_manager import ParquetBundle

            return ParquetBundle
        except Exception:
            return None

    def _deserialize_wrapper(
        self,
        wrapper: dict[str, Any],
        *,
        cache_root: Path | None,
        bundle_cls: Any | None,
    ) -> Any:
        """Execute deserialize wrapper.



        Args:

            wrapper: Input value used by this callable.

            cache_root: Input value used by this callable.

            bundle_cls: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        kind = wrapper.get("_cache_kind")
        if kind == "bundle_ref":
            return self._deserialize_bundle_ref(wrapper, bundle_cls=bundle_cls)
        if kind == "parquet_ref":
            return self._deserialize_parquet_ref(wrapper, cache_root=cache_root)
        if kind == "bytes":
            return wrapper.get("value", b"")
        if kind == "msgpack":
            return wrapper.get("value")
        if kind == "object":
            return self._deserialize_object_payload(wrapper)
        if kind == "pickle":
            return self._deserialize_pickle_payload(wrapper)
        return None

    @staticmethod
    def _deserialize_bundle_ref(wrapper: dict[str, Any], *, bundle_cls: Any | None) -> Any:
        """Execute deserialize bundle ref.



        Args:

            wrapper: Input value used by this callable.

            bundle_cls: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        if bundle_cls is None:
            return None
        parsed_path = wrapper.get("parsed_parquet_path")
        return bundle_cls(
            source_path=Path(wrapper.get("source_path", "")),
            ext=wrapper.get("ext", ""),
            file_id=wrapper.get("file_id", ""),
            raw_parquet_path=Path(wrapper.get("raw_parquet_path", "")),
            parsed_parquet_path=Path(parsed_path) if parsed_path else None,
            parsed_kind=wrapper.get("parsed_kind", "none"),
            metadata=wrapper.get("metadata", {}),
            dirty=bool(wrapper.get("dirty", False)),
        )

    @staticmethod
    def _deserialize_parquet_ref(wrapper: dict[str, Any], *, cache_root: Path | None) -> Any:
        """Execute deserialize parquet ref.



        Args:

            wrapper: Input value used by this callable.

            cache_root: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

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

    @staticmethod
    def _deserialize_object_payload(wrapper: dict[str, Any]) -> Any:
        """Execute deserialize object payload.



        Args:

            wrapper: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

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

    @staticmethod
    def _deserialize_pickle_payload(wrapper: dict[str, Any]) -> Any:
        """Execute deserialize pickle payload.



        Args:

            wrapper: Input value used by this callable.



        Returns:

            Return value produced by the callable.



        Raises:

            Exception: Propagates domain-specific failures with context.

        """

        payload = wrapper.get("payload", b"")
        digest = wrapper.get("sha256")
        if digest and sha256(payload).hexdigest() != digest:
            raise ValueError("Cache pickle payload hash mismatch")
        return pickle.loads(payload)

    def _deserialize_without_wrapper(self, data: bytes) -> Any:
        """Execute deserialize without wrapper.



        Args:

            data: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        if data.startswith(self._MAGIC_PICKLE):
            return pickle.loads(data[len(self._MAGIC_PICKLE) :])
        if MSGSPEC_AVAILABLE and msgspec is not None:
            try:
                return msgspec.msgpack.decode(data)
            except Exception:
                pass
        return pickle.loads(data)
