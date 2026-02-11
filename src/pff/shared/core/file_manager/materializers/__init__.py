"""Materializer registry and factory function.

Maps parsed_kind values to materializer implementations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .base import Materializer
from .implementations import (
    BytesMaterializer,
    ContainerMaterializer,
    JsonYamlMaterializer,
    TabularMaterializer,
    TextMaterializer,
    ZipMaterializer,
    ZstdMaterializer,
)

if TYPE_CHECKING:
    from ..bundles import ParquetBundle


_MATERIALIZERS: dict[str, Materializer] = {
    "tabular": TabularMaterializer(),
    "json": JsonYamlMaterializer(),
    "yaml": JsonYamlMaterializer(),
    "text": TextMaterializer(),
    "bytes": BytesMaterializer(),
    "container": ContainerMaterializer(),
}


def get_materializer(parsed_kind: str) -> Materializer | None:
    """Get materializer for the given parsed_kind.

    Args:
        parsed_kind: The bundle's parsed_kind value.

    Returns:
        Materializer instance or None if not found.
    """
    return _MATERIALIZERS.get(parsed_kind)


def register_materializer(parsed_kind: str, materializer: Materializer) -> None:
    """Register a custom materializer for a parsed_kind.

    Args:
        parsed_kind: The parsed_kind to register.
        materializer: The materializer instance.
    """
    _MATERIALIZERS[parsed_kind] = materializer


def materialize_bundle(bundle: ParquetBundle, **kwargs: Any) -> Any:
    """Convert a ParquetBundle to its native Python representation.

    Uses the Strategy pattern to dispatch to the appropriate materializer
    based on the bundle's parsed_kind.

    Args:
        bundle: ParquetBundle to materialize.
        **kwargs: Additional options passed to the handler/materializer.

    Returns:
        Native Python object.
    """

    if "mmap_mode" in kwargs and bundle.source_path.exists():
        from ..handlers import get_handler

        handler = get_handler(bundle.ext)
        if handler:
            return handler.read(bundle.source_path, **kwargs)

    materializer = get_materializer(bundle.parsed_kind)
    if materializer:
        return materializer.materialize(bundle)

    if bundle.ext in {".zip"}:
        return ZipMaterializer().materialize(bundle)

    if bundle.ext in {".zst", ".zstd"}:
        return ZstdMaterializer().materialize(bundle)

    from ..handlers import get_handler
    from ..utils import read_raw_bytes

    raw = read_raw_bytes(bundle.raw_parquet_path)
    handler = get_handler(bundle.ext)
    if handler:
        return handler.load_bytes(raw, **kwargs)
    return raw


__all__ = [
    "Materializer",
    "TabularMaterializer",
    "JsonYamlMaterializer",
    "TextMaterializer",
    "BytesMaterializer",
    "ContainerMaterializer",
    "ZipMaterializer",
    "ZstdMaterializer",
    "get_materializer",
    "register_materializer",
    "materialize_bundle",
]
