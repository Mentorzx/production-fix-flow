"""YAML file handler using ruamel.yaml for round-trip preservation."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import ruamel.yaml
import aiofile

from .base import FileHandler
from ..utils import ensure_dir
from ..async_io import async_ensure_dir, write_async_text


class YAMLHandler(FileHandler):
    """Estado da arte YAML handler with round-trip preservation.

    Features:
    - Preserves comments, formatting, and key order
    - Thread-safe via instance-level YAML objects
    - Custom tag support
    """

    def __init__(self) -> None:
        """Initialize with a reusable ruamel YAML instance."""
        self._yaml = ruamel.yaml.YAML(typ="rt")
        self._yaml.preserve_quotes = True
        self._yaml.indent(mapping=2, sequence=4, offset=2)
        self._yaml.allow_duplicate_keys = False

    def read(
        self, path: Path | io.BytesIO, custom_tags: dict | None = None, **kwargs: Any
    ) -> Any:
        """Deserialize YAML content with optional custom tag support.

        Args:
            path: YAML file path or in-memory buffer.
            custom_tags: Optional dict mapping tag strings to constructor functions.
            **kwargs: Reserved for future options.

        Returns:
            Parsed YAML data.
        """
        if custom_tags:
            yaml_instance = ruamel.yaml.YAML(typ="rt")
            yaml_instance.preserve_quotes = True
            yaml_instance.indent(mapping=2, sequence=4, offset=2)
            yaml_instance.allow_duplicate_keys = False
            for tag, constructor in custom_tags.items():
                yaml_instance.constructor.add_constructor(tag, constructor)

            if isinstance(path, io.BytesIO):
                return yaml_instance.load(path.read().decode("utf-8"))
            with path.open("r", encoding="utf-8") as f:
                return yaml_instance.load(f)

        if isinstance(path, io.BytesIO):
            return self._yaml.load(path.read().decode("utf-8"))
        with path.open("r", encoding="utf-8") as f:
            return self._yaml.load(f)

    def save(self, obj: Any, path: Path, **kwargs: Any) -> None:
        """Serialize object to YAML file, creating dirs if needed."""
        ensure_dir(path)
        with path.open("w", encoding="utf-8") as f:
            self._yaml.dump(obj, f)

    async def async_read(
        self, path: Path, custom_tags: dict | None = None, **kwargs: Any
    ) -> Any:
        """Asynchronously deserialize YAML content using real async I/O."""
        async with aiofile.async_open(path, "r", encoding="utf-8") as f:
            content = await f.read()

        if custom_tags:
            yaml_instance = ruamel.yaml.YAML(typ="rt")
            yaml_instance.preserve_quotes = True
            yaml_instance.indent(mapping=2, sequence=4, offset=2)
            yaml_instance.allow_duplicate_keys = False
            for tag, constructor in custom_tags.items():
                yaml_instance.constructor.add_constructor(tag, constructor)
            return yaml_instance.load(content)
        return self._yaml.load(content)

    async def async_save(self, obj: Any, path: Path, **kwargs: Any) -> None:
        """Asynchronously serialize object to YAML using real async I/O.

        Raises:
            ValueError: If the object cannot be serialized to YAML.
        """
        await async_ensure_dir(path)

        buffer = io.StringIO()
        try:
            self._yaml.dump(obj, buffer)
            await write_async_text(path, buffer.getvalue(), encoding="utf-8")
        except (ruamel.yaml.YAMLError, TypeError) as exc:
            raise ValueError(
                f"YAML serialization failed for {path}; object not YAML-safe"
            ) from exc

    def load_bytes(self, raw: bytes, **kwargs: Any) -> Any:
        """Load YAML from raw bytes."""
        return self._yaml.load(raw.decode("utf-8"))
