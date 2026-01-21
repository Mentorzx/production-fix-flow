"""YAML file handler using ruamel.yaml for round-trip preservation."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import aiofile
import ruamel.yaml

from ..async_io import async_ensure_dir, write_async_text
from ..utils import ensure_dir
from .base import FileHandler


class YAMLHandler(FileHandler):
    """Estado da arte YAML handler with round-trip preservation.

    Features:
    - Preserves comments, formatting, and key order
    - Thread-safe via instance-level YAML objects
    - Custom tag support
    """

    def __init__(self) -> None:
        """Initialize YAML handler."""
                                                                               
                                                  
                                                                                                 
        pass

    def _get_yaml(self, custom_tags: dict | None = None) -> ruamel.yaml.YAML:
        yaml = ruamel.yaml.YAML(typ="rt")
        yaml.preserve_quotes = True
        yaml.indent(mapping=2, sequence=4, offset=2)
        yaml.allow_duplicate_keys = False

        if custom_tags:
            for tag, constructor in custom_tags.items():
                yaml.constructor.add_constructor(tag, constructor)
        return yaml

    def read(self, path: Path | io.BytesIO, custom_tags: dict | None = None, **kwargs: Any) -> Any:
        """Deserialize YAML content with optional custom tag support.

        Args:
            path: YAML file path or in-memory buffer.
            custom_tags: Optional dict mapping tag strings to constructor functions.
            **kwargs: Reserved for future options.

        Returns:
            Parsed YAML data.
        """
        yaml = self._get_yaml(custom_tags)

        if isinstance(path, io.BytesIO):
            return yaml.load(path.read().decode("utf-8"))
        with path.open("r", encoding="utf-8") as f:
            return yaml.load(f)

    def save(self, obj: Any, path: Path, **kwargs: Any) -> None:
        """Serialize object to YAML file, creating dirs if needed."""
        ensure_dir(path)
        yaml = self._get_yaml()
        with path.open("w", encoding="utf-8") as f:
            yaml.dump(obj, f)

    async def async_read(self, path: Path, custom_tags: dict | None = None, **kwargs: Any) -> Any:
        """Asynchronously deserialize YAML content using real async I/O."""
        async with aiofile.async_open(path, "r", encoding="utf-8") as f:
            content = await f.read()

        yaml = self._get_yaml(custom_tags)
        return yaml.load(content)

    async def async_save(self, obj: Any, path: Path, **kwargs: Any) -> None:
        """Asynchronously serialize object to YAML using real async I/O.

        Raises:
            ValueError: If the object cannot be serialized to YAML.
        """
        await async_ensure_dir(path)
        yaml = self._get_yaml()

        buffer = io.StringIO()
        try:
            yaml.dump(obj, buffer)
            await write_async_text(path, buffer.getvalue(), encoding="utf-8")
        except (ruamel.yaml.YAMLError, TypeError) as exc:
            raise ValueError(f"YAML serialization failed for {path}; object not YAML-safe") from exc

    def load_bytes(self, raw: bytes, **kwargs: Any) -> Any:
        """Load YAML from raw bytes."""
        yaml = self._get_yaml()
        return yaml.load(raw.decode("utf-8"))
