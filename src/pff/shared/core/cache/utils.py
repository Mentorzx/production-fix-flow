"""Cache module utilities and helpers."""

from __future__ import annotations

import functools
import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from hashlib import blake2b
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, ParamSpec, TypeVar, cast

import orjson
from filelock import FileLock

from .constants import (
    ATOMIC_WRITE_RETRY_COUNT,
    ATOMIC_WRITE_RETRY_DELAY,
    DEFAULT_TEMPLATE_TTL_DAYS,
)

P = ParamSpec("P")
R = TypeVar("R")


class JsonSafeEncoder:
    """Ensures objects can be safely JSON-encoded for cache key generation."""

    @staticmethod
    def make_json_safe(obj: Any) -> Any:
        """
        Convert an object to a JSON-safe representation.

        Args:
            obj: Any object to make JSON-safe

        Returns:
            A JSON-serializable version of the object
        """
        try:
            orjson.dumps(obj)
            return obj
        except (TypeError, OverflowError):
            return repr(obj)


class FunctionCallHasher:
    """Generates unique hashes for function calls."""

    @staticmethod
    def hash_function_call(
        function: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> str:
        """
        Generate a unique hash for a function call with its arguments.

        Args:
            function: The function being called
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            A hexadecimal hash string
        """
        encoder = JsonSafeEncoder()
        make_safe = encoder.make_json_safe

        payload = {
            "fn": f"{function.__module__}.{function.__qualname__}",
            "args": [make_safe(arg) for arg in args],
            "kwargs": {key: make_safe(value) for key, value in kwargs.items()},
        }

        serialized = orjson.dumps(payload, option=orjson.OPT_NON_STR_KEYS)

        return blake2b(serialized, digest_size=16).hexdigest()


class AtomicFileWriter:
    """Provides atomic file write operations."""

    @staticmethod
    def write_atomically(path: Path, data: bytes) -> None:
        """
        Write data to a file atomically to prevent partial writes.

        Ensures data integrity by writing to a temporary file first,
        then atomically replacing the target file.

        Args:
            path: Target file path
            data: Binary data to write

        Raises:
            OSError: If the write operation fails
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        with NamedTemporaryFile(dir=path.parent, delete=False) as temp_file:
            temp_file.write(data)
            temp_file.flush()
            os.fsync(temp_file.fileno())
            temp_path = Path(temp_file.name)

        lock_path = f"{path}.lock"
        with FileLock(lock_path):
            for attempt in range(ATOMIC_WRITE_RETRY_COUNT):
                try:
                    temp_path.replace(path)
                    return
                except PermissionError:
                    if attempt < ATOMIC_WRITE_RETRY_COUNT - 1:
                        time.sleep(ATOMIC_WRITE_RETRY_DELAY)

            temp_path.replace(path)


class TemplatePatternNormalizer:
    """Handles URL normalization and template extraction."""

    UUID_PATTERN = re.compile(
        r"[\da-fA-F]{8}-[\da-fA-F]{4}-[\da-fA-F]{4}-[\da-fA-F]{4}-[\da-fA-F]{12}"
    )
    HEX_ID_PATTERN = re.compile(r"[a-fA-F0-9]{16,}")
    MSISDN_PATTERN = re.compile(r"55\d{11,13}")
    LONG_NUMBER_PATTERN = re.compile(r"/\d{6,}/")

    def normalize_url(self, url: str) -> str:
        """
        Normalize a URL by replacing dynamic values with placeholders.

        Args:
            url: The URL to normalize

        Returns:
            Normalized URL with placeholders
        """
        normalized = url

        normalized = self.MSISDN_PATTERN.sub("55{msisdn}", normalized)
        normalized = re.sub(
            r"communicationId=55\d{11,13}", "communicationId=55{msisdn}", normalized
        )

        normalized = re.sub(r"=[\w\.\-\+]+", "={value}", normalized)

        normalized = self.UUID_PATTERN.sub("/{uuid}", normalized)
        normalized = self.LONG_NUMBER_PATTERN.sub("/{number}/", normalized)
        normalized = self.HEX_ID_PATTERN.sub("{hex_id}", normalized)

        return normalized

    def extract_template(self, url: str, known_values: dict[str, str]) -> str:
        """
        Extract a template from a URL by replacing known values.

        Args:
            url: The URL to process
            known_values: Known variable values to replace

        Returns:
            Template string with placeholders
        """
        template = url

        for variable_name, value in known_values.items():
            if value:
                template = template.replace(value, f"{{{variable_name}}}")

        template = self.MSISDN_PATTERN.sub("55{msisdn}", template)
        template = self.UUID_PATTERN.sub("{uuid}", template)
        template = self.HEX_ID_PATTERN.sub("{hex_id}", template)
        template = re.sub(r"/\d{6,}/", "/{id}/", template)

        return template


@dataclass
class CacheEntry:
    """Base class for cache entries with expiration support."""

    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    expires_at: float | None = None

    def is_expired(self) -> bool:
        """Check if this entry has expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    def touch(self) -> None:
        """Update the last access time."""
        self.last_accessed = time.time()


@dataclass(kw_only=True)
class HttpTemplateEntry(CacheEntry):
    """Cache entry for HTTP request templates."""

    template: str
    endpoint_type: str
    method: str = "GET"
    headers: dict[str, str] = field(default_factory=dict)
    success_count: int = 0
    variables: list[str] = field(default_factory=list)

    def __post_init__(self):
        """Initialize computed fields after dataclass initialization."""
        super().__init__()

        if not self.variables:
            self.variables = self._extract_template_variables()

        if self.expires_at is None:
            self.expires_at = self.created_at + (DEFAULT_TEMPLATE_TTL_DAYS * 24 * 3600)

    def _extract_template_variables(self) -> list[str]:
        """Extract variable names from the template string."""
        pattern = r"\{(\w+)\}"
        return re.findall(pattern, self.template)


def create_memory_cache(maxsize: int = 128):
    """
    Create an in-memory LRU cache decorator.

    Args:
        maxsize: Maximum number of items to cache

    Returns:
        Decorator function for caching
    """

    def decorator(function: Callable[P, R]) -> Callable[P, R]:
        return cast(Callable[P, R], functools.lru_cache(maxsize=maxsize)(function))

    return decorator
