"""Binary file handlers: BinHandler, PickleHandler, NumPyHandler."""

from __future__ import annotations

import io
import os
import pickle
from pathlib import Path
from typing import Any

import joblib
import msgspec
import numpy as np

from ...logging import logger
from ..async_io import async_ensure_dir
from ..utils import encode_msgpack, ensure_dir, memory_map_file
from .base import FileHandler


class BinHandler(FileHandler):
    """Handler for binary model files (.bin, .pt).

    Attempts msgspec decoding first, falls back to joblib.
    """

    def read(self, path: Path | io.BytesIO, **kw: Any) -> Any:
        """Read binary file, trying msgspec first then joblib fallback."""
        if isinstance(path, io.BytesIO):
            try:
                raw = path.getvalue()
                return msgspec.msgpack.decode(raw)
            except Exception as exc:
                logger.debug(f"msgspec decode failed for buffer: {exc}")
                path.seek(0)
                return joblib.load(path)
        else:
            p = Path(path)
            try:
                with memory_map_file(p) as mm:
                    return msgspec.msgpack.decode(mm)
            except Exception as e:
                logger.debug(f"msgspec decode failed: {e!s}")

            return joblib.load(p)

    def save(self, obj: Any, path: Path, **kw: Any) -> None:
        """Save object to binary file using msgspec or joblib fallback."""
        ensure_dir(path)
        if isinstance(obj, (bytes, bytearray, memoryview)):
            path.write_bytes(bytes(obj))
            return

        try:
            encoded = encode_msgpack(obj)
            path.write_bytes(encoded)
        except (TypeError, msgspec.EncodeError):
            allow_pickle = (
                bool(kw.pop("allow_pickle", False))
                or os.getenv("PFF_FILE_MANAGER_ALLOW_PICKLE", "") == "1"
            )
            if not allow_pickle:
                raise ValueError("Object not MessagePack-safe and pickle fallback is disabled.")
            logger.warning("MessagePack failed; using pickle fallback due to allow_pickle")
            joblib.dump(obj, path, protocol=pickle.HIGHEST_PROTOCOL)

    async def async_read(self, path: Path, **kw: Any) -> Any:
        """Async read delegates to sync implementation."""
        return self.read(path, **kw)

    async def async_save(self, obj: Any, path: Path, **kw: Any) -> None:
        """Async save delegates to sync implementation."""
        await async_ensure_dir(path)
        self.save(obj, path, **kw)


class PickleHandler(FileHandler):
    """Pickle handler for Python objects.

    WARNING: Unpickling data from untrusted sources is unsafe.
    Only use with trusted, validated data.
    """

    def read(self, path: Path | io.BytesIO, **kwargs: Any) -> Any:
        """Deserialize a Python object from a pickle file or buffer.

        WARNING: Unpickling data from untrusted sources is unsafe.
        """
        if isinstance(path, io.BytesIO):
            return pickle.load(path)
        with Path(path).open("rb") as f:
            return pickle.load(f)

    def save(self, obj: Any, path: Path, **kwargs: Any) -> None:
        """Serialize a Python object with pickle."""
        ensure_dir(path)
        with path.open("wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

    async def async_read(self, path: Path, **kwargs: Any) -> Any:
        """Async read delegates to sync implementation."""
        return self.read(path, **kwargs)

    async def async_save(self, obj: Any, path: Path, **kwargs: Any) -> None:
        """Async save delegates to sync implementation."""
        await async_ensure_dir(path)
        self.save(obj, path, **kwargs)


class NumPyHandler(FileHandler):
    """Handler for NumPy .npy files."""

    def read(self, path: Path | io.BytesIO, **kwargs: Any) -> np.ndarray:
        """Load a NumPy array from .npy file."""
        if isinstance(path, io.BytesIO):
            return np.load(path, allow_pickle=False)

        return np.load(path, allow_pickle=False, **kwargs)

    def save(self, obj: Any, path: Path, **kwargs: Any) -> None:
        """Save a NumPy array to .npy file."""
        ensure_dir(path)
        np.save(path, obj, allow_pickle=False)

    async def async_read(self, path: Path, **kwargs: Any) -> np.ndarray:
        """Async read delegates to sync implementation."""
        return self.read(path, **kwargs)

    async def async_save(self, obj: Any, path: Path, **kwargs: Any) -> None:
        """Async save delegates to sync implementation."""
        await async_ensure_dir(path)
        self.save(obj, path, **kwargs)
