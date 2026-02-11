"""ZIP file utilities for container handling."""

from __future__ import annotations

import asyncio
import io
import mmap
import zipfile
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from ....acceleration.concurrency import ConcurrencyManager
from ...logging import logger
from ..handlers import SUPPORTED_EXTS, get_handler
from ..utils import fast_suffix


@lru_cache(maxsize=64)
def get_cached_zip_members(
    zip_path: Path,
    supported_exts: tuple[str, ...],
    stat_sig: tuple[int, int],
) -> list[str]:
    """Get filtered list of ZIP members with caching.

    Args:
        zip_path: Path to ZIP file.
        supported_exts: Tuple of supported extensions.
        stat_sig: Tuple of (mtime_ns, size) for cache invalidation.

    Returns:
        List of member names with supported extensions.
    """
    source = ZipPathSource(zip_path)
    return source.list_members(supported_exts)


def process_zip_entry(
    item: tuple[str, bytes], handler_kwargs: dict[str, Any]
) -> tuple[str, Any]:
    """Process a single entry from a ZIP archive using the appropriate handler.

    Args:
        item: Tuple of (entry_name, raw_bytes).
        handler_kwargs: Kwargs to pass to handler.

    Returns:
        Tuple of (entry_name, parsed_content or None on failure).
    """
    name, raw = item
    suffix = fast_suffix(name)
    handler = get_handler(suffix)
    if handler:
        try:
            return name, handler.load_bytes(raw, **handler_kwargs)
        except Exception as exc:
            logger.debug(
                f"Failed to process ZIP entry name={name} suffix={suffix}: {exc}"
            )
            return name, None
    return name, raw


class ZipSource:
    """Abstract ZIP source for path or in-memory ZIP payloads."""

    def list_members(self, supported_exts: Iterable[str]) -> list[str]:
        raise NotImplementedError

    def iter_members(self, members: Iterable[str]) -> Iterator[tuple[str, bytes]]:
        raise NotImplementedError


@dataclass(frozen=True)
class ZipPathSource(ZipSource):
    """ZIP source backed by a filesystem path."""

    zip_path: Path
    use_mmap: bool = True

    def _open_zip(self):
        if not self.use_mmap:
            return zipfile.ZipFile(self.zip_path, "r")
        fd = self.zip_path.open("rb")
        mm = mmap.mmap(fd.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            zf = zipfile.ZipFile(mm, "r")
        except Exception:
            mm.close()
            fd.close()
            raise
        zf._pff_mmap = mm
        zf._pff_fd = fd
        return zf

    def list_members(self, supported_exts: Iterable[str]) -> list[str]:
        supported = set(supported_exts)
        zf = self._open_zip()
        try:
            return [
                m
                for m in zf.namelist()
                if not m.endswith("/") and fast_suffix(m) in supported
            ]
        finally:
            zf.close()
            mm = getattr(zf, "_pff_mmap", None)
            fd = getattr(zf, "_pff_fd", None)
            if mm is not None:
                mm.close()
            if fd is not None:
                fd.close()

    def iter_members(self, members: Iterable[str]) -> Iterator[tuple[str, bytes]]:
        zf = self._open_zip()
        try:
            for name in members:
                try:
                    yield name, zf.read(name)
                except Exception as exc:
                    logger.debug(f"Failed to read ZIP entry {name}: {exc}")
        finally:
            zf.close()
            mm = getattr(zf, "_pff_mmap", None)
            fd = getattr(zf, "_pff_fd", None)
            if mm is not None:
                mm.close()
            if fd is not None:
                fd.close()


@dataclass(frozen=True)
class ZipBytesSource(ZipSource):
    """ZIP source backed by in-memory bytes."""

    data: bytes

    def list_members(self, supported_exts: Iterable[str]) -> list[str]:
        supported = set(supported_exts)
        with zipfile.ZipFile(io.BytesIO(self.data), "r") as zf:
            return [
                m
                for m in zf.namelist()
                if not m.endswith("/") and fast_suffix(m) in supported
            ]

    def iter_members(self, members: Iterable[str]) -> Iterator[tuple[str, bytes]]:
        with zipfile.ZipFile(io.BytesIO(self.data), "r") as zf:
            for name in members:
                try:
                    yield name, zf.read(name)
                except Exception as exc:
                    logger.debug(f"Failed to read ZIP entry {name}: {exc}")


def iter_zip_entries(
    source: ZipSource, members: Iterable[str]
) -> Iterator[tuple[str, bytes]]:
    """Iterate over ZIP entries using a ZipSource."""
    yield from source.iter_members(members)


def _read_members_chunk(
    source: ZipSource, members: list[str]
) -> list[tuple[str, bytes]]:
    return list(source.iter_members(members))


def _read_and_process_members_chunk(
    source: ZipSource,
    members: list[str],
    handler_kwargs: dict[str, Any],
) -> list[tuple[str, Any]]:
    return [
        process_zip_entry(item, handler_kwargs) for item in source.iter_members(members)
    ]


def _load_zip_from_source(
    source: ZipSource,
    members: list[str],
    *,
    parallel: bool,
    task_type: str,
    chunk_size: int,
    handler_kwargs: dict[str, Any],
    fuse_processing: bool,
) -> dict[str, Any]:
    cm = ConcurrencyManager()
    if parallel and len(members) > 1:
        chunks = [
            members[i : i + chunk_size] for i in range(0, len(members), chunk_size)
        ]
        if fuse_processing:
            read_args_fused: list[tuple[ZipSource, list[str], dict[str, Any]]] = [
                (source, chunk, handler_kwargs) for chunk in chunks
            ]
            chunk_results = cm.execute_sync(
                _read_and_process_members_chunk,
                read_args_fused,
                task_type=task_type,
                max_workers=cm.hardware.logical_cores,
                desc="Loading ZIP entries",
            )
            result = [item for chunk in chunk_results for item in chunk if chunk]
            return dict(result)
        read_args_simple: list[tuple[ZipSource, list[str]]] = [
            (source, chunk) for chunk in chunks
        ]
        chunk_results = cm.execute_sync(
            _read_members_chunk,
            read_args_simple,
            task_type=task_type,
            max_workers=cm.hardware.logical_cores,
            desc="Reading ZIP entries",
        )
        raw_entries = [item for chunk in chunk_results for item in chunk if chunk]
    else:
        raw_entries = list(source.iter_members(members))

    if fuse_processing:
        return dict(process_zip_entry(entry, handler_kwargs) for entry in raw_entries)

    args_list = [(entry, handler_kwargs) for entry in raw_entries]
    processed = cm.execute_sync(
        process_zip_entry,
        args_list,
        task_type=task_type,
        max_workers=cm.hardware.logical_cores,
        desc="Parsing ZIP entries",
    )
    return dict(processed)


async def load_zip_from_bytes(
    data: bytes,
    *,
    parallel: bool,
    task_type: str,
    chunk_size: int,
    handler_kwargs: dict[str, Any],
    fuse_processing: bool,
    max_members: int | None = None,
) -> dict[str, Any]:
    """Load ZIP contents from bytes asynchronously."""
    source = ZipBytesSource(data)
    members = source.list_members(SUPPORTED_EXTS)
    if max_members:
        members = members[:max_members]
    if task_type == "process":
        task_type = "thread"
    return await asyncio.to_thread(
        _load_zip_from_source,
        source,
        members,
        parallel=parallel,
        task_type=task_type,
        chunk_size=chunk_size,
        handler_kwargs=handler_kwargs,
        fuse_processing=fuse_processing,
    )


def load_zip_from_path(
    zip_path: Path,
    *,
    parallel: bool,
    task_type: str,
    chunk_size: int,
    handler_kwargs: dict[str, Any],
    fuse_processing: bool,
    max_members: int | None = None,
    use_mmap: bool = True,
) -> dict[str, Any]:
    """Load ZIP contents from a path."""
    source = ZipPathSource(zip_path, use_mmap=use_mmap)
    members = source.list_members(SUPPORTED_EXTS)
    if max_members:
        members = members[:max_members]
    return _load_zip_from_source(
        source,
        members,
        parallel=parallel,
        task_type=task_type,
        chunk_size=chunk_size,
        handler_kwargs=handler_kwargs,
        fuse_processing=fuse_processing,
    )
