"""FileManager facade for parquet-first I/O."""

from __future__ import annotations

import asyncio
import hashlib
import io
import mmap
import shutil
import zipfile
from collections.abc import Iterable
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

from pff.shared.ops.global_interrupt_manager import (
    get_interrupt_manager,
    should_stop,
)

from .bundles import ParquetBundle
from .config import get_streaming_threshold_bytes
from .container import load_zip_from_path
from .handlers import SUPPORTED_EXTS, get_handler
from .ingestion import ingest as ingest_file
from .parquet_io import stream_raw_parquet_to_path
from .utils import memory_map_file


class FileManager:
    """Facade for parquet-first file I/O."""

    def __init__(self) -> None:
        """Initialize and register cleanup callback."""
        get_interrupt_manager().register_callback(
            self._cleanup_on_interrupt, label="file_manager_flush"
        )

    def _cleanup_on_interrupt(self) -> None:
        """Callback for graceful shutdown."""
        pass

    @staticmethod
    def supported_extensions() -> set[str]:
        return {".zip", *SUPPORTED_EXTS}

    @staticmethod
    def supports_extension(ext: str) -> bool:
        return ext.lower() in FileManager.supported_extensions()

    @staticmethod
    def assert_supported_path(
        path: str | Path,
        *,
        allowed_exts: Iterable[str] | None = None,
    ) -> str:
        ext = Path(path).suffix.lower()
        if allowed_exts is not None:
            allowed = {e.lower() for e in allowed_exts}
            if ext not in allowed:
                raise ValueError(f"Unsupported extension: {ext}")
            return ext
        if not FileManager.supports_extension(ext):
            raise ValueError(f"Unsupported extension: {ext}")
        return ext

    @staticmethod
    def same_extension(path_a: str | Path, path_b: str | Path) -> bool:
        return Path(path_a).suffix.lower() == Path(path_b).suffix.lower()

    @staticmethod
    def ingest(
        path: str | Path,
        *,
        build_parsed: bool = True,
        cache: bool = True,
        **kwargs: Any,
    ) -> ParquetBundle:
        return ingest_file(path, build_parsed=build_parsed, cache=cache, **kwargs)

    @staticmethod
    def ingest_directory(
        dir_path: Path,
        *,
        build_parsed: bool = True,
        cache: bool = True,
        **kwargs: Any,
    ) -> dict[str, ParquetBundle]:
        files = [
            p
            for p in dir_path.rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
        ]
        bundles: dict[str, ParquetBundle] = {}
        for p in files:
            rel = str(p.relative_to(dir_path))
            bundles[rel] = FileManager.ingest(
                p, build_parsed=build_parsed, cache=cache, **kwargs
            )
        return bundles

    @staticmethod
    def export(
        bundle_or_path: ParquetBundle | str | Path,
        dest_path: str | Path,
        *,
        prefer_raw_if_pristine: bool = True,
        **kwargs: Any,
    ) -> None:
        bundle = (
            bundle_or_path
            if isinstance(bundle_or_path, ParquetBundle)
            else FileManager.ingest(bundle_or_path, build_parsed=True, cache=True)
        )
        dest = Path(dest_path)
        dest_ext = dest.suffix.lower()

        if prefer_raw_if_pristine and not bundle.dirty and dest_ext == bundle.ext:
            sha = stream_raw_parquet_to_path(bundle.raw_parquet_path, dest)
            expected = bundle.metadata.get("sha256")
            if expected and sha != expected:
                raise ValueError(
                    "RAW export hash mismatch; source integrity compromised"
                )
            return

        if dest_ext == ".zip" and bundle.parsed_kind == "container":
            FileManager._export_container_to_zip(bundle, dest)
            return

        if dest_ext in {".parquet", ".pq", ".parq"} and bundle.parsed_kind == "tabular":
            if bundle.parsed_parquet_path is None:
                raise ValueError("Parsed parquet not available for export")
            shutil.copyfile(bundle.parsed_parquet_path, dest)
            return

        handler = get_handler(dest_ext)
        if handler is None:
            raise ValueError(f"Unsupported export extension: {dest.suffix}")
        handler.save(bundle.to_native(), dest, **kwargs)

    @staticmethod
    def read(
        path: str | Path | ParquetBundle,
        *,
        return_native: bool = False,
        build_parsed: bool = True,
        cache: bool = True,
        **kwargs: Any,
    ) -> Any:
        if isinstance(path, ParquetBundle):
            return path.to_native() if return_native else path
        p = Path(path)
        if p.is_dir():
            if return_native:
                return FileManager.load_directory(p, **kwargs)
            return FileManager.ingest_directory(
                p, build_parsed=build_parsed, cache=cache, **kwargs
            )
        bundle = FileManager.ingest(p, build_parsed=build_parsed, cache=cache, **kwargs)
        return bundle.to_native(**kwargs) if return_native else bundle

    @staticmethod
    def read_streaming_if_large(
        path: str | Path,
        *,
        streaming_threshold_bytes: int | None = None,
        return_native: bool = False,
        build_parsed: bool = True,
        cache: bool = True,
        **kwargs: Any,
    ) -> Any:
        p = Path(path)
        if "streaming" not in kwargs and p.exists() and p.is_file():
            threshold = streaming_threshold_bytes or get_streaming_threshold_bytes()
            if p.stat().st_size > threshold:
                kwargs["streaming"] = True
        return FileManager.read(
            p,
            return_native=return_native,
            build_parsed=build_parsed,
            cache=cache,
            **kwargs,
        )

    @staticmethod
    def save(obj: Any, path: str | Path, **kwargs: Any) -> None:
        if should_stop():
            return
        if isinstance(obj, ParquetBundle):
            FileManager.export(obj, path, **kwargs)
            return
        p = Path(path)
        handler = get_handler(p.suffix.lower())
        if handler is None:
            raise ValueError(f"Unsupported extension: {p.suffix}")
        handler.save(obj, p, **kwargs)

    @staticmethod
    async def async_read(path: str | Path, **kwargs: Any) -> Any:
        return await asyncio.to_thread(FileManager.read, path, **kwargs)

    @staticmethod
    async def async_save(obj: Any, path: str | Path, **kwargs: Any) -> None:
        if isinstance(obj, ParquetBundle):
            await asyncio.to_thread(FileManager.export, obj, path, **kwargs)
            return
        p = Path(path)
        handler = get_handler(p.suffix.lower())
        if handler is None:
            raise ValueError(f"Unsupported extension: {p.suffix}")
        await handler.async_save(obj, p, **kwargs)

    @staticmethod
    def exists(path: str | Path) -> bool:
        return Path(path).exists()

    @staticmethod
    def ensure_dir(path: str | Path) -> Path:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @staticmethod
    def ensure_parent_dir(path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p.parent

    @staticmethod
    def glob(path: str | Path, pattern: str) -> list[Path]:
        p = Path(path)
        try:
            return sorted(p.glob(pattern))
        except Exception:
            return []

    @staticmethod
    def read_bytes(path: str | Path) -> bytes:
        return Path(path).read_bytes()

    @staticmethod
    def read_tail_bytes(path: str | Path, *, max_bytes: int = 65536) -> bytes:
        if max_bytes <= 0:
            return b""
        p = Path(path)
        if not p.exists() or not p.is_file():
            return b""
        with p.open("rb") as handle:
            handle.seek(0, io.SEEK_END)
            size = handle.tell()
            offset = max(0, size - max_bytes)
            handle.seek(offset, io.SEEK_SET)
            return handle.read()

    @staticmethod
    def write_bytes(data: bytes, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)

    @staticmethod
    def write_text(data: str, path: str | Path, *, encoding: str = "utf-8") -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(data, encoding=encoding)

    @staticmethod
    def scan_csv(pattern: str, **kwargs: Any) -> pl.LazyFrame:
        return pl.scan_csv(pattern, **kwargs)

    @staticmethod
    def scan_parquet(pattern: str, **kwargs: Any) -> pl.LazyFrame:
        return pl.scan_parquet(pattern, **kwargs)

    @staticmethod
    def scan_ndjson(pattern: str, **kwargs: Any) -> pl.LazyFrame:
        return pl.scan_ndjson(pattern, **kwargs)

    @staticmethod
    def scan_directory(dir_path: Path, **kwargs: Any) -> pl.LazyFrame:
        if next(dir_path.glob("*.parquet"), None):
            return pl.scan_parquet(str(dir_path / "*.parquet"), **kwargs)
        if next(dir_path.glob("*.csv"), None):
            return pl.scan_csv(str(dir_path / "*.csv"), **kwargs)
        if next(dir_path.glob("*.ndjson"), None):
            return pl.scan_ndjson(str(dir_path / "*.ndjson"), **kwargs)
        raise ValueError(
            f"Directory '{dir_path}' contains no single, scannable file type."
        )

    @staticmethod
    def adaptive_scan(path: str | Path, **kwargs: Any) -> pl.LazyFrame:
        p = Path(path)
        streaming = kwargs.pop("streaming", True)
        suffix = p.suffix.lower()
        if suffix in {".csv", ".tsv"}:
            handler = get_handler(".csv")
            if handler is None:
                raise ValueError(f"Handler for CSV not found: {suffix}")
            lazy_frame = handler.read(p, lazy=True, streaming=streaming, **kwargs)
            return (
                lazy_frame
                if isinstance(lazy_frame, pl.LazyFrame)
                else lazy_frame.lazy()
            )
        if suffix in {".parquet", ".pq", ".parq"}:
            handler = get_handler(".parquet")
            if handler is None:
                raise ValueError(f"Handler for Parquet not found: {suffix}")
            lazy_frame = handler.read(p, lazy=True, streaming=streaming, **kwargs)
            return (
                lazy_frame
                if isinstance(lazy_frame, pl.LazyFrame)
                else lazy_frame.lazy()
            )
        if suffix in {".ndjson", ".jsonl"}:
            handler = get_handler(".ndjson")
            if handler is None:
                raise ValueError(f"Handler for NDJSON not found: {suffix}")
            lazy_frame = handler.read(p, lazy=True, streaming=streaming, **kwargs)
            return (
                lazy_frame
                if isinstance(lazy_frame, pl.LazyFrame)
                else lazy_frame.lazy()
            )
        raise ValueError(f"Adaptive scan not supported for extension {suffix}")

    @staticmethod
    @contextmanager
    def memory_map(path: str | Path, *, access: int = mmap.ACCESS_READ):
        p = Path(path)
        with memory_map_file(p) as mm:
            yield mm

    @staticmethod
    def load_directory(dir_path: Path, **kwargs: Any) -> pl.DataFrame | dict[str, Any]:
        try:
            return FileManager.scan_directory(dir_path, **kwargs).collect()
        except ValueError:
            files = [
                p
                for p in dir_path.rglob("*")
                if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
            ]
            return {
                str(p.relative_to(dir_path)): FileManager.read(
                    p, return_native=True, **kwargs
                )
                for p in files
            }

    @staticmethod
    def load_zip(zip_path: str | Path, **kwargs: Any) -> dict[str, Any]:
        p = Path(zip_path)
        return load_zip_from_path(
            p,
            parallel=bool(kwargs.pop("parallel", False)),
            task_type=str(kwargs.pop("task_type", "thread")),
            chunk_size=int(kwargs.pop("chunk_size", 16)),
            handler_kwargs=kwargs,
            fuse_processing=bool(kwargs.pop("fuse_processing", True)),
            max_members=kwargs.pop("max_members", None),
            use_mmap=bool(kwargs.pop("use_mmap", True)),
        )

    @staticmethod
    def get_hash(path: Path, block_size: int = 65536) -> str:
        hasher = hashlib.md5()
        try:
            with open(path, "rb") as f:
                buf = f.read(block_size)
                while len(buf) > 0:
                    hasher.update(buf)
                    buf = f.read(block_size)
            return hasher.hexdigest()
        except FileNotFoundError:
            return ""

    @staticmethod
    def delete_directory(path: Path, *, ignore_errors: bool = False) -> bool:
        if not path.exists():
            return False
        try:
            shutil.rmtree(path, ignore_errors=ignore_errors)
            return True
        except OSError:
            if not ignore_errors:
                raise
            return False

    @staticmethod
    def delete_file(path: Path, *, ignore_errors: bool = False) -> bool:
        if not path.exists():
            return False
        try:
            path.unlink()
            return True
        except OSError:
            if not ignore_errors:
                raise
            return False

    @staticmethod
    def copy_file(src: Path, dest: Path, *, preserve_metadata: bool = True) -> Path:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if preserve_metadata:
            return Path(shutil.copy2(src, dest))
        return Path(shutil.copy(src, dest))

    @staticmethod
    def copy_directory(src: Path, dest: Path, *, dirs_exist_ok: bool = True) -> Path:
        return Path(shutil.copytree(src, dest, dirs_exist_ok=dirs_exist_ok))

    @staticmethod
    def get_timestamp() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    @staticmethod
    def read_text(path: str | Path, *, encoding: str = "utf-8") -> str:
        return Path(path).read_text(encoding=encoding)

    @staticmethod
    def json_dumps(obj: Any, *, sort_keys: bool = False) -> str:
        import orjson

        if sort_keys:
            return orjson.dumps(obj, option=orjson.OPT_SORT_KEYS).decode("utf-8")
        import msgspec

        return msgspec.json.encode(obj).decode("utf-8")

    @staticmethod
    def json_loads(s: str | bytes) -> Any:
        import msgspec

        if isinstance(s, str):
            s = s.encode("utf-8")
        return msgspec.json.decode(s)

    @staticmethod
    def query(query: str, **kwargs: Any) -> pl.DataFrame:
        import duckdb

        return duckdb.query(query, **kwargs).pl()

    @staticmethod
    def _export_container_to_zip(bundle: ParquetBundle, dest: Path) -> None:
        entries = bundle.to_native()
        dest.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for name, content in entries.items():
                if isinstance(content, pl.DataFrame):
                    buf = io.BytesIO()
                    content.write_parquet(buf)
                    zf.writestr(name, buf.getvalue())
                elif isinstance(content, (dict, list)):
                    zf.writestr(name, FileManager.json_dumps(content))
                elif isinstance(content, str):
                    zf.writestr(name, content)
                elif isinstance(content, (bytes, bytearray)):
                    zf.writestr(name, bytes(content))
                else:
                    zf.writestr(name, str(content))
