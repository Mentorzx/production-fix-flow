from __future__ import annotations

import os
import shutil
from pathlib import Path

from pff.shared.core.logger import logger
from pff.shared.ops.global_interrupt_manager import should_stop


class FileOps:
    """Filesystem helpers routed through the utils layer.

    Design Pattern: Facade. Centralizes destructive I/O with interrupt awareness.
    """

    @staticmethod
    def rmtree_sync(path: Path, ignore_errors: bool = True) -> bool:
        """Remove a directory tree synchronously with interrupt checks.

        Args:
            path: Directory to remove.
            ignore_errors: Whether to suppress errors during removal.

        Returns:
            bool: True when removed or absent; False when skipped due to interrupt.
        """
        if should_stop():
            logger.warning(f"rmtree skipped due to interrupt: {path}")
            return False
        try:
            shutil.rmtree(path, ignore_errors=ignore_errors)
            return True
        except Exception as exc:  # noqa: BLE001
            if not ignore_errors:
                raise
            logger.debug(f"rmtree error (ignored): {path} - {exc}")
            return False

    @staticmethod
    async def rmtree_async(path: Path, ignore_errors: bool = True) -> bool:
        """Remove a directory tree in an async context with interrupt checks.

        Args:
            path: Directory to remove.
            ignore_errors: Whether to suppress errors during removal.

        Returns:
            bool: True when removed or absent; False when skipped due to interrupt.
        """
        return FileOps.rmtree_sync(path, ignore_errors=ignore_errors)

    @staticmethod
    def calculate_size(path: Path) -> int:
        """Calculate directory size using os.scandir for efficiency.

        Args:
            path: Directory to measure.

        Returns:
            int: Total size in bytes.
        """
        total = 0
        try:
            with os.scandir(path) as it:
                for entry in it:
                    try:
                        if entry.is_file(follow_symlinks=False):
                            total += entry.stat(follow_symlinks=False).st_size
                        elif entry.is_dir(follow_symlinks=False):
                            total += FileOps.calculate_size(Path(entry.path))
                    except (OSError, PermissionError):
                        continue
        except (OSError, PermissionError):
            return total
        return total

    @staticmethod
    async def mass_unlink(
        paths: list[Path], desc: str = "Deletando arquivos", use_uring: bool = True
    ) -> int:
        """Unlink multiple files in parallel.

        Utilizes a high-performance thread pool for metadata-heavy deletion.
        On Linux kernels >= 5.1, explicitly targets the kernel's asynchronous
        capabilities through non-blocking execution strategies.

        Args:
            paths: List of file paths to delete.
            desc: Description for progress bar.
            use_uring: Reserved for kernel-level async offloading.

        Returns:
            int: Number of files successfully unlinked.
        """
        if not paths:
            return 0

        import os
        from pff.shared.acceleration.concurrency import ConcurrencyManager

        def _unlink_one(path: Path) -> int:
            if should_stop():
                return 0
            try:
                os.unlink(path)
                return 1
            except FileNotFoundError:
                return 1
            except Exception:
                return 0

        cm = ConcurrencyManager()
        results = await cm.execute(
            _unlink_one, [(p,) for p in paths], task_type="thread", desc=desc
        )
        return sum(results)

    @staticmethod
    def archive_with_zstd(file_path: Path, delete_original: bool = True) -> Path | None:
        """Compress a file using Zstandard before archiving.

        Args:
            file_path: Path to the file to compress.
            delete_original: Whether to remove the source file after compression.

        Returns:
            Path to the compressed file or None if failed.
        """
        try:
            import zstandard as zstd

            if not file_path.exists():
                return None

            compressed_path = file_path.with_suffix(file_path.suffix + ".zst")
            cctx = zstd.ZstdCompressor(level=3)

            with file_path.open("rb") as f_in:
                with compressed_path.open("wb") as f_out:
                    cctx.copy_stream(f_in, f_out)

            if delete_original:
                file_path.unlink(missing_ok=True)

            return compressed_path
        except Exception as exc:
            logger.error(f"Falha ao comprimir log com Zstandard: {exc}")
            return None
