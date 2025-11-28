from __future__ import annotations

import asyncio
import datetime
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import polars as pl

from pff import settings
from ..core.file_manager import FileManager
from ..core.logger import logger

# ────────────────────────── helpers ────────────────────────── #
_ROT_TMPL = "{stem}_{idx:04d}{suffix}"


def _make_rotated_path(path: Path, idx: int) -> Path:
    return path.with_name(f"{path.stem}_{idx:04d}{path.suffix}")


class BufferedWriter:
    """
    AsyncBufferedWriter is a fully asynchronous buffered writer for CSV, JSONL, and Parquet files.
    This class accumulates rows in memory and periodically flushes them to disk based on
    row count or elapsed time. It supports file rotation, concurrent writes via an asyncio queue,
    and can be used as an async context manager. The writer supports writing dictionaries,
    sequences, or polars DataFrames, and automatically handles file creation and format.
    
    Parameters:
        dest (str | Path): Destination file path. Supported extensions: .csv, .jsonl, .parquet.
        flush_rows (int, optional): Number of rows to buffer before flushing to disk. Default is 5,000.
        flush_secs (int, optional): Maximum seconds to wait before flushing buffer. Default is 30.
        rotation (int | None, optional): If set, rotates file after writing this many rows. Default is None.
        max_queue (int, optional): Maximum number of items in the write queue. Default is 50,000.
    
    Methods:
        write(row): Enqueue a row (dict, sequence, or DataFrame) for writing.
        close(): Stop the writer task and flush remaining data.
        force_flush(): Force a flush of the current buffer.
        write_async(rows): Asynchronously enqueue multiple rows for writing.
    
    Context Manager:
        Can be used with 'async with' statement to ensure proper resource cleanup.
    
    Raises:
        ValueError: If the file extension is not supported.
    """

    def __init__(
        self,
        dest: str | Path,
        *,
        flush_rows: int = 5_000,
        flush_secs: int = 30,
        rotation: int | None = None,
        max_queue: int = 50_000,
    ) -> None:
        raw_dest = Path(dest)
        self.dest = raw_dest if raw_dest.is_absolute() else settings.ROOT_DIR / raw_dest
        self.dest.parent.mkdir(parents=True, exist_ok=True)
        self.ext = self.dest.suffix.lower()
        if self.ext not in {".csv", ".jsonl", ".parquet"}:
            raise ValueError("Somente .csv, .jsonl ou .parquet suportados")

        self.flush_rows = flush_rows
        self.flush_secs = flush_secs
        self.rotation = rotation
        self._rot_idx = 0
        self._file_manager = FileManager()

        self._last_flush = time.time()
        self._row_count = 0
        self._current_target = self.dest
        self._frames: list[pl.DataFrame] = []
        self._buffer: list[Any] = []

    async def write(self, row: dict[str, Any] | Sequence[Any] | pl.DataFrame) -> None:
        self._buffer.append(row)
        time_cond = (time.time() - self._last_flush) >= self.flush_secs
        if len(self._buffer) >= self.flush_rows or (self._buffer and time_cond):
            await self._flush(self._buffer)
            self._buffer.clear()
            self._last_flush = time.time()

    async def close(self) -> None:
        if self._buffer:
            await self._flush(self._buffer)
            self._buffer.clear()
            self._last_flush = time.time()
        await self._finalize_target()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):  # noqa: ANN001
        await self.close()

    async def _flush(self, rows: list[Any]) -> None:
        if not rows:
            return
        incoming_rows = len(rows)
        if self.rotation and (self._row_count + incoming_rows) > self.rotation:
            await self._finalize_target()
            self._rot_idx += 1
            self._row_count = 0
            self._current_target = _make_rotated_path(self.dest, self._rot_idx)

        df_rows: list[pl.DataFrame] = []
        for r in rows:
            if isinstance(r, pl.DataFrame):
                df_rows.append(r)
            elif isinstance(r, dict):
                df_rows.append(pl.DataFrame([r]))
            else:
                df_rows.append(pl.DataFrame([list(r)]))

        if df_rows:
            new_df = pl.concat(df_rows, how="diagonal")
            self._frames.append(new_df)

        self._row_count += incoming_rows
        logger.debug(f"Flushed {len(rows)} row(s) -> {self._current_target.name}")

    async def force_flush(self) -> None:
        """Force a flush of the current buffer."""
        if self._buffer:
            await self._flush(self._buffer)
            self._buffer.clear()
        self._last_flush = time.time()

    async def write_async(self, rows: Iterable[Any]) -> None:
        """Asynchronously enqueue multiple rows for writing."""
        tasks = [self.write(row) for row in rows]
        await asyncio.gather(*tasks)

    async def _finalize_target(self) -> None:
        if not self._frames:
            self._file_manager.save(pl.DataFrame([]), self._current_target)
            return

        combined = pl.concat(self._frames, how="vertical")
        self._file_manager.save(combined, self._current_target)
        self._frames.clear()


class ResultCollector:
    """
    ResultCollector is a utility class for collecting, deduplicating, and exporting result rows to CSV or XLSX files.
    
    Attributes:
        _writer (AsyncBufferedWriter): Internal writer for buffering and writing rows to a temporary CSV file.
        _seen (set[str]): Set of MSISDNs already processed to avoid duplicates.
        exec_id (str): Identifier for the current execution, used in output filenames.
        _tmp_path (Path): Path to the temporary file used for intermediate storage.
    
    Args:
        exec_id (str): Unique identifier for the execution session.
        flush_rows (int, optional): Number of rows to buffer before flushing to disk. Defaults to 2,000.
        rotation (int | None, optional): Optional file rotation parameter. Defaults to None.
    
    Methods:
        append_row(msisdn: str, request: str, result: str, obs: str | dict[str, list[str]]) -> None:
            Appends a result row to the collector, formatting observations as needed and deduplicating by MSISDN.
        has_row(msisdn: str) -> bool:
            Checks if a row with the given MSISDN has already been added.
        save(path: str | Path | None = None, *, fmt: str | None = None) -> Path:
            Saves the collected results to a file. If no path is provided, saves to 'outputs/{timestamp}_{exec_id}.xlsx'.
            Supports CSV or XLSX output based on file extension or 'fmt' parameter.
            Cleans up the temporary file after saving.
    """

    _writer: BufferedWriter
    _seen: set[str]

    def __init__(
        self,
        exec_id: str,
        *,
        flush_rows: int = 2_000,
        rotation: int | None = None,
    ) -> None:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        tmp_name = f"{ts}_{exec_id}.parquet"
        self._tmp_path = settings.OUTPUTS_DIR / "temp" / "result_collector" / tmp_name
        self._tmp_path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = BufferedWriter(
            self._tmp_path, flush_rows=flush_rows, rotation=rotation
        )
        self._seen = set()
        self.exec_id = exec_id
        self._file_manager = FileManager()

    async def append_row(
        self,
        msisdn: str,
        request: str,
        result: str,
        obs: str | dict[str, list[str]],
    ) -> None:
        if isinstance(obs, dict):
            partes: list[str] = []
            for chave, lista in obs.items():
                partes.append(f"{chave}: {', '.join(lista) if lista else 'nenhum'}")
            obs_txt = " | ".join(partes)
        else:
            obs_txt = str(obs)

        row = {
            "MSISDN": msisdn,
            "Solicitacao": request,
            "Resultado": result,
            "Observacoes": obs_txt,
        }
        await self._writer.write(row)
        self._seen.add(msisdn)

    def has_row(self, msisdn: str) -> bool:
        return msisdn in self._seen

    async def save(
        self,
        path: str | Path | None = None,
        *,
        fmt: str | None = None,
    ) -> Path:
        await self._writer.close()
        
        if not self._tmp_path.exists():
            logger.warning("No records captured; skipping save.")
            return Path()
        
        if path is None:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            path = settings.OUTPUTS_DIR / "results" / f"{ts}_{self.exec_id}.xlsx"
        
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)

        # Read the temporary file (fast synchronous operation)
        df_to_save = self._file_manager.read(self._tmp_path)
        
        # Write to final destination (potentially slow async operation)
        await self._file_manager.async_save(df_to_save, dest)

        logger.info(f"Resultado salvo em {dest}")
        
        try:
            self._tmp_path.unlink(missing_ok=True)  # Clean up temp file
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"Failed to remove tmp file {self._tmp_path}: {exc}")

        return dest


__all__ = ["BufferedWriter", "ResultCollector"]
