from __future__ import annotations

import asyncio
import datetime
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import polars as pl

from pff.utils import FileManager, logger

# ────────────────────────── helpers ────────────────────────── #
_ROT_TMPL = "{stem}_{idx:04d}{suffix}"


def _make_rotated_path(path: Path, idx: int) -> Path:
    return path.with_name(_ROT_TMPL.format(stem=path.stem, idx=idx, suffix=path.suffix))


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
        self.dest = Path(dest)
        self.dest.parent.mkdir(parents=True, exist_ok=True)
        self.ext = self.dest.suffix.lower()
        if self.ext not in {".csv", ".jsonl", ".parquet"}:
            raise ValueError("Somente .csv, .jsonl ou .parquet suportados")

        self.flush_rows = flush_rows
        self.flush_secs = flush_secs
        self.rotation = rotation
        self._rot_idx = 0
        self._file_manager = FileManager()

        self._q: asyncio.Queue[Any] = asyncio.Queue(maxsize=max_queue)
        self._stop_event = asyncio.Event()
        self._last_flush = time.time()
        self._row_count = 0
        self._worker_task: asyncio.Task[None] | None = None
        
        # Start the worker task
        self._worker_task = asyncio.create_task(self._worker())

    async def write(self, row: dict[str, Any] | Sequence[Any] | pl.DataFrame) -> None:
        await self._q.put(row)

    async def close(self) -> None:
        self._stop_event.set()
        await self._q.join()
        if not self.dest.exists():
            await self._file_manager.async_save(pl.DataFrame([]), self.dest)
        if self._worker_task:
            await self._worker_task

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):  # noqa: ANN001
        await self.close()

    async def _worker(self) -> None:
        buffer: list[Any] = []
        
        while not self._stop_event.is_set() or not self._q.empty():
            try:
                # Use wait_for to allow periodic checking of flush conditions
                item = await asyncio.wait_for(self._q.get(), timeout=1.0)
                buffer.append(item)
                self._q.task_done()
            except asyncio.TimeoutError:
                # Timeout is expected, continue to check flush conditions
                pass

            # Check flush conditions
            time_cond = (time.time() - self._last_flush) >= self.flush_secs
            if len(buffer) >= self.flush_rows or (buffer and time_cond):
                await self._flush(buffer)
                buffer.clear()
                self._last_flush = time.time()

        # Final flush of remaining data
        if buffer:
            await self._flush(buffer)

    async def _flush(self, rows: list[Any]) -> None:
        if not rows:
            return

        # Handle file rotation
        if self.rotation and self._row_count >= self.rotation:
            self._rot_idx += 1
            self.dest = _make_rotated_path(self.dest, self._rot_idx)
            self._row_count = 0
            logger.info(f"Arquivo rotacionado -> {self.dest.name}")

        # Convert buffer rows to DataFrame
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

            # If file exists, read and concatenate with new data
            if self.dest.exists():
                existing_df = self._file_manager.read(self.dest)  # Fast synchronous read
                combined_df = pl.concat([existing_df, new_df], how="diagonal")
            else:
                combined_df = new_df

            # Use async save to avoid blocking the event loop
            await self._file_manager.async_save(combined_df, self.dest)

        self._row_count += len(rows)
        logger.debug(f"Flushed {len(rows)} row(s) -> {self.dest.name}")

    async def force_flush(self) -> None:
        """Force a flush of the current buffer."""
        await self._q.join()
        self._last_flush = 0

    async def write_async(self, rows: Iterable[Any]) -> None:
        """Asynchronously enqueue multiple rows for writing."""
        tasks = [self.write(row) for row in rows]
        await asyncio.gather(*tasks)


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
        self._tmp_path = Path(tempfile.gettempdir()) / tmp_name
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
            "Solicitação": request,
            "Resultado": result,
            "Observações": obs_txt,
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
            logger.warning("Nenhum registro gravado; skip save.")
            return Path()
        
        if path is None:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            path = Path("outputs") / f"{ts}_{self.exec_id}.xlsx"
        
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
            logger.debug(f"Não consegui remover tmp {self._tmp_path}: {exc}")

        return dest


__all__ = ["BufferedWriter", "ResultCollector"]
