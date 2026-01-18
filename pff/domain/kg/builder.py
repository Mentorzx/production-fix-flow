from __future__ import annotations

import argparse
import asyncio
import itertools
import os
import random
import re
import sys
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from pff.domain.ports.persistence.kg_ports import KGSplitsPort

import polars as pl
import pyarrow.parquet as pq

from pff import settings
from pff.shared.core.config import INGESTION_CONFIG_PATH
from pff.shared import (
    CacheManager,
    ConcurrencyManager,
    FileManager,
    logger,
    progress_bar,
)
from pff.shared.core.file_manager import ParquetBundle
from pff.shared.core.file_manager.handlers.parquet import iter_parquet_structs
from pff.shared.acceleration.loop_accelerator import LoopAccelerator

DEFAULT_ENCODING = "utf-8"
DEFAULT_SOURCE = "."
DEFAULT_OUTPUT_DIR = "outputs/kg"
_KV = re.compile(r"""\s*["']?([^"':\t]+)["']?\s*:\s*["']?([^"']+)["']?\s*,?\s*$""")
_SKIP_LINES = {"{", "}", "[", "]", "},", "],", "{}", "[]"}
_SKIP_VALUES = {"{", "[", "{}", "[]"}


def _clean(text: str) -> str:
    if "\t" not in text:
        return text.strip()
    return text.replace("\t", " ").strip()


def _ensure_dir(path: Path) -> None:
    FileManager().ensure_dir(path)


def _load_ingestion_config() -> dict[str, Any]:
    fm = FileManager()
    try:
        payload = fm.read(INGESTION_CONFIG_PATH)
        cfg = payload.to_native() if isinstance(payload, ParquetBundle) else payload
        if isinstance(cfg, dict):
            return cfg.get("ingestion", cfg)
    except FileNotFoundError as exc:
        logger.warning(f"Ingestion config not found at {INGESTION_CONFIG_PATH}: {exc}")
    except (OSError, ValueError) as exc:
        logger.warning(
            f"Failed to load ingestion config from {INGESTION_CONFIG_PATH}: {exc}",
            exc_info=True,
        )
    return {}


def _resolve_path(path_like: str | Path, *, base: Path) -> Path:
    candidate = Path(path_like)
    if not candidate.is_absolute():
        candidate = base / candidate
    return candidate.expanduser().resolve()


CACHE = CacheManager()


class KGBuilder:
    """
    KGBuilder is a utility class for constructing and serializing
    knowledge graphs from various data sources.
    """

    def __init__(
        self,
        source_path: str | Path | None,
        output_dir: str | Path | None,
        max_members: int | None = None,
        *,
        parallel: bool = True,
        workers: int | None = None,
        disk_cache: bool = False,
        splits_repo: KGSplitsPort | None = None,
        seed: int | None = 42,
    ) -> None:
        cfg = _load_ingestion_config()
        default_source = cfg.get(
            "correct_zip_path", settings.DATA_DIR / "models" / "correct.parquet"
        )
        default_output = cfg.get("output_dir", settings.OUTPUTS_DIR / "kg" / "graph")
        staging_default = cfg.get("temp_output_dir", settings.OUTPUTS_DIR / "temp" / "kg_ingestion")
        ratios_cfg = cfg.get("split_ratios", {"train": 0.8, "valid": 0.1, "test": 0.1})
        batch_size_cfg = cfg.get("batch_size", 50000)
        graph_subdir = cfg.get("graph_subdir", "kg")

        self.source_path = _resolve_path(source_path or default_source, base=settings.ROOT_DIR)
        resolved_output = _resolve_path(output_dir or default_output, base=settings.OUTPUTS_DIR)
        if not resolved_output.is_relative_to(settings.OUTPUTS_DIR):
            resolved_output = settings.OUTPUTS_DIR / resolved_output.name
        nested_outputs = settings.OUTPUTS_DIR / "outputs"
        if resolved_output in (nested_outputs, settings.OUTPUTS_DIR):
            resolved_output = settings.OUTPUTS_DIR / graph_subdir
            logger.warning(
                f"Output directory pointed to nested 'outputs'; normalizing to {resolved_output}"
            )
        self.output_dir = resolved_output
        _ensure_dir(self.output_dir)

        staging_dir = _resolve_path(staging_default, base=settings.OUTPUTS_DIR)
        if not staging_dir.is_relative_to(settings.OUTPUTS_DIR):
            staging_dir = settings.OUTPUTS_DIR / staging_dir.name
        self._staging_dir = staging_dir
        _ensure_dir(self._staging_dir)

        self.fm = FileManager()
        self.max_members = max_members
        self.parallel = parallel
        self.max_workers = workers or min(os.cpu_count() or 1, 8)
        self.chunk_size = max(int(batch_size_cfg), 1)
        self.split_ratios = self._normalize_ratios(ratios_cfg)
        self.splits_repo = splits_repo
        self.rng = random.Random(seed)

        self._stats = SimpleNamespace(total_members=0, total_triples=0)
        self._split_counts: dict[str, int] = {k: 0 for k in self.split_ratios}

        self._buffers: dict[str, list[tuple[str, str, str]]] = {
            split: [] for split in self.split_ratios
        }
        self._chunk_indices: dict[str, int] = {split: 0 for split in self.split_ratios}
        self._pending_tasks: list[asyncio.Task] = []

        self._cached_convert = (
            CACHE.disk(ttl=None)(self._convert_to_triples)
            if disk_cache
            else self._convert_to_triples
        )

    async def run(self) -> None:
        """Faz todo o fluxo: load  parse  split  salvar."""
        await self._load_and_parse()
        await self._serialise()
        logger.success(" Construção finalizada.")

    async def extract_triples(self) -> list[tuple[str, str, str]]:
        """Public helper to load and return triples without serializacao em disco."""
        collector: list[tuple[str, str, str]] = []
        self._stats = SimpleNamespace(total_members=0, total_triples=0)
        await self._load_and_parse(collector=collector, persist=False)
        return collector

    @staticmethod
    def _normalize_ratios(ratios: Mapping[str, Any]) -> dict[str, float]:
        defaults = {"train": 0.8, "valid": 0.1, "test": 0.1}
        try:
            cleaned = {k: float(v) for k, v in ratios.items()}
        except (TypeError, ValueError):
            return defaults
        total = sum(v for v in cleaned.values() if v > 0)
        if total <= 0:
            return defaults
        normalized = {k: max(0.0, v) / total for k, v in cleaned.items()}
        return normalized or defaults

    def _select_split(self) -> str:
        rnd = self.rng.random()
        cumulative = 0.0
        for split, ratio in self.split_ratios.items():
            cumulative += ratio
            if rnd <= cumulative:
                return split
        return next(iter(self.split_ratios))

    def _batch_select_splits(self, n: int) -> list[str]:
        splits = list(self.split_ratios.keys())
        weights = list(self.split_ratios.values())
        return self.rng.choices(splits, weights=weights, k=n)

    def _flush_split(self, split: str) -> None:
        buffer = self._buffers.get(split, [])
        if not buffer:
            return

        df = pl.DataFrame(buffer, schema=["s", "p", "o"], orient="row")
        chunk_path = self._staging_dir / f"{split}_{self._chunk_indices[split]}.parquet"

        task = asyncio.create_task(self.fm.async_save(df, chunk_path))
        self._pending_tasks.append(task)

        self._chunk_indices[split] += 1
        buffer.clear()

    async def _wait_for_tasks(self) -> None:
        if not self._pending_tasks:
            return
        logger.debug(f"Aguardando {len(self._pending_tasks)} escritas em background…")
        await asyncio.gather(*self._pending_tasks)
        self._pending_tasks.clear()

    async def _flush_all_buffers(self) -> None:
        for split in self._buffers:
            self._flush_split(split)
        await self._wait_for_tasks()

    def _buffer_triples(
        self,
        triples: list[tuple[str, str, str]],
        collector: list[tuple[str, str, str]] | None,
    ) -> None:
        if not triples:
            return

        if collector is not None:
            collector.extend(triples)

        splits = self._batch_select_splits(len(triples))
        for triple, split in zip(triples, splits):
            self._split_counts[split] += 1
            self._stats.total_triples += 1
            self._buffers[split].append(triple)
            if len(self._buffers[split]) >= self.chunk_size:
                self._flush_split(split)

    def _iter_parquet_json_entries(
        self,
        parquet_path: Path,
        *,
        max_members: int | None,
    ) -> tuple[Iterable[tuple[str, Any]], int | None]:
        parquet_file = pq.ParquetFile(parquet_path)
        total_rows = parquet_file.metadata.num_rows if parquet_file.metadata else None
        batch_size = max(512, min(self.chunk_size, 8192))

        def _iter() -> Iterable[tuple[str, Any]]:
            seen = 0
            for source_name, row_dict in iter_parquet_structs(parquet_path, batch_size=batch_size):
                name = source_name or f"row_{seen}"
                yield (name, row_dict)
                seen += 1
                if max_members is not None and seen >= max_members:
                    return

        if max_members is not None and total_rows is not None:
            total_rows = min(total_rows, max_members)
        elif max_members is not None and total_rows is None:
            total_rows = max_members
        return _iter(), total_rows

    async def _load_and_parse(
        self,
        collector: list[tuple[str, str, str]] | None = None,
        *,
        persist: bool = True,
    ) -> None:
        if not self.fm.exists(self.source_path):
            sys.exit(f"Missing source file: {self.source_path}")

        logger.info(f"▶ Lendo {self.source_path.name}")
        content: Any = await FileManager.async_read(self.source_path)

        # Specialized fast path for tabular parquet files (vectorized)
        if isinstance(content, ParquetBundle) and content.parsed_kind == "tabular":
            logger.debug("Usando fast-path vetorizado para arquivo tabular…")
            df = content.to_native()
            if not isinstance(df, pl.DataFrame):
                df = pl.read_parquet(content.parsed_parquet_path or content.source_path)

            if self.max_members:
                df = df.head(self.max_members)

            triples = self._vectorized_tabular_to_triples(df)

            if persist:
                self._buffer_triples(triples, collector)
            elif collector is not None:
                collector.extend(triples)
                self._stats.total_triples += len(triples)
            self._stats.total_members += len(df)

            logger.info(
                f" {self._stats.total_members:,} linha(s) processadas – "
                f"{self._stats.total_triples:,} triplas no total (vetorizado)"
            )
            await self._wait_for_tasks()
            return

        members_total: int | None = None
        members: Sequence[tuple[str, Any]] | Iterable[tuple[str, Any]]
        if isinstance(content, ParquetBundle) and content.parsed_kind == "container":
            members = content.iter_entries()
            members_total = content.metadata.get("entries")
            if self.max_members:
                members = itertools.islice(members, self.max_members)
                if members_total is not None:
                    members_total = min(int(members_total), self.max_members)
                else:
                    members_total = self.max_members
        else:
            native = content.to_native() if isinstance(content, ParquetBundle) else content
            if isinstance(native, dict):
                members = list(native.items())
            else:
                members = [(self.source_path.name, native)]
            if self.max_members:
                members = list(members)[: self.max_members]
            members_total = len(members) if isinstance(members, list) else None

        members_list = list(members)
        members_total = len(members_list)

        parsed: list[tuple[str, list[tuple[str, str, str]]]] = []
        use_parallel = self.parallel and persist and members_total > 5000

        if use_parallel:
            logger.debug(f"Processando {members_total} membro(s) em pool…")
            cm = ConcurrencyManager()
            parsed = await cm.execute(
                self._cached_convert,
                [(c, n) for n, c in members_list],
                task_type="process",
                max_workers=self.max_workers,
                desc="Parseando",
            )
        else:
            parsed = [
                self._cached_convert(entry_content, name)
                for name, entry_content in progress_bar(
                    members_list, desc="parseando", total=members_total
                )
            ]

        for _, triples in parsed:
            if persist:
                self._buffer_triples(triples, collector)
            elif collector is not None:
                collector.extend(triples)
                self._stats.total_triples += len(triples)
            self._stats.total_members += 1

        logger.info(
            f" {self._stats.total_members:,} membro(s) processados – "
            f"{self._stats.total_triples:,} triplas no total"
        )
        await self._wait_for_tasks()

    def _vectorized_tabular_to_triples(self, df: pl.DataFrame) -> list[tuple[str, str, str]]:
        """Convert tabular dataframe to triples using vectorized operations."""
        id_col = (
            "id" if "id" in df.columns else ("externalId" if "externalId" in df.columns else None)
        )
        if id_col is None:
            df = df.with_row_index("_subject")
            id_col = "_subject"

        # Filter out private columns and the ID column itself from predicates
        cols = [c for c in df.columns if not c.startswith("_") and c != id_col]

        # Cast all columns to string to support complex types in melt/unpivot
        # For complex types (list/struct), we use json_encode if available
        exprs = []
        for c in df.columns:
            if c.startswith("_"):
                continue

            dtype = df.schema[c]
            if isinstance(dtype, pl.Struct):
                # Vectorized JSON encoding for structs
                exprs.append(pl.col(c).struct.json_encode().alias(c))
            elif isinstance(dtype, pl.List):
                # Robust fallback for lists (less efficient but works)
                exprs.append(pl.col(c).map_elements(str, return_dtype=pl.Utf8).alias(c))
            else:
                exprs.append(pl.col(c).cast(pl.Utf8).alias(c))

        subset = df.select([id_col] + cols).with_columns(exprs)

        # Melt into (s, p, o) format
        melted = subset.melt(id_vars=id_col, variable_name="p", value_name="o")
        melted = melted.rename({id_col: "s"}).drop_nulls()

        # Efficiently clean and filter
        bad_patterns = ["1970-01-01", "9999-12-31"]

        # Ensure all columns are strings and clean whitespace/tabs
        cleaned = melted.with_columns(
            [
                pl.col("s").str.replace("\t", " ").str.strip_chars(),
                pl.col("p").str.replace("\t", " ").str.strip_chars(),
                pl.col("o").str.replace("\t", " ").str.strip_chars(),
            ]
        )

        mask = pl.lit(True)
        for pat in bad_patterns:
            mask = mask & (~pl.col("s").str.contains(pat, literal=True))
            mask = mask & (~pl.col("o").str.contains(pat, literal=True))

        # Return as rows (list of tuples)
        return list(cleaned.filter(mask).iter_rows())

    def _convert_to_triples(self, obj: Any, subject: str) -> tuple[str, list[tuple[str, str, str]]]:
        triples: list[tuple[str, str, str]] = []
        accelerator = LoopAccelerator()

        if isinstance(obj, pl.DataFrame):
            cleaned_df = obj.select(
                [
                    pl.col("s").cast(pl.Utf8).str.replace("\t", " ").str.strip_chars(),
                    pl.col("p").cast(pl.Utf8).str.replace("\t", " ").str.strip_chars(),
                    pl.col("o").cast(pl.Utf8).str.replace("\t", " ").str.strip_chars(),
                ]
            )

            bad_patterns = ["1970-01-01", "9999-12-31"]

            valid_mask = pl.lit(True)
            for col in ["s", "p", "o"]:
                for pat in bad_patterns:
                    valid_mask = valid_mask & (~pl.col(col).str.contains(pat, literal=True))

            filtered_df = cleaned_df.filter(valid_mask)

            triples.extend(filtered_df.rows())
            return subject, triples

        if isinstance(obj, list):

            def _build_from_dict(item: Any) -> tuple[str, str, str] | None:
                if not isinstance(item, dict):
                    return None
                s_val = _clean(str(item.get("s", "")))
                p_val = _clean(str(item.get("p", "")))
                o_val = _clean(str(item.get("o", "")))
                if not (s_val and p_val and o_val):
                    return None
                if (
                    "1970-01-01" in s_val
                    or "9999-12-31" in s_val
                    or "1970-01-01" in p_val
                    or "9999-12-31" in p_val
                    or "1970-01-01" in o_val
                    or "9999-12-31" in o_val
                ):
                    return None
                return (s_val, p_val, o_val)

            triples.extend([t for t in accelerator.map(_build_from_dict, obj) if t])
            return subject, triples

        if isinstance(obj, dict) and {"s", "p", "o"} <= obj.keys():
            s = _clean(str(obj["s"]))
            p = _clean(str(obj["p"]))
            o = _clean(str(obj["o"]))
            if s and p and o:
                if not (
                    "1970-01-01" in s
                    or "9999-12-31" in s
                    or "1970-01-01" in p
                    or "9999-12-31" in p
                    or "1970-01-01" in o
                    or "9999-12-31" in o
                ):
                    triples.append((s, p, o))
            return subject, triples

        if isinstance(obj, dict):
            entity_id = obj.get("id") or obj.get("externalId") or subject
            current = _clean(str(entity_id)) if entity_id else subject

            stack: list[tuple[str, str, Any]] = []
            for key, value in obj.items():
                if not key.startswith("_"):
                    stack.append((current, key, value))

            while stack:
                subj, pred, val = stack.pop()

                if val is None:
                    continue

                if isinstance(val, (str, int, float, bool)):
                    val_str = _clean(str(val))
                    if val_str and val_str not in _SKIP_VALUES:
                        pred_clean = _clean(pred)
                        if not (
                            "1970-01-01" in subj
                            or "9999-12-31" in subj
                            or "1970-01-01" in pred_clean
                            or "9999-12-31" in pred_clean
                            or "1970-01-01" in val_str
                            or "9999-12-31" in val_str
                        ):
                            triples.append((subj, pred_clean, val_str))

                elif isinstance(val, dict):
                    for k, v in val.items():
                        if not k.startswith("_"):
                            stack.append((subj, f"{pred}.{k}", v))

                elif isinstance(val, list):
                    for item in val:
                        stack.append((subj, pred, item))

            return subject, triples

        if not isinstance(obj, str):
            return subject, triples

        lines = obj.splitlines()
        for line in lines:
            if not line or line.strip() in _SKIP_LINES:
                continue
            match = _KV.match(line)
            if match:
                pred, val = match.groups()
                val_clean = _clean(val)
                if val_clean and val_clean not in _SKIP_VALUES:
                    pred_clean = _clean(pred)
                    if not (
                        "1970-01-01" in subject
                        or "9999-12-31" in subject
                        or "1970-01-01" in pred_clean
                        or "9999-12-31" in pred_clean
                        or "1970-01-01" in val_clean
                        or "9999-12-31" in val_clean
                    ):
                        triples.append((subject, pred_clean, val_clean))
        return subject, triples

    async def _serialise(self) -> None:
        await self._flush_all_buffers()

        for split in self.split_ratios:
            output_path = self.output_dir / f"{split}.parquet"
            chunk_files = list(self._staging_dir.glob(f"{split}_*.parquet"))

            if not chunk_files:
                logger.warning(f"Nenhuma tripla para o split {split}")
                continue

            lf = pl.scan_parquet(chunk_files)
            lf.sink_parquet(output_path, compression="lz4", row_group_size=100000)

            logger.info(
                f" Salvo {self._split_counts.get(split, 0)} triplas em {output_path.name} (disco)"
            )

        if self.splits_repo:
            try:
                splits = {}
                for split in self.split_ratios:
                    path = self.output_dir / f"{split}.parquet"
                    if path.exists():
                        splits[split] = pl.read_parquet(path)

                await self.splits_repo.save_splits(
                    train_df=splits.get("train", pl.DataFrame()),
                    valid_df=splits.get("valid", pl.DataFrame()),
                    test_df=splits.get("test", pl.DataFrame()),
                    source=self.source_path.name,
                )
                logger.success(" Splits salvos no PostgreSQL")
            except Exception as exc:
                logger.error(f"Erro ao salvar no repositório de splits: {exc}", exc_info=True)
