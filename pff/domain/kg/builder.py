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
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pff.domain.ports.persistence.kg_ports import KGSplitsPort

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from pff.shared import (
    CacheManager,
    ConcurrencyManager,
    FileManager,
    logger,
    progress_bar,
)
from pff.shared.acceleration.loop_accelerator import LoopAccelerator
from pff.shared.core.config import INGESTION_CONFIG_PATH, settings
from pff.shared.core.config_loader import load_config
from pff.shared.core.file_manager import ParquetBundle

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
    try:
        cfg = load_config(INGESTION_CONFIG_PATH)
        if isinstance(cfg, dict):
            return cfg.get("ingestion", cfg)
    except FileNotFoundError as exc:
        logger.warning(
            f"component=kg_builder event=config_missing path={INGESTION_CONFIG_PATH} error={exc}"
        )
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
        staging_default = cfg.get(
            "temp_output_dir", settings.OUTPUTS_DIR / "temp" / "kg_ingestion"
        )
        ratios_cfg = cfg.get("split_ratios", {"train": 0.8, "valid": 0.1, "test": 0.1})
        batch_size_cfg = cfg.get("batch_size", 50000)
        graph_subdir = cfg.get("graph_subdir", "kg")

        self.source_path = _resolve_path(
            source_path or default_source, base=settings.ROOT_DIR
        )
        resolved_output = _resolve_path(
            output_dir or default_output, base=settings.OUTPUTS_DIR
        )
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
        self.np_rng = np.random.default_rng(seed)

        self._stats = SimpleNamespace(total_members=0, total_triples=0)
        self._split_counts: dict[str, int] = {k: 0 for k in self.split_ratios}

        self._buffers: dict[str, list[pl.DataFrame]] = {
            split: [] for split in self.split_ratios
        }
        self._buffer_counts: dict[str, int] = {split: 0 for split in self.split_ratios}
        self._chunk_indices: dict[str, int] = {split: 0 for split in self.split_ratios}
        self._pending_tasks: list[asyncio.Task] = []

        self._cached_convert = (
            CACHE.disk(ttl=None)(self._convert_to_triples)
            if disk_cache
            else self._convert_to_triples
        )

        self._split_thresholds: list[tuple[float, str]] = []
        acc = 0.0
        sorted_ratios = sorted(self.split_ratios.items())
        for split, ratio in sorted_ratios:
            acc += ratio
            self._split_thresholds.append((acc, split))

    async def run(self) -> None:
        """Execute the full pipeline: load, parse, split, and serialize."""
        await self._load_and_parse()
        await self._serialise()
        logger.success(" Construção finalizada.")

    async def extract_triples(self) -> list[tuple[str, str, str]]:
        """Public helper to load and return triples without disk serialization."""
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

    def _flush_split(self, split: str) -> None:
        dfs = self._buffers.get(split, [])
        if not dfs:
            return

        df = pl.concat(dfs)
        chunk_path = self._staging_dir / f"{split}_{self._chunk_indices[split]}.parquet"

        task = asyncio.create_task(self.fm.async_save(df, chunk_path))
        self._pending_tasks.append(task)

        self._chunk_indices[split] += 1
        self._buffers[split].clear()
        self._buffer_counts[split] = 0

    async def _wait_for_tasks(self) -> None:
        if not self._pending_tasks:
            return
        logger.debug(f"Waiting for {len(self._pending_tasks)} background writes…")
        await asyncio.gather(*self._pending_tasks)
        self._pending_tasks.clear()

    async def _flush_all_buffers(self) -> None:
        for split in self._buffers:
            self._flush_split(split)
        await self._wait_for_tasks()

    def _buffer_triples(
        self,
        triples: list[tuple[str, str, str]] | pl.DataFrame,
        collector: list[tuple[str, str, str]] | None,
    ) -> None:
        if isinstance(triples, list):
            if not triples:
                return
            if collector is not None:
                collector.extend(triples)
                return

            df = pl.DataFrame(triples, schema=["s", "p", "o"], orient="row")
        else:
            df = triples
            if df.is_empty():
                return
            if collector is not None:
                collector.extend(df.iter_rows())
                return

        vals = self.np_rng.random(len(df))
        df = df.with_columns(pl.lit(vals).alias("_rnd"))

        default_split = self._split_thresholds[-1][1]
        expr = None

        if self._split_thresholds:
            first_thresh, first_split = self._split_thresholds[0]
            expr = pl.when(pl.col("_rnd") <= first_thresh).then(pl.lit(first_split))

            for thresh, split in self._split_thresholds[1:]:
                expr = expr.when(pl.col("_rnd") <= thresh).then(pl.lit(split))

            expr = expr.otherwise(pl.lit(default_split)).alias("_split")
        else:
            expr = pl.lit(next(iter(self.split_ratios))).alias("_split")

        df = df.with_columns(expr)
        partitions = df.partition_by("_split", as_dict=True)

        for key_tuple, part_df in partitions.items():
            split = str(key_tuple[0])
            clean_part = part_df.drop(["_rnd", "_split"])
            count = len(clean_part)

            self._split_counts[split] += count

            self._stats.total_triples += count

            self._buffers[split].append(clean_part)
            self._buffer_counts[split] += count

            if self._buffer_counts[split] >= self.chunk_size:
                self._flush_split(split)

    @staticmethod
    def _is_hidden_column(name: str) -> bool:
        return any(part.startswith("_") for part in name.split("."))

    def _flatten_struct_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        while True:
            struct_cols = [
                col for col, dtype in df.schema.items() if isinstance(dtype, pl.Struct)
            ]
            if not struct_cols:
                return df
            for col in struct_cols:
                dtype = df.schema[col]
                field_names = [field.name for field in dtype.fields]
                renamed = [f"{col}.{name}" for name in field_names]
                df = df.with_columns(
                    pl.col(col).struct.rename_fields(renamed).alias(col)
                )
                df = df.unnest(col)

    def _clean_triples_frame(self, df: pl.DataFrame, subject_col: str) -> pl.DataFrame:
        cleaned = df.select(
            [
                pl.col(subject_col).cast(pl.Utf8).str.strip_chars().alias("s"),
                pl.col("p").cast(pl.Utf8).str.strip_chars(),
                pl.col("o").cast(pl.Utf8).str.strip_chars(),
            ]
        )

        def _bad_date(expr: pl.Expr) -> pl.Expr:
            left = expr.str.contains("1970-01-01", literal=True).fill_null(False)
            right = expr.str.contains("9999-12-31", literal=True).fill_null(False)
            return left | right

        mask = (
            pl.col("s").is_not_null()
            & pl.col("p").is_not_null()
            & pl.col("o").is_not_null()
            & (pl.col("s").str.len_chars() > 0)
            & (pl.col("p").str.len_chars() > 0)
            & (pl.col("o").str.len_chars() > 0)
            & (~pl.col("o").is_in(list(_SKIP_VALUES)))
            & (~_bad_date(pl.col("s")))
            & (~_bad_date(pl.col("p")))
            & (~_bad_date(pl.col("o")))
        )

        return cleaned.filter(mask)

    def _collect_triples_frames(
        self, df: pl.DataFrame, *, subject_col: str
    ) -> list[pl.DataFrame]:
        df = self._flatten_struct_columns(df)

        drop_cols = [
            col
            for col in df.columns
            if col != subject_col and self._is_hidden_column(col)
        ]
        if drop_cols:
            df = df.drop(drop_cols)

        list_cols = [
            col
            for col, dtype in df.schema.items()
            if col != subject_col and isinstance(dtype, pl.List)
        ]

        frames: list[pl.DataFrame] = []

        for col in list_cols:
            dtype = df.schema.get(col)
            if isinstance(dtype, pl.List) and isinstance(dtype.inner, pl.Struct):
                base = df.select([subject_col, pl.col(col).alias(col)]).with_columns(
                    pl.int_ranges(0, pl.col(col).list.len()).alias("_idx")
                )
                exploded = base.explode([col, "_idx"]).filter(pl.col(col).is_not_null())
                if not exploded.is_empty():
                    struct_fields = [field.name for field in dtype.inner.fields]
                    item_id_candidates = []
                    if "id" in struct_fields:
                        item_id_candidates.append(
                            pl.col(col).struct.field("id").cast(pl.Utf8)
                        )
                    if "externalId" in struct_fields:
                        item_id_candidates.append(
                            pl.col(col).struct.field("externalId").cast(pl.Utf8)
                        )
                    item_id_candidates.append(
                        pl.col(subject_col).cast(pl.Utf8)
                        + pl.lit(f"_{col}_")
                        + pl.col("_idx").cast(pl.Utf8)
                    )

                    item_id_expr = pl.coalesce(item_id_candidates).alias("_item_id")
                    exploded = exploded.with_columns(item_id_expr)

                    edges = exploded.select(
                        [
                            pl.col(subject_col),
                            pl.lit(col).alias("p"),
                            pl.col("_item_id").alias("o"),
                        ]
                    )
                    frames.append(self._clean_triples_frame(edges, subject_col))

                    item_df = exploded.select(
                        [pl.col("_item_id").alias(subject_col)]
                        + [
                            pl.col(col).struct.field(name).alias(name)
                            for name in struct_fields
                        ]
                    )
                    frames.extend(
                        self._collect_triples_frames(item_df, subject_col=subject_col)
                    )
                continue

            exploded = df.select([subject_col, pl.col(col).alias(col)]).explode(col)
            exploded = exploded.filter(pl.col(col).is_not_null())
            frames.extend(
                self._collect_triples_frames(exploded, subject_col=subject_col)
            )

        if list_cols:
            df = df.drop(list_cols)

        primitive_cols = [col for col in df.columns if col != subject_col]
        if primitive_cols:
            melted = df.unpivot(
                index=subject_col,
                on=primitive_cols,
                variable_name="p",
                value_name="o",
            )
            frames.append(self._clean_triples_frame(melted, subject_col))

        return frames

    def _vectorized_entity_to_triples(self, df: pl.DataFrame) -> pl.DataFrame:
        if "_row_id" not in df.columns:
            df = df.with_row_index("_row_id")

        subject_candidates = []
        for col in ("id", "externalId", "_source_name"):
            if col in df.columns:
                subject_candidates.append(pl.col(col).cast(pl.Utf8))
        subject_candidates.append(pl.col("_row_id").cast(pl.Utf8))

        df = df.with_columns(pl.coalesce(subject_candidates).alias("_subject"))

        frames = self._collect_triples_frames(df, subject_col="_subject")
        if not frames:
            return pl.DataFrame(schema=["s", "p", "o"])
        return pl.concat(frames, how="vertical")

    def _vectorized_triples_from_columns(
        self,
        df: pl.DataFrame,
        *,
        column_map: Mapping[str, str] | None = None,
    ) -> pl.DataFrame:
        if column_map:
            df = df.rename(column_map)
        df = df.select(["s", "p", "o"])
        return self._clean_triples_frame(df, "s")

    async def _load_and_parse(
        self,
        collector: list[tuple[str, str, str]] | None = None,
        *,
        persist: bool = True,
    ) -> None:
        if not self.fm.exists(self.source_path):
            sys.exit(f"Missing source file: {self.source_path}")

        logger.info(f"▶ Lendo {self.source_path.name}")

        ext = self.source_path.suffix.lower()
        if ext in (".tsv", ".csv", ".txt"):
            sep = "\t" if ext in (".tsv", ".txt") else ","
            df = self.fm.read(
                self.source_path,
                separator=sep,
                has_header=False,
                new_columns=["s", "p", "o"],
                quote_char=None,
                ignore_errors=True,
                return_native=True,
            )

            if not isinstance(df, pl.DataFrame):
                logger.error(f"Failed to load CSV/TSV: unexpected format {type(df)}")
                return

            df = df.select(
                [
                    pl.col("s").cast(pl.Utf8).str.strip_chars(),
                    pl.col("p").cast(pl.Utf8).str.strip_chars(),
                    pl.col("o").cast(pl.Utf8).str.strip_chars(),
                ]
            ).drop_nulls()

            if self.max_members:
                df = df.head(self.max_members)

            if persist:
                self._buffer_triples(df, collector)
            elif collector is not None:
                collector.extend(df.iter_rows())
                self._stats.total_triples += len(df)

            self._stats.total_members += len(df)
            logger.info(f" {len(df):,} triplas carregadas via Polars CSV engine")
            await self._wait_for_tasks()
            return

        if ext in (".jsonl", ".ndjson"):
            df = self.fm.read(self.source_path, return_native=True)

            if not isinstance(df, pl.DataFrame):
                logger.error(f"Failed to load NDJSON: unexpected format {type(df)}")
                return

            if {"s", "p", "o"}.issubset(set(df.columns)):
                df = df.select(
                    [
                        pl.col("s").cast(pl.Utf8).str.strip_chars(),
                        pl.col("p").cast(pl.Utf8).str.strip_chars(),
                        pl.col("o").cast(pl.Utf8).str.strip_chars(),
                    ]
                ).drop_nulls()

                if self.max_members:
                    df = df.head(self.max_members)

                if persist:
                    self._buffer_triples(df, collector)
                elif collector is not None:
                    collector.extend(df.iter_rows())
                    self._stats.total_triples += len(df)

                self._stats.total_members += len(df)
                logger.info(f" {len(df):,} triplas carregadas via Polars NDJSON engine")
            await self._wait_for_tasks()
            return

        content: Any = await FileManager.async_read(self.source_path)

        members_total: int | None = None
        members: Sequence[tuple[str, Any]] | Iterable[tuple[str, Any]]
        if isinstance(content, ParquetBundle) and content.parsed_kind == "tabular":
            parquet_path = content.parsed_parquet_path or content.source_path
            await self._load_parquet_tabular(
                parquet_path,
                collector=collector,
                persist=persist,
            )
            await self._wait_for_tasks()
            return
        elif isinstance(content, ParquetBundle) and content.parsed_kind == "container":
            members = content.iter_entries()
            members_total = content.metadata.get("entries")
            if self.max_members:
                members = itertools.islice(members, self.max_members)
                if members_total is not None:
                    members_total = min(int(members_total), self.max_members)
                else:
                    members_total = self.max_members
        else:
            native = (
                content.to_native() if isinstance(content, ParquetBundle) else content
            )
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
            logger.debug(f"Processing {members_total} member(s) in pool…")
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

    async def _load_parquet_tabular(
        self,
        parquet_path: Path,
        *,
        collector: list[tuple[str, str, str]] | None,
        persist: bool,
    ) -> None:
        parquet_file = pq.ParquetFile(parquet_path)
        schema = parquet_file.schema_arrow
        schema_names = set(schema.names)

        has_spo = {"s", "p", "o"}.issubset(schema_names)
        has_hrt = {"head", "relation", "tail"}.issubset(schema_names)
        has_raw_json = "_raw_json" in schema_names
        has_struct = any(
            pa.types.is_struct(field.type)
            or pa.types.is_list(field.type)
            or pa.types.is_large_list(field.type)
            for field in schema
        )

        if has_spo:
            columns = ["s", "p", "o"]
            column_map = None
        elif has_hrt:
            columns = ["head", "relation", "tail"]
            column_map = {"head": "s", "relation": "p", "tail": "o"}
        elif has_raw_json and not has_struct:
            columns = ["_raw_json"]
            if "_parse_error" in schema_names:
                columns.append("_parse_error")
            if "_source_name" in schema_names:
                columns.append("_source_name")
            column_map = None
        else:
            columns = [name for name in schema.names if name != "_raw_json"]
            column_map = None

        batch_size = max(1024, min(self.chunk_size, 8192))
        remaining = self.max_members

        for batch in parquet_file.iter_batches(columns=columns, batch_size=batch_size):
            df = pl.from_arrow(batch, rechunk=False)
            if "_parse_error" in df.columns:
                df = df.filter(pl.col("_parse_error").is_null())
            if "_raw_json" in df.columns:
                df = df.filter(pl.col("_raw_json").is_not_null())

            if remaining is not None:
                df = df.head(remaining)

            if df.is_empty():
                if remaining is not None and remaining <= 0:
                    break
                continue

            remaining = remaining - len(df) if remaining is not None else None

            if has_spo or has_hrt:
                triples = self._vectorized_triples_from_columns(
                    df, column_map=column_map
                )
            else:
                rowwise_df: pl.DataFrame | None = None
                if "_raw_json" in df.columns:
                    decoded = df.with_columns(
                        pl.col("_raw_json")
                        .str.json_decode(dtype=pl.Struct)
                        .alias("_decoded")
                    )
                    dtype = decoded.schema["_decoded"]
                    if not isinstance(dtype, pl.Struct):
                        raise ValueError(
                            f"JSON decode did not result in Struct (path={parquet_path})"
                        )
                    decoded = decoded.drop(["_raw_json"]).unnest("_decoded")
                    decoded_columns = set(decoded.columns)
                    raw_columns = set(df.columns) - {"_raw_json"}
                    if decoded_columns.issubset(raw_columns):
                        rowwise_df = df
                    elif {"s", "p", "o"}.issubset(decoded_columns):
                        triples = self._vectorized_triples_from_columns(decoded)
                    elif {"head", "relation", "tail"}.issubset(decoded_columns):
                        triples = self._vectorized_triples_from_columns(
                            decoded,
                            column_map={"head": "s", "relation": "p", "tail": "o"},
                        )
                    else:
                        decoded_has_struct = any(
                            isinstance(dtype, (pl.Struct, pl.List))
                            for dtype in decoded.schema.values()
                        )
                        if decoded_has_struct:
                            rowwise_df = decoded
                        else:
                            triples = self._vectorized_entity_to_triples(decoded)
                else:
                    if has_struct:
                        rowwise_df = df
                    else:
                        triples = self._vectorized_entity_to_triples(df)

                if rowwise_df is not None:
                    batch_triples: list[tuple[str, str, str]] = []
                    base_index = self._stats.total_members
                    for offset, row in enumerate(rowwise_df.iter_rows(named=True)):
                        raw_json = (
                            row.get("_raw_json") if isinstance(row, dict) else None
                        )
                        if isinstance(raw_json, str):
                            try:
                                row = FileManager.json_loads(raw_json)
                            except Exception:
                                row = raw_json
                        _, row_triples = self._cached_convert(
                            row, f"row_{base_index + offset}"
                        )
                        if row_triples:
                            batch_triples.extend(row_triples)
                    triples = batch_triples

            if persist:
                self._buffer_triples(triples, collector)
            elif collector is not None:
                if isinstance(triples, list):
                    collector.extend(triples)
                    self._stats.total_triples += len(triples)
                else:
                    collector.extend(triples.iter_rows())
                    self._stats.total_triples += len(triples)

            self._stats.total_members += len(df)

            if remaining is not None and remaining <= 0:
                break

        logger.info(
            f" {self._stats.total_members:,} linha(s) processadas – "
            f"{self._stats.total_triples:,} triplas no total"
        )

    def _convert_to_triples(
        self, obj: Any, subject: str
    ) -> tuple[str, list[tuple[str, str, str]]]:
        triples: list[tuple[str, str, str]] = []
        clean = _clean
        is_instance = isinstance

        if is_instance(obj, str):
            trimmed = obj.strip()
            if (trimmed.startswith("{") and trimmed.endswith("}")) or (
                trimmed.startswith("[") and trimmed.endswith("]")
            ):
                try:
                    obj = FileManager.json_loads(trimmed)
                except Exception:
                    pass

        if is_instance(obj, pl.DataFrame):
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
                    valid_mask = valid_mask & (
                        ~pl.col(col).str.contains(pat, literal=True)
                    )

            filtered_df = cleaned_df.filter(valid_mask)

            triples.extend(filtered_df.rows())
            return subject, triples

        if is_instance(obj, list):
            accelerator = LoopAccelerator()

            def _build_from_dict(item: Any) -> tuple[str, str, str] | None:
                if not is_instance(item, dict):
                    return None
                s_val = clean(str(item.get("s", "")))
                p_val = clean(str(item.get("p", "")))
                o_val = clean(str(item.get("o", "")))
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

        if is_instance(obj, dict) and {"s", "p", "o"} <= obj.keys():
            s = clean(str(obj["s"]))
            p = clean(str(obj["p"]))
            o = clean(str(obj["o"]))
            if s and p and o:
                triples.append((s, p, o))
            return subject, triples

        if is_instance(obj, dict):
            entity_id = obj.get("id") or obj.get("externalId") or subject
            current = (
                (
                    str(entity_id).strip()
                    if not is_instance(entity_id, str)
                    else entity_id.strip()
                )
                if entity_id
                else subject
            )

            stack: list[tuple[str, str, Any]] = [
                (current, k, v) for k, v in obj.items() if k[:1] != "_"
            ]
            triples_append = triples.append

            _BAD_1 = "1970-01-01"
            _BAD_2 = "9999-12-31"

            while stack:
                subj, pred, val = stack.pop()

                if val is None:
                    continue

                if is_instance(val, str):
                    val_str = val.strip()
                    if val_str and val_str not in _SKIP_VALUES:
                        pred_clean = pred.strip()
                        if not (
                            _BAD_1 in subj
                            or _BAD_2 in subj
                            or _BAD_1 in pred_clean
                            or _BAD_2 in pred_clean
                            or _BAD_1 in val_str
                            or _BAD_2 in val_str
                        ):
                            triples_append((subj, pred_clean, val_str))
                elif (
                    is_instance(val, bool)
                    or is_instance(val, int)
                    or is_instance(val, float)
                ):
                    pred_clean = pred.strip()
                    if not (
                        _BAD_1 in subj
                        or _BAD_2 in subj
                        or _BAD_1 in pred_clean
                        or _BAD_2 in pred_clean
                    ):
                        triples_append((subj, pred_clean, str(val)))

                elif is_instance(val, dict):
                    for k, v in val.items():
                        if k[:1] != "_":
                            stack.append((subj, f"{pred}.{k}", v))

                elif is_instance(val, list):
                    for idx, item in enumerate(val):
                        if is_instance(item, dict):
                            item_id = (
                                item.get("id")
                                or item.get("externalId")
                                or f"{subj}_{pred}_{idx}"
                            )
                            item_subj = (
                                str(item_id).strip()
                                if not is_instance(item_id, str)
                                else item_id.strip()
                            )
                            pred_clean = pred.strip()
                            if item_subj and pred_clean:
                                if not (
                                    _BAD_1 in subj
                                    or _BAD_2 in subj
                                    or _BAD_1 in pred_clean
                                    or _BAD_2 in pred_clean
                                    or _BAD_1 in item_subj
                                    or _BAD_2 in item_subj
                                ):
                                    triples_append((subj, pred_clean, item_subj))
                            for k, v in item.items():
                                if k[:1] != "_":
                                    stack.append((item_subj, k, v))
                        else:
                            stack.append((subj, pred, item))

            return subject, triples

        if not is_instance(obj, str):
            return subject, triples

        lines = obj.splitlines()
        for line in lines:
            trimmed_line = line.strip()
            if not trimmed_line or trimmed_line in _SKIP_LINES:
                continue

            parts = trimmed_line.split("\t")
            if len(parts) == 3:
                s_val, p_val, o_val = [clean(p) for p in parts]
                if s_val and p_val and o_val:
                    triples.append((s_val, p_val, o_val))
                continue

            if trimmed_line.startswith("{") and trimmed_line.endswith("}"):
                try:
                    line_obj = FileManager.json_loads(trimmed_line)
                    if (
                        is_instance(line_obj, dict)
                        and {"s", "p", "o"} <= line_obj.keys()
                    ):
                        s = clean(str(line_obj["s"]))
                        p = clean(str(line_obj["p"]))
                        o = clean(str(line_obj["o"]))
                        if s and p and o:
                            triples.append((s, p, o))
                        continue
                except Exception:
                    pass

            match = _KV.match(line)
            if match:
                pred, val = match.groups()
                val_clean = clean(val)
                if val_clean and val_clean not in _SKIP_VALUES:
                    pred_clean = clean(pred)
                    if not (
                        "1970-01-01" in subject
                        or "9999-12-31" in subject
                        or "1970-01-01" in pred_clean
                        or "9999-12-31" in pred_clean
                        or "1970-01-01" in val_clean
                        or "9999-12-31" in val_clean
                    ):
                        triples.append((subject, pred_clean, val_clean))

        if triples:
            logger.debug(
                f"Extracted {len(triples)} triples from string member {subject}"
            )
        return subject, triples

    async def _serialise(self) -> None:
        await self._flush_all_buffers()

        for split in self.split_ratios:
            output_path = self.output_dir / f"{split}.parquet"
            chunk_files = list(self._staging_dir.glob(f"{split}_*.parquet"))

            if not chunk_files:
                logger.warning(f"No triples for split {split}")
                continue

            lf = pl.scan_parquet(chunk_files)
            lf.sink_parquet(output_path, compression="lz4", row_group_size=100000)

            logger.info(
                f" Salvo {self._split_counts.get(split, 0)} triplas em {output_path.name} (disco)"
            )

        if self.splits_repo:
            try:
                for split in self.split_ratios:
                    path = self.output_dir / f"{split}.parquet"
                    if self.fm.exists(path):
                        df = self.fm.read(path, return_native=True)
                        await self.splits_repo.save_split(
                            split_name=split, df=df, split_type="raw"
                        )
                logger.success(
                    "component_name=kg_builder stop_reason=step_completion message='Splits salvos no PostgreSQL'"
                )
            except Exception as exc:
                logger.error(
                    f"Failed to save to splits repository: {exc}", exc_info=True
                )

        stats = {
            "total_members": self._stats.total_members,
            "total_triples": self._stats.total_triples,
            "splits": self._split_counts,
            "ratios": self.split_ratios,
            "timestamp": datetime.now().isoformat(),
        }
        self.fm.save(stats, self.output_dir / "stats.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="PFF Knowledge Graph Builder")
    parser.add_argument(
        "source", nargs="?", help="Caminho para o arquivo ou diretório fonte"
    )
    parser.add_argument("--output", "-o", help="Diretório de saída")
    parser.add_argument(
        "--max-members", "-n", type=int, help="Limite de membros a processar"
    )
    parser.add_argument(
        "--no-parallel", action="store_true", help="Desativa processamento paralelo"
    )
    parser.add_argument("--workers", "-w", type=int, help="Número de workers")
    parser.add_argument(
        "--disk-cache", action="store_true", help="Ativa cache em disco"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed para reprodutibilidade"
    )

    ns = parser.parse_args()

    builder = KGBuilder(
        source_path=ns.source,
        output_dir=ns.output,
        max_members=ns.max_members,
        parallel=not ns.no_parallel,
        workers=ns.workers,
        disk_cache=ns.disk_cache,
        seed=ns.seed,
    )

    asyncio.run(builder.run())


if __name__ == "__main__":
    main()
