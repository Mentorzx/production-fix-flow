"""Provide module-level functionality for the PFF codebase.



Notes:

    File: src/pff/domain/kg/builder.py

"""

from __future__ import annotations

import argparse
import asyncio
import itertools
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
import orjson
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from pff_rust import convert_to_triples as rust_convert_to_triples

from pff.shared import (
    CacheManager,
    ConcurrencyManager,
    FileManager,
    logger,
    progress_bar,
)
from pff.shared.system.probe import get_safe_cpu_count
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
    """Execute clean.



    Args:

        text: Input value used by this callable.



    Returns:

        Return value produced by the callable.

    """

    if "\t" not in text:
        return text.strip()
    return text.replace("\t", " ").strip()


def _load_ingestion_config() -> dict[str, Any]:
    """Execute load ingestion config.



    Returns:

        Return value produced by the callable.

    """

    try:
        cfg = load_config(INGESTION_CONFIG_PATH)
        return cfg.get("ingestion", cfg)  # type: ignore[return-value, no-any-return]
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
    """Execute resolve path.



    Args:

        path_like: Input value used by this callable.

        base: Input value used by this callable.



    Returns:

        Return value produced by the callable.

    """

    candidate = Path(path_like)
    if not candidate.is_absolute():
        candidate = base / candidate
    return candidate.expanduser().resolve()


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
        file_manager: FileManager | None = None,
        cache_manager: CacheManager | None = None,
    ) -> None:
        """Execute init.



        Args:

            source_path: Input value used by this callable.

            output_dir: Input value used by this callable.

            max_members: Optional input value.

            parallel: Optional input value.

            workers: Optional input value.

            disk_cache: Optional input value.

            splits_repo: Optional input value.

            seed: Optional input value.

            file_manager: Optional input value.

            cache_manager: Optional input value.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

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
        self.fm = file_manager or FileManager()
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
        self.fm.ensure_dir(self.output_dir)

        staging_dir = _resolve_path(staging_default, base=settings.OUTPUTS_DIR)
        if not staging_dir.is_relative_to(settings.OUTPUTS_DIR):
            staging_dir = settings.OUTPUTS_DIR / staging_dir.name
        self._staging_dir = staging_dir
        self.fm.ensure_dir(self._staging_dir)
        self.max_members = max_members
        self.parallel = parallel
        self.max_workers = workers or min(get_safe_cpu_count(logical=True), 8)
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
        self._cache_manager = cache_manager or CacheManager()

        self._cached_convert = (
            self._cache_manager.disk(ttl=None)(self._convert_to_triples)
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
        """Execute normalize ratios.



        Args:

            ratios: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

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
        """Execute flush split.



        Args:

            split: Input value used by this callable.

        """

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
        """Execute wait for tasks."""

        if not self._pending_tasks:
            return
        logger.debug(f"Waiting for {len(self._pending_tasks)} background writes…")
        await asyncio.gather(*self._pending_tasks)
        self._pending_tasks.clear()

    async def _flush_all_buffers(self) -> None:
        """Execute flush all buffers."""

        for split in self._buffers:
            self._flush_split(split)
        await self._wait_for_tasks()

    def _buffer_triples(
        self,
        triples: list[tuple[str, str, str]] | pl.DataFrame,
        collector: list[tuple[str, str, str]] | None,
    ) -> None:
        """Execute buffer triples.



        Args:

            triples: Input value used by this callable.

            collector: Input value used by this callable.

        """

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
                expr = expr.when(pl.col("_rnd") <= thresh).then(pl.lit(split))  # type: ignore[assignment]

            expr = expr.otherwise(pl.lit(default_split)).alias("_split")  # type: ignore[assignment]
        else:
            expr = pl.lit(next(iter(self.split_ratios))).alias("_split")  # type: ignore[assignment]

        df = df.with_columns(expr)
        partitions = df.partition_by("_split", as_dict=True)

        for key_tuple, part_df in partitions.items():  # type: ignore[assignment]
            split = str(key_tuple[0])
            clean_part = part_df.drop(["_rnd", "_split"])  # type: ignore[assignment]
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
        """Execute flatten struct columns.



        Args:

            df: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        while True:
            struct_cols = [
                col for col, dtype in df.schema.items() if isinstance(dtype, pl.Struct)
            ]
            if not struct_cols:
                return df
            for col in struct_cols:
                dtype = df.schema[col]
                field_names = [field.name for field in dtype.fields]  # type: ignore[attr-defined]
                renamed = [f"{col}.{name}" for name in field_names]
                df = df.with_columns(
                    pl.col(col).struct.rename_fields(renamed).alias(col)
                )
                df = df.unnest(col)

    def _clean_triples_frame(self, df: pl.DataFrame, subject_col: str) -> pl.DataFrame:
        """Execute clean triples frame.



        Args:

            df: Input value used by this callable.

            subject_col: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

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
        """Execute collect triples frames.



        Args:

            df: Input value used by this callable.

            subject_col: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

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
        """Execute vectorized entity to triples.



        Args:

            df: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

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
        """Execute vectorized triples from columns.



        Args:

            df: Input value used by this callable.

            column_map: Optional input value.



        Returns:

            Return value produced by the callable.

        """

        if column_map:
            df = df.rename(column_map)
        df = df.select(["s", "p", "o"])
        return self._clean_triples_frame(df, "s")

    def _commit_triples_batch(
        self,
        triples: pl.DataFrame | list[tuple[str, str, str]],
        collector: list[tuple[str, str, str]] | None,
        *,
        persist: bool,
    ) -> None:
        """Execute commit triples batch.



        Args:

            triples: Input value used by this callable.

            collector: Input value used by this callable.

            persist: Input value used by this callable.

        """

        if persist:
            self._buffer_triples(triples, collector)
            return

        if collector is None:
            return

        if isinstance(triples, list):
            collector.extend(triples)
            self._stats.total_triples += len(triples)
            return

        collector.extend(triples.iter_rows())
        self._stats.total_triples += len(triples)

    def _process_delimited_source(
        self,
        *,
        separator: str,
        collector: list[tuple[str, str, str]] | None,
        persist: bool,
    ) -> bool:
        """Execute process delimited source.



        Args:

            separator: Input value used by this callable.

            collector: Input value used by this callable.

            persist: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        df = self.fm.read(
            self.source_path,
            separator=separator,
            has_header=False,
            new_columns=["s", "p", "o"],
            quote_char=None,
            ignore_errors=True,
            return_native=True,
        )

        if not isinstance(df, pl.DataFrame):
            logger.error(f"Failed to load CSV/TSV: unexpected format {type(df)}")
            return True

        df = df.select(
            [
                pl.col("s").cast(pl.Utf8).str.strip_chars(),
                pl.col("p").cast(pl.Utf8).str.strip_chars(),
                pl.col("o").cast(pl.Utf8).str.strip_chars(),
            ]
        ).drop_nulls()

        if self.max_members:
            df = df.head(self.max_members)

        self._commit_triples_batch(df, collector, persist=persist)
        self._stats.total_members += len(df)
        logger.info(f" {len(df):,} triplas carregadas via Polars CSV engine")
        return True

    def _process_ndjson_source(
        self,
        collector: list[tuple[str, str, str]] | None,
        *,
        persist: bool,
    ) -> bool:
        """Execute process ndjson source.



        Args:

            collector: Input value used by this callable.

            persist: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        df = self.fm.read(self.source_path, return_native=True)

        if not isinstance(df, pl.DataFrame):
            logger.error(f"Failed to load NDJSON: unexpected format {type(df)}")
            return True

        if not {"s", "p", "o"}.issubset(set(df.columns)):
            return True

        df = df.select(
            [
                pl.col("s").cast(pl.Utf8).str.strip_chars(),
                pl.col("p").cast(pl.Utf8).str.strip_chars(),
                pl.col("o").cast(pl.Utf8).str.strip_chars(),
            ]
        ).drop_nulls()

        if self.max_members:
            df = df.head(self.max_members)

        self._commit_triples_batch(df, collector, persist=persist)
        self._stats.total_members += len(df)
        logger.info(f" {len(df):,} triplas carregadas via Polars NDJSON engine")
        return True

    async def _load_members_content(
        self,
        collector: list[tuple[str, str, str]] | None,
        *,
        persist: bool,
    ) -> tuple[list[tuple[str, Any]], int]:
        """Execute load members content.



        Args:

            collector: Input value used by this callable.

            persist: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        content: Any = await self.fm.async_read(self.source_path)
        members_total: int | None = None
        members: Sequence[tuple[str, Any]] | Iterable[tuple[str, Any]]

        if isinstance(content, ParquetBundle) and content.parsed_kind == "tabular":
            parquet_path = content.parsed_parquet_path or content.source_path
            await self._load_parquet_tabular(
                parquet_path,
                collector=collector,
                persist=persist,
            )
            return [], 0

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
            native = (
                content.to_native() if isinstance(content, ParquetBundle) else content
            )
            if isinstance(native, dict):
                members = list(native.items())
            else:
                members = [(self.source_path.name, native)]
            if self.max_members:
                members = list(members)[: self.max_members]
            members_total = len(members)

        members_list = list(members)
        if members_total is None:
            members_total = len(members_list)
        return members_list, members_total

    async def _parse_members(
        self,
        members_list: list[tuple[str, Any]],
        members_total: int,
        collector: list[tuple[str, str, str]] | None,
        *,
        persist: bool,
    ) -> None:
        """Execute parse members.



        Args:

            members_list: Input value used by this callable.

            members_total: Input value used by this callable.

            collector: Input value used by this callable.

            persist: Input value used by this callable.

        """

        use_parallel = self.parallel and persist and members_total > 5000
        parsed: list[tuple[str, list[tuple[str, str, str]]]]

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
            self._commit_triples_batch(triples, collector, persist=persist)
            self._stats.total_members += 1

    async def _load_and_parse(
        self,
        collector: list[tuple[str, str, str]] | None = None,
        *,
        persist: bool = True,
    ) -> None:
        """Execute load and parse.



        Args:

            collector: Optional input value.

            persist: Optional input value.

        """

        if not self.fm.exists(self.source_path):
            sys.exit(f"Missing source file: {self.source_path}")

        logger.info(f"▶ Lendo {self.source_path.name}")

        ext = self.source_path.suffix.lower()
        if ext in (".tsv", ".csv", ".txt"):
            separator = "\t" if ext in (".tsv", ".txt") else ","
            self._process_delimited_source(
                separator=separator,
                collector=collector,
                persist=persist,
            )
            await self._wait_for_tasks()
            return

        if ext in (".jsonl", ".ndjson"):
            self._process_ndjson_source(collector, persist=persist)
            await self._wait_for_tasks()
            return

        members_list, members_total = await self._load_members_content(
            collector, persist=persist
        )
        if not members_list and members_total == 0:
            await self._wait_for_tasks()
            return

        await self._parse_members(
            members_list,
            members_total,
            collector,
            persist=persist,
        )

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
        """Execute load parquet tabular.



        Args:

            parquet_path: Input value used by this callable.

            collector: Input value used by this callable.

            persist: Input value used by this callable.

        """

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
        columns, column_map = self._resolve_parquet_columns(
            schema=schema,
            schema_names=schema_names,
            has_spo=has_spo,
            has_hrt=has_hrt,
            has_raw_json=has_raw_json,
            has_struct=has_struct,
        )

        batch_size = max(1024, min(self.chunk_size, 8192))
        remaining = self.max_members

        for batch in parquet_file.iter_batches(columns=columns, batch_size=batch_size):
            df_raw = pl.from_arrow(batch, rechunk=False)
            df = df_raw if isinstance(df_raw, pl.DataFrame) else df_raw.to_frame()
            if "_parse_error" in df.columns:  # type: ignore[union-attr]
                df = df.filter(pl.col("_parse_error").is_null())  # type: ignore[arg-type]
            if "_raw_json" in df.columns:  # type: ignore[union-attr]
                df = df.filter(pl.col("_raw_json").is_not_null())  # type: ignore[arg-type]

            if remaining is not None:
                df = df.head(remaining)

            if df.is_empty():
                if remaining is not None and remaining <= 0:
                    break
                continue

            remaining = remaining - len(df) if remaining is not None else None

            if has_spo or has_hrt:
                triples = self._vectorized_triples_from_columns(
                    df,
                    column_map=column_map,  # type: ignore[arg-type]
                )
            else:
                triples = self._extract_non_tabular_triples(
                    df=df,
                    parquet_path=parquet_path,
                    has_struct=has_struct,
                )

            self._commit_triples_batch(triples, collector, persist=persist)

            self._stats.total_members += len(df)

            if remaining is not None and remaining <= 0:
                break

        logger.info(
            f" {self._stats.total_members:,} linha(s) processadas – "
            f"{self._stats.total_triples:,} triplas no total"
        )

    @staticmethod
    def _resolve_parquet_columns(
        *,
        schema: pa.Schema,
        schema_names: set[str],
        has_spo: bool,
        has_hrt: bool,
        has_raw_json: bool,
        has_struct: bool,
    ) -> tuple[list[str], Mapping[str, str] | None]:
        """Execute resolve parquet columns.



        Args:

            schema: Input value used by this callable.

            schema_names: Input value used by this callable.

            has_spo: Input value used by this callable.

            has_hrt: Input value used by this callable.

            has_raw_json: Input value used by this callable.

            has_struct: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        if has_spo:
            return ["s", "p", "o"], None

        if has_hrt:
            return ["head", "relation", "tail"], {
                "head": "s",
                "relation": "p",
                "tail": "o",
            }

        if has_raw_json and not has_struct:
            columns = ["_raw_json"]
            if "_parse_error" in schema_names:
                columns.append("_parse_error")
            if "_source_name" in schema_names:
                columns.append("_source_name")
            return columns, None

        return [name for name in schema.names if name != "_raw_json"], None

    def _extract_non_tabular_triples(
        self,
        *,
        df: pl.DataFrame,
        parquet_path: Path,
        has_struct: bool,
    ) -> pl.DataFrame | list[tuple[str, str, str]]:
        """Execute extract non tabular triples.



        Args:

            df: Input value used by this callable.

            parquet_path: Input value used by this callable.

            has_struct: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        if "_raw_json" in df.columns:
            return self._extract_from_raw_json_batch(df, parquet_path=parquet_path)
        if has_struct:
            return self._extract_rowwise_triples(df)
        return self._vectorized_entity_to_triples(df)

    def _extract_from_raw_json_batch(
        self,
        df: pl.DataFrame,
        *,
        parquet_path: Path,
    ) -> pl.DataFrame | list[tuple[str, str, str]]:
        """Execute extract from raw json batch.



        Args:

            df: Input value used by this callable.

            parquet_path: Input value used by this callable.



        Returns:

            Return value produced by the callable.



        Raises:

            Exception: Propagates domain-specific failures with context.

        """

        decoded = df.with_columns(
            pl.col("_raw_json").str.json_decode(dtype=pl.Struct).alias("_decoded")
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
            return self._extract_rowwise_triples(df)
        if {"s", "p", "o"}.issubset(decoded_columns):
            return self._vectorized_triples_from_columns(decoded)
        if {"head", "relation", "tail"}.issubset(decoded_columns):
            return self._vectorized_triples_from_columns(
                decoded,
                column_map={"head": "s", "relation": "p", "tail": "o"},
            )

        decoded_has_struct = any(
            isinstance(column_dtype, (pl.Struct, pl.List))
            for column_dtype in decoded.schema.values()
        )
        if decoded_has_struct:
            return self._extract_rowwise_triples(decoded)
        return self._vectorized_entity_to_triples(decoded)

    def _extract_rowwise_triples(
        self, rowwise_df: pl.DataFrame
    ) -> list[tuple[str, str, str]]:
        """Execute extract rowwise triples.



        Args:

            rowwise_df: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        batch_triples: list[tuple[str, str, str]] = []
        base_index = self._stats.total_members
        has_raw_json = "_raw_json" in rowwise_df.columns
        for offset, row in enumerate(rowwise_df.iter_rows(named=True)):
            payload = self._normalize_row_payload(row) if has_raw_json else row
            _, row_triples = self._cached_convert(payload, f"row_{base_index + offset}")
            if row_triples:
                batch_triples.extend(row_triples)
        return batch_triples

    def _normalize_row_payload(self, row: Any) -> Any:
        """Execute normalize row payload.



        Args:

            row: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        raw_json = row.get("_raw_json") if isinstance(row, dict) else None
        if not isinstance(raw_json, str):
            return row
        try:
            return self.fm.json_loads(raw_json)
        except Exception:
            return raw_json

    @staticmethod
    def _has_blocked_date(value: str) -> bool:
        return "1970-01-01" in value or "9999-12-31" in value

    def _extract_triples_from_dataframe(
        self, obj: pl.DataFrame
    ) -> list[tuple[str, str, str]]:
        """Execute extract triples from dataframe.



        Args:

            obj: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

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

        return cleaned_df.filter(valid_mask).rows()

    def _extract_triples_from_list(self, obj: list[Any]) -> list[tuple[str, str, str]]:
        """Execute extract triples from list.



        Args:

            obj: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        clean = _clean

        def _build_from_dict(item: Any) -> tuple[str, str, str] | None:
            """Execute build from dict.



            Args:

                item: Input value used by this callable.



            Returns:

                Return value produced by the callable.

            """

            if not isinstance(item, dict):
                return None
            s_val = clean(str(item.get("s", "")))
            p_val = clean(str(item.get("p", "")))
            o_val = clean(str(item.get("o", "")))
            if not (s_val and p_val and o_val):
                return None
            if (
                self._has_blocked_date(s_val)
                or self._has_blocked_date(p_val)
                or self._has_blocked_date(o_val)
            ):
                return None
            return (s_val, p_val, o_val)

        accelerator: Any = LoopAccelerator()
        return [t for t in accelerator.map(_build_from_dict, obj) if t]

    def _extract_triples_from_dict_graph(
        self, obj: dict[str, Any], subject: str
    ) -> list[tuple[str, str, str]]:
        """Execute extract triples from dict graph.



        Args:

            obj: Input value used by this callable.

            subject: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        triples: list[tuple[str, str, str]] = []
        current = self._normalize_entity_subject(
            obj.get("id") or obj.get("externalId") or subject, subject
        )
        stack: list[tuple[str, str, Any]] = [
            (current, key, value) for key, value in obj.items() if key[:1] != "_"
        ]

        while stack:
            subj, pred, val = stack.pop()
            self._process_graph_value(subj, pred, val, stack, triples)

        return triples

    def _extract_triples_from_text(
        self, obj: str, subject: str
    ) -> list[tuple[str, str, str]]:
        """Execute extract triples from text.



        Args:

            obj: Input value used by this callable.

            subject: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        triples: list[tuple[str, str, str]] = []
        clean = _clean
        for line in obj.splitlines():
            trimmed_line = line.strip()
            if not trimmed_line or trimmed_line in _SKIP_LINES:
                continue

            tab_triple = self._parse_tab_triple(trimmed_line, clean)
            if tab_triple is not None:
                triples.append(tab_triple)
                continue

            json_triple = self._parse_json_triple(trimmed_line, clean)
            if json_triple is not None:
                triples.append(json_triple)
                continue

            kv_triple = self._parse_key_value_triple(line, subject, clean)
            if kv_triple is None:
                continue
            triples.append(kv_triple)

        if triples:
            logger.debug(
                f"Extracted {len(triples)} triples from string member {subject}"
            )
        return triples

    def _normalize_entity_subject(self, value: Any, default: str) -> str:
        """Execute normalize entity subject.



        Args:

            value: Input value used by this callable.

            default: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        if not value:
            return default
        if isinstance(value, str):
            normalized = value.strip()
        else:
            normalized = str(value).strip()
        return normalized or default

    def _has_blocked_values(self, *values: str) -> bool:
        return any(self._has_blocked_date(value) for value in values)

    def _append_graph_triple(
        self,
        triples: list[tuple[str, str, str]],
        subject: str,
        predicate: str,
        obj_value: str,
    ) -> None:
        """Execute append graph triple.



        Args:

            triples: Input value used by this callable.

            subject: Input value used by this callable.

            predicate: Input value used by this callable.

            obj_value: Input value used by this callable.

        """

        predicate_clean = predicate.strip()
        object_clean = obj_value.strip()
        if not subject or not predicate_clean or not object_clean:
            return
        if self._has_blocked_values(subject, predicate_clean, object_clean):
            return
        triples.append((subject, predicate_clean, object_clean))

    def _process_graph_value(
        self,
        subject: str,
        predicate: str,
        value: Any,
        stack: list[tuple[str, str, Any]],
        triples: list[tuple[str, str, str]],
    ) -> None:
        """Execute process graph value.



        Args:

            subject: Input value used by this callable.

            predicate: Input value used by this callable.

            value: Input value used by this callable.

            stack: Input value used by this callable.

            triples: Input value used by this callable.

        """

        if value is None:
            return
        if isinstance(value, str):
            value_str = value.strip()
            if value_str and value_str not in _SKIP_VALUES:
                self._append_graph_triple(triples, subject, predicate, value_str)
            return
        if isinstance(value, (bool, int, float)):
            self._append_graph_triple(triples, subject, predicate, str(value))
            return
        if isinstance(value, dict):
            for key, nested_value in value.items():
                if key[:1] != "_":
                    stack.append((subject, f"{predicate}.{key}", nested_value))
            return
        if isinstance(value, list):
            self._process_graph_list(subject, predicate, value, stack, triples)

    def _process_graph_list(
        self,
        subject: str,
        predicate: str,
        values: list[Any],
        stack: list[tuple[str, str, Any]],
        triples: list[tuple[str, str, str]],
    ) -> None:
        """Execute process graph list.



        Args:

            subject: Input value used by this callable.

            predicate: Input value used by this callable.

            values: Input value used by this callable.

            stack: Input value used by this callable.

            triples: Input value used by this callable.

        """

        for index, item in enumerate(values):
            if not isinstance(item, dict):
                stack.append((subject, predicate, item))
                continue
            item_id = (
                item.get("id")
                or item.get("externalId")
                or f"{subject}_{predicate}_{index}"
            )
            item_subject = self._normalize_entity_subject(
                item_id, f"{subject}_{predicate}_{index}"
            )
            self._append_graph_triple(triples, subject, predicate, item_subject)
            for key, nested_value in item.items():
                if key[:1] != "_":
                    stack.append((item_subject, key, nested_value))

    def _parse_tab_triple(
        self,
        trimmed_line: str,
        clean: Any,
    ) -> tuple[str, str, str] | None:
        """Execute parse tab triple.



        Args:

            trimmed_line: Input value used by this callable.

            clean: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        parts = trimmed_line.split("\t")
        if len(parts) != 3:
            return None
        s_val, p_val, o_val = [clean(part) for part in parts]
        if not (s_val and p_val and o_val):
            return None
        return (s_val, p_val, o_val)

    def _parse_json_triple(
        self,
        trimmed_line: str,
        clean: Any,
    ) -> tuple[str, str, str] | None:
        """Execute parse json triple.



        Args:

            trimmed_line: Input value used by this callable.

            clean: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        if not (trimmed_line.startswith("{") and trimmed_line.endswith("}")):
            return None
        try:
            line_obj = self.fm.json_loads(trimmed_line)
        except Exception:
            return None
        if not (isinstance(line_obj, dict) and {"s", "p", "o"} <= line_obj.keys()):
            return None
        s_val = clean(str(line_obj["s"]))
        p_val = clean(str(line_obj["p"]))
        o_val = clean(str(line_obj["o"]))
        if not (s_val and p_val and o_val):
            return None
        return (s_val, p_val, o_val)

    def _parse_key_value_triple(
        self,
        line: str,
        subject: str,
        clean: Any,
    ) -> tuple[str, str, str] | None:
        """Execute parse key value triple.



        Args:

            line: Input value used by this callable.

            subject: Input value used by this callable.

            clean: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        match = _KV.match(line)
        if not match:
            return None
        predicate, value = match.groups()
        value_clean = clean(value)
        if not value_clean or value_clean in _SKIP_VALUES:
            return None
        predicate_clean = clean(predicate)
        if self._has_blocked_values(subject, predicate_clean, value_clean):
            return None
        return (subject, predicate_clean, value_clean)

    def _serialize_for_rust_triples(self, obj: Any) -> str:
        """Execute serialize for rust triples.



        Args:

            obj: Input value used by this callable.



        Returns:

            Return value produced by the callable.



        Raises:

            Exception: Propagates domain-specific failures with context.

        """

        if isinstance(obj, str):
            return obj
        if isinstance(obj, pl.DataFrame):
            obj = obj.to_dicts()
        try:
            return orjson.dumps(obj).decode(DEFAULT_ENCODING)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"Unsupported payload type for Rust triple conversion: {type(obj).__name__}"
            ) from exc

    def _convert_to_triples(
        self, obj: Any, subject: str
    ) -> tuple[str, list[tuple[str, str, str]]]:
        payload = self._serialize_for_rust_triples(obj)
        triples = rust_convert_to_triples(payload, subject)
        return subject, triples

    async def _serialise(self) -> None:
        """Execute serialise."""

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
    """Execute main.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

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
