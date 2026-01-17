from __future__ import annotations

import argparse
import asyncio
import os
import random
import re
import sys
import itertools
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, TYPE_CHECKING
from collections.abc import Mapping, Sequence, Iterable

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


# ───────────────────────── helpers ──────────────────────────── #
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


# ───────────────────────── core builder ─────────────────────── #
CACHE = CacheManager()


class KGBuilder:
    """
    KGBuilder is a utility class for constructing and serializing
    knowledge graphs from various data sources.
    Args:
        source_path (str | Path): Path to the input data file or directory.
        output_dir (str | Path): Directory where output files will be saved.
        max_members (int | None, optional): Maximum number of members to process.
            Defaults to None (process all).
        parallel (bool, optional): Whether to process members in parallel.
            Defaults to True.
        workers (int | None, optional): Number of worker threads for parallel
            processing. Defaults to min(os.cpu_count(), 4).
        disk_cache (bool, optional): Whether to enable disk caching for member
            conversion. Defaults to False.
    Attributes:
        source_path (Path): Resolved path to the input data.
        output_dir (Path): Resolved path to the output directory.
        fm (FileManager): File manager instance for loading and saving files.
        max_members (int | None): Maximum number of members to process.
        parallel (bool): Whether to process members in parallel.
        max_workers (int): Number of worker threads for parallel processing.
        _triples (list[tuple[str, str, str]]): Accumulated list of triples.
        _stats (SimpleNamespace): Statistics about processed members and triples.
        _cached_convert (Callable): Conversion function, optionally wrapped with disk cache.
    Methods:
        run():
            Executes the full pipeline: load, parse, split, and save triples.
        _load_and_parse():
            Loads the source data, parses it into triples, and accumulates
            statistics.
        _convert_to_triples(obj, subject):
            Converts a single data member into a list of triples, supporting
            multiple input formats.
        _serialise():
            Shuffles and splits triples into train/valid/test sets, saves them to
            disk, and writes statistics.
    Raises:
        SystemExit: If the source file is missing or no valid triples are found.
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
        # Guard against nested "outputs/outputs" paths and flatten to OUTPUTS_DIR/<graph_subdir>
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

        # stats
        self._stats = SimpleNamespace(total_members=0, total_triples=0)
        self._split_counts: dict[str, int] = {k: 0 for k in self.split_ratios}

        # streaming buffers
        self._buffers: dict[str, list[tuple[str, str, str]]] = {
            split: [] for split in self.split_ratios
        }
        self._chunk_indices: dict[str, int] = {split: 0 for split in self.split_ratios}
        self._pending_tasks: list[asyncio.Task] = []

        self._cached_convert = (
            CACHE.disk_cache(ttl=None)(self._convert_to_triples)
            if disk_cache
            else self._convert_to_triples
        )

    # ───────────────────── API pública ───────────────────── #
    async def run(self) -> None:
        """Faz todo o fluxo: load  parse  split  salvar."""
        await self._load_and_parse()
        await self._serialise()
        logger.success(" Construção finalizada.")  # type: ignore[attr-defined] (loguru)

    async def extract_triples(self) -> list[tuple[str, str, str]]:
        """Public helper to load and return triples without serializacao em disco."""
        collector: list[tuple[str, str, str]] = []
        self._stats = SimpleNamespace(total_members=0, total_triples=0)
        await self._load_and_parse(collector=collector, persist=False)
        return collector

    # ───────────────────── internals ────────────────────── #
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
        """Iterate over parquet entries yielding (name, content) tuples.

        Supports both legacy parquets with _raw_json column and optimized
        parquets with struct columns only. Returns dict objects directly
        when struct columns are available (faster, no JSON parsing needed).
        """
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
        elif isinstance(content, ParquetBundle) and content.parsed_kind == "tabular":
            parquet_path = content.parsed_parquet_path or content.source_path
            members, members_total = self._iter_parquet_json_entries(
                parquet_path,
                max_members=self.max_members,
            )
        else:
            native = content.to_native() if isinstance(content, ParquetBundle) else content
            if isinstance(native, dict):
                members = list(native.items())
            else:
                members = [(self.source_path.name, native)]
            if self.max_members:
                members = list(members)[: self.max_members]
            members_total = len(members) if isinstance(members, list) else None

        # Convert to list once for processing
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

    def _convert_to_triples(self, obj: Any, subject: str) -> tuple[str, list[tuple[str, str, str]]]:
        triples: list[tuple[str, str, str]] = []
        # LoopAccelerator removed as Polars vectorization is faster

        if isinstance(obj, pl.DataFrame):
            # Optimized Vectorized Cleaning
            # 1. Select and Clean columns in Rust/Polars engine
            cleaned_df = obj.select(
                [
                    pl.col("s").cast(pl.Utf8).str.replace("\t", " ").str.strip_chars(),
                    pl.col("p").cast(pl.Utf8).str.replace("\t", " ").str.strip_chars(),
                    pl.col("o").cast(pl.Utf8).str.replace("\t", " ").str.strip_chars(),
                ]
            )

            # 2. Filter invalid values (1970/9999) vectorially
            # Logic: valid if NONE of the columns contain the bad strings
            # (Using negation of "any column has bad string")
            # Note: The original logic returned None for the whole row if any part was bad.

            # Define bad patterns
            bad_patterns = ["1970-01-01", "9999-12-31"]

            # Build filter expression
            # We want rows where S is valid AND P is valid AND O is valid
            # Valid means: does not contain "1970..." AND does not contain "9999..."

            valid_mask = pl.lit(True)
            for col in ["s", "p", "o"]:
                for pat in bad_patterns:
                    valid_mask = valid_mask & (~pl.col(col).str.contains(pat, literal=True))

            # Apply filter
            filtered_df = cleaned_df.filter(valid_mask)

            # 3. Export to list of tuples (fastest path)
            # rows() returns list of tuples
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
                        stack.append((subj, f"{pred}.{k}", v))

                elif isinstance(val, list):
                    for i, item in enumerate(val):
                        if isinstance(item, dict):
                            item_id = (
                                item.get("id") or item.get("externalId") or f"{subj}_{pred}_{i}"
                            )
                            item_subj = _clean(str(item_id))
                            pred_clean = _clean(pred)
                            if not (
                                "1970-01-01" in subj
                                or "9999-12-31" in subj
                                or "1970-01-01" in pred_clean
                                or "9999-12-31" in pred_clean
                                or "1970-01-01" in item_subj
                                or "9999-12-31" in item_subj
                            ):
                                triples.append((subj, pred_clean, item_subj))
                            for k, v in item.items():
                                stack.append((item_subj, k, v))
                        else:
                            stack.append((subj, pred, item))

            return subject, triples

        if not isinstance(obj, str):
            return subject, triples

        current = subject
        for idx, raw in enumerate(obj.splitlines()):
            line = raw.strip()
            if not line or line in _SKIP_LINES or line.strip('",') in _SKIP_LINES:
                continue

            if m := _KV.match(line):
                pred, val = map(str.strip, m.groups())
                if val in _SKIP_VALUES or not val:
                    continue
                if not (
                    "1970-01-01" in current
                    or "9999-12-31" in current
                    or "1970-01-01" in pred
                    or "9999-12-31" in pred
                    or "1970-01-01" in val
                    or "9999-12-31" in val
                ):
                    if pred.lower() == "id":
                        current = val
                        triples.append((_clean(current), "id", _clean(val)))
                    else:
                        triples.append((_clean(current), _clean(pred), _clean(val)))
                continue

            if "\t" in line and not line.startswith('"'):
                parts = [_clean(p) for p in line.split("\t", 2)]
                if len(parts) == 3:
                    if not (
                        "1970-01-01" in parts[0]
                        or "9999-12-31" in parts[0]
                        or "1970-01-01" in parts[1]
                        or "9999-12-31" in parts[1]
                        or "1970-01-01" in parts[2]
                        or "9999-12-31" in parts[2]
                    ):
                        current = parts[0]
                        triples.append((parts[0], parts[1], parts[2]))
                    continue

            if not line.startswith('"'):
                parts = [_clean(p) for p in line.split(maxsplit=2)]
                if len(parts) == 3:
                    if not (
                        "1970-01-01" in parts[0]
                        or "9999-12-31" in parts[0]
                        or "1970-01-01" in parts[1]
                        or "9999-12-31" in parts[1]
                        or "1970-01-01" in parts[2]
                        or "9999-12-31" in parts[2]
                    ):
                        current = parts[0]
                        triples.append((parts[0], parts[1], parts[2]))
                        continue

            if not ("1970-01-01" in line or "9999-12-31" in line):
                triples.append((_clean(current), f"line_{idx}", _clean(line)))

        return subject, triples

    # ------------------------------------------------------ #
    async def _serialise(self) -> None:
        """
        Serializes the collected triples into train, validation, and test Parquet files,
        saves to PostgreSQL, and saves dataset statistics.

        Splits are made incrementally to avoid O(n) memory usage.
        """
        if self._stats.total_triples == 0:
            sys.exit("No valid triples found – aborting.")

        await self._flush_all_buffers()

        split_paths: dict[str, Path] = {}
        _ensure_dir(self.output_dir)
        _ensure_dir(self._staging_dir)

        for split in self.split_ratios:
            chunk_paths = sorted(self._staging_dir.glob(f"{split}_*.parquet"))
            if not chunk_paths:
                logger.warning(f" No data for split '{split}', skipping materialization")
                continue
            lf = pl.scan_parquet([str(p) for p in chunk_paths])
            output_path = self.output_dir / f"{split}.parquet"
            self.fm.save(lf, output_path)
            split_paths[split] = output_path
            logger.info(
                f" Salvo {self._split_counts.get(split, 0)} triplas em {output_path.name} (disco)"
            )

        if self.splits_repo:
            try:
                for split_name, path in split_paths.items():
                    bundle = self.fm.read(path, streaming=True)
                    df = (
                        bundle.lazyframe().collect(engine="streaming")
                        if isinstance(bundle, ParquetBundle)
                        else bundle
                    )
                    if df is not None:
                        await self.splits_repo.save_split(split_name, df, split_type="raw")

                if split_paths:
                    logger.success(" Triplas salvas no PostgreSQL (acesso rápido)")
            except Exception as e:
                logger.warning(f"Failed to persist splits to PostgreSQL: {e}", exc_info=True)
                logger.info("Dados salvos apenas em disco (modo_alternativo)")
        else:
            logger.debug("splits_repo not available; using disk-only mode")

        stats = {
            "total_members": self._stats.total_members,
            "total_triples": self._stats.total_triples,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        for split, count in self._split_counts.items():
            stats[f"{split}_count"] = count

        df = pl.DataFrame([stats])
        self.fm.save(df, self.output_dir / "stats.parquet")

        try:
            if self.fm.exists(self._staging_dir):
                self.fm.delete_directory(self._staging_dir)
        except OSError as e:
            logger.warning(f"Failed to cleanup staging directory: {e}")

        train_count = self._split_counts.get("train", 0)
        valid_count = self._split_counts.get("valid", 0)
        test_count = self._split_counts.get("test", 0)
        logger.success(
            f" {self._stats.total_triples:,} triplas salvas "
            f"(treino: {train_count} | validação: {valid_count} | teste: {test_count})"
        )


# ───────────────────────── CLI entry-point ─────────────────────── #
def _parse_argv(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        prog="pff.domain.kg",
        description="Constrói grafo (train/valid/test) a partir de arquivo, pasta ou ZIP.",
    )
    parser.add_argument("--source", default=DEFAULT_SOURCE, help="Fonte (arquivo/pasta/ZIP)")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR, help="Pasta de saída")
    parser.add_argument("--max-members", type=int, default=None, metavar="N")
    parser.add_argument("--no-parallel", action="store_true", help="Desliga parsing paralelo")
    parser.add_argument("--workers", type=int, default=None, metavar="W")
    parser.add_argument("--disk-cache", action="store_true", help="Cachear parse em disco")

    ns = parser.parse_args(args=argv)
    return ns


async def cli_main(argv: list[str] | None = None) -> None:
    ns = _parse_argv(argv)
    await KGBuilder(
        ns.source,
        ns.output,
        max_members=ns.max_members,
        parallel=not ns.no_parallel,
        workers=ns.workers,
        disk_cache=ns.disk_cache,
    ).run()


if __name__ == "__main__":
    asyncio.run(cli_main())
