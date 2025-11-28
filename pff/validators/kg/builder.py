from __future__ import annotations

import argparse
import asyncio
import os
import random
import re
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import polars as pl

from pff import settings
from pff.config import INGESTION_CONFIG_PATH
from pff.utils import (
    CacheManager,
    ConcurrencyManager,
    FileManager,
    logger,
    progress_bar,
)
from pff.utils.acceleration.loop_accelerator import LoopAccelerator

DEFAULT_ENCODING = "utf-8"
_KV = re.compile(r"""\s*["']?([^"':\t]+)["']?\s*:\s*["']?([^"']+)["']?\s*,?\s*$""")
_SKIP_LINES = {"{", "}", "[", "]", "},", "],", "{}", "[]"}
_SKIP_VALUES = {"{", "[", "{}", "[]"}


# ───────────────────────── helpers ──────────────────────────── #
def _clean(text: str) -> str:
    return text.replace("\t", " ").strip()


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _load_ingestion_config() -> dict[str, Any]:
    fm = FileManager()
    try:
        cfg = fm.read(INGESTION_CONFIG_PATH)
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
    KGBuilder is a utility class for constructing and serializing knowledge graphs from various data sources.
    Args:
        source_path (str | Path): Path to the input data file or directory.
        output_dir (str | Path): Directory where output files will be saved.
        max_members (int | None, optional): Maximum number of members to process. Defaults to None (process all).
        parallel (bool, optional): Whether to process members in parallel. Defaults to True.
        workers (int | None, optional): Number of worker threads for parallel processing. Defaults to min(os.cpu_count(), 4).
        disk_cache (bool, optional): Whether to enable disk caching for member conversion. Defaults to False.
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
            Loads the source data, parses it into triples, and accumulates statistics.
        _convert_to_triples(obj, subject):
            Converts a single data member into a list of triples, supporting multiple input formats.
        _serialise():
            Shuffles and splits triples into train/valid/test sets, saves them to disk, and writes statistics.
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
    ) -> None:
        cfg = _load_ingestion_config()
        default_source = cfg.get(
            "correct_zip_path", settings.DATA_DIR / "models" / "correct.zip"
        )
        default_output = cfg.get("output_dir", settings.OUTPUTS_DIR / "kg" / "graph")
        staging_default = cfg.get(
            "temp_output_dir", settings.OUTPUTS_DIR / "temp" / "kg_ingestion"
        )
        ratios_cfg = cfg.get(
            "split_ratios", {"train": 0.8, "valid": 0.1, "test": 0.1}
        )
        batch_size_cfg = cfg.get("batch_size", 50000)

        self.source_path = _resolve_path(
            source_path or default_source, base=settings.ROOT_DIR
        )
        resolved_output = _resolve_path(
            output_dir or default_output, base=settings.OUTPUTS_DIR
        )
        if not resolved_output.is_relative_to(settings.OUTPUTS_DIR):
            resolved_output = settings.OUTPUTS_DIR / resolved_output.name
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

        # stats
        self._stats = SimpleNamespace(total_members=0, total_triples=0)
        self._split_counts: dict[str, int] = {k: 0 for k in self.split_ratios}

        # streaming buffers
        self._buffers: dict[str, list[tuple[str, str, str]]] = {
            split: [] for split in self.split_ratios
        }
        self._chunk_indices: dict[str, int] = {split: 0 for split in self.split_ratios}

        # cache decorator (no-op se disk_cache=False)
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
        rnd = random.random()
        cumulative = 0.0
        for split, ratio in self.split_ratios.items():
            cumulative += ratio
            if rnd <= cumulative:
                return split
        return next(iter(self.split_ratios))

    def _flush_split(self, split: str) -> None:
        buffer = self._buffers.get(split, [])
        if not buffer:
            return
        df = pl.DataFrame(buffer, schema=["s", "p", "o"], orient="row")
        chunk_path = self._staging_dir / f"{split}_{self._chunk_indices[split]}.parquet"
        self.fm.save(df, chunk_path)
        self._chunk_indices[split] += 1
        buffer.clear()

    def _flush_all_buffers(self) -> None:
        for split in self._buffers:
            self._flush_split(split)

    def _buffer_triples(
        self,
        triples: list[tuple[str, str, str]],
        collector: list[tuple[str, str, str]] | None,
    ) -> None:
        if collector is not None:
            collector.extend(triples)
        for triple in triples:
            split = self._select_split()
            self._split_counts[split] += 1
            self._stats.total_triples += 1
            self._buffers[split].append(triple)
            if len(self._buffers[split]) >= self.chunk_size:
                self._flush_split(split)

    async def _load_and_parse(
        self, collector: list[tuple[str, str, str]] | None = None, *, persist: bool = True
    ) -> None:
        if not self.source_path.exists():
            sys.exit(f"Missing source file: {self.source_path}")

        logger.info(f"▶ Lendo {self.source_path.name}")
        if self.source_path.suffix.lower() == ".zip":
            content: Any = await self.fm.load_zip(self.source_path, task_type="thread")
        else:
            content: Any = self.fm.read(self.source_path)

        if isinstance(content, dict):
            members: Sequence[tuple[str, Any]] = list(content.items())
        else:
            members = [(self.source_path.name, content)]

        if self.max_members:
            members = members[: self.max_members]

        parsed: list[tuple[str, list[tuple[str, str, str]]]] = []
        if self.parallel and len(members) > 1 and persist:
            logger.debug(f"Processando {len(members)} membro(s) em pool…")
            cm = ConcurrencyManager()
            parsed = await cm.execute(
                self._cached_convert,
                [(c, n) for n, c in members],
                task_type="process",
                max_workers=self.max_workers,
                desc="Parseando",
            )
        else:
            parsed = [
                self._cached_convert(content, name)
                for name, content in progress_bar(members, desc="parseando")
            ]

        for _, triples in parsed:
            if persist:
                self._buffer_triples(triples, collector)
            elif collector is not None:
                collector.extend(triples)
                self._stats.total_triples += len(triples)
            self._stats.total_members += 1

        logger.info(
            " {} membro(s) processados – {} triplas no total",
            f"{self._stats.total_members:,}",
            f"{self._stats.total_triples:,}",
        )

    def _convert_to_triples(self, obj: Any, subject: str) -> tuple[str, list[tuple[str, str, str]]]:
        triples: list[tuple[str, str, str]] = []
        accelerator = LoopAccelerator()

        # DataFrame ----------------------------------
        if isinstance(obj, pl.DataFrame):
            rows = obj.select(["s", "p", "o"]).rows()

            def _build_df_triple(row: tuple[Any, Any, Any]) -> tuple[str, str, str] | None:
                s, p, o = row
                if any("1970-01-01" in str(x) or "9999-12-31" in str(x) for x in [s, p, o]):
                    return None
                return (_clean(str(s)), _clean(str(p)), _clean(str(o)))

            triples.extend([t for t in accelerator.map(_build_df_triple, rows) if t])
            return subject, triples

        # List[dict] ---------------------------------
        if isinstance(obj, list):
            def _build_from_dict(item: Any) -> tuple[str, str, str] | None:
                if not isinstance(item, dict):
                    return None
                s_val = _clean(str(item.get("s", "")))
                p_val = _clean(str(item.get("p", "")))
                o_val = _clean(str(item.get("o", "")))
                if any("1970-01-01" in x or "9999-12-31" in x for x in [s_val, p_val, o_val]):
                    return None
                if not (s_val and p_val and o_val):
                    return None
                return (s_val, p_val, o_val)

            triples.extend([t for t in accelerator.map(_build_from_dict, obj) if t])
            return subject, triples

        # Single dict --------------------------------
        if isinstance(obj, dict) and {"s", "p", "o"} <= obj.keys():
            s = _clean(str(obj["s"]))
            p = _clean(str(obj["p"]))
            o = _clean(str(obj["o"]))
            # Validar a tripla
            if not any("1970-01-01" in x or "9999-12-31" in x for x in [s, p, o]):
                if s and p and o:
                    triples.append((s, p, o))
            return subject, triples

        # Plain text ---------------------------------
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
                # Validar a tripla
                if not any("1970-01-01" in x or "9999-12-31" in x for x in [current, pred, val]):
                    if pred.lower() == "id":
                        current = val
                        triples.append((_clean(current), "id", _clean(val)))
                    else:
                        triples.append((_clean(current), _clean(pred), _clean(val)))
                continue

            if "\t" in line and not line.startswith('"'):
                parts = [_clean(p) for p in line.split("\t", 2)]
                if len(parts) == 3:
                    # Validar a tripla
                    if not any("1970-01-01" in p or "9999-12-31" in p for p in parts):
                        current = parts[0]
                        triples.append((parts[0], parts[1], parts[2]))
                    continue

            if not line.startswith('"'):
                parts = [_clean(p) for p in line.split(maxsplit=2)]
                if len(parts) == 3:
                    # Validar a tripla
                    if not any("1970-01-01" in p or "9999-12-31" in p for p in parts):
                        current = parts[0]
                        triples.append((parts[0], parts[1], parts[2]))
                        continue

            # Para linhas genéricas, validar também
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

        self._flush_all_buffers()

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

        # Save to PostgreSQL (primary storage)
        try:
            from pff.db.repositories import KGSplitsRepository

            repo = KGSplitsRepository()

            for split_name, path in split_paths.items():
                df = self.fm.read(path, streaming=True)
                if df is not None:
                    await repo.save_split(split_name, "raw", df, source=self.source_path.name)

            if split_paths:
                logger.success(" Triplas salvas no PostgreSQL (acesso rápido)")
        except Exception as e:
            logger.warning(f"Failed to persist splits to PostgreSQL: {e}", exc_info=True)
            logger.info("Dados salvos apenas em disco (fallback)")

        stats = {
            "total_members": self._stats.total_members,
            "total_triples": self._stats.total_triples,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        for split, count in self._split_counts.items():
            stats[f"{split}_count"] = count

        self.fm.save(stats, self.output_dir / "stats.json")

        logger.success(
            f" {self._stats.total_triples:,} triplas salvas "
            f"(treino: {self._split_counts.get('train', 0)} | validação: {self._split_counts.get('valid', 0)} | "
            f"teste: {self._split_counts.get('test', 0)})"
        )


# ───────────────────────── CLI entry-point ─────────────────────── #
def _parse_argv(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        prog="pff.validators.kg",
        description="Constrói grafo (train/valid/test) a partir de arquivo, pasta ou ZIP.",
    )
    parser.add_argument(
        "--source", default=DEFAULT_SOURCE, help="Fonte (arquivo/pasta/ZIP)"
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR, help="Pasta de saída")
    parser.add_argument("--max-members", type=int, default=None, metavar="N")
    parser.add_argument(
        "--no-parallel", action="store_true", help="Desliga parsing paralelo"
    )
    parser.add_argument("--workers", type=int, default=None, metavar="W")
    parser.add_argument(
        "--disk-cache", action="store_true", help="Cachear parse em disco"
    )

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
