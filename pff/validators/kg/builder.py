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
from typing import Any, Iterable, Sequence

import polars as pl

from pff.utils import (
    CacheManager,
    ConcurrencyManager,
    FileManager,
    logger,
    progress_bar,
)

DEFAULT_ENCODING = "utf-8"
TRAIN_RATIO, VALID_RATIO, TEST_RATIO = 0.8, 0.1, 0.1
_KV = re.compile(r"""\s*["']?([^"':\t]+)["']?\s*:\s*["']?([^"']+)["']?\s*,?\s*$""")
_SKIP_LINES = {"{", "}", "[", "]", "},", "],", "{}", "[]"}
_SKIP_VALUES = {"{", "[", "{}", "[]"}

DEFAULT_SOURCE = Path("data/models/correct.zip")
DEFAULT_OUTPUT_DIR = Path("data/models/kg/graph")


# ───────────────────────── helpers ──────────────────────────── #
def _clean(text: str) -> str:
    return text.replace("\t", " ").strip()


def _iter_rows(df: pl.DataFrame) -> Iterable[tuple[str, str, str]]:  # type: ignore[type-var]
    for col in ("s", "p", "o"):
        if col not in df.columns:
            df = df.with_columns([pl.lit(None).cast(pl.Utf8).alias(col)])
    for s, p, o in df.select(["s", "p", "o"]).rows():
        yield (
            str(s) if s is not None else "",
            str(p) if p is not None else "",
            str(o) if o is not None else "",
        )


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


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
        source_path: str | Path,
        output_dir: str | Path,
        max_members: int | None = None,
        *,
        parallel: bool = True,
        workers: int | None = None,
        disk_cache: bool = False,
    ) -> None:
        self.source_path = Path(source_path).expanduser().resolve()
        self.output_dir = Path(output_dir).expanduser().resolve()
        _ensure_dir(self.output_dir)

        self.fm = FileManager()
        self.max_members = max_members
        self.parallel = parallel
        self.max_workers = workers or min(os.cpu_count() or 1, 8)

        # stats
        self._triples: list[tuple[str, str, str]] = []
        self._stats = SimpleNamespace(total_members=0, total_triples=0)

        # cache decorator (no-op se disk_cache=False)
        self._cached_convert = (
            CACHE.disk_cache(ttl=None)(self._convert_to_triples)
            if disk_cache
            else self._convert_to_triples
        )

    # ───────────────────── API pública ───────────────────── #
    async def run(self) -> None:
        """Faz todo o fluxo: load ➜ parse ➜ split ➜ salvar."""
        await self._load_and_parse()
        self._serialise()
        logger.success("✅ Construção finalizada.")  # type: ignore[attr-defined] (loguru)

    # ───────────────────── internals ────────────────────── #
    async def _load_and_parse(self) -> None:
        if not self.source_path.exists():
            sys.exit(f"Arquivo ausente: {self.source_path}")

        logger.info(f"▶ Lendo {self.source_path.name}")
        content: Any = self.fm.read(self.source_path)

        if isinstance(content, dict):
            members: Sequence[tuple[str, Any]] = list(content.items())
        else:
            members = [(self.source_path.name, content)]

        if self.max_members:
            members = members[: self.max_members]

        if self.parallel and len(members) > 1:
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
            self._triples.extend(triples)
            self._stats.total_members += 1
            self._stats.total_triples += len(triples)

        logger.info(
            "✓ {} membro(s) processados – {} triplas no total",
            f"{self._stats.total_members:,}",
            f"{self._stats.total_triples:,}",
        )

    def _convert_to_triples(self, obj: Any, subject: str) -> tuple[str, list[tuple[str, str, str]]]:
        triples: list[tuple[str, str, str]] = []

        # DataFrame ----------------------------------
        if isinstance(obj, pl.DataFrame):
            for s, p, o in _iter_rows(obj):
                if any("1970-01-01" in str(x) or "9999-12-31" in str(x) for x in [s, p, o]):
                    continue
                triples.append((_clean(s), _clean(p), _clean(o)))
            return subject, triples

        # List[dict] ---------------------------------
        if isinstance(obj, list):
            for d in obj:
                if isinstance(d, dict):
                    s = _clean(str(d.get("s", "")))
                    p = _clean(str(d.get("p", "")))
                    o = _clean(str(d.get("o", "")))
                    if any("1970-01-01" in x or "9999-12-31" in x for x in [s, p, o]):
                        continue
                    if s and p and o:
                        triples.append((s, p, o))
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
    def _serialise(self) -> None:
        """
        Serializes the collected triples into train, validation, and test Parquet files, and saves dataset statistics.
        Splits the list of triples into training, validation, and test sets according to predefined ratios.
        Each split is saved as a Parquet file in the specified output directory. If no valid triples are found,
        the process is aborted. Also saves a JSON file with statistics about the dataset, including counts and a timestamp.
        Raises:
            SystemExit: If no valid triples are found.
        """
        if not self._triples:
            sys.exit("Nenhuma tripla válida encontrada – abortando.")
        random.shuffle(self._triples)
        total = len(self._triples)
        train_end = int(total * TRAIN_RATIO)
        valid_end = train_end + int(total * VALID_RATIO)
        train_triples = self._triples[:train_end]
        valid_triples = self._triples[train_end:valid_end]
        test_triples = self._triples[valid_end:]

        def _dump_parquet(path: Path, triples: Sequence[tuple[str, str, str]]) -> None:
            if not triples:
                return
            df = pl.DataFrame(triples, schema=["s", "p", "o"], orient="row")
            _ensure_dir(path.parent)
            df.write_parquet(path)
            logger.info(f"Salvo {df.height} triplas em {path.name}")

        _ensure_dir(self.output_dir)
        _dump_parquet(self.output_dir / "train.parquet", train_triples)
        _dump_parquet(self.output_dir / "valid.parquet", valid_triples)
        _dump_parquet(self.output_dir / "test.parquet", test_triples)

        stats = {
            "total_members": self._stats.total_members,
            "total_triples": self._stats.total_triples,
            "train_count": len(train_triples),
            "valid_count": len(valid_triples),
            "test_count": len(test_triples),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        self.fm.save(stats, self.output_dir / "stats.json")

        logger.success(
            f"✔ {total:,} triplas salvas "
            f"(treino: {len(train_triples)} | validação: {len(valid_triples)} | "
            f"teste: {len(test_triples)})"
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