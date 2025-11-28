import json  # Keep for JSONDecodeError exception type
import os
import re
from math import floor
from pathlib import Path
from typing import Any, Mapping

from genson import SchemaBuilder

from pff.utils.concurrency import ConcurrencyManager
from pff.utils.file_manager import FileManager
from pff.utils import logger
from pff.utils.acceleration.loop_accelerator import LoopAccelerator

_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}(?:T.*)?$")
_UUID_PATTERN = re.compile(
    r"^(?:[0-9A-Fa-f]{32}|[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-"
    r"[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12})$"
)


class SchemaGenerator:
    """Generates JSON schemas from sample data files.

    Design Patterns Applied:
        - **Builder Pattern:** Uses internal SchemaBuilder to incrementally
          construct complex JSON schemas from collected value samples.
        - **Strategy Pattern:** Employs LoopAccelerator for traversal,
          allowing selection of vectorized/parallel strategies.
        - **Visitor Pattern (implicit):** Recursively visits nested structures
          to collect value occurrences by JSON path.

    Performance Optimizations:
        - LoopAccelerator.map() for accelerated nested structure traversal.
        - FileManager.json_loads() for msgspec-based fast JSON parsing.
        - Compiled regex patterns for value classification.

    Attributes:
        source_path: Path to source JSON/JSONL files.
        output_path: Path for generated schema output.
    """
        self.source_path = source_path
        self.output_path = output_path
        self._schema_builder = SchemaBuilder()
        self._value_occurrences: dict[tuple[str, ...], set[str]] = {}

    def _collect_values_by_path(self, obj: Any, prefix: tuple[str, ...]) -> None:
        # Use LoopAccelerator.map to speed traversal over nested structures
        accelerator = LoopAccelerator()

        if isinstance(obj, dict):
            accelerator.map(
                lambda kv: self._collect_values_by_path(kv[1], prefix + (kv[0],)),
                list(obj.items()),
            )
        elif isinstance(obj, list):
            accelerator.map(
                lambda e: self._collect_values_by_path(e, prefix),
                list(obj),
            )
        else:
            self._value_occurrences.setdefault(prefix, set()).add(str(obj))

    def _parse_json_line(self, line: str, file_path: Path, line_no: int) -> Any:
        try:
            # Sprint 16.5: Use FileManager for faster JSON parsing (msgspec)
            return FileManager.json_loads(line)
        except json.JSONDecodeError as e:
            msg = f"Erro no arquivo '{file_path}', linha {line_no}: {e.msg}"
            raise json.JSONDecodeError(msg, e.doc, e.pos) from e

    def _process_single_json_file(self, file_path: Path) -> None:
        text = file_path.read_text(encoding="utf-8")
        for i, raw in enumerate(text.splitlines(), start=1):
            line = raw.strip()
            if not line:
                continue
            obj = self._parse_json_line(line, file_path, i)
            self._collect_values_by_path(obj, ())
            self._schema_builder.add_object(obj)

    def _process_directory(self) -> None:
        paths = sorted(self.source_path.glob("*.json"))
        if not paths:
            logger.warning(f"[404] Nenhum arquivo .json em '{self.source_path}'")
            return
        workers = max(1, floor((os.cpu_count() or 1) * 0.85))
        cm = ConcurrencyManager()
        cm.execute(
            self._process_single_json_file,
            [(p,) for p in paths],
            task_type="io_thread",
            max_workers=workers,
            desc=f"Processando {len(paths)} JSON",
        )

    def _process_zip_file(self) -> None:
        contents = FileManager.read(self.source_path)
        members = {
            name: content
            for name, content in contents.items()
            if name.lower().endswith((".json", ".txt"))
        }
        if not members:
            logger.warning(f"[404] Nenhum .json/.txt em '{self.source_path}'")
            return
        for name, content in members.items():
            ext = Path(name).suffix.lower()
            if ext == ".txt":
                txt = (
                    content
                    if isinstance(content, str)
                    else content.decode("utf-8", "ignore")
                )
                txt = txt.strip()
                if not txt:
                    continue
                try:
                    # Sprint 16.5: Use FileManager for faster JSON parsing (msgspec)
                    obj = FileManager.json_loads(txt)
                except json.JSONDecodeError as e:
                    logger.warning(f"[IGNORED] Invalid JSON in '{name}': {e.msg}")
                    continue
                self._collect_values_by_path(obj, ())
                self._schema_builder.add_object(obj)
            else:
                obj = content
                if not isinstance(obj, dict):
                    continue
                self._collect_values_by_path(obj, ())
                self._schema_builder.add_object(obj)

    def _insert_enums(
        self, schema: Mapping[str, Any], prefix: tuple[str, ...] = ()
    ) -> None:
        props = schema.get("properties")
        if isinstance(props, dict):
            for key, subsch in props.items():
                newp = prefix + (key,)
                vals = self._value_occurrences.get(newp)
                if vals and 1 < len(vals) <= 10:
                    if not all(_DATE_PATTERN.match(v) for v in vals) and not all(
                        _UUID_PATTERN.match(v) for v in vals
                    ):
                        subsch.setdefault("enum", sorted(vals))
                self._insert_enums(subsch, newp)
        if schema.get("type") == "array":
            items = schema.get("items")
            if isinstance(items, dict):
                self._insert_enums(items, prefix)
        for defs in ("definitions", "$defs"):
            sec = schema.get(defs)
            if isinstance(sec, dict):
                for k, v in sec.items():
                    self._insert_enums(v, prefix + (k,))

    def generate(self) -> None:
        if self.source_path.is_dir():
            self._process_directory()
        elif self.source_path.suffix.lower() == ".zip":
            self._process_zip_file()
        else:
            raise FileNotFoundError(f"404: {self.source_path}")
        schema = self._schema_builder.to_schema()
        self._insert_enums(schema)
        FileManager.save(schema, self.output_path)
        logger.info(f"Esquema salvo em '{self.output_path}'")


def generate_schema(src: str, dest: str) -> None:
    gen = SchemaGenerator(Path(src), Path(dest))
    gen.generate()


if __name__ == "__main__":
    import sys

    generate_schema(sys.argv[1], sys.argv[2])
