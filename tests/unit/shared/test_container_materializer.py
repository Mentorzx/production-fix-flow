from __future__ import annotations

from pathlib import Path

import polars as pl

from pff.shared.core.file_manager import FileManager, ParquetBundle
from pff.shared.core.file_manager.materializers.implementations import (
    ContainerMaterializer,
)


def _build_container_bundle(tmp_path: Path) -> ParquetBundle:
    parsed_path = tmp_path / "container_parsed.parquet"
    raw_path = tmp_path / "container_raw.parquet"

    df = pl.DataFrame(
        {
            "entry_name": ["entry_0", "entry_1"],
            "entry_ext": [".txt", ".txt"],
            "payload_kind": ["text", "text"],
            "payload_msgpack": pl.Series(
                "payload_msgpack", [None, None], dtype=pl.Binary
            ),
            "payload_text": ["alpha", "beta"],
            "payload_bytes": pl.Series("payload_bytes", [None, None], dtype=pl.Binary),
            "payload_parquet_path": pl.Series(
                "payload_parquet_path", [None, None], dtype=pl.Utf8
            ),
        }
    )
    FileManager.save(df, parsed_path)

    raw_df = pl.DataFrame({"chunk_bytes": [b""]})
    FileManager.save(raw_df, raw_path)

    return ParquetBundle(
        source_path=tmp_path / "source.zip",
        ext=".zip",
        file_id="test",
        raw_parquet_path=raw_path,
        parsed_parquet_path=parsed_path,
        parsed_kind="container",
    )


def test_container_materializer_materialize_text_entries(tmp_path) -> None:
    bundle = _build_container_bundle(tmp_path)
    materializer = ContainerMaterializer()

    result = materializer.materialize(bundle)

    assert result == {"entry_0": "alpha", "entry_1": "beta"}


def test_parquet_bundle_iter_entries(tmp_path) -> None:
    bundle = _build_container_bundle(tmp_path)

    entries = dict(bundle.iter_entries())

    assert entries == {"entry_0": "alpha", "entry_1": "beta"}
