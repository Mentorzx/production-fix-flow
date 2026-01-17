from pathlib import Path

import pytest
import polars as pl

from pff.domain.kg.builder import KGBuilder


@pytest.mark.asyncio
async def test_extract_triples_returns_parsed_rows(tmp_path: Path) -> None:
    parquet_path = tmp_path / "mini.parquet"
    out_dir = tmp_path / "out"
    content = """{
    "id": "customer_1",
    "relation": "friend_1",
    "status": "active"
}"""

    pl.DataFrame(
        [
            {
                "_raw_json": content,
                "_source_name": "sample.txt",
                "_parse_error": None,
            }
        ]
    ).write_parquet(parquet_path)

    builder = KGBuilder(source_path=parquet_path, output_dir=out_dir, parallel=False)
    triples = await builder.extract_triples()

    assert ("customer_1", "relation", "friend_1") in triples
    assert ("customer_1", "status", "active") in triples
    assert builder._stats.total_triples == len(triples)
