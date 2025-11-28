import zipfile
from pathlib import Path

import pytest

from pff.validators.kg.builder import KGBuilder


@pytest.mark.asyncio
async def test_extract_triples_returns_parsed_rows(tmp_path: Path) -> None:
    zip_path = tmp_path / "mini.zip"
    out_dir = tmp_path / "out"
    content = "customer_1 relation friend_1\ncustomer_1 status active\n"

    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("sample.txt", content)

    builder = KGBuilder(source_path=zip_path, output_dir=out_dir, parallel=False)
    triples = await builder.extract_triples()

    assert ("customer_1", "relation", "friend_1") in triples
    assert builder._stats.total_triples == len(triples)
