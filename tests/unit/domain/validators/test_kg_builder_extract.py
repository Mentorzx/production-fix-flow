import json
from collections import Counter
from pathlib import Path

import polars as pl
import pytest

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


@pytest.mark.asyncio
async def test_vectorized_parquet_raw_json_matches_convert(tmp_path: Path) -> None:
    parquet_path = tmp_path / "raw_json.parquet"
    out_dir = tmp_path / "out_raw"

    payload = {
        "id": "customer_1",
        "externalId": "ext_1",
        "status": [{"status": "active", "validFor": {"startDateTime": "2024-01-01"}}],
        "account": [
            {
                "id": "acc_1",
                "characteristic": [
                    {
                        "charSpecExternalId": "customerTaxCategory",
                        "value": [{"value": "12"}],
                    }
                ],
            }
        ],
        "triples": [{"s": "subj_1", "p": "rel_1", "o": "obj_1"}],
        "tags": ["a", "b"],
        "_internal": "skip",
    }

    pl.DataFrame(
        [
            {
                "_raw_json": json.dumps(payload),
                "_source_name": "source_1",
                "_parse_error": None,
            }
        ]
    ).write_parquet(parquet_path)

    builder = KGBuilder(source_path=parquet_path, output_dir=out_dir, parallel=False)
    _, expected = builder._convert_to_triples(payload, "source_1")
    triples = await builder.extract_triples()

    assert Counter(triples) == Counter(expected)


@pytest.mark.asyncio
async def test_vectorized_parquet_struct_columns_matches_convert(
    tmp_path: Path,
) -> None:
    parquet_path = tmp_path / "structs.parquet"
    out_dir = tmp_path / "out_struct"

    payload = {
        "id": "customer_2",
        "externalId": "ext_2",
        "status": [{"status": "inactive", "validFor": {"startDateTime": "2024-02-01"}}],
        "account": [
            {
                "id": "acc_2",
                "characteristic": [
                    {
                        "charSpecExternalId": "taxJurisdictionCode",
                        "value": [{"value": "23"}],
                    }
                ],
            }
        ],
        "triples": [{"s": "subj_2", "p": "rel_2", "o": "obj_2"}],
        "tags": ["x", "y"],
        "_internal": "skip",
    }

    pl.DataFrame([payload | {"_source_name": "source_2"}]).write_parquet(parquet_path)

    builder = KGBuilder(source_path=parquet_path, output_dir=out_dir, parallel=False)
    _, expected = builder._convert_to_triples(payload, "source_2")
    triples = await builder.extract_triples()

    assert Counter(triples) == Counter(expected)


@pytest.mark.asyncio
async def test_vectorized_parquet_struct_columns_preserve_list_edges(
    tmp_path: Path,
) -> None:
    parquet_path = tmp_path / "structs_edges.parquet"
    out_dir = tmp_path / "out_struct_edges"

    payload = {
        "id": "customer_4",
        "account": [{"id": "acc_4", "status": "active"}],
        "_source_name": "source_4",
    }

    pl.DataFrame([payload]).write_parquet(parquet_path)

    builder = KGBuilder(source_path=parquet_path, output_dir=out_dir, parallel=False)
    triples = await builder.extract_triples()

    assert ("customer_4", "account", "acc_4") in triples
    assert ("acc_4", "status", "active") in triples


def test_convert_list_of_dicts_creates_entity_edges(tmp_path: Path) -> None:
    parquet_path = tmp_path / "dummy.parquet"
    out_dir = tmp_path / "out_dummy"

    pl.DataFrame([{"id": "row"}]).write_parquet(parquet_path)

    payload = {
        "id": "customer_3",
        "account": [{"id": "acc_3", "status": "active"}],
    }

    builder = KGBuilder(source_path=parquet_path, output_dir=out_dir, parallel=False)
    _, triples = builder._convert_to_triples(payload, "source_3")

    assert ("customer_3", "account", "acc_3") in triples
    assert ("acc_3", "status", "active") in triples


def test_vectorized_entity_to_triples_handles_list_struct_without_ids(
    tmp_path: Path,
) -> None:
    parquet_path = tmp_path / "dummy_struct.parquet"
    out_dir = tmp_path / "out_struct_no_id"

    pl.DataFrame([{"id": "row"}]).write_parquet(parquet_path)

    payload = {
        "id": "customer_5",
        "tags": [{"value": "a"}, {"value": "b"}],
    }

    builder = KGBuilder(source_path=parquet_path, output_dir=out_dir, parallel=False)
    vectorized = builder._vectorized_entity_to_triples(pl.DataFrame([payload]))
    _, expected = builder._convert_to_triples(payload, "row_0")

    assert Counter(vectorized.iter_rows()) == Counter(expected)
