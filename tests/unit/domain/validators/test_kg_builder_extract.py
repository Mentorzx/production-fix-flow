"""Provide module-level functionality for the PFF codebase.



Notes:

    File: tests/unit/domain/validators/test_kg_builder_extract.py

"""

from collections import Counter
from pathlib import Path

import orjson
import polars as pl
import pytest

from pff.domain.kg.builder import KGBuilder


@pytest.mark.asyncio
async def test_extract_triples_returns_parsed_rows(tmp_path: Path) -> None:
    """Execute test extract triples returns parsed rows.



    Args:

        tmp_path: Input value used by this callable.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

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
    """Execute test vectorized parquet raw json matches convert.



    Args:

        tmp_path: Input value used by this callable.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

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
                "_raw_json": orjson.dumps(payload).decode("utf-8"),
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
    """Execute test vectorized parquet struct columns matches convert.



    Args:

        tmp_path: Input value used by this callable.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

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
    """Execute test vectorized parquet struct columns preserve list edges.



    Args:

        tmp_path: Input value used by this callable.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

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
    """Execute test convert list of dicts creates entity edges.



    Args:

        tmp_path: Input value used by this callable.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

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
    """Execute test vectorized entity to triples handles list struct without ids.



    Args:

        tmp_path: Input value used by this callable.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

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


def test_convert_to_triples_uses_rust_kernel_for_serializable_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Execute test convert to triples uses rust kernel for serializable payload.



    Args:

        tmp_path: Input value used by this callable.

        monkeypatch: Input value used by this callable.



    Returns:

        Return value produced by the callable.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    parquet_path = tmp_path / "dummy_rust.parquet"
    out_dir = tmp_path / "out_rust"
    pl.DataFrame([{"id": "row"}]).write_parquet(parquet_path)

    builder = KGBuilder(source_path=parquet_path, output_dir=out_dir, parallel=False)

    called: dict[str, str] = {}

    def _fake_rust_convert(payload: str, subject: str) -> list[tuple[str, str, str]]:
        called["payload"] = payload
        called["subject"] = subject
        return [("s1", "p1", "o1")]

    monkeypatch.setattr("pff.domain.kg.builder.rust_convert_to_triples", _fake_rust_convert)
    _, triples = builder._convert_to_triples({"id": "cust_1", "status": "active"}, "source_9")

    assert triples == [("s1", "p1", "o1")]
    assert called["subject"] == "source_9"
    assert called["payload"].startswith("{")


def test_extract_rowwise_skips_raw_json_normalization_when_absent(tmp_path: Path) -> None:
    """Should avoid per-row raw JSON normalization when `_raw_json` column is missing."""
    parquet_path = tmp_path / "dummy_rowwise.parquet"
    out_dir = tmp_path / "out_rowwise"
    pl.DataFrame([{"id": "row"}]).write_parquet(parquet_path)

    builder = KGBuilder(source_path=parquet_path, output_dir=out_dir, parallel=False)
    rowwise_df = pl.DataFrame({"s": ["A", "B"], "p": ["r", "r"], "o": ["B", "C"]})

    def _fail_on_normalize(_row):
        raise AssertionError("_normalize_row_payload must not be called")

    builder._normalize_row_payload = _fail_on_normalize  # type: ignore[method-assign]
    builder._cached_convert = lambda row, _subject: ("", [(row["s"], row["p"], row["o"])])  # type: ignore[assignment]
    builder._stats.total_members = 0

    triples = builder._extract_rowwise_triples(rowwise_df)
    assert triples == [("A", "r", "B"), ("B", "r", "C")]
