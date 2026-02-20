"""Provide module-level functionality for the PFF codebase.



Notes:

    File: tests/unit/domain/services/test_polars_extensions.py

"""

from __future__ import annotations

from pff.application.services.polars_extensions import ResponseToDataFrameConverter


def test_json_to_dataframe_from_json_string() -> None:
    """Execute test json to dataframe from json string."""

    payload = '[{"id": 1, "name": "a"}, {"id": 2, "name": "b"}]'

    df = ResponseToDataFrameConverter.json_to_dataframe(payload)

    assert df is not None
    assert df.height == 2
    assert "_source_type" in df.columns


def test_json_to_dataframe_returns_none_for_invalid_json() -> None:
    """Execute test json to dataframe returns none for invalid json."""

    df = ResponseToDataFrameConverter.json_to_dataframe("{invalid-json")

    assert df is None
