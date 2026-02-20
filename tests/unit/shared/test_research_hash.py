"""Provide module-level functionality for the PFF codebase.



Notes:

    File: tests/unit/shared/test_research_hash.py

"""

from pff.shared.research import _hash_json_for_cache


def test_hash_json_for_cache_sorted_keys() -> None:
    """Execute test hash json for cache sorted keys.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    data_a = {"b": 2, "a": 1}
    data_b = {"a": 1, "b": 2}

    assert _hash_json_for_cache(data_a) == _hash_json_for_cache(data_b)
