from pff.shared.research import _hash_json_for_cache


def test_hash_json_for_cache_sorted_keys() -> None:
    data_a = {"b": 2, "a": 1}
    data_b = {"a": 1, "b": 2}

    assert _hash_json_for_cache(data_a) == _hash_json_for_cache(data_b)
