"""Tests for _TripleIndexStrategy Rust conversion paths."""

import pytest

from pff.shared.research import _TripleIndexStrategy


def test_triple_index_strategy_uses_rust_kernel(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute test triple index strategy uses rust kernel.



    Args:

        monkeypatch: Input value used by this callable.



    Returns:

        Return value produced by the callable.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    strategy = _TripleIndexStrategy()
    called: dict[str, str] = {}

    def _fake_rust_convert(payload: str, subject: str):
        called["payload"] = payload
        called["subject"] = subject
        return [("s1", "p1", "o1")]

    monkeypatch.setattr("pff.shared.research.rust_convert_to_triples", _fake_rust_convert)

    triples = strategy._normalize_to_triples_optimized({"id": "cust_1", "status": "active"})

    assert triples == [("s1", "p1", "o1")]
    assert called["subject"] == "entity_0"
    assert called["payload"].startswith("{")


def test_triple_index_strategy_rejects_invalid_string_payload() -> None:
    """Execute test triple index strategy rejects invalid string payload.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    strategy = _TripleIndexStrategy()

    with pytest.raises(TypeError):
        strategy._normalize_to_triples_optimized("not-json")
