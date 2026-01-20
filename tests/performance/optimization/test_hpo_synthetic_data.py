from __future__ import annotations

from pff.infrastructure.hpo.trials.data_loader import load_synthetic_kg_data


def test_load_synthetic_kg_data_is_deterministic() -> None:
    train_a, valid_a, info_a = load_synthetic_kg_data()
    train_b, valid_b, info_b = load_synthetic_kg_data()

    assert info_a["source"] == "synthetic"
    assert info_a == info_b
    assert train_a.to_dicts() == train_b.to_dicts()
    assert valid_a.to_dicts() == valid_b.to_dicts()
