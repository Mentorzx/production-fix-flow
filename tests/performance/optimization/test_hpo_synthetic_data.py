"""Provide module-level functionality for the PFF codebase.



Notes:

    File: tests/performance/optimization/test_hpo_synthetic_data.py

"""

from __future__ import annotations

from pff.infrastructure.hpo.trials.data_loader import load_synthetic_kg_data


def test_load_synthetic_kg_data_is_deterministic() -> None:
    """Execute test load synthetic kg data is deterministic.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    train_a, valid_a, info_a = load_synthetic_kg_data()
    train_b, valid_b, info_b = load_synthetic_kg_data()

    assert info_a["source"] == "synthetic"
    assert info_a == info_b
    assert train_a.to_dicts() == train_b.to_dicts()
    assert valid_a.to_dicts() == valid_b.to_dicts()
