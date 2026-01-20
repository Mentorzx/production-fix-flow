"""Test split consistency validation is deterministic.

This test verifies that bug G (split consistency) is addressed.
The validate_split_consistency function should produce deterministic hashes.
"""

from __future__ import annotations

import polars as pl


def test_split_source_determinism() -> None:
    """Hash computation must be deterministic for identical data."""
    from pff.infrastructure.hpo.trials.data_loader import validate_split_consistency

    train = pl.DataFrame({"s": ["a", "b"], "p": ["r1", "r1"], "o": ["c", "d"]})
    valid = pl.DataFrame({"s": ["a"], "p": ["r1"], "o": ["e"]})

    stats1 = validate_split_consistency(train, valid, "test")
    stats2 = validate_split_consistency(train, valid, "test")

    assert stats1["hash"] == stats2["hash"], "Hash must be deterministic"
    assert stats1["hash_short"] == stats1["hash"][:16]


def test_split_stats_are_correct() -> None:
    """Validate that stats are computed correctly."""
    from pff.infrastructure.hpo.trials.data_loader import validate_split_consistency

    train = pl.DataFrame(
        {
            "s": ["a", "b", "c"],
            "p": ["r1", "r2", "r1"],
            "o": ["d", "e", "f"],
        }
    )
    valid = pl.DataFrame(
        {
            "s": ["a"],
            "p": ["r2"],
            "o": ["g"],
        }
    )

    stats = validate_split_consistency(train, valid, "unit_test")

    assert stats["train_triples"] == 3
    assert stats["valid_triples"] == 1
    assert stats["total_triples"] == 4
    assert stats["source"] == "unit_test"
    # Entities: a, b, c, d, e, f, g = 7
    assert stats["entities"] == 7
    # Relations: r1, r2 = 2
    assert stats["relations"] == 2


def test_different_data_different_hash() -> None:
    """Different data must produce different hashes."""
    from pff.infrastructure.hpo.trials.data_loader import validate_split_consistency

    train1 = pl.DataFrame({"s": ["a"], "p": ["r1"], "o": ["b"]})
    valid1 = pl.DataFrame({"s": ["c"], "p": ["r1"], "o": ["d"]})

    train2 = pl.DataFrame({"s": ["x"], "p": ["r1"], "o": ["y"]})
    valid2 = pl.DataFrame({"s": ["z"], "p": ["r1"], "o": ["w"]})

    stats1 = validate_split_consistency(train1, valid1, "source1")
    stats2 = validate_split_consistency(train2, valid2, "source2")

    assert stats1["hash"] != stats2["hash"], "Different data should have different hash"
