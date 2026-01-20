"""Unit tests for preprocessing strategies.

Tests cover:
- DeduplicationStrategy: removes exact duplicates
- SelfLoopRemovalStrategy: removes s==o triples
- InverseRelationStrategy: adds inverses with proper suffix

Design Pattern: Tests follow AAA (Arrange-Act-Assert) structure.
Uses Polars DataFrames with columns [s, p, o] per module convention.
"""

import polars as pl
import pytest

from pff.domain.kg.preprocessing.strategies import (
    DeduplicationStrategy,
    InverseRelationStrategy,
    SelfLoopRemovalStrategy,
)


@pytest.fixture
def sample_triples() -> pl.DataFrame:
    """Create sample triples for testing.

    Contains:
    - One duplicate: (A, r1, B) appears twice
    - One self-loop: (A, r5, A)
    """
    return pl.DataFrame(
        {
            "s": ["A", "B", "C", "A", "D", "A"],
            "p": ["r1", "r2", "r3", "r1", "r4", "r5"],
            "o": ["B", "C", "D", "B", "D", "A"],
        }
    )


class TestDeduplicationStrategy:
    """Tests for DeduplicationStrategy."""

    def test_removes_exact_duplicates(self, sample_triples: pl.DataFrame) -> None:
        """Deduplication removes exact duplicates (A, r1, B) appears twice."""
        strategy = DeduplicationStrategy()

        original_count = len(sample_triples)
        result = strategy.process(sample_triples)

        assert len(result.data) == original_count - 1
        # Verify only one (A, r1, B) triple remains
        filtered = result.data.filter(
            (pl.col("s") == "A") & (pl.col("p") == "r1") & (pl.col("o") == "B")
        )
        assert len(filtered) == 1

    def test_preserves_unique_triples(self) -> None:
        """Deduplication preserves all unique triples."""
        unique_df = pl.DataFrame(
            {
                "s": ["A", "B", "C"],
                "p": ["r1", "r2", "r3"],
                "o": ["X", "Y", "Z"],
            }
        )
        strategy = DeduplicationStrategy()

        result = strategy.process(unique_df)

        assert len(result.data) == 3
        assert result.stats["duplicates_removed"] == 0

    def test_empty_dataframe(self) -> None:
        """Deduplication handles empty dataframe gracefully."""
        empty_df = pl.DataFrame(
            {
                "s": pl.Series([], dtype=pl.Utf8),
                "p": pl.Series([], dtype=pl.Utf8),
                "o": pl.Series([], dtype=pl.Utf8),
            }
        )
        strategy = DeduplicationStrategy()

        result = strategy.process(empty_df)

        assert len(result.data) == 0
        assert result.stats["initial_triples"] == 0

    def test_stats_correct(self, sample_triples: pl.DataFrame) -> None:
        """Deduplication returns correct statistics."""
        strategy = DeduplicationStrategy()

        result = strategy.process(sample_triples)

        assert result.stats["initial_triples"] == 6
        assert result.stats["duplicates_removed"] == 1
        assert result.stats["final_triples"] == 5
        assert result.stats["duplicate_percentage"] > 0


class TestSelfLoopRemovalStrategy:
    """Tests for SelfLoopRemovalStrategy."""

    def test_removes_self_loops(self, sample_triples: pl.DataFrame) -> None:
        """Self-loop removal removes triples where s == o."""
        strategy = SelfLoopRemovalStrategy()

        result = strategy.process(sample_triples)

        # Fixture has (D, r4, D) and (A, r5, A) as self-loops = 2 total
        # Verify no self-loops remain
        loops = result.data.filter(pl.col("s") == pl.col("o"))
        assert len(loops) == 0
        # 6 original rows - 2 self-loops = 4 remaining
        assert len(result.data) == 4
        assert result.stats["self_loops_removed"] == 2

    def test_preserves_non_self_loops(self) -> None:
        """Self-loop removal preserves all non-self-loop triples."""
        no_loops = pl.DataFrame(
            {
                "s": ["A", "B", "C"],
                "p": ["r1", "r2", "r3"],
                "o": ["X", "Y", "Z"],
            }
        )
        strategy = SelfLoopRemovalStrategy()

        result = strategy.process(no_loops)

        assert len(result.data) == 3
        assert result.stats["self_loops_removed"] == 0

    def test_all_self_loops(self) -> None:
        """Self-loop removal handles dataframe with all self-loops."""
        all_loops = pl.DataFrame(
            {
                "s": ["A", "B", "C"],
                "p": ["r1", "r2", "r3"],
                "o": ["A", "B", "C"],
            }
        )
        strategy = SelfLoopRemovalStrategy()

        result = strategy.process(all_loops)

        assert len(result.data) == 0
        assert result.stats["self_loops_removed"] == 3

    def test_allowed_reflexive_relations(self) -> None:
        """Self-loops preserved for allowed reflexive relations."""
        df = pl.DataFrame(
            {
                "s": ["A", "B"],
                "p": ["sameAs", "r1"],
                "o": ["A", "B"],
            }
        )
        strategy = SelfLoopRemovalStrategy(allowed_reflexive_relations={"sameAs"})

        result = strategy.process(df)

        # r1 self-loop removed, sameAs preserved
        assert len(result.data) == 1
        assert result.data["p"][0] == "sameAs"


class TestInverseRelationStrategy:
    """Tests for InverseRelationStrategy."""

    def test_adds_inverse_relations(self) -> None:
        """Inverse strategy adds inverse for each triple."""
        df = pl.DataFrame(
            {
                "s": ["A", "B"],
                "p": ["r1", "r2"],
                "o": ["X", "Y"],
            }
        )
        strategy = InverseRelationStrategy()

        result = strategy.process(df)

        assert len(result.data) == 4
        relations = result.data["p"].to_list()
        assert "r1_inv" in relations
        assert "r2_inv" in relations

    def test_inverse_structure(self) -> None:
        """Inverse triple has swapped s/o and suffixed relation."""
        df = pl.DataFrame(
            {
                "s": ["A"],
                "p": ["r1"],
                "o": ["B"],
            }
        )
        strategy = InverseRelationStrategy()

        result = strategy.process(df)
        inverses = result.data.filter(pl.col("p").str.ends_with("_inv"))

        assert len(inverses) == 1
        assert inverses["s"][0] == "B"  # Original object becomes subject
        assert inverses["o"][0] == "A"  # Original subject becomes object
        assert inverses["p"][0] == "r1_inv"

    def test_custom_inverse_suffix(self) -> None:
        """Inverse strategy uses custom suffix."""
        df = pl.DataFrame(
            {
                "s": ["A"],
                "p": ["r1"],
                "o": ["B"],
            }
        )
        strategy = InverseRelationStrategy(suffix="_reverse")

        result = strategy.process(df)

        relations = result.data["p"].to_list()
        assert "r1_reverse" in relations

    def test_stats_correct(self) -> None:
        """Inverse strategy returns correct statistics."""
        df = pl.DataFrame(
            {
                "s": ["A", "B"],
                "p": ["r1", "r2"],
                "o": ["X", "Y"],
            }
        )
        strategy = InverseRelationStrategy()

        result = strategy.process(df)

        assert result.stats["initial_triples"] == 2
        assert result.stats["final_triples"] == 4
        assert result.stats["inverse_triples_added"] == 2
        assert result.stats["original_relations"] == 2
        assert result.stats["final_relations"] == 4
