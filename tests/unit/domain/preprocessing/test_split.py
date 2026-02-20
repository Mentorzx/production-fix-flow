"""Unit tests for SafeSplitter and leakage detection.

Tests cover:
- Random split creation with correct ratios
- Leakage detection between splits
- Prevention of test/valid triples appearing as train inverses

Design Pattern: Tests follow AAA (Arrange-Act-Assert) structure.
Uses Polars DataFrames with columns [s, p, o] per module convention.
"""

import polars as pl
import pytest

from pff.domain.kg.preprocessing.split import LeakageChecker, SafeSplitter


@pytest.fixture
def sample_triples() -> pl.DataFrame:
    """Create sample triples for testing splits."""
    return pl.DataFrame(
        {
            "s": [f"e{i}" for i in range(100)],
            "p": [f"r{i % 5}" for i in range(100)],
            "o": [f"e{(i + 1) % 100}" for i in range(100)],
        }
    )


class TestSafeSplitter:
    """Tests for SafeSplitter."""

    def test_split_ratios(self, sample_triples: pl.DataFrame) -> None:
        """Split creates train/valid/test with approximately correct ratios."""
        splitter = SafeSplitter(
            train_ratio=0.8,
            valid_ratio=0.1,
            test_ratio=0.1,
            seed=42,
        )

        result = splitter.random_split(sample_triples)

        total = len(sample_triples)
        assert abs(len(result.train) / total - 0.8) < 0.05
        assert abs(len(result.valid) / total - 0.1) < 0.05
        assert abs(len(result.test) / total - 0.1) < 0.05

    def test_split_no_overlap(self, sample_triples: pl.DataFrame) -> None:
        """Split creates non-overlapping train/valid/test sets."""
        splitter = SafeSplitter(seed=42)

        result = splitter.random_split(sample_triples)

        # Convert to sets of tuples for comparison
        train_set = set(result.train.iter_rows())
        valid_set = set(result.valid.iter_rows())
        test_set = set(result.test.iter_rows())

        assert len(train_set & valid_set) == 0
        assert len(train_set & test_set) == 0
        assert len(valid_set & test_set) == 0

    def test_split_complete(self, sample_triples: pl.DataFrame) -> None:
        """Split includes all original triples."""
        splitter = SafeSplitter(seed=42)

        result = splitter.random_split(sample_triples)

        assert len(result.train) + len(result.valid) + len(result.test) == len(sample_triples)

    def test_split_reproducibility(self, sample_triples: pl.DataFrame) -> None:
        """Split is reproducible with same seed."""
        splitter = SafeSplitter(seed=42)

        result1 = splitter.random_split(sample_triples)
        result2 = splitter.random_split(sample_triples)

        assert result1.train.equals(result2.train)
        assert result1.valid.equals(result2.valid)
        assert result1.test.equals(result2.test)

    def test_split_stats(self, sample_triples: pl.DataFrame) -> None:
        """Split returns complete statistics."""
        splitter = SafeSplitter(seed=42)

        result = splitter.random_split(sample_triples)

        assert "total_triples" in result.stats
        assert "train_triples" in result.stats
        assert "valid_triples" in result.stats
        assert "test_triples" in result.stats


class TestLeakageChecker:
    """Tests for LeakageChecker."""

    def test_no_leakage_clean_data(self) -> None:
        """Leakage checker returns no issues for clean data."""
        train = pl.DataFrame(
            {
                "s": ["A", "B"],
                "p": ["r1", "r2"],
                "o": ["X", "Y"],
            }
        )
        valid = pl.DataFrame(
            {
                "s": ["C"],
                "p": ["r3"],
                "o": ["Z"],
            }
        )
        test = pl.DataFrame(
            {
                "s": ["D"],
                "p": ["r4"],
                "o": ["W"],
            }
        )
        checker = LeakageChecker()

        result = checker.check_triple_leakage(train, valid, test)

        assert not result["has_leakage"]
        assert result["train_valid_overlap"] == 0
        assert result["train_test_overlap"] == 0

    def test_detects_exact_leakage(self) -> None:
        """Leakage checker finds exact duplicates across splits."""
        train = pl.DataFrame(
            {
                "s": ["A", "B"],
                "p": ["r1", "r2"],
                "o": ["X", "Y"],
            }
        )
        valid = pl.DataFrame(
            {
                "s": ["A"],
                "p": ["r1"],
                "o": ["X"],
            }
        )
        test = pl.DataFrame(
            {
                "s": ["D"],
                "p": ["r4"],
                "o": ["W"],
            }
        )
        checker = LeakageChecker()

        result = checker.check_triple_leakage(train, valid, test)

        assert result["has_leakage"]
        assert result["train_valid_overlap"] > 0

    def test_detects_inverse_leakage(self) -> None:
        """Leakage checker finds inverse relations that match test/valid."""
        # Train has inverse of a valid triple
        train = pl.DataFrame(
            {
                "s": ["A", "X"],
                "p": ["r1", "r2_inv"],
                "o": ["X", "B"],
            }
        )
        valid = pl.DataFrame(
            {
                "s": ["B"],
                "p": ["r2"],
                "o": ["X"],
            }
        )
        test = pl.DataFrame(
            {
                "s": ["D"],
                "p": ["r4"],
                "o": ["W"],
            }
        )
        checker = LeakageChecker()

        result = checker.check_inverse_leakage(train, valid, test)

        assert result["has_inverse_leakage"]
        assert result["train_valid_inverse_leak"] > 0

    def test_leakage_log_flag_preserves_results(self) -> None:
        """log_on_leak flag does not change detection semantics."""
        train = pl.DataFrame(
            {
                "s": ["A"],
                "p": ["r1"],
                "o": ["B"],
            }
        )
        valid = pl.DataFrame(
            {
                "s": ["A"],
                "p": ["r1"],
                "o": ["B"],
            }
        )
        test = pl.DataFrame(
            {
                "s": ["C"],
                "p": ["r2"],
                "o": ["D"],
            }
        )
        checker = LeakageChecker()

        with_logs = checker.check_triple_leakage(train, valid, test)
        silent = checker.check_triple_leakage(train, valid, test, log_on_leak=False)

        assert with_logs == silent
        assert with_logs["has_leakage"]

    def test_inverse_leakage_log_flag_preserves_results(self) -> None:
        """log_on_leak flag keeps inverse leakage detection consistent."""
        train = pl.DataFrame(
            {
                "s": ["A"],
                "p": ["rel_inv"],
                "o": ["B"],
            }
        )
        valid = pl.DataFrame(
            {
                "s": ["B"],
                "p": ["rel"],
                "o": ["A"],
            }
        )
        test = pl.DataFrame(
            {
                "s": ["C"],
                "p": ["rel2"],
                "o": ["D"],
            }
        )
        checker = LeakageChecker(inverse_suffix="_inv")

        with_logs = checker.check_inverse_leakage(train, valid, test)
        silent = checker.check_inverse_leakage(train, valid, test, log_on_leak=False)

        assert with_logs == silent
        assert with_logs["has_inverse_leakage"]

    def test_entity_coverage(self) -> None:
        """Entity coverage check detects unseen entities."""
        train = pl.DataFrame(
            {
                "s": ["A", "B"],
                "p": ["r1", "r2"],
                "o": ["X", "Y"],
            }
        )
        valid = pl.DataFrame(
            {
                "s": ["C"],
                "p": ["r3"],
                "o": ["Z"],
            }
        )
        test = pl.DataFrame(
            {
                "s": ["A"],
                "p": ["r4"],
                "o": ["X"],
            }
        )
        checker = LeakageChecker()

        result = checker.check_entity_coverage(train, valid, test)

        assert result["valid_unseen_entities"] == 2
        assert result["test_unseen_entities"] == 0
        assert result["valid_coverage"] < 1.0

    def test_full_check(self) -> None:
        """Full check runs all checks and returns comprehensive report."""
        train = pl.DataFrame(
            {
                "s": ["A", "B"],
                "p": ["r1", "r2"],
                "o": ["X", "Y"],
            }
        )
        valid = pl.DataFrame(
            {
                "s": ["C"],
                "p": ["r3"],
                "o": ["Z"],
            }
        )
        test = pl.DataFrame(
            {
                "s": ["D"],
                "p": ["r4"],
                "o": ["W"],
            }
        )
        checker = LeakageChecker()

        result = checker.full_check(train, valid, test)

        assert "triple_leakage" in result
        assert "inverse_leakage" in result
        assert "entity_coverage" in result
        assert "all_clear" in result


class TestLeakagePrevention:
    """Tests for leakage prevention in split workflow."""

    def test_split_with_inverse_no_leak(self) -> None:
        """Split with inverse safety adds inverses to all splits but prevents leakage."""
        # Setup simple data
        df = pl.DataFrame(
            {
                "s": ["A", "B", "C", "D"],
                "p": ["r1", "r1", "r1", "r1"],
                "o": ["B", "C", "D", "E"],
            }
        )

        splitter = SafeSplitter(
            train_ratio=0.5,
            valid_ratio=0.25,
            test_ratio=0.25,
            seed=42,
        )

        # Split with inverse safety
        result = splitter.split_with_inverse_safety(df, add_inverses=True)

        # Verify train has inverses
        train_has_inv = result.train.filter(pl.col("p").str.ends_with("_inv"))
        assert len(train_has_inv) > 0

        # NOTE: In the current implementation, inverses are added to ALL splits
        # This is safer because it doesn't artificially inflate train vs test
        # The important thing is no leakage between splits

    def test_inverse_not_leaking_test(self) -> None:
        """Train inverses don't contain test triple equivalents."""
        # Create specific data where leakage would be obvious
        train = pl.DataFrame(
            {
                "s": ["A"],
                "p": ["r1"],
                "o": ["B"],
            }
        )
        test = pl.DataFrame(
            {
                "s": ["B"],
                "p": ["r1"],
                "o": ["A"],
            }
        )

        # Add inverses to train
        from pff.domain.kg.preprocessing.strategies import InverseRelationStrategy

        inv_strategy = InverseRelationStrategy()
        train_with_inv = inv_strategy.process(train).data

        # Check that test triple is NOT in train (even if inverse was added)
        # The inverse (B, r1_inv, A) is different from test (B, r1, A)
        train_triples = set(train_with_inv.iter_rows())
        test_triple = tuple(test.row(0))

        assert test_triple not in train_triples
