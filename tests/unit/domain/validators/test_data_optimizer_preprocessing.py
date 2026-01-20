"""Tests for TelecomDataOptimizer preprocessing methods.

This module tests the new data quality preprocessing capabilities:
- Duplicate removal
- Self-loop removal
- Inverse relation augmentation

Pattern: AAA (Arrange, Act, Assert) for all test cases.
"""

import polars as pl

from pff.domain.kg.data_optimizer import (
    OptimizationConfig,
    TelecomDataOptimizer,
)


class TestRemoveDuplicates:
    """Tests for duplicate triple removal."""

    def test_removes_exact_duplicates(self):
        """Should remove exact duplicate triples."""
        config = OptimizationConfig(
            min_entity_degree=1,
            min_relation_support=1,
            remove_duplicates=True,
            remove_self_loops=False,
            add_inverse_relations=False,
            balance_relations=False,
        )
        optimizer = TelecomDataOptimizer(config)

        # Arrange: DataFrame with duplicates
        df = pl.DataFrame(
            {
                "s": ["A", "A", "B", "A", "C"],
                "p": ["r1", "r1", "r2", "r1", "r3"],
                "o": ["B", "B", "C", "B", "D"],
            }
        )  # A-r1-B appears 3 times

        # Act
        result = optimizer.remove_duplicates(df)

        # Assert
        assert len(result) == 3  # Only 3 unique triples
        # Use set comparison for values as order depends on implementation/index
        assert set(result["s"].to_list()) == {"A", "B", "C"}

    def test_preserves_unique_triples(self):
        """Should preserve all unique triples."""
        config = OptimizationConfig(
            min_entity_degree=1,
            min_relation_support=1,
            remove_duplicates=True,
            remove_self_loops=False,
            add_inverse_relations=False,
            balance_relations=False,
        )
        optimizer = TelecomDataOptimizer(config)

        # Arrange: All unique triples
        df = pl.DataFrame(
            {
                "s": ["A", "B", "C"],
                "p": ["r1", "r2", "r3"],
                "o": ["X", "Y", "Z"],
            }
        )

        # Act
        result = optimizer.remove_duplicates(df)

        # Assert
        assert len(result) == 3

    def test_handles_empty_dataframe(self):
        """Should handle empty DataFrame gracefully."""
        config = OptimizationConfig(
            min_entity_degree=1,
            min_relation_support=1,
            remove_duplicates=True,
            remove_self_loops=False,
            add_inverse_relations=False,
            balance_relations=False,
        )
        optimizer = TelecomDataOptimizer(config)

        # Arrange
        df = pl.DataFrame({"s": [], "p": [], "o": []})

        # Act
        result = optimizer.remove_duplicates(df)

        # Assert
        assert len(result) == 0


class TestRemoveSelfLoops:
    """Tests for self-loop removal."""

    def test_removes_self_loops(self):
        """Should remove triples where subject == object."""
        config = OptimizationConfig(
            min_entity_degree=1,
            min_relation_support=1,
            remove_duplicates=False,
            remove_self_loops=True,
            add_inverse_relations=False,
            balance_relations=False,
        )
        optimizer = TelecomDataOptimizer(config)

        # Arrange: DataFrame with self-loops
        df = pl.DataFrame(
            {
                "s": ["A", "B", "C", "D"],
                "p": ["r1", "r2", "r3", "r4"],
                "o": ["A", "C", "C", "E"],  # A->A and C->C are self-loops
            }
        )

        # Act
        result = optimizer.remove_self_loops(df)

        # Assert
        assert len(result) == 2
        assert "A" not in result["s"].to_list() or result.filter(pl.col("s") == "A")[
            "o"
        ].to_list() != ["A"]

    def test_preserves_non_self_loops(self):
        """Should preserve triples where subject != object."""
        config = OptimizationConfig(
            min_entity_degree=1,
            min_relation_support=1,
            remove_duplicates=False,
            remove_self_loops=True,
            add_inverse_relations=False,
            balance_relations=False,
        )
        optimizer = TelecomDataOptimizer(config)

        # Arrange: No self-loops
        df = pl.DataFrame(
            {
                "s": ["A", "B", "C"],
                "p": ["r1", "r2", "r3"],
                "o": ["X", "Y", "Z"],
            }
        )

        # Act
        result = optimizer.remove_self_loops(df)

        # Assert
        assert len(result) == 3


class TestAddInverseRelations:
    """Tests for inverse relation augmentation."""

    def test_doubles_triple_count(self):
        """Should double the number of triples."""
        config = OptimizationConfig(
            min_entity_degree=1,
            min_relation_support=1,
            remove_duplicates=False,
            remove_self_loops=False,
            add_inverse_relations=True,
            inverse_relation_suffix="_inv",
            balance_relations=False,
        )
        optimizer = TelecomDataOptimizer(config)

        # Arrange
        df = pl.DataFrame(
            {
                "s": ["A", "B"],
                "p": ["r1", "r2"],
                "o": ["X", "Y"],
            }
        )

        # Act
        result = optimizer.add_inverse_relations(df)

        # Assert
        assert len(result) == 4  # 2 original + 2 inverse

    def test_creates_inverse_triples_correctly(self):
        """Should swap subject and object for inverse triples."""
        config = OptimizationConfig(
            min_entity_degree=1,
            min_relation_support=1,
            remove_duplicates=False,
            remove_self_loops=False,
            add_inverse_relations=True,
            inverse_relation_suffix="_inv",
            balance_relations=False,
        )
        optimizer = TelecomDataOptimizer(config)

        # Arrange: (A, r1, B) should become (B, r1_inv, A)
        df = pl.DataFrame(
            {
                "s": ["A"],
                "p": ["r1"],
                "o": ["B"],
            }
        )

        # Act
        result = optimizer.add_inverse_relations(df)

        # Assert
        inverse = result.filter(pl.col("p") == "r1_inv")
        assert len(inverse) == 1
        assert inverse["s"][0] == "B"
        assert inverse["o"][0] == "A"

    def test_doubles_relation_count(self):
        """Should double the number of unique relations."""
        config = OptimizationConfig(
            min_entity_degree=1,
            min_relation_support=1,
            remove_duplicates=False,
            remove_self_loops=False,
            add_inverse_relations=True,
            inverse_relation_suffix="_inv",
            balance_relations=False,
        )
        optimizer = TelecomDataOptimizer(config)

        # Arrange: 2 unique relations
        df = pl.DataFrame(
            {
                "s": ["A", "B", "C"],
                "p": ["r1", "r2", "r1"],
                "o": ["X", "Y", "Z"],
            }
        )

        # Act
        result = optimizer.add_inverse_relations(df)

        # Assert
        assert result["p"].n_unique() == 4  # r1, r2, r1_inv, r2_inv

    def test_custom_suffix(self):
        """Should use custom suffix for inverse relations."""
        config = OptimizationConfig(
            min_entity_degree=1,
            min_relation_support=1,
            remove_duplicates=False,
            remove_self_loops=False,
            add_inverse_relations=True,
            inverse_relation_suffix="_reverse",
            balance_relations=False,
        )
        optimizer = TelecomDataOptimizer(config)

        # Arrange
        df = pl.DataFrame(
            {
                "s": ["A"],
                "p": ["rel"],
                "o": ["B"],
            }
        )

        # Act
        result = optimizer.add_inverse_relations(df)

        # Assert
        assert "rel_reverse" in result["p"].to_list()


class TestOptimizationConfigFromMapping:
    """Tests for OptimizationConfig.from_mapping()."""

    def test_loads_new_preprocessing_fields(self):
        """Should load new preprocessing config fields."""
        mapping = {
            "min_entity_degree": 1,
            "min_relation_support": 1,
            "remove_duplicates": False,
            "remove_self_loops": True,
            "add_inverse_relations": True,
            "inverse_relation_suffix": "_rev",
        }

        config = OptimizationConfig.from_mapping(mapping)

        assert config.remove_duplicates is False
        assert config.remove_self_loops is True
        assert config.add_inverse_relations is True
        assert config.inverse_relation_suffix == "_rev"

    def test_defaults_for_missing_fields(self):
        """Should use defaults for missing preprocessing fields."""
        config = OptimizationConfig.from_mapping(
            {
                "min_entity_degree": 1,
                "min_relation_support": 1,
            }
        )

        # New defaults should all be True for maximum data quality
        assert config.remove_duplicates is True
        assert config.remove_self_loops is True
        assert config.add_inverse_relations is True
        assert config.inverse_relation_suffix == "_inv"


class TestFullPipeline:
    """Integration tests for the full preprocessing pipeline."""

    def test_pipeline_order_is_correct(self):
        """Pipeline should apply steps in correct order:
        1. Remove duplicates (before inverses)
        2. Remove self-loops (before inverses)
        3. Add inverses (after cleanup)
        4. Filter sparse entities
        5. Balance relations
        """
        config = OptimizationConfig(
            remove_duplicates=True,
            remove_self_loops=True,
            add_inverse_relations=True,
            balance_relations=False,  # Skip for simplicity
            min_entity_degree=1,  # Keep all entities
            min_relation_support=1,
        )
        optimizer = TelecomDataOptimizer(config)

        # Arrange: Data with duplicates, self-loops, and valid triples
        df = pl.DataFrame(
            {
                "s": ["A", "A", "B", "C", "D", "E"],
                "p": ["r1", "r1", "r2", "r3", "r4", "r5"],
                "o": [
                    "B",
                    "B",
                    "C",
                    "C",
                    "E",
                    "F",
                ],  # A-r1-B is duplicate, C-r3-C is self-loop
            }
        )

        # Act - apply each step manually to verify order
        step1 = optimizer.remove_duplicates(df)  # Should have 5 triples
        step2 = optimizer.remove_self_loops(step1)  # Should have 4 triples
        step3 = optimizer.add_inverse_relations(step2)  # Should have 8 triples

        # Assert
        assert len(step1) == 5, f"After dedup: expected 5, got {len(step1)}"
        assert len(step2) == 4, f"After self-loop removal: expected 4, got {len(step2)}"
        assert len(step3) == 8, f"After inverse addition: expected 8, got {len(step3)}"

    def test_no_duplicate_inverses_created(self):
        """Should not create duplicate inverse triples when input has duplicates."""
        config = OptimizationConfig(
            remove_duplicates=True,
            remove_self_loops=False,
            add_inverse_relations=True,
            balance_relations=False,
            min_entity_degree=1,
            min_relation_support=1,
        )
        optimizer = TelecomDataOptimizer(config)

        # Arrange: Data with many duplicates (simulating 62% duplicate rate)
        df = pl.DataFrame(
            {
                "s": ["A"] * 5 + ["B"] * 3,
                "p": ["r1"] * 5 + ["r2"] * 3,
                "o": ["X"] * 5 + ["Y"] * 3,  # 5 copies of (A,r1,X), 3 copies of (B,r2,Y)
            }
        )

        # Act
        step1 = optimizer.remove_duplicates(df)
        step2 = optimizer.add_inverse_relations(step1)

        # Assert: Should have exactly 4 triples (2 unique + 2 inverses)
        assert len(step1) == 2, "Should have 2 unique triples"
        assert len(step2) == 4, "Should have 4 total triples (2 + 2 inverses)"
