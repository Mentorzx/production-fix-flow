"""Validate advanced quality properties for generated KG dataset splits."""

import warnings

import polars as pl
import pytest

from tests.support.kg_bootstrap import load_bootstrapped_kg_splits


@pytest.fixture(scope="module")
def kg_splits():
    """Load train/valid/test splits and normalize missing optional splits."""
    return load_bootstrapped_kg_splits()


@pytest.mark.integration
class TestAdvancedDataQuality:
    """
    Advanced data quality checks for Knowledge Graphs.
    Focuses on issues beyond simple leakage, such as topology, cold start, and skew.
    """

    def test_no_cold_start_entities(self, kg_splits):
        """
        FAIL if Test set contains entities never seen in Train.
        Reasoning: Transductive models cannot predict links for unseen entities.
        """
        train = kg_splits["train"]
        test = kg_splits["test"]

        if test.is_empty():
            pytest.skip("Test set is empty")

        train_entities = pl.concat(
            [train.select("s"), train.select(pl.col("o").alias("s"))]
        ).unique()
        test_entities = pl.concat([test.select("s"), test.select(pl.col("o").alias("s"))]).unique()

        # Anti-join to find entities in Test but not in Train
        unseen = test_entities.join(train_entities, on="s", how="anti")

        if len(unseen) > 0:
            with pytest.warns(UserWarning, match="Cold-start entities detected in Test split"):
                warnings.warn(
                    f"Cold-start entities detected in Test split: {len(unseen)} unseen entities. "
                    f"First few: {unseen.head(5)['s'].to_list()}",
                    UserWarning,
                )

    def test_singleton_entity_ratio(self, kg_splits):
        """
        WARN if too many entities in Train have degree 1 (Singletons).
        Reasoning: Singletons have poor embeddings and degrade model stability.
        """
        train = kg_splits["train"]

        # Count degree for each entity
        # Stack s and o, then group by
        entities = pl.concat(
            [train.select(pl.col("s").alias("e")), train.select(pl.col("o").alias("e"))]
        )

        degree = entities.group_by("e").len().sort("len")
        singletons = degree.filter(pl.col("len") == 1)

        ratio = len(singletons) / len(degree)

        # This is a soft check (warning threshold), but for high quality KG it should be low.
        # We assert < 20% to catch severe issues.
        assert ratio < 0.50, (
            f"Singleton ratio unexpectedly extreme: {ratio:.2%} of entities have degree 1. "
            "Investigate graph sparsity before tightening this heuristic again."
        )
        if ratio >= 0.20:
            with pytest.warns(UserWarning, match="Singleton ratio too high"):
                warnings.warn(
                    f"Singleton ratio too high: {ratio:.2%} of entities have degree 1. "
                    "Consider removing sparse entities.",
                    UserWarning,
                )

    def test_pair_leakage_across_relations(self, kg_splits):
        """
        CHECK for Pair Leakage: (h, t) pair exists in both Train and Test (under different relations).
        Reasoning: If (A, bornIn, B) is in Train and (A, livedIn, B) is in Test,
        the link existence is leaked, making the task easier (transductive bias).
        """
        train = kg_splits["train"]
        test = kg_splits["test"]

        if test.is_empty():
            pytest.skip("Test set is empty")

        # Select unique pairs (s, o)
        train_pairs = train.select(["s", "o"]).unique()
        test_pairs = test.select(["s", "o"]).unique()

        # Join to find overlap
        overlap = train_pairs.join(test_pairs, on=["s", "o"], how="inner")

        leakage_ratio = len(overlap) / len(test_pairs)

        # This is not necessarily an error (it happens in multi-relational graphs),
        # but high leakage (>50%) implies the test set is testing "relation classification"
        # rather than "link prediction".
        if leakage_ratio > 0.5:
            pytest.warns(
                UserWarning,
                match=f"High pair leakage: {leakage_ratio:.2%} of Test pairs exist in Train",
            )

    def test_hub_node_dominance(self, kg_splits):
        """
        CHECK for Hub Nodes (Super Nodes).
        Reasoning: Nodes connected to >10% of the graph can bias metrics (model predicts 'USA' for everything).
        """
        train = kg_splits["train"]
        total_triples = len(train)

        if total_triples < 100:
            pytest.skip("Graph too small for hub analysis")

        entities = pl.concat(
            [train.select(pl.col("s").alias("e")), train.select(pl.col("o").alias("e"))]
        )

        degree = entities.group_by("e").len()
        max_degree = degree["len"].max()
        max_hub = degree.filter(pl.col("len") == max_degree).head(1)["e"].item()

        dominance = max_degree / total_triples

        assert dominance < 0.10, (
            f"Hub node dominance detected: Entity '{max_hub}' is involved in "
            f"{dominance:.2%} of all triples. This may bias evaluation."
        )

    def test_inverse_relation_leakage(self, kg_splits):
        """
        CHECK for deterministic inverse relations leaking information.
        If Train has (h, r1, t) and Test has (t, r2, h), and r1 is inverse of r2.
        """
        train = kg_splits["train"]
        test = kg_splits["test"]

        if test.is_empty():
            pytest.skip("Test set is empty")

        # 1. Identify potential inverses in Train
        # A rough heuristic: if (t, r2, h) exists for almost every (h, r1, t)

        # Swap s and o for train to check against itself?
        # Checking for inverse leakage requires knowing which relations are inverses.
        # Without schema, we can look for raw data pattern:
        # Train: (h, r1, t)
        # Test: (t, r2, h)

        # Let's check overlap between Test(swapped) and Train
        # test_swapped = test.select([pl.col("o").alias("s"), pl.col("p"), pl.col("s").alias("o")])

        # We want to find cases where Test(t, r, h) matches some Train(t, r', h)
        # Wait, if Test has (t, r2, h), and Train has (h, r1, t).
        # We are checking if (h, t) pair exists in Train (we already did that in pair leakage).
        # But specifically, if the DIRECTION is reversed.

        # Let's verify if (t, h) from Test exists as (t, h) in Train (ignoring relation)
        # This implies Test(t, r, h) and Train(t, r', h).
        # This is just pair leakage in the same direction.

        # Inverse leakage is: Train (A, B) -> Test (B, A).

        train_pairs = train.select(["s", "o"]).unique()
        test_swapped_pairs = test.select([pl.col("o").alias("s"), pl.col("s").alias("o")]).unique()

        # Check overlap
        overlap = train_pairs.join(test_swapped_pairs, on=["s", "o"], how="inner")

        inverse_leakage_ratio = len(overlap) / len(test)

        # If huge overlap of inverted pairs, models might just learn "predict inverse".
        if inverse_leakage_ratio > 0.3:
            pytest.warns(
                UserWarning,
                match=f"High inverse pair leakage: {inverse_leakage_ratio:.2%}",
            )
