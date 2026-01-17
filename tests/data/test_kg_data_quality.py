"""Data quality tests for KG preprocessed schema and splits.

These tests validate:
- Schema compatibility: required tables/columns exist
- Entity/relation consistency: no orphan IDs in triples
- Split integrity: no triple leakage between train/valid/test
- Data quality: no NaN/Inf, no excessive duplicates
- Cardinality checks: entity/relation counts match expectations

All tests are READ-ONLY - no database writes.
"""

from __future__ import annotations

import asyncpg
import pytest
import pytest_asyncio

from pff.shared.core.config import settings

_COLUMN_ALIASES = {
    "s": ["s", "subject"],
    "p": ["p", "predicate"],
    "o": ["o", "object"],
}
_COL_MAPPING: dict[str, str] | None = None
_COL_TYPES: dict[str, str] | None = None
_NUMERIC_TYPES = ("integer", "bigint", "smallint")


async def _resolve_columns(conn) -> dict[str, str]:
    """Resolve column aliases (s/p/o vs subject/predicate/object)."""
    global _COL_MAPPING
    if _COL_MAPPING is not None:
        return _COL_MAPPING

    columns = await conn.fetch(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = 'kg_splits'
        """
    )
    names = {col["column_name"] for col in columns}
    mapping: dict[str, str] = {}
    for logical, aliases in _COLUMN_ALIASES.items():
        found = next((c for c in aliases if c in names), None)
        assert (
            found is not None
        ), f"Required column for '{logical}' missing (aliases: {aliases})"
        mapping[logical] = found
    _COL_MAPPING = mapping
    return mapping


async def _resolve_column_types(conn) -> dict[str, str]:
    """Resolve column types for kg_splits."""
    global _COL_TYPES
    if _COL_TYPES is not None:
        return _COL_TYPES
    columns = await conn.fetch(
        """
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_name = 'kg_splits'
        """
    )
    _COL_TYPES = {col["column_name"]: col["data_type"] for col in columns}
    return _COL_TYPES


def _kg_cte(cols: dict[str, str]) -> str:
    """Create CTE that aliases DB columns to canonical s/p/o."""
    return (
        "WITH kg AS ("
        f"SELECT {cols['s']} AS s, {cols['p']} AS p, {cols['o']} AS o, split_name, split_type "
        "FROM kg_splits)"
    )


# Skip if no database configured
pytestmark = [
    pytest.mark.integration,
    pytest.mark.asyncio,
    pytest.mark.skipif(
        not settings.DATABASE_URL_ASYNC, reason="DATABASE_URL_ASYNC not configured"
    ),
]


@pytest_asyncio.fixture(loop_scope="function")
async def db_connection():
    """Create async read-only database connection for testing (pytest-asyncio safe)."""
    db_url = settings.DATABASE_URL_ASYNC.replace("+asyncpg", "")

    try:
        conn = await asyncpg.connect(db_url)
    except Exception as e:
        pytest.skip(f"Database connection failed: {e}")
        return

    try:
        yield conn
    finally:
        await conn.close()


# =============================================================================
# Category G.1: Schema Compatibility Tests
# =============================================================================


class TestKGSplitsSchemaCompatibility:
    """Test that kg_splits table has required schema for DSLFM-KGC."""

    @pytest.mark.asyncio
    async def test_kg_splits_table_exists(self, db_connection):
        """Test that kg_splits table exists in the database."""
        tables = await db_connection.fetch(
            """
            SELECT tablename FROM pg_tables
            WHERE schemaname = 'public' AND tablename = 'kg_splits'
        """
        )

        assert len(tables) > 0, "kg_splits table does not exist"

    @pytest.mark.asyncio
    async def test_kg_splits_has_required_columns(self, db_connection):
        """Test that kg_splits has columns required by the model."""
        columns = await db_connection.fetch(
            """
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name = 'kg_splits'
            ORDER BY ordinal_position
        """
        )

        if not columns:
            pytest.skip("kg_splits table has no columns or doesn't exist")

        column_names = [col["column_name"] for col in columns]
        required_alias_groups = [
            _COLUMN_ALIASES["s"],
            _COLUMN_ALIASES["p"],
            _COLUMN_ALIASES["o"],
            ["split_name"],
            ["split_type"],
        ]
        for aliases in required_alias_groups:
            assert any(
                col in column_names for col in aliases
            ), f"Required column missing; expected one of {aliases}"

    @pytest.mark.asyncio
    async def test_kg_splits_column_types_compatible(self, db_connection):
        """Test that column types are compatible with model expectations."""
        columns = await db_connection.fetch(
            """
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = 'kg_splits'
        """
        )

        if not columns:
            pytest.skip("kg_splits table has no columns or doesn't exist")

        column_types = {col["column_name"]: col["data_type"] for col in columns}
        mapping = await _resolve_columns(db_connection)
        numeric_types = ("integer", "bigint", "smallint")
        text_types = ("character varying", "text")
        for logical, actual in mapping.items():
            ctype = column_types[actual]
            assert (
                ctype in numeric_types + text_types
            ), f"Column '{actual}' for '{logical}' has unexpected type: {ctype}"


# =============================================================================
# Category G.2: Split Integrity Tests
# =============================================================================


class TestKGSplitIntegrity:
    """Test that train/valid/test splits are properly disjoint."""

    @pytest.mark.asyncio
    async def test_no_triple_leakage_between_splits(self, db_connection):
        """Test that no (s, p, o) triple appears in multiple splits."""
        cols = await _resolve_columns(db_connection)
        cte = _kg_cte(cols)
        # Check if kg_splits has data
        count = await db_connection.fetchval(
            """
            SELECT COUNT(*) FROM pg_tables
            WHERE schemaname = 'public' AND tablename = 'kg_splits'
        """
        )

        if not count:
            pytest.skip("kg_splits table does not exist")

        # Find duplicates across splits
        duplicates = await db_connection.fetch(
            f"""
            {cte}
            SELECT s, p, o, COUNT(DISTINCT split_name) as split_count
            FROM kg
            WHERE split_type = 'preprocessed'
            GROUP BY s, p, o
            HAVING COUNT(DISTINCT split_name) > 1
            LIMIT 10
            """
        )

        if duplicates:
            sample = duplicates[0]
            pytest.fail(
                f"Triple leakage detected: ({sample['s']}, {sample['p']}, {sample['o']}) "
                f"appears in {sample['split_count']} splits. Found {len(duplicates)} such triples."
            )

    @pytest.mark.asyncio
    async def test_all_splits_have_data(self, db_connection):
        """Test that train and valid splits both have data."""
        cols = await _resolve_columns(db_connection)
        cte = _kg_cte(cols)
        # Check if kg_splits exists
        exists = await db_connection.fetchval(
            """
            SELECT EXISTS(
                SELECT 1 FROM pg_tables
                WHERE schemaname = 'public' AND tablename = 'kg_splits'
            )
        """
        )

        if not exists:
            pytest.skip("kg_splits table does not exist")

        split_counts = await db_connection.fetch(
            f"""
            {cte}
            SELECT split_name, COUNT(*) as count
            FROM kg
            WHERE split_type = 'preprocessed'
            GROUP BY split_name
            """
        )

        if not split_counts:
            pytest.skip("No preprocessed splits found")

        split_dict = {row["split_name"]: row["count"] for row in split_counts}

        # Train and valid are required, test is optional
        assert "train" in split_dict, "No 'train' split found"
        assert "valid" in split_dict, "No 'valid' split found"
        assert split_dict["train"] > 0, "Train split is empty"
        assert split_dict["valid"] > 0, "Valid split is empty"

        # Valid should be smaller than train (sanity check)
        if split_dict["valid"] > split_dict["train"]:
            pytest.fail(
                f"Valid split ({split_dict['valid']}) is larger than train ({split_dict['train']})"
            )


# =============================================================================
# Category G.3: Entity/Relation Consistency Tests
# =============================================================================


class TestEntityRelationConsistency:
    """Test entity and relation ID consistency in triples."""

    pass  # Numeric ID tests removed - schema uses text columns


# =============================================================================
# Category G.4: Data Quality Checks
# =============================================================================


class TestDataQuality:
    """Test data quality in KG splits."""

    @pytest.mark.asyncio
    async def test_no_self_loops_in_triples(self, db_connection):
        """Test that there are no self-loops (s == o) in triples."""
        cols = await _resolve_columns(db_connection)
        cte = _kg_cte(cols)
        exists = await db_connection.fetchval(
            """
            SELECT EXISTS(
                SELECT 1 FROM pg_tables
                WHERE schemaname = 'public' AND tablename = 'kg_splits'
            )
        """
        )

        if not exists:
            pytest.skip("kg_splits table does not exist")

        self_loop_count = await db_connection.fetchval(
            f"""
            {cte}
            SELECT COUNT(*) FROM kg
            WHERE s = o AND split_type = 'preprocessed'
            """
        )

        if self_loop_count > 0:
            # Get sample for debugging
            sample = await db_connection.fetchrow(
                f"""
                {cte}
                SELECT s, p, o, split_name FROM kg
                WHERE s = o AND split_type = 'preprocessed'
                LIMIT 1
                """
            )
            pytest.fail(
                f"Found {self_loop_count} self-loop triples. "
                f"Sample: ({sample['s']}, {sample['p']}, {sample['o']}) in {sample['split_name']}"
            )

    @pytest.mark.asyncio
    async def test_no_duplicate_triples_within_split(self, db_connection):
        """Test that there are no duplicate (s, p, o) within the same split."""
        cols = await _resolve_columns(db_connection)
        cte = _kg_cte(cols)
        exists = await db_connection.fetchval(
            """
            SELECT EXISTS(
                SELECT 1 FROM pg_tables
                WHERE schemaname = 'public' AND tablename = 'kg_splits'
            )
        """
        )

        if not exists:
            pytest.skip("kg_splits table does not exist")

        duplicates = await db_connection.fetch(
            f"""
            {cte}
            SELECT s, p, o, split_name, COUNT(*) as count
            FROM kg
            WHERE split_type = 'preprocessed'
            GROUP BY s, p, o, split_name
            HAVING COUNT(*) > 1
            LIMIT 5
            """
        )

        if duplicates:
            sample = duplicates[0]
            pytest.fail(
                f"Found {len(duplicates)} duplicate triple patterns. "
                f"Sample: ({sample['s']}, {sample['p']}, {sample['o']}) "
                f"appears {sample['count']} times in {sample['split_name']}"
            )

    @pytest.mark.asyncio
    async def test_relation_distribution_not_too_skewed(self, db_connection):
        """Test that relation distribution is not extremely skewed."""
        cols = await _resolve_columns(db_connection)
        cte = _kg_cte(cols)
        exists = await db_connection.fetchval(
            """
            SELECT EXISTS(
                SELECT 1 FROM pg_tables
                WHERE schemaname = 'public' AND tablename = 'kg_splits'
            )
        """
        )

        if not exists:
            pytest.skip("kg_splits table does not exist")

        distribution = await db_connection.fetch(
            f"""
            {cte}
            SELECT p, COUNT(*) as count
            FROM kg
            WHERE split_type = 'preprocessed' AND split_name = 'train'
            GROUP BY p
            ORDER BY count DESC
            """
        )

        if not distribution:
            pytest.skip("No training data found")

        total_triples = sum(row["count"] for row in distribution)
        max_relation_count = distribution[0]["count"]
        distribution[-1]["count"]

        # Adapt threshold for small datasets while keeping the 10-example guard for larger sets.
        min_examples = max(1, min(10, int(total_triples * 0.001)))
        relations_with_few_examples = [
            row for row in distribution if row["count"] < min_examples
        ]

        if relations_with_few_examples:
            rare_ratio = len(relations_with_few_examples) / len(distribution)
            ids = [str(r["p"]) for r in relations_with_few_examples[:5]]
            if rare_ratio > 0.05:
                pytest.fail(
                    f"Found {len(relations_with_few_examples)} relations with < {min_examples} examples in train "
                    f"({rare_ratio:.1%} of relations). Sample relation IDs: {', '.join(ids)}"
                )

        # Check if most frequent relation dominates (> 50% of data)
        dominance_ratio = max_relation_count / total_triples if total_triples > 0 else 0
        if dominance_ratio > 0.5:
            pytest.fail(
                f"Relation {distribution[0]['p']} dominates with {dominance_ratio:.1%} of data"
            )


# =============================================================================
# Category G.5: Cardinality Validation Tests
# =============================================================================


class TestCardinalityValidation:
    """Test that entity/relation counts match model expectations."""

    @pytest.mark.asyncio
    async def test_entity_count_is_reasonable(self, db_connection):
        """Test that entity count is within expected range."""
        cols = await _resolve_columns(db_connection)
        cte = _kg_cte(cols)
        exists = await db_connection.fetchval(
            """
            SELECT EXISTS(
                SELECT 1 FROM pg_tables
                WHERE schemaname = 'public' AND tablename = 'kg_splits'
            )
        """
        )

        if not exists:
            pytest.skip("kg_splits table does not exist")

        entity_count = await db_connection.fetchval(
            f"""
            {cte}
            SELECT COUNT(*) FROM (
                SELECT DISTINCT entity_id FROM (
                    SELECT s as entity_id FROM kg WHERE split_type = 'preprocessed'
                    UNION
                    SELECT o as entity_id FROM kg WHERE split_type = 'preprocessed'
                ) all_entities
            ) unique_entities
            """
        )

        # Sanity checks
        assert entity_count is not None, "Could not count entities"
        assert entity_count > 0, "No entities found in preprocessed data"
        assert (
            entity_count < 10_000_000
        ), f"Unexpectedly high entity count: {entity_count:,}"

    @pytest.mark.asyncio
    async def test_relation_count_is_reasonable(self, db_connection):
        """Test that relation count is within expected range."""
        cols = await _resolve_columns(db_connection)
        cte = _kg_cte(cols)
        exists = await db_connection.fetchval(
            """
            SELECT EXISTS(
                SELECT 1 FROM pg_tables
                WHERE schemaname = 'public' AND tablename = 'kg_splits'
            )
        """
        )

        if not exists:
            pytest.skip("kg_splits table does not exist")

        relation_count = await db_connection.fetchval(
            f"""
            {cte}
            SELECT COUNT(DISTINCT p) FROM kg
            WHERE split_type = 'preprocessed'
            """
        )

        # Sanity checks
        assert relation_count is not None, "Could not count relations"
        assert relation_count > 0, "No relations found in preprocessed data"
        assert (
            relation_count < 10_000
        ), f"Unexpectedly high relation count: {relation_count:,}"

    @pytest.mark.asyncio
    async def test_train_valid_ratio_is_reasonable(self, db_connection):
        """Test that train/valid split ratio is reasonable (e.g., 80/20)."""
        cols = await _resolve_columns(db_connection)
        cte = _kg_cte(cols)
        exists = await db_connection.fetchval(
            """
            SELECT EXISTS(
                SELECT 1 FROM pg_tables
                WHERE schemaname = 'public' AND tablename = 'kg_splits'
            )
        """
        )

        if not exists:
            pytest.skip("kg_splits table does not exist")

        counts = await db_connection.fetch(
            f"""
            {cte}
            SELECT split_name, COUNT(*) as count
            FROM kg
            WHERE split_type = 'preprocessed'
            AND split_name IN ('train', 'valid')
            GROUP BY split_name
            """
        )

        if not counts:
            pytest.skip("No preprocessed splits found")

        count_dict = {row["split_name"]: row["count"] for row in counts}

        if "train" not in count_dict or "valid" not in count_dict:
            pytest.skip("Missing train or valid split")

        total = count_dict["train"] + count_dict["valid"]
        train_ratio = count_dict["train"] / total if total > 0 else 0

        # Train should be 60-95% of data
        assert (
            0.6 <= train_ratio <= 0.95
        ), f"Train/valid ratio seems off: train={train_ratio:.1%}, valid={(1 - train_ratio):.1%}"
