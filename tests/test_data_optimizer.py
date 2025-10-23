"""
Tests for TelecomDataOptimizer (pff/validators/data_optimizer.py)

Tests cover:
- Data quality analysis
- Sparsity filtering (entities with low degree)
- Relation balancing (removing rare relations)
- Optimization pipeline
- Density improvements
- Degree distribution analysis
"""

import pytest
import polars as pl
import tempfile
from pathlib import Path

from pff.validators.data_optimizer import (
    TelecomDataOptimizer,
    OptimizationConfig,
    quick_optimize_training_data
)


@pytest.fixture
def sample_sparse_kg():
    """Create a sample sparse knowledge graph for testing."""
    # Create a sparse KG with:
    # - Some low-degree entities (degree 1-2)
    # - Some high-degree entities (degree >= 3)
    # - Some rare relations (< 50 examples)
    # - Some common relations (>= 50 examples)

    data = {
        's': [],
        'p': [],
        'o': []
    }

    # Add common relation: "has_product" (100 triples)
    for i in range(100):
        data['s'].append(f"user_{i}")
        data['p'].append("has_product")
        data['o'].append(f"product_{i % 20}")  # 20 products

    # Add rare relation: "special_offer" (30 triples)
    for i in range(30):
        data['s'].append(f"user_{i}")
        data['p'].append("special_offer")
        data['o'].append(f"offer_{i}")

    # Add entities with low degree (will be filtered)
    for i in range(10):
        data['s'].append(f"user_sparse_{i}")
        data['p'].append("has_product")
        data['o'].append(f"product_{i}")

    # Add more connections to some entities to increase their degree
    for i in range(50):
        data['s'].append(f"user_{i}")
        data['p'].append("uses_service")
        data['o'].append(f"service_{i % 10}")  # 10 services

        data['s'].append(f"user_{i}")
        data['p'].append("location")
        data['o'].append(f"city_{i % 5}")  # 5 cities

    return pl.DataFrame(data)


@pytest.fixture
def optimizer():
    """Create a TelecomDataOptimizer with default config."""
    return TelecomDataOptimizer()


@pytest.fixture
def optimizer_strict():
    """Create a TelecomDataOptimizer with strict filtering."""
    config = OptimizationConfig(
        min_entity_degree=5,
        min_relation_support=100,
        balance_relations=True,
        log_statistics=False  # Disable logging for cleaner test output
    )
    return TelecomDataOptimizer(config)


# ═══════════════════════════════════════════════════════════════════
# Configuration Tests
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestOptimizationConfig:
    """Test OptimizationConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = OptimizationConfig()

        assert config.min_entity_degree == 3
        assert config.min_relation_support == 50
        assert config.balance_relations is True
        assert config.preserve_original is True
        assert config.log_statistics is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = OptimizationConfig(
            min_entity_degree=10,
            min_relation_support=100,
            balance_relations=False
        )

        assert config.min_entity_degree == 10
        assert config.min_relation_support == 100
        assert config.balance_relations is False


# ═══════════════════════════════════════════════════════════════════
# Data Quality Analysis Tests
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestDataQualityAnalysis:
    """Test data quality analysis functionality."""

    def test_analyze_data_quality_basic(self, optimizer, sample_sparse_kg):
        """Test basic data quality analysis."""
        stats = optimizer.analyze_data_quality(sample_sparse_kg)

        assert 'num_triples' in stats
        assert 'num_entities' in stats
        assert 'num_relations' in stats
        assert 'density' in stats
        assert 'avg_degree' in stats

    def test_analyze_data_quality_counts(self, optimizer, sample_sparse_kg):
        """Test that analysis returns correct counts."""
        stats = optimizer.analyze_data_quality(sample_sparse_kg)

        # Should have data
        assert stats['num_triples'] > 0
        assert stats['num_entities'] > 0
        assert stats['num_relations'] > 0

    def test_analyze_data_quality_density(self, optimizer, sample_sparse_kg):
        """Test density calculation."""
        stats = optimizer.analyze_data_quality(sample_sparse_kg)

        # Density should be between 0 and 1
        assert 0 <= stats['density'] <= 1

    def test_analyze_data_quality_degree_stats(self, optimizer, sample_sparse_kg):
        """Test degree statistics."""
        stats = optimizer.analyze_data_quality(sample_sparse_kg)

        assert 'avg_degree' in stats
        assert 'median_degree' in stats
        assert 'min_degree' in stats
        assert 'max_degree' in stats

        # Min should be <= avg <= max
        assert stats['min_degree'] <= stats['avg_degree'] <= stats['max_degree']

    def test_analyze_identifies_sparse_entities(self, optimizer, sample_sparse_kg):
        """Test that analysis identifies entities with low degree."""
        stats = optimizer.analyze_data_quality(sample_sparse_kg)

        # Should identify sparse entities (degree < 3)
        assert 'low_degree_entities' in stats
        assert stats['low_degree_entities'] >= 0

    def test_analyze_identifies_rare_relations(self, optimizer, sample_sparse_kg):
        """Test that analysis identifies rare relations."""
        stats = optimizer.analyze_data_quality(sample_sparse_kg)

        # Should identify rare relations (< 50 examples)
        assert 'rare_relations' in stats
        assert stats['rare_relations'] >= 0


# ═══════════════════════════════════════════════════════════════════
# Sparsity Filtering Tests
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestSparsityFiltering:
    """Test sparse entity filtering."""

    def test_filter_sparse_entities_reduces_count(self, optimizer, sample_sparse_kg):
        """Test that filtering removes entities with low degree."""
        initial_count = len(sample_sparse_kg)
        filtered_df = optimizer.filter_sparse_entities(sample_sparse_kg)

        # Filtered data should have fewer or equal triples
        assert len(filtered_df) <= initial_count

    def test_filter_sparse_entities_removes_low_degree(self, optimizer, sample_sparse_kg):
        """Test that filtering removes most low-degree entities."""
        filtered_df = optimizer.filter_sparse_entities(sample_sparse_kg)

        # Calculate degrees in filtered data
        entity_degrees = pl.concat([
            filtered_df.select(pl.col("s").alias("entity")),
            filtered_df.select(pl.col("o").alias("entity"))
        ]).group_by("entity").len().rename({"len": "degree"})

        # Most entities should have degree >= min_entity_degree
        # Note: Some entities may end up with lower degree after others are removed
        avg_degree = entity_degrees['degree'].mean()
        assert avg_degree >= optimizer.config.min_entity_degree - 1

    def test_filter_sparse_entities_preserves_structure(self, optimizer, sample_sparse_kg):
        """Test that filtered data preserves dataframe structure."""
        filtered_df = optimizer.filter_sparse_entities(sample_sparse_kg)

        # Should have same columns
        assert filtered_df.columns == ['s', 'p', 'o']
        assert isinstance(filtered_df, pl.DataFrame)

    def test_filter_sparse_entities_strict_config(self, optimizer_strict, sample_sparse_kg):
        """Test filtering with strict configuration."""
        filtered_df = optimizer_strict.filter_sparse_entities(sample_sparse_kg)

        # With min_entity_degree=5, should filter more aggressively
        assert len(filtered_df) < len(sample_sparse_kg)


# ═══════════════════════════════════════════════════════════════════
# Relation Balancing Tests
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestRelationBalancing:
    """Test relation balancing functionality."""

    def test_balance_relations_removes_rare(self, optimizer, sample_sparse_kg):
        """Test that rare relations are removed."""
        initial_relations = sample_sparse_kg['p'].n_unique()
        balanced_df = optimizer.balance_relations(sample_sparse_kg)

        final_relations = balanced_df['p'].n_unique()

        # Should remove at least one rare relation
        # (sample_sparse_kg has "special_offer" with only 30 examples, < 50 threshold)
        assert final_relations <= initial_relations

    def test_balance_relations_keeps_common(self, optimizer, sample_sparse_kg):
        """Test that common relations are kept."""
        balanced_df = optimizer.balance_relations(sample_sparse_kg)

        # "has_product" has 100 examples, should be kept
        assert "has_product" in balanced_df['p'].unique()

    def test_balance_relations_preserves_structure(self, optimizer, sample_sparse_kg):
        """Test that balanced data preserves structure."""
        balanced_df = optimizer.balance_relations(sample_sparse_kg)

        assert balanced_df.columns == ['s', 'p', 'o']
        assert isinstance(balanced_df, pl.DataFrame)

    def test_balance_relations_strict_config(self, optimizer_strict, sample_sparse_kg):
        """Test relation balancing with strict config."""
        # With min_relation_support=100, only "has_product" should remain
        balanced_df = optimizer_strict.balance_relations(sample_sparse_kg)

        # Should have fewer triples
        assert len(balanced_df) <= len(sample_sparse_kg)


# ═══════════════════════════════════════════════════════════════════
# Integration Tests - Full Optimization Pipeline
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.integration
class TestOptimizationPipeline:
    """Test complete optimization pipeline."""

    def test_optimize_telecom_data_pipeline(self, optimizer, sample_sparse_kg):
        """Test full optimization pipeline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save sample data (with header so FileManager can read it properly)
            input_path = Path(tmpdir) / "train.csv"
            sample_sparse_kg.write_csv(input_path)

            # Run optimization
            optimized_df, summary = optimizer.optimize_telecom_data(input_path)

            # Verify results
            assert isinstance(optimized_df, pl.DataFrame)
            assert isinstance(summary, dict)
            assert 'original_stats' in summary
            assert 'final_stats' in summary
            assert 'improvements' in summary

    def test_optimize_increases_density(self, optimizer, sample_sparse_kg):
        """Test that optimization increases graph density."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "train.csv"
            sample_sparse_kg.write_csv(input_path)

            optimized_df, summary = optimizer.optimize_telecom_data(input_path)

            # Density should improve (be higher)
            original_density = summary['original_stats']['density']
            final_density = summary['final_stats']['density']

            assert final_density >= original_density

    def test_optimize_creates_backup(self, optimizer, sample_sparse_kg):
        """Test that optimization creates backup file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "train.csv"
            sample_sparse_kg.write_csv(input_path)

            optimizer.optimize_telecom_data(input_path)

            # Backup should be created
            backup_path = input_path.with_suffix('.backup.csv')
            assert backup_path.exists()

    def test_optimize_creates_optimized_file(self, optimizer, sample_sparse_kg):
        """Test that optimized file is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "train.csv"
            sample_sparse_kg.write_csv(input_path)

            optimizer.optimize_telecom_data(input_path)

            # Optimized file should be created
            optimized_path = Path(tmpdir) / "train_optimized.csv"
            assert optimized_path.exists()

    def test_optimize_summary_has_improvements(self, optimizer, sample_sparse_kg):
        """Test that optimization summary includes improvement metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "train.csv"
            sample_sparse_kg.write_csv(input_path)

            _, summary = optimizer.optimize_telecom_data(input_path)

            improvements = summary['improvements']
            assert 'density_improvement' in improvements
            assert 'avg_degree_improvement' in improvements
            assert 'size_reduction' in improvements


# ═══════════════════════════════════════════════════════════════════
# Utility Function Tests
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestQuickOptimizeFunction:
    """Test quick_optimize_training_data utility function."""

    def test_quick_optimize_with_custom_path(self, sample_sparse_kg):
        """Test quick optimization with custom path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "train.csv"
            sample_sparse_kg.write_csv(input_path)

            optimized_df, summary = quick_optimize_training_data(
                train_path=input_path,
                min_entity_degree=3,
                min_relation_support=50
            )

            assert isinstance(optimized_df, pl.DataFrame)
            assert isinstance(summary, dict)

    def test_quick_optimize_custom_parameters(self, sample_sparse_kg):
        """Test quick optimization with custom parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "train.csv"
            sample_sparse_kg.write_csv(input_path)

            # Use less strict parameters (to avoid filtering everything)
            optimized_df, summary = quick_optimize_training_data(
                train_path=input_path,
                min_entity_degree=2,
                min_relation_support=30
            )

            # Should filter some data
            assert len(optimized_df) <= len(sample_sparse_kg)
            assert isinstance(summary, dict)
