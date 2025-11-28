"""Tests for RotatE Mapping Utilities.

Tests for the mapping utilities module that handles entity and relation
mappings for RotatE models.

Author: PFF Team
Date: 2025-11-26
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest


class TestLoadMappings:
    """Tests for load_mappings function."""

    @pytest.fixture
    def mapping_files(self, tmp_path: Path) -> tuple[Path, Path]:
        """Create mapping files and return their paths."""
        # Entity mappings
        entity_df = pl.DataFrame({
            "id": [0, 1, 2, 3],
            "label": ["user:1", "user:2", "device:1", "device:2"],
        })
        entity_path = tmp_path / "entity_mappings.parquet"
        entity_df.write_parquet(entity_path)
        
        # Relation mappings
        relation_df = pl.DataFrame({
            "id": [0, 1, 2],
            "label": ["uses", "owns", "connects_to"],
        })
        relation_path = tmp_path / "relation_mappings.parquet"
        relation_df.write_parquet(relation_path)
        
        return entity_path, relation_path

    def test_load_mappings_from_parquet(self, mapping_files: tuple[Path, Path]):
        """Test loading mappings from parquet files."""
        from pff.validators.rotate.mapping_utils import load_mappings
        
        entity_path, relation_path = mapping_files
        entity_to_idx, idx_to_entity, relation_to_idx, idx_to_relation = load_mappings(
            entity_path, relation_path
        )
        
        assert len(entity_to_idx) == 4
        assert len(relation_to_idx) == 3
        assert entity_to_idx["user:1"] == 0
        assert relation_to_idx["uses"] == 0
        assert idx_to_entity[0] == "user:1"
        assert idx_to_relation[0] == "uses"

    def test_load_mappings_with_idx_column(self, tmp_path: Path):
        """Test loading mappings with idx column format."""
        # Entity mappings with idx format
        entity_df = pl.DataFrame({
            "idx": [0, 1, 2],
            "entity": ["e1", "e2", "e3"],
        })
        entity_path = tmp_path / "entity_mappings.parquet"
        entity_df.write_parquet(entity_path)
        
        # Relation mappings with idx format
        relation_df = pl.DataFrame({
            "idx": [0, 1],
            "relation": ["r1", "r2"],
        })
        relation_path = tmp_path / "relation_mappings.parquet"
        relation_df.write_parquet(relation_path)
        
        from pff.validators.rotate.mapping_utils import load_mappings
        
        entity_to_idx, _, relation_to_idx, _ = load_mappings(entity_path, relation_path)
        
        assert entity_to_idx["e1"] == 0
        assert relation_to_idx["r1"] == 0

    def test_load_mappings_with_index_column(self, tmp_path: Path):
        """Test loading mappings with index column format."""
        entity_df = pl.DataFrame({
            "index": [0, 1],
            "name": ["entity_a", "entity_b"],
        })
        entity_path = tmp_path / "entity_mappings.parquet"
        entity_df.write_parquet(entity_path)
        
        relation_df = pl.DataFrame({
            "index": [0],
            "name": ["relation_a"],
        })
        relation_path = tmp_path / "relation_mappings.parquet"
        relation_df.write_parquet(relation_path)
        
        from pff.validators.rotate.mapping_utils import load_mappings
        
        entity_to_idx, _, relation_to_idx, _ = load_mappings(entity_path, relation_path)
        
        assert entity_to_idx["entity_a"] == 0
        assert relation_to_idx["relation_a"] == 0

    def test_load_mappings_missing_entity_file_raises_error(self, tmp_path: Path):
        """Test that missing entity mapping file raises error."""
        from pff.validators.rotate.mapping_utils import load_mappings
        
        # Only create relation mappings
        relation_df = pl.DataFrame({"id": [0], "label": ["r1"]})
        relation_path = tmp_path / "relation_mappings.parquet"
        relation_df.write_parquet(relation_path)
        
        entity_path = tmp_path / "entity_mappings.parquet"  # Does not exist
        
        with pytest.raises(FileNotFoundError):
            load_mappings(entity_path, relation_path)

    def test_load_mappings_missing_relation_file_raises_error(self, tmp_path: Path):
        """Test that missing relation mapping file raises error."""
        from pff.validators.rotate.mapping_utils import load_mappings
        
        # Only create entity mappings
        entity_df = pl.DataFrame({"id": [0], "label": ["e1"]})
        entity_path = tmp_path / "entity_mappings.parquet"
        entity_df.write_parquet(entity_path)
        
        relation_path = tmp_path / "relation_mappings.parquet"  # Does not exist
        
        with pytest.raises(FileNotFoundError):
            load_mappings(entity_path, relation_path)


class TestParseMappingDf:
    """Tests for _parse_mapping_df function."""

    def test_parse_id_label_format(self):
        """Test parsing dataframe with id/label columns."""
        from pff.validators.rotate.mapping_utils import _parse_mapping_df
        
        df = pl.DataFrame({
            "id": [0, 1, 2],
            "label": ["a", "b", "c"],
        })
        
        name_to_idx, idx_to_name = _parse_mapping_df(df, "entity")
        
        assert name_to_idx == {"a": 0, "b": 1, "c": 2}
        assert idx_to_name == {0: "a", 1: "b", 2: "c"}

    def test_parse_idx_entity_format(self):
        """Test parsing dataframe with idx/entity columns."""
        from pff.validators.rotate.mapping_utils import _parse_mapping_df
        
        df = pl.DataFrame({
            "idx": [10, 20],
            "entity": ["x", "y"],
        })
        
        name_to_idx, idx_to_name = _parse_mapping_df(df, "entity")
        
        assert name_to_idx == {"x": 10, "y": 20}
        assert idx_to_name == {10: "x", 20: "y"}

    def test_parse_idx_relation_format(self):
        """Test parsing dataframe with idx/relation columns."""
        from pff.validators.rotate.mapping_utils import _parse_mapping_df
        
        df = pl.DataFrame({
            "idx": [0, 1],
            "relation": ["rel1", "rel2"],
        })
        
        name_to_idx, idx_to_name = _parse_mapping_df(df, "relation")
        
        assert name_to_idx == {"rel1": 0, "rel2": 1}
        assert idx_to_name == {0: "rel1", 1: "rel2"}

    def test_parse_index_name_format(self):
        """Test parsing dataframe with index/name columns."""
        from pff.validators.rotate.mapping_utils import _parse_mapping_df
        
        df = pl.DataFrame({
            "index": [5, 6, 7],
            "name": ["n1", "n2", "n3"],
        })
        
        name_to_idx, idx_to_name = _parse_mapping_df(df, "entity")
        
        assert name_to_idx == {"n1": 5, "n2": 6, "n3": 7}
        assert idx_to_name == {5: "n1", 6: "n2", 7: "n3"}

    def test_parse_fallback_to_first_columns(self):
        """Test that parsing falls back to first two columns."""
        from pff.validators.rotate.mapping_utils import _parse_mapping_df
        
        # Columns don't match known conventions, should use first two
        df = pl.DataFrame({
            "col_a": [0, 1],
            "col_b": ["x", "y"],
        })
        
        name_to_idx, idx_to_name = _parse_mapping_df(df, "entity")
        
        assert name_to_idx == {"x": 0, "y": 1}

    def test_parse_empty_dataframe(self):
        """Test parsing empty dataframe."""
        from pff.validators.rotate.mapping_utils import _parse_mapping_df
        
        df = pl.DataFrame({
            "id": [],
            "label": [],
        }).cast({"id": pl.Int64, "label": pl.Utf8})
        
        name_to_idx, idx_to_name = _parse_mapping_df(df, "entity")
        
        assert name_to_idx == {}
        assert idx_to_name == {}


class TestMappingsIntegration:
    """Integration tests for mapping utilities."""

    def test_mappings_roundtrip(self, tmp_path: Path):
        """Test that mappings can be saved and loaded correctly."""
        # Create original mappings
        original_entities = {"e1": 0, "e2": 1, "e3": 2}
        original_relations = {"r1": 0, "r2": 1}
        
        # Save to parquet
        entity_df = pl.DataFrame({
            "id": list(original_entities.values()),
            "label": list(original_entities.keys()),
        })
        entity_path = tmp_path / "entity_mappings.parquet"
        entity_df.write_parquet(entity_path)
        
        relation_df = pl.DataFrame({
            "id": list(original_relations.values()),
            "label": list(original_relations.keys()),
        })
        relation_path = tmp_path / "relation_mappings.parquet"
        relation_df.write_parquet(relation_path)
        
        # Load back
        from pff.validators.rotate.mapping_utils import load_mappings
        
        loaded_entity_to_idx, _, loaded_relation_to_idx, _ = load_mappings(
            entity_path, relation_path
        )
        
        assert loaded_entity_to_idx == original_entities
        assert loaded_relation_to_idx == original_relations

    def test_mappings_with_special_characters(self, tmp_path: Path):
        """Test mappings with special characters in labels."""
        entity_df = pl.DataFrame({
            "id": [0, 1, 2],
            "label": ["user:123", "device/456", "item-789"],
        })
        entity_path = tmp_path / "entity_mappings.parquet"
        entity_df.write_parquet(entity_path)
        
        relation_df = pl.DataFrame({
            "id": [0],
            "label": ["connects:to"],
        })
        relation_path = tmp_path / "relation_mappings.parquet"
        relation_df.write_parquet(relation_path)
        
        from pff.validators.rotate.mapping_utils import load_mappings
        
        entity_to_idx, _, relation_to_idx, _ = load_mappings(entity_path, relation_path)
        
        assert entity_to_idx["user:123"] == 0
        assert entity_to_idx["device/456"] == 1
        assert relation_to_idx["connects:to"] == 0

    def test_mappings_preserve_order(self, tmp_path: Path):
        """Test that mappings preserve index order."""
        # Create with non-sequential ids
        entity_df = pl.DataFrame({
            "id": [5, 10, 15, 20],
            "label": ["e5", "e10", "e15", "e20"],
        })
        entity_path = tmp_path / "entity_mappings.parquet"
        entity_df.write_parquet(entity_path)
        
        relation_df = pl.DataFrame({
            "id": [100, 200],
            "label": ["r100", "r200"],
        })
        relation_path = tmp_path / "relation_mappings.parquet"
        relation_df.write_parquet(relation_path)
        
        from pff.validators.rotate.mapping_utils import load_mappings
        
        entity_to_idx, _, relation_to_idx, _ = load_mappings(entity_path, relation_path)
        
        assert entity_to_idx["e5"] == 5
        assert entity_to_idx["e20"] == 20
        assert relation_to_idx["r100"] == 100

    def test_mappings_large_scale(self, tmp_path: Path):
        """Test mappings with large number of entities."""
        n_entities = 10000
        n_relations = 50
        
        entity_df = pl.DataFrame({
            "id": list(range(n_entities)),
            "label": [f"entity_{i}" for i in range(n_entities)],
        })
        entity_path = tmp_path / "entity_mappings.parquet"
        entity_df.write_parquet(entity_path)
        
        relation_df = pl.DataFrame({
            "id": list(range(n_relations)),
            "label": [f"relation_{i}" for i in range(n_relations)],
        })
        relation_path = tmp_path / "relation_mappings.parquet"
        relation_df.write_parquet(relation_path)
        
        from pff.validators.rotate.mapping_utils import load_mappings
        
        entity_to_idx, _, relation_to_idx, _ = load_mappings(entity_path, relation_path)
        
        assert len(entity_to_idx) == n_entities
        assert len(relation_to_idx) == n_relations
        assert entity_to_idx["entity_9999"] == 9999


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
