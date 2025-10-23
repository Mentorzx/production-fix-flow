from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from pff.utils import FileManager, logger

"""
TransE Mapping Utilities

This module provides utilities for creating, loading, and validating entity/relation
mappings for TransE knowledge graph embeddings. It ensures consistency across
training, validation, and test sets.
"""


def create_raw_mappings(graph_dir: Path, output_dir: Path) -> tuple[Path, Path]:
    """
    Create entity and relation mappings from raw graph files.

    This function processes train_raw.txt, valid_raw.txt, and test_raw.txt files
    to create comprehensive entity and relation mappings that cover all splits.

    Args:
        graph_dir: Directory containing raw graph files (*.txt)
        output_dir: Directory where mappings will be saved

    Returns:
        Tuple of (entity_map_path, relation_map_path)

    Raises:
        FileNotFoundError: If required graph files are not found
        ValueError: If no entities or relations are found
    """
    file_manager = FileManager()

    logger.info("🔧 Creating raw mappings for TransE...")

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all entities and relations from all splits
    all_entities = set()
    all_relations = set()

    # Expected raw files
    raw_files = ["train_raw.txt", "valid_raw.txt", "test_raw.txt"]
    files_found = 0

    for filename in raw_files:
        file_path = graph_dir / filename

        if not file_path.exists():
            logger.warning(f"⚠️ File {filename} not found in {graph_dir}")
            continue

        files_found += 1
        logger.info(f"   Processing {filename}...")

        try:
            # Read file as TSV (tab-separated values)
            df = pl.read_csv(
                file_path,
                separator="\t",
                has_header=False,
                new_columns=["head", "relation", "tail"],
                schema_overrides={
                    "head": pl.Utf8,
                    "relation": pl.Utf8,
                    "tail": pl.Utf8,
                },
            )

            # Collect unique entities and relations
            entities_in_file = set(df["head"].to_list() + df["tail"].to_list())
            relations_in_file = set(df["relation"].to_list())

            all_entities.update(entities_in_file)
            all_relations.update(relations_in_file)

            logger.info(
                f"     Found: {len(entities_in_file):,} entities, "
                f"{len(relations_in_file)} relations"
            )

        except Exception as e:
            logger.error(f"❌ Error processing {filename}: {e}")
            raise

    if files_found == 0:
        raise FileNotFoundError(
            f"No raw graph files found in {graph_dir}. "
            "Expected files: train_raw.txt, valid_raw.txt, test_raw.txt"
        )

    if not all_entities:
        raise ValueError("No entities found in graph files!")

    if not all_relations:
        raise ValueError("No relations found in graph files!")

    logger.info(f"📊 Total unique entities: {len(all_entities):,}")
    logger.info(f"📊 Total unique relations: {len(all_relations)}")

    # Create entity mapping with sequential IDs
    entity_map = pl.DataFrame(
        {
            "id": range(len(all_entities)),
            "label": sorted(list(all_entities)),  # Sort for consistency
        }
    )

    # Create relation mapping with sequential IDs
    relation_map = pl.DataFrame(
        {
            "id": range(len(all_relations)),
            "label": sorted(list(all_relations)),  # Sort for consistency
        }
    )

    # Save mappings
    entity_map_path = output_dir / "transe_entity_map_raw.parquet"
    relation_map_path = output_dir / "transe_relation_map_raw.parquet"

    file_manager.save(entity_map, entity_map_path)
    file_manager.save(relation_map, relation_map_path)

    logger.success(
        f"✅ Mappings created successfully:\n"
        f"   Entity map: {entity_map_path}\n"
        f"   Relation map: {relation_map_path}"
    )

    return entity_map_path, relation_map_path


def load_mappings(
    entity_map_path: Path, relation_map_path: Path
) -> tuple[dict[str, int], dict[int, str], dict[str, int], dict[int, str]]:
    """
    Load entity and relation mappings from parquet files.

    Args:
        entity_map_path: Path to entity mapping parquet file
        relation_map_path: Path to relation mapping parquet file

    Returns:
        Tuple of (entity_to_idx, idx_to_entity, relation_to_idx, idx_to_relation)

    Raises:
        FileNotFoundError: If mapping files don't exist
        ValueError: If mapping files are corrupted or empty
    """
    file_manager = FileManager()

    logger.info("📂 Loading TransE mappings...")

    # Check if files exist
    if not entity_map_path.exists():
        raise FileNotFoundError(f"Entity mapping not found: {entity_map_path}")

    if not relation_map_path.exists():
        raise FileNotFoundError(f"Relation mapping not found: {relation_map_path}")

    # Load mappings
    try:
        entity_df = file_manager.read(entity_map_path)
        relation_df = file_manager.read(relation_map_path)
    except Exception as e:
        raise ValueError(f"Error loading mapping files: {e}")

    # Validate mapping structure
    if not {"id", "label"}.issubset(entity_df.columns):
        raise ValueError(
            f"Invalid entity mapping structure. "
            f"Expected columns: ['id', 'label'], got: {entity_df.columns}"
        )

    if not {"id", "label"}.issubset(relation_df.columns):
        raise ValueError(
            f"Invalid relation mapping structure. "
            f"Expected columns: ['id', 'label'], got: {relation_df.columns}"
        )

    # Check for empty mappings
    if len(entity_df) == 0:
        raise ValueError("Entity mapping is empty!")

    if len(relation_df) == 0:
        raise ValueError("Relation mapping is empty!")

    # Create bidirectional dictionaries
    entity_to_idx = dict(zip(entity_df["label"], entity_df["id"]))
    idx_to_entity = dict(zip(entity_df["id"], entity_df["label"]))
    relation_to_idx = dict(zip(relation_df["label"], relation_df["id"]))
    idx_to_relation = dict(zip(relation_df["id"], relation_df["label"]))

    logger.info(
        f"✅ Mappings loaded successfully:\n"
        f"   Entities: {len(entity_to_idx):,}\n"
        f"   Relations: {len(relation_to_idx)}"
    )

    return entity_to_idx, idx_to_entity, relation_to_idx, idx_to_relation


def validate_mappings(
    entity_to_idx: dict[str, int],
    relation_to_idx: dict[str, int],
    data_path: Path | None = None,
) -> dict[str, Any]:
    """
    Validate entity and relation mappings for consistency.

    This function checks:
    - Sequential IDs starting from 0
    - No duplicate IDs or labels
    - Optionally validates against data files

    Args:
        entity_to_idx: Entity to index mapping
        relation_to_idx: Relation to index mapping
        data_path: Optional path to data file for validation

    Returns:
        Dictionary with validation results and statistics

    Raises:
        ValueError: If validation fails
    """
    logger.info("🔍 Validating TransE mappings...")

    results = {"valid": True, "issues": [], "statistics": {}}

    # Check entity mapping
    entity_ids = sorted(entity_to_idx.values())
    expected_entity_ids = list(range(len(entity_ids)))

    if entity_ids != expected_entity_ids:
        results["valid"] = False
        results["issues"].append(
            f"Entity IDs are not sequential. "
            f"Expected: 0-{len(entity_ids) - 1}, "
            f"Got: {entity_ids[0]}-{entity_ids[-1]}"
        )

    # Check for duplicate entity IDs
    if len(entity_ids) != len(set(entity_ids)):
        results["valid"] = False
        results["issues"].append("Duplicate entity IDs found!")

    # Check relation mapping
    relation_ids = sorted(relation_to_idx.values())
    expected_relation_ids = list(range(len(relation_ids)))

    if relation_ids != expected_relation_ids:
        results["valid"] = False
        results["issues"].append(
            f"Relation IDs are not sequential. "
            f"Expected: 0-{len(relation_ids) - 1}, "
            f"Got: {relation_ids[0]}-{relation_ids[-1]}"
        )

    # Check for duplicate relation IDs
    if len(relation_ids) != len(set(relation_ids)):
        results["valid"] = False
        results["issues"].append("Duplicate relation IDs found!")

    # Validate against data if provided
    if data_path and data_path.exists():
        logger.info(f"   Validating against data: {data_path}")

        try:
            if data_path.suffix == ".parquet":
                df = pl.read_parquet(data_path)
                entities_in_data = set(df["s"].to_list() + df["o"].to_list())
                relations_in_data = set(df["p"].to_list())
            elif data_path.suffix == ".txt":
                df = pl.read_csv(
                    data_path,
                    separator="\t",
                    has_header=False,
                    new_columns=["s", "p", "o"],
                )
                entities_in_data = set(df["s"].to_list() + df["o"].to_list())
                relations_in_data = set(df["p"].to_list())
            else:
                logger.warning(f"⚠️ Unsupported file format: {data_path.suffix}")
                entities_in_data = set()
                relations_in_data = set()

            # Check coverage
            unmapped_entities = entities_in_data - set(entity_to_idx.keys())
            unmapped_relations = relations_in_data - set(relation_to_idx.keys())

            if unmapped_entities:
                results["valid"] = False
                results["issues"].append(
                    f"Found {len(unmapped_entities)} unmapped entities in data. "
                    f"Examples: {list(unmapped_entities)[:5]}"
                )

            if unmapped_relations:
                results["valid"] = False
                results["issues"].append(
                    f"Found {len(unmapped_relations)} unmapped relations in data. "
                    f"Examples: {list(unmapped_relations)[:5]}"
                )

            results["statistics"]["data_coverage"] = {
                "entities_in_data": len(entities_in_data),
                "relations_in_data": len(relations_in_data),
                "unmapped_entities": len(unmapped_entities),
                "unmapped_relations": len(unmapped_relations),
            }

        except Exception as e:
            logger.warning(f"⚠️ Could not validate against data: {e}")

    # Add general statistics
    results["statistics"]["mapping_sizes"] = {
        "total_entities": len(entity_to_idx),
        "total_relations": len(relation_to_idx),
    }

    # Log results
    if results["valid"]:
        logger.success("✅ Mappings are valid!")
    else:
        logger.error("❌ Mapping validation failed:")
        for issue in results["issues"]:
            logger.error(f"   - {issue}")

    return results


def convert_graph_to_indices(
    graph_path: Path,
    entity_to_idx: dict[str, int],
    relation_to_idx: dict[str, int],
    output_path: Path | None = None,
    use_optimized: bool = False,
) -> np.ndarray:
    """
    Convert a graph file to indexed numpy array format.

    Args:
        graph_path: Path to graph file (parquet or txt)
        entity_to_idx: Entity to index mapping
        relation_to_idx: Relation to index mapping
        output_path: Optional path to save the indexed array
        use_optimized: Whether to look for optimized version of the file

    Returns:
        Numpy array of shape (num_triples, 3) with columns [head_idx, rel_idx, tail_idx]

    Raises:
        FileNotFoundError: If graph file doesn't exist
        ValueError: If entities/relations in graph are not in mappings
    """
    logger.info(f"🔄 Converting graph to indices: {graph_path.name}")

    # Check for optimized version if requested
    if use_optimized:
        optimized_path = (
            graph_path.parent / f"{graph_path.stem}_optimized{graph_path.suffix}"
        )
        if optimized_path.exists():
            logger.info(f"   Using optimized version: {optimized_path.name}")
            graph_path = optimized_path

    if not graph_path.exists():
        raise FileNotFoundError(f"Graph file not found: {graph_path}")

    # Load graph data
    try:
        if graph_path.suffix == ".parquet":
            df = pl.read_parquet(graph_path)
        elif graph_path.suffix == ".txt":
            df = pl.read_csv(
                graph_path,
                separator="\t",
                has_header=False,
                new_columns=["s", "p", "o"],
            )
        else:
            raise ValueError(f"Unsupported file format: {graph_path.suffix}")
    except Exception as e:
        raise ValueError(f"Error loading graph file: {e}")

    # Convert to indices
    indexed_triples = []
    unmapped_entities = set()
    unmapped_relations = set()

    for row in df.iter_rows(named=True):
        head, relation, tail = row["s"], row["p"], row["o"]

        # Check if all elements are in mappings
        if head not in entity_to_idx:
            unmapped_entities.add(head)
            continue
        if tail not in entity_to_idx:
            unmapped_entities.add(tail)
            continue
        if relation not in relation_to_idx:
            unmapped_relations.add(relation)
            continue

        # Add indexed triple
        indexed_triples.append(
            [entity_to_idx[head], relation_to_idx[relation], entity_to_idx[tail]]
        )

    # Report unmapped elements
    if unmapped_entities:
        logger.warning(
            f"⚠️ Skipped {len(unmapped_entities)} unmapped entities. "
            f"Examples: {list(unmapped_entities)[:5]}"
        )

    if unmapped_relations:
        logger.warning(
            f"⚠️ Skipped {len(unmapped_relations)} unmapped relations. "
            f"Examples: {list(unmapped_relations)[:5]}"
        )

    # Convert to numpy array
    indexed_array = np.array(indexed_triples, dtype=np.int64)

    logger.info(
        f"✅ Converted {len(indexed_array):,} / {len(df):,} triples "
        f"({len(indexed_array) / len(df) * 100:.1f}% success rate)"
    )

    # Save if output path provided
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, indexed_array)
        logger.info(f"   Saved to: {output_path}")

    return indexed_array


def merge_mappings(
    mapping_paths: list[Path], output_path: Path, mapping_type: str = "entity"
) -> pl.DataFrame:
    """
    Merge multiple mapping files, ensuring unique IDs.

    Args:
        mapping_paths: List of paths to mapping files
        output_path: Path to save merged mapping
        mapping_type: Type of mapping ("entity" or "relation")

    Returns:
        Merged mapping DataFrame
    """
    file_manager = FileManager()

    logger.info(f"🔄 Merging {len(mapping_paths)} {mapping_type} mappings...")

    all_labels = set()

    # Collect all unique labels
    for path in mapping_paths:
        if path.exists():
            df = file_manager.read(path)
            all_labels.update(df["label"].to_list())

    # Create merged mapping with new sequential IDs
    merged_df = pl.DataFrame(
        {"id": range(len(all_labels)), "label": sorted(list(all_labels))}
    )

    # Save merged mapping
    file_manager.save(merged_df, output_path)

    logger.info(f"✅ Merged {len(all_labels):,} unique {mapping_type}s")

    return merged_df


def create_reverse_mappings(
    entity_to_idx: dict[str, int], relation_to_idx: dict[str, int]
) -> tuple[dict[int, str], dict[int, str]]:
    """
    Create reverse mappings (index to label).

    Args:
        entity_to_idx: Entity to index mapping
        relation_to_idx: Relation to index mapping

    Returns:
        Tuple of (idx_to_entity, idx_to_relation)
    """
    idx_to_entity = {idx: entity for entity, idx in entity_to_idx.items()}
    idx_to_relation = {idx: relation for relation, idx in relation_to_idx.items()}

    return idx_to_entity, idx_to_relation


def save_mappings_to_checkpoint(
    checkpoint_path: Path,
    entity_to_idx: dict[str, int],
    relation_to_idx: dict[str, int],
    additional_data: dict[str, Any] | None = None,
) -> None:
    """
    Save mappings to a checkpoint file.

    Args:
        checkpoint_path: Path to save checkpoint
        entity_to_idx: Entity to index mapping
        relation_to_idx: Relation to index mapping
        additional_data: Additional data to include in checkpoint
    """
    import torch

    checkpoint_data = {
        "entity_to_idx": entity_to_idx,
        "relation_to_idx": relation_to_idx,
        "num_entities": len(entity_to_idx),
        "num_relations": len(relation_to_idx),
    }

    if additional_data:
        checkpoint_data.update(additional_data)

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_data, checkpoint_path)

    logger.info(f"💾 Mappings saved to checkpoint: {checkpoint_path}")
