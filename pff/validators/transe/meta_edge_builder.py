from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from pff import settings
from pff.utils import FileManager, logger
from pff.utils.global_interrupt_manager import should_stop

"""
Meta Edge Builder for TransE

This module provides utilities for building meta-edges and
extracting meta-paths from knowledge graphs.
"""


class MetaEdgeBuilder:
    """
    Builder for meta-edges and meta-paths in knowledge graphs.

    Meta-edges represent patterns of connections between entity types,
    useful for understanding graph structure and improving embeddings.
    """

    def __init__(self, kg_config_path: Path | None = None):
        """
        Initialize meta edge builder.

        Args:
            kg_config_path: Path to KG configuration
        """
        self.file_manager = FileManager()
        self.kg_config_path = kg_config_path or settings.CONFIG_DIR / "kg.yaml"

        # Storage for meta information
        self.entity_types: dict[str, str] = {}
        self.relation_types: dict[str, tuple[str, str]] = {}
        self.meta_edges: set[tuple[str, str, str]] = set()
        self.meta_paths: list[list[str]] = []

        logger.info("✅ MetaEdgeBuilder initialized")

    def extract_entity_types(self, df: pl.DataFrame) -> dict[str, str]:
        """
        Extract entity types from knowledge graph.

        Args:
            df: DataFrame with columns [s, p, o]

        Returns:
            Dictionary mapping entity to type
        """
        logger.info("🔍 Extracting entity types...")

        entity_types = {}

        # Simple heuristic: entities with similar prefixes have same type
        all_entities = set(df["s"].to_list() + df["o"].to_list())

        for entity in all_entities:
            if should_stop():
                break

            # Extract type based on prefix patterns
            if entity.startswith("product_"):
                entity_types[entity] = "Product"
            elif entity.startswith("char_"):
                entity_types[entity] = "Characteristic"
            elif entity.startswith("offer_"):
                entity_types[entity] = "Offer"
            elif entity.startswith("status_"):
                entity_types[entity] = "Status"
            elif entity.startswith("date_"):
                entity_types[entity] = "Date"
            elif entity.startswith("id_"):
                entity_types[entity] = "Identifier"
            else:
                # Default type based on first token
                parts = entity.split("_")
                if parts:
                    entity_types[entity] = parts[0].capitalize()
                else:
                    entity_types[entity] = "Entity"

        self.entity_types = entity_types

        # Log type distribution
        type_counts = defaultdict(int)
        for entity_type in entity_types.values():
            type_counts[entity_type] += 1

        logger.info(f"✅ Found {len(type_counts)} entity types:")
        for entity_type, count in sorted(type_counts.items(), key=lambda x: -x[1])[:10]:
            logger.info(f"   {entity_type}: {count:,} entities")

        return entity_types

    def extract_relation_types(self, df: pl.DataFrame) -> dict[str, tuple[str, str]]:
        """
        Extract relation types (domain and range).

        Args:
            df: DataFrame with columns [s, p, o]

        Returns:
            Dictionary mapping relation to (domain_type, range_type)
        """
        logger.info("🔍 Extracting relation types...")

        if not self.entity_types:
            self.extract_entity_types(df)

        relation_types = {}

        # Group by relation
        for relation in df["p"].unique().to_list():
            if should_stop():
                break

            # Get all triples with this relation
            rel_df = df.filter(pl.col("p") == relation)

            # Find most common types for head and tail
            head_types = defaultdict(int)
            tail_types = defaultdict(int)

            for row in rel_df.iter_rows(named=True):
                head = row["s"]
                tail = row["o"]

                if head in self.entity_types:
                    head_types[self.entity_types[head]] += 1
                if tail in self.entity_types:
                    tail_types[self.entity_types[tail]] += 1

            # Get most common types
            if head_types and tail_types:
                domain_type = max(head_types.items(), key=lambda x: x[1])[0]
                range_type = max(tail_types.items(), key=lambda x: x[1])[0]
                relation_types[relation] = (domain_type, range_type)

        self.relation_types = relation_types

        logger.info(f"✅ Found {len(relation_types)} relation types")

        return relation_types

    def build_meta_edges(self, df: pl.DataFrame) -> set[tuple[str, str, str]]:
        """
        Build meta-edges from knowledge graph.

        Args:
            df: DataFrame with columns [s, p, o]

        Returns:
            Set of meta-edges (source_type, relation, target_type)
        """
        logger.info("🔧 Building meta-edges...")

        if not self.relation_types:
            self.extract_relation_types(df)

        meta_edges = set()

        for relation, (domain_type, range_type) in self.relation_types.items():
            meta_edge = (domain_type, relation, range_type)
            meta_edges.add(meta_edge)

            # Also add inverse for symmetric relations
            if domain_type == range_type:
                meta_edges.add((range_type, f"{relation}_inv", domain_type))

        self.meta_edges = meta_edges

        logger.info(f"✅ Built {len(meta_edges)} meta-edges")

        return meta_edges

    def find_meta_paths(
        self, df: pl.DataFrame, max_length: int = 3, min_support: int = 10
    ) -> list[list[str]]:
        """
        Find frequent meta-paths in the graph.

        Args:
            df: DataFrame with columns [s, p, o]
            max_length: Maximum path length
            min_support: Minimum occurrences

        Returns:
            List of meta-paths
        """
        logger.info(f"🔍 Finding meta-paths (max_length={max_length})...")

        if not self.entity_types:
            self.extract_entity_types(df)

        # Build adjacency list
        adjacency = defaultdict(list)

        for row in df.iter_rows(named=True):
            if should_stop():
                break

            head = row["s"]
            relation = row["p"]
            tail = row["o"]

            adjacency[head].append((relation, tail))

        # Find paths using BFS
        path_counts = defaultdict(int)

        # Sample entities to start paths
        entities = list(adjacency.keys())
        sample_size = min(1000, len(entities))
        sampled_entities = np.random.choice(entities, sample_size, replace=False)

        for start_entity in sampled_entities:
            if should_stop():
                break

            # BFS to find paths
            queue = [(start_entity, [])]
            visited = {start_entity}

            while queue:
                current, path = queue.pop(0)

                if len(path) >= max_length:
                    continue

                # Get entity type
                current_type = self.entity_types.get(current, "Unknown")

                # Explore neighbors
                for relation, neighbor in adjacency.get(current, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        neighbor_type = self.entity_types.get(neighbor, "Unknown")

                        # Create meta-path
                        new_path = path + [(current_type, relation, neighbor_type)]
                        meta_path = tuple(f"{t1}-{r}-{t2}" for t1, r, t2 in new_path)
                        path_counts[meta_path] += 1

                        queue.append((neighbor, new_path))

        # Filter by support
        frequent_paths = [
            list(path) for path, count in path_counts.items() if count >= min_support
        ]

        self.meta_paths = frequent_paths

        logger.info(f"✅ Found {len(frequent_paths)} frequent meta-paths")

        # Log top paths
        sorted_paths = sorted(path_counts.items(), key=lambda x: -x[1])[:10]

        logger.info("Top 10 meta-paths:")
        for path, count in sorted_paths:
            logger.info(f"   {' -> '.join(path)}: {count} occurrences")

        return frequent_paths

    def save_meta_information(self, output_dir: Path | None = None) -> None:
        """
        Save extracted meta information.

        Args:
            output_dir: Output directory (uses default if None)
        """
        if output_dir is None:
            output_dir = settings.OUTPUTS_DIR / "transe" / "meta"

        output_dir.mkdir(parents=True, exist_ok=True)

        # Save entity types
        if self.entity_types:
            entity_types_df = pl.DataFrame(
                {
                    "entity": list(self.entity_types.keys()),
                    "type": list(self.entity_types.values()),
                }
            )
            self.file_manager.save(entity_types_df, output_dir / "entity_types.parquet")

        # Save relation types
        if self.relation_types:
            relation_data = []
            for rel, (domain, range_) in self.relation_types.items():
                relation_data.append(
                    {"relation": rel, "domain": domain, "range": range_}
                )
            relation_types_df = pl.DataFrame(relation_data)
            self.file_manager.save(
                relation_types_df, output_dir / "relation_types.parquet"
            )

        # Save meta-edges
        if self.meta_edges:
            meta_edges_data = [
                {"source_type": s, "relation": r, "target_type": t}
                for s, r, t in self.meta_edges
            ]
            meta_edges_df = pl.DataFrame(meta_edges_data)
            self.file_manager.save(meta_edges_df, output_dir / "meta_edges.parquet")

        # Save meta-paths
        if self.meta_paths:
            meta_paths_data = {
                "meta_paths": self.meta_paths,
                "count": len(self.meta_paths),
            }
            self.file_manager.save(meta_paths_data, output_dir / "meta_paths.json")

        logger.info(f"✅ Meta information saved to: {output_dir}")

    def analyze_graph_structure(self, df: pl.DataFrame) -> dict[str, Any]:
        """
        Analyze graph structure using meta information.

        Args:
            df: DataFrame with columns [s, p, o]

        Returns:
            Dictionary with analysis results
        """
        logger.info("📊 Analyzing graph structure...")

        # Extract all meta information
        self.extract_entity_types(df)
        self.extract_relation_types(df)
        self.build_meta_edges(df)
        self.find_meta_paths(df)

        # Calculate statistics
        analysis = {
            "num_entities": len(self.entity_types),
            "num_entity_types": len(set(self.entity_types.values())),
            "num_relations": len(self.relation_types),
            "num_meta_edges": len(self.meta_edges),
            "num_meta_paths": len(self.meta_paths),
            "entity_type_distribution": defaultdict(int),
            "relation_type_distribution": defaultdict(int),
        }

        # Entity type distribution
        for entity_type in self.entity_types.values():
            analysis["entity_type_distribution"][entity_type] += 1

        # Relation type distribution
        for domain, range_ in self.relation_types.values():
            key = f"{domain} -> {range_}"
            analysis["relation_type_distribution"][key] += 1

        # Convert defaultdicts to regular dicts
        analysis["entity_type_distribution"] = dict(
            analysis["entity_type_distribution"]
        )
        analysis["relation_type_distribution"] = dict(
            analysis["relation_type_distribution"]
        )

        logger.info("✅ Graph structure analysis complete")

        return analysis


def extract_meta_information(
    data_path: Path, output_dir: Path | None = None
) -> dict[str, Any]:
    """
    Extract and save meta information from a knowledge graph.

    Args:
        data_path: Path to graph data (parquet file)
        output_dir: Output directory for meta information

    Returns:
        Dictionary with extraction results
    """
    logger.info(f"🚀 Extracting meta information from: {data_path}")

    # Load data
    file_manager = FileManager()
    df = file_manager.read(data_path)

    # Initialize builder
    builder = MetaEdgeBuilder()

    # Analyze structure
    analysis = builder.analyze_graph_structure(df)

    # Save results
    builder.save_meta_information(output_dir)

    logger.success("✅ Meta information extraction complete!")

    return analysis
