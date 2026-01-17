"""
Advanced Preprocessing Strategies for KG Data (SOTA Implementation).

Design Pattern: Strategy Pattern
- HubDownsamplingStrategy: Reduce dominance of high-degree nodes
- SemanticInverseStrategy: Create semantically named inverse relations
- EntityResolutionStrategy: Deduplicate similar entities
- RelationCardinalityClassifier: Classify 1:1, 1:N, N:1, N:N patterns
- PathCountingStrategy: Count k-hop paths for feature extraction
- TextualizationStrategy: Generate BERT-ready text from triples

References:
- Entity Resolution: Christophides et al. (2020) "End-to-End Entity Resolution"
- KG Preprocessing: Bordes et al. (2013) "Translating Embeddings"
- Graph Sampling: Hamilton et al. (2017) "Inductive Representation Learning"
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field

import polars as pl
from scipy import sparse
import numpy as np

from pff.shared import logger

from .strategies import PreprocessingStrategy, ProcessingResult


# ═══════════════════════════════════════════════════════════════════════════
# (A) HUB DOWNSAMPLING - Reduce dominance of high-degree nodes
# ═══════════════════════════════════════════════════════════════════════════


class HubDownsamplingStrategy(PreprocessingStrategy):
    """Downsample edges from hub nodes to reduce their dominance.

    Hub nodes (high-degree entities) can dominate embedding learning by
    appearing in too many training samples. This strategy reduces their
    influence by randomly sampling a subset of their edges.

    Strategy: For entities above the hub threshold, randomly sample
    max_edges_per_hub edges, preserving edge diversity where possible.

    References:
        - GraphSAINT: Graph Sampling Based Inductive Learning (ICLR 2020)
        - Hamilton et al. "Inductive Representation Learning on Large Graphs"
    """

    def __init__(
        self,
        percentile: float = 0.99,
        max_edges_per_hub: int | None = None,
        sampling_factor: float = 0.1,
        seed: int = 42,
    ) -> None:
        """Initialize hub downsampling strategy.

        Args:
            percentile: Percentile threshold for hub detection (default: 99th)
            max_edges_per_hub: Maximum edges to keep per hub (if None, uses median)
            sampling_factor: Fraction of hub edges to keep (alternative to max_edges)
            seed: Random seed for reproducibility
        """
        self.percentile = percentile
        self.max_edges_per_hub = max_edges_per_hub
        self.sampling_factor = sampling_factor
        self.seed = seed

    @property
    def name(self) -> str:
        return "hub_downsampling"

    def process(self, df: pl.DataFrame) -> ProcessingResult:
        """Downsample edges from hub nodes.

        Args:
            df: DataFrame with columns [s, p, o]

        Returns:
            ProcessingResult with downsampled data
        """
        random.seed(self.seed)
        initial_count = len(df)

        subject_degrees = df.group_by("s").agg(pl.len().alias("out_degree"))

        object_degrees = df.group_by("o").agg(pl.len().alias("in_degree"))

        # Merge to get total degree
        entity_degrees = (
            subject_degrees.join(
                object_degrees.rename({"o": "s"}), on="s", how="full", coalesce=True
            )
            .with_columns(
                [
                    pl.col("out_degree").fill_null(0),
                    pl.col("in_degree").fill_null(0),
                ]
            )
            .with_columns((pl.col("out_degree") + pl.col("in_degree")).alias("total_degree"))
        )

        # Identify hubs
        hub_threshold = entity_degrees["total_degree"].quantile(self.percentile)
        if hub_threshold is None or hub_threshold == 0:
            logger.info("[HUB DOWNSAMPLING] Nenhum hub detectado, dados inalterados")
            return ProcessingResult(data=df, stats={"hub_threshold": 0, "n_hubs": 0})

        hub_entities = set(
            entity_degrees.filter(pl.col("total_degree") >= hub_threshold)["s"].to_list()
        )

        if not hub_entities:
            logger.info("[HUB DOWNSAMPLING] Nenhum hub detectado, dados inalterados")
            return ProcessingResult(
                data=df, stats={"hub_threshold": float(hub_threshold), "n_hubs": 0}
            )

        # Determine max edges per hub
        if self.max_edges_per_hub is not None:
            max_edges = self.max_edges_per_hub
        else:
            # Use median degree as target
            median_degree = entity_degrees["total_degree"].median()
            max_edges = int(median_degree) if median_degree else 50

        # Separate hub edges from non-hub edges
        is_hub_subject = df["s"].is_in(list(hub_entities))
        is_hub_object = df["o"].is_in(list(hub_entities))
        hub_edge_mask = is_hub_subject | is_hub_object

        non_hub_edges = df.filter(~hub_edge_mask)
        hub_edges = df.filter(hub_edge_mask)

        # Downsample hub edges by entity
        sampled_hub_edges = []
        for entity in hub_entities:
            entity_edges = hub_edges.filter((pl.col("s") == entity) | (pl.col("o") == entity))

            n_edges = len(entity_edges)
            if n_edges <= max_edges:
                sampled_hub_edges.append(entity_edges)
            else:
                # Random sample preserving relation diversity
                sample_indices = random.sample(range(n_edges), max_edges)
                sampled_hub_edges.append(entity_edges[sample_indices])

        # Combine results
        if sampled_hub_edges:
            sampled_df = pl.concat(sampled_hub_edges).unique(subset=["s", "p", "o"])
            result_df = pl.concat([non_hub_edges, sampled_df]).unique(subset=["s", "p", "o"])
        else:
            result_df = non_hub_edges

        final_count = len(result_df)
        edges_removed = initial_count - final_count

        stats = {
            "initial_triples": initial_count,
            "final_triples": final_count,
            "edges_removed": edges_removed,
            "hub_threshold": float(hub_threshold),
            "n_hubs": len(hub_entities),
            "max_edges_per_hub": max_edges,
        }

        logger.info(
            f"[HUB DOWNSAMPLING] {len(hub_entities):,} hubs identificados "
            f"(threshold={hub_threshold:.0f}), {edges_removed:,} arestas removidas"
        )

        return ProcessingResult(data=result_df, stats=stats)


# ═══════════════════════════════════════════════════════════════════════════
# (C) SEMANTIC INVERSE RELATIONS
# ═══════════════════════════════════════════════════════════════════════════


class SemanticInverseStrategy(PreprocessingStrategy):
    """Add inverse relations with semantic naming.

    Instead of generic `_inv` suffix, this strategy attempts to map
    relations to their semantic inverses when known, falling back to
    suffix for unknown relations.

    Examples:
        - worksIn → employs
        - hasParent → hasChild
        - partOf → hasPart
        - locatedIn → contains
    """

    # Default semantic inverse mappings (telecom/general domain)
    DEFAULT_SEMANTIC_INVERSES: dict[str, str] = {
        # Organizational
        "worksIn": "employs",
        "worksFor": "employs",
        "employedBy": "employs",
        "memberOf": "hasMember",
        "belongsTo": "contains",
        # Hierarchical
        "hasParent": "hasChild",
        "parentOf": "childOf",
        "subclassOf": "superclassOf",
        "partOf": "hasPart",
        "containedIn": "contains",
        # Spatial
        "locatedIn": "contains",
        "locatedAt": "locationOf",
        "nearTo": "nearTo",  # Symmetric
        # Telecom-specific
        "subscribedTo": "hasSubscriber",
        "usesService": "providedTo",
        "hasProvider": "provides",
        "connectedTo": "connectedFrom",
        "routedThrough": "routes",
        # Generic
        "hasType": "typeOf",
        "hasValue": "valueOf",
        "relatedTo": "relatedTo",  # Symmetric
    }

    def __init__(
        self,
        semantic_mappings: dict[str, str] | None = None,
        fallback_suffix: str = "_inv",
        case_insensitive: bool = True,
    ) -> None:
        """Initialize semantic inverse strategy.

        Args:
            semantic_mappings: Custom relation → inverse mappings
            fallback_suffix: Suffix for unmapped relations
            case_insensitive: Match relation names case-insensitively
        """
        self.mappings = {**self.DEFAULT_SEMANTIC_INVERSES}
        if semantic_mappings:
            self.mappings.update(semantic_mappings)

        self.fallback_suffix = fallback_suffix
        self.case_insensitive = case_insensitive

        # Build case-insensitive lookup if needed
        if case_insensitive:
            self._lookup = {k.lower(): v for k, v in self.mappings.items()}
        else:
            self._lookup = self.mappings

    @property
    def name(self) -> str:
        return "semantic_inverse"

    def _get_inverse_name(self, relation: str) -> str:
        """Get inverse relation name (semantic or fallback).

        Args:
            relation: Original relation name

        Returns:
            Inverse relation name
        """
        lookup_key = relation.lower() if self.case_insensitive else relation
        if lookup_key in self._lookup:
            return self._lookup[lookup_key]
        return f"{relation}{self.fallback_suffix}"

    def process(self, df: pl.DataFrame) -> ProcessingResult:
        """Add semantically named inverse triples.

        Args:
            df: DataFrame with columns [s, p, o]

        Returns:
            ProcessingResult with original + inverse triples
        """
        initial_count = len(df)
        original_relations = df["p"].n_unique()

        unique_relations = df["p"].unique().to_list()
        inverse_map = {r: self._get_inverse_name(r) for r in unique_relations}

        semantic_count = sum(
            1
            for r in unique_relations
            if (r.lower() if self.case_insensitive else r) in self._lookup
        )

        inverse_df = df.with_columns(
            [
                pl.col("o").alias("s_new"),
                pl.col("p").replace(inverse_map).alias("p_new"),
                pl.col("s").alias("o_new"),
            ]
        ).select(
            [
                pl.col("s_new").alias("s"),
                pl.col("p_new").alias("p"),
                pl.col("o_new").alias("o"),
            ]
        )

        # Concatenate original + inverse
        result_df = pl.concat([df, inverse_df])

        final_count = len(result_df)
        final_relations = result_df["p"].n_unique()

        stats = {
            "initial_triples": initial_count,
            "final_triples": final_count,
            "inverse_triples_added": len(inverse_df),
            "original_relations": original_relations,
            "final_relations": final_relations,
            "semantic_mappings_used": semantic_count,
            "fallback_mappings_used": len(unique_relations) - semantic_count,
        }

        metadata = {
            "inverse_mapping": inverse_map,
            "semantic_relations": [
                r
                for r in unique_relations
                if (r.lower() if self.case_insensitive else r) in self._lookup
            ],
        }

        logger.info(
            f"[INVERSAS SEMANTICAS] {len(inverse_df):,} triplas adicionadas, "
            f"{semantic_count}/{len(unique_relations)} relacoes com nome semantico"
        )

        return ProcessingResult(data=result_df, stats=stats, metadata=metadata)


# ═══════════════════════════════════════════════════════════════════════════
# (E) ENTITY RESOLUTION - Deduplicate similar entities
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class EntityCluster:
    """A cluster of similar entities to be merged."""

    canonical: str
    members: set[str] = field(default_factory=set)
    similarity: float = 0.0


class EntityResolutionStrategy(PreprocessingStrategy):
    """Resolve duplicate entities using blocking + similarity matching.

    This implements a simplified but effective entity resolution pipeline:
    1. Blocking: Group entities by first N characters to reduce comparisons
    2. Similarity: Compute Jaccard similarity on character n-grams
    3. Clustering: Group entities above threshold into clusters
    4. Merging: Replace cluster members with canonical entity

    References:
        - Christophides et al. (2020) "End-to-End Entity Resolution for Big Data"
        - Papadakis et al. (2016) "Comparative Analysis of Approximate Blocking"
    """

    def __init__(
        self,
        min_similarity: float = 0.8,
        blocking_key_length: int = 3,
        ngram_size: int = 3,
        max_cluster_size: int = 100,
        canonical_strategy: str = "shortest",
    ) -> None:
        """Initialize entity resolution strategy.

        Args:
            min_similarity: Minimum Jaccard similarity to consider match (0-1)
            blocking_key_length: First N chars for blocking key
            ngram_size: Character n-gram size for similarity computation
            max_cluster_size: Maximum entities per cluster (avoid mega-clusters)
            canonical_strategy: How to pick canonical entity:
                - "shortest": Shortest string (often cleanest)
                - "longest": Longest string (most descriptive)
                - "most_frequent": Most common in triples
        """
        self.min_similarity = min_similarity
        self.blocking_key_length = blocking_key_length
        self.ngram_size = ngram_size
        self.max_cluster_size = max_cluster_size
        self.canonical_strategy = canonical_strategy

    @property
    def name(self) -> str:
        return "entity_resolution"

    def _get_ngrams(self, s: str) -> set[str]:
        """Get character n-grams from string.

        Args:
            s: Input string

        Returns:
            Set of n-gram strings
        """
        s_lower = s.lower()
        if len(s_lower) < self.ngram_size:
            return {s_lower}
        return {s_lower[i : i + self.ngram_size] for i in range(len(s_lower) - self.ngram_size + 1)}

    def _jaccard_similarity(self, s1: str, s2: str) -> float:
        """Compute Jaccard similarity between two strings using n-grams.

        Args:
            s1: First string
            s2: Second string

        Returns:
            Jaccard similarity (0-1)
        """
        ngrams1 = self._get_ngrams(s1)
        ngrams2 = self._get_ngrams(s2)

        intersection = len(ngrams1 & ngrams2)
        union = len(ngrams1 | ngrams2)

        return intersection / union if union > 0 else 0.0

    def _create_blocks(self, entities: list[str]) -> dict[str, list[str]]:
        """Create blocking groups by first N characters.

        Args:
            entities: List of entity strings

        Returns:
            Dictionary mapping blocking key to entity list
        """
        blocks: dict[str, list[str]] = defaultdict(list)
        for entity in entities:
            key = entity[: self.blocking_key_length].lower()
            blocks[key].append(entity)
        return blocks

    def _select_canonical(
        self,
        members: set[str],
        entity_counts: dict[str, int],
    ) -> str:
        """Select canonical entity from cluster members.

        Args:
            members: Set of cluster member entities
            entity_counts: Count of triple appearances per entity

        Returns:
            Selected canonical entity
        """
        if not members:
            raise ValueError("Cannot select canonical from empty cluster")

        if self.canonical_strategy == "shortest":
            return min(members, key=len)
        elif self.canonical_strategy == "longest":
            return max(members, key=len)
        elif self.canonical_strategy == "most_frequent":
            return max(members, key=lambda e: entity_counts.get(e, 0))
        else:
            return min(members, key=len)

    def _cluster_entities(
        self,
        blocks: dict[str, list[str]],
        entity_counts: dict[str, int],
    ) -> list[EntityCluster]:
        """Cluster similar entities within blocks.

        Args:
            blocks: Blocking groups
            entity_counts: Triple count per entity

        Returns:
            List of entity clusters
        """
        clusters: list[EntityCluster] = []
        processed: set[str] = set()

        for block_entities in blocks.values():
            # Skip singleton blocks
            if len(block_entities) < 2:
                continue

            # Pairwise comparison within block
            for i, e1 in enumerate(block_entities):
                if e1 in processed:
                    continue

                cluster_members = {e1}

                for e2 in block_entities[i + 1 :]:
                    if e2 in processed:
                        continue

                    sim = self._jaccard_similarity(e1, e2)
                    if sim >= self.min_similarity:
                        cluster_members.add(e2)

                        # Limit cluster size
                        if len(cluster_members) >= self.max_cluster_size:
                            break

                if len(cluster_members) > 1:
                    canonical = self._select_canonical(cluster_members, entity_counts)
                    clusters.append(
                        EntityCluster(
                            canonical=canonical,
                            members=cluster_members,
                            similarity=self.min_similarity,
                        )
                    )
                    processed.update(cluster_members)

        return clusters

    def process(self, df: pl.DataFrame) -> ProcessingResult:
        """Resolve duplicate entities in the KG.

        Args:
            df: DataFrame with columns [s, p, o]

        Returns:
            ProcessingResult with resolved entities
        """
        initial_count = len(df)

        # Get all unique entities
        subjects = set(df["s"])
        objects = set(df["o"])
        all_entities = list(subjects | objects)
        initial_entities = len(all_entities)

        # Skip if too few entities
        if initial_entities < 10:
            logger.info("[ENTITY RESOLUTION] Poucos entidades, pulando resolucao")
            return ProcessingResult(
                data=df,
                stats={"initial_entities": initial_entities, "clusters_found": 0},
            )

        subject_counts = df.group_by("s").len().to_dict()
        object_counts = df.group_by("o").len().to_dict()
        entity_counts: dict[str, int] = defaultdict(int)
        for s, c in zip(subject_counts["s"], subject_counts["len"]):
            entity_counts[s] += c
        for o, c in zip(object_counts["o"], object_counts["len"]):
            entity_counts[o] += c

        blocks = self._create_blocks(all_entities)

        clusters = self._cluster_entities(blocks, entity_counts)

        if not clusters:
            logger.info("[ENTITY RESOLUTION] Nenhum cluster encontrado")
            return ProcessingResult(
                data=df,
                stats={
                    "initial_entities": initial_entities,
                    "final_entities": initial_entities,
                    "clusters_found": 0,
                },
            )

        # Build entity mapping (member → canonical)
        entity_mapping: dict[str, str] = {}
        for cluster in clusters:
            for member in cluster.members:
                if member != cluster.canonical:
                    entity_mapping[member] = cluster.canonical

        # Apply mapping to dataframe
        result_df = df.with_columns(
            [
                pl.col("s").replace(entity_mapping).alias("s"),
                pl.col("o").replace(entity_mapping).alias("o"),
            ]
        )

        # Deduplicate after resolution
        result_df = result_df.unique(subset=["s", "p", "o"])

        final_count = len(result_df)

        # Count final entities
        final_subjects = set(result_df["s"].unique().to_list())
        final_objects = set(result_df["o"].unique().to_list())
        final_entities = len(final_subjects | final_objects)

        stats = {
            "initial_triples": initial_count,
            "final_triples": final_count,
            "triples_merged": initial_count - final_count,
            "initial_entities": initial_entities,
            "final_entities": final_entities,
            "entities_merged": initial_entities - final_entities,
            "clusters_found": len(clusters),
            "avg_cluster_size": sum(len(c.members) for c in clusters) / len(clusters),
        }

        metadata = {
            "entity_mapping": entity_mapping,
            "clusters": [
                {
                    "canonical": c.canonical,
                    "members": list(c.members),
                    "similarity": c.similarity,
                }
                for c in clusters[:100]  # Limit for metadata
            ],
        }

        logger.info(
            f"[ENTITY RESOLUTION] {len(clusters):,} clusters, "
            f"{stats['entities_merged']:,} entidades unificadas, "
            f"{stats['triples_merged']:,} triplas mescladas"
        )

        return ProcessingResult(data=result_df, stats=stats, metadata=metadata)


# ═══════════════════════════════════════════════════════════════════════════
# (F) RELATION CARDINALITY CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class CardinalityStats:
    """Statistics for relation cardinality classification."""

    relation: str
    avg_heads_per_tail: float
    avg_tails_per_head: float
    cardinality: str  # "1:1", "1:N", "N:1", "N:N"
    triple_count: int


class RelationCardinalityClassifier(PreprocessingStrategy):
    """Classify relations by their cardinality pattern (1:1, 1:N, N:1, N:N).

    Cardinality patterns are important for KG embedding models:
    - 1:1 relations (e.g., hasBirthDate): One head maps to one tail
    - 1:N relations (e.g., hasChild): One head maps to many tails
    - N:1 relations (e.g., bornIn): Many heads map to one tail
    - N:N relations (e.g., friendOf): Many-to-many mapping

    This information can be used as features or to adjust scoring functions.

    Reference:
        Wang et al. (2014) "Knowledge Graph Embedding by Translating on Hyperplanes"
    """

    def __init__(self, threshold: float = 1.5) -> None:
        """Initialize cardinality classifier.

        Args:
            threshold: Ratio threshold to distinguish 1 from N
                       (e.g., 1.5 means avg > 1.5 is considered "many")
        """
        self.threshold = threshold

    @property
    def name(self) -> str:
        return "relation_cardinality"

    def _classify_cardinality(
        self,
        avg_heads_per_tail: float,
        avg_tails_per_head: float,
    ) -> str:
        """Classify cardinality based on averages.

        Args:
            avg_heads_per_tail: Average number of heads per unique tail
            avg_tails_per_head: Average number of tails per unique head

        Returns:
            Cardinality string: "1:1", "1:N", "N:1", or "N:N"
        """
        many_heads = avg_heads_per_tail > self.threshold
        many_tails = avg_tails_per_head > self.threshold

        if not many_heads and not many_tails:
            return "1:1"
        elif not many_heads and many_tails:
            return "1:N"
        elif many_heads and not many_tails:
            return "N:1"
        else:
            return "N:N"

    def process(self, df: pl.DataFrame) -> ProcessingResult:
        """Classify cardinality for all relations.

        Args:
            df: DataFrame with columns [s, p, o]

        Returns:
            ProcessingResult with cardinality features in metadata
        """
        # Compute heads per tail for each relation
        heads_per_tail = (
            df.group_by(["p", "o"])
            .agg(pl.n_unique("s").alias("n_heads"))
            .group_by("p")
            .agg(pl.mean("n_heads").alias("avg_heads_per_tail"))
        )

        # Compute tails per head for each relation
        tails_per_head = (
            df.group_by(["p", "s"])
            .agg(pl.n_unique("o").alias("n_tails"))
            .group_by("p")
            .agg(pl.mean("n_tails").alias("avg_tails_per_head"))
        )

        relation_counts = df.group_by("p").len().rename({"len": "triple_count"})

        cardinality_df = heads_per_tail.join(tails_per_head, on="p", how="inner").join(
            relation_counts, on="p", how="inner"
        )

        cardinality_stats: list[CardinalityStats] = []
        cardinality_counts: dict[str, int] = {"1:1": 0, "1:N": 0, "N:1": 0, "N:N": 0}

        for row in cardinality_df.iter_rows(named=True):
            cardinality = self._classify_cardinality(
                row["avg_heads_per_tail"],
                row["avg_tails_per_head"],
            )
            cardinality_counts[cardinality] += 1
            cardinality_stats.append(
                CardinalityStats(
                    relation=row["p"],
                    avg_heads_per_tail=row["avg_heads_per_tail"],
                    avg_tails_per_head=row["avg_tails_per_head"],
                    cardinality=cardinality,
                    triple_count=row["triple_count"],
                )
            )

        # Build relation → cardinality mapping
        cardinality_mapping = {s.relation: s.cardinality for s in cardinality_stats}

        stats = {
            "total_relations": len(cardinality_stats),
            "cardinality_distribution": cardinality_counts,
            "threshold": self.threshold,
        }

        metadata = {
            "cardinality_mapping": cardinality_mapping,
            "cardinality_stats": [
                {
                    "relation": s.relation,
                    "avg_heads_per_tail": s.avg_heads_per_tail,
                    "avg_tails_per_head": s.avg_tails_per_head,
                    "cardinality": s.cardinality,
                    "triple_count": s.triple_count,
                }
                for s in sorted(cardinality_stats, key=lambda x: x.triple_count, reverse=True)
            ],
        }

        logger.info(
            f"[CARDINALIDADE] Relacoes classificadas: "
            f"1:1={cardinality_counts['1:1']}, "
            f"1:N={cardinality_counts['1:N']}, "
            f"N:1={cardinality_counts['N:1']}, "
            f"N:N={cardinality_counts['N:N']}"
        )

        return ProcessingResult(data=df, stats=stats, metadata=metadata)


# ═══════════════════════════════════════════════════════════════════════════
# (F) PATH COUNTING - K-hop features for entities
# ═══════════════════════════════════════════════════════════════════════════


class PathCountingStrategy(PreprocessingStrategy):
    """Count k-hop paths for entities as features.

    Path counting provides structural features that capture:
    - Local neighborhood density (1-hop)
    - Extended connectivity (2-hop, 3-hop)
    - Reachability patterns

    Uses sparse matrix multiplication for efficient computation.

    Reference:
        Bordes et al. (2013) "Translating Embeddings for Modeling Multi-relational Data"
    """

    def __init__(self, max_hops: int = 2) -> None:
        """Initialize path counting strategy.

        Args:
            max_hops: Maximum number of hops to count (default: 2)
        """
        self.max_hops = max_hops

    @property
    def name(self) -> str:
        return "path_counting"

    def process(self, df: pl.DataFrame) -> ProcessingResult:
        """Compute k-hop path counts for all entities.

        Args:
            df: DataFrame with columns [s, p, o]

        Returns:
            ProcessingResult with path features in metadata
        """
        # Build entity index
        all_subjects = set(df["s"])
        all_objects = set(df["o"])
        all_entities = sorted(all_subjects | all_objects)
        entity_to_idx = {e: i for i, e in enumerate(all_entities)}
        n_entities = len(all_entities)

        if n_entities == 0:
            logger.info("[PATH COUNTING] Nenhuma entidade, pulando")
            return ProcessingResult(data=df, stats={"n_entities": 0, "max_hops": self.max_hops})

        # Build adjacency matrix efficiently using Polars native mapping
        rows_series = df["s"].replace(entity_to_idx)
        cols_series = df["o"].replace(entity_to_idx)

        rows = rows_series.to_numpy()
        cols = cols_series.to_numpy()
        data = np.ones(len(rows), dtype=np.float32)

        adj = sparse.csr_matrix(
            (data, (rows, cols)), shape=(n_entities, n_entities), dtype=np.float32
        )

        # Make symmetric (undirected paths)
        adj_sym = adj + adj.T
        adj_sym.data[:] = 1

        path_features: dict[str, dict[str, int]] = {e: {} for e in all_entities}
        current_paths = adj_sym.copy()
        for hop in range(1, self.max_hops + 1):
            path_counts = np.array(current_paths.sum(axis=1)).flatten()

            for i, entity in enumerate(all_entities):
                path_features[entity][f"{hop}_hop_paths"] = int(path_counts[i])

            if hop < self.max_hops:
                current_paths = current_paths @ adj_sym

        avg_paths = {
            f"avg_{h}_hop": np.mean([f[f"{h}_hop_paths"] for f in path_features.values()])
            for h in range(1, self.max_hops + 1)
        }

        # Convert to DataFrame
        path_df = pl.DataFrame([{"entity": e, **counts} for e, counts in path_features.items()])

        stats = {
            "n_entities": n_entities,
            "max_hops": self.max_hops,
            **avg_paths,
        }

        metadata = {
            "path_features": path_df,
            "entity_to_idx": entity_to_idx,
        }

        logger.info(
            f"[PATH COUNTING] {n_entities:,} entidades, "
            f"1-hop avg={avg_paths.get('avg_1_hop', 0):.1f}, "
            f"2-hop avg={avg_paths.get('avg_2_hop', 0):.1f}"
        )

        return ProcessingResult(data=df, stats=stats, metadata=metadata)


# ═══════════════════════════════════════════════════════════════════════════
# (F) TEXTUALIZATION - BERT-ready text generation
# ═══════════════════════════════════════════════════════════════════════════


class TextualizationStrategy(PreprocessingStrategy):
    """Generate natural language text from KG triples for BERT.

    Converts triples to readable sentences that BERT can encode,
    enabling the text encoder component of DSLFM to learn semantic
    representations.

    Examples:
        - (John, worksIn, Google) → "John works in Google"
        - (Alice, hasChild, Bob) → "Alice has child Bob"

    Reference:
        Yao et al. (2019) "KG-BERT: BERT for Knowledge Graph Completion"
    """

    # Default templates for common relation patterns
    DEFAULT_TEMPLATES: dict[str, str] = {
        # Work/Organization
        "worksIn": "{head} works in {tail}",
        "worksFor": "{head} works for {tail}",
        "employs": "{head} employs {tail}",
        "memberOf": "{head} is a member of {tail}",
        # Family
        "hasChild": "{head} has child {tail}",
        "hasParent": "{head} has parent {tail}",
        "marriedTo": "{head} is married to {tail}",
        # Location
        "locatedIn": "{head} is located in {tail}",
        "bornIn": "{head} was born in {tail}",
        "livesIn": "{head} lives in {tail}",
        # Type/Category
        "hasType": "{head} has type {tail}",
        "instanceOf": "{head} is an instance of {tail}",
        "subclassOf": "{head} is a subclass of {tail}",
        # Telecom
        "subscribedTo": "{head} is subscribed to {tail}",
        "usesService": "{head} uses service {tail}",
        "hasProvider": "{head} has provider {tail}",
        "connectedTo": "{head} is connected to {tail}",
        # Generic
        "relatedTo": "{head} is related to {tail}",
        "hasValue": "{head} has value {tail}",
    }

    def __init__(
        self,
        templates: dict[str, str] | None = None,
        default_template: str = "{head} {relation} {tail}",
        humanize_relation: bool = True,
    ) -> None:
        """Initialize textualization strategy.

        Args:
            templates: Custom relation → template mappings
            default_template: Template for unmapped relations
            humanize_relation: Convert camelCase to "camel case"
        """
        self.templates = {**self.DEFAULT_TEMPLATES}
        if templates:
            self.templates.update(templates)
        self.default_template = default_template
        self.humanize_relation = humanize_relation

    @property
    def name(self) -> str:
        return "textualization"

    def _humanize(self, s: str) -> str:
        """Convert camelCase to human readable.

        Args:
            s: Input string (e.g., "hasChild")

        Returns:
            Humanized string (e.g., "has child")
        """
        import re

        # Insert space before capitals
        result = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)
        # Handle consecutive capitals
        result = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", result)
        return result.lower()

    def _textualize(self, head: str, relation: str, tail: str) -> str:
        """Convert triple to natural language.

        Args:
            head: Head entity
            relation: Relation name
            tail: Tail entity

        Returns:
            Natural language sentence
        """
        template = self.templates.get(relation)
        if template:
            return template.format(head=head, tail=tail)

        # Fallback: humanize relation
        if self.humanize_relation:
            humanized_rel = self._humanize(relation)
        else:
            humanized_rel = relation

        return self.default_template.format(head=head, relation=humanized_rel, tail=tail)

    def process(self, df: pl.DataFrame) -> ProcessingResult:
        """Generate text representations for all triples.

        Args:
            df: DataFrame with columns [s, p, o]

        Returns:
            ProcessingResult with text column added
        """
        # Generate text for each triple
        texts = [self._textualize(row["s"], row["p"], row["o"]) for row in df.iter_rows(named=True)]

        # Add text column
        result_df = df.with_columns(pl.Series("text", texts))

        # Count template usage
        unique_relations = df["p"].unique().to_list()
        template_used = sum(1 for r in unique_relations if r in self.templates)

        stats = {
            "total_triples": len(df),
            "unique_relations": len(unique_relations),
            "template_coverage": (template_used / len(unique_relations) if unique_relations else 0),
            "avg_text_length": sum(len(t) for t in texts) / len(texts) if texts else 0,
        }

        metadata = {
            "templates_used": [r for r in unique_relations if r in self.templates],
            "unmapped_relations": [r for r in unique_relations if r not in self.templates],
        }

        logger.info(
            f"[TEXTUALIZACAO] {len(df):,} triplas convertidas, "
            f"cobertura templates={stats['template_coverage']:.1%}"
        )

        return ProcessingResult(data=result_df, stats=stats, metadata=metadata)


# ═══════════════════════════════════════════════════════════════════════════
# EXPORTS
# ═══════════════════════════════════════════════════════════════════════════

__all__ = [
    "HubDownsamplingStrategy",
    "SemanticInverseStrategy",
    "EntityResolutionStrategy",
    "RelationCardinalityClassifier",
    "PathCountingStrategy",
    "TextualizationStrategy",
    "EntityCluster",
    "CardinalityStats",
]
