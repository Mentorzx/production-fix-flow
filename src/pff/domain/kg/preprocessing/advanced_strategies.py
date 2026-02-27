"""
Advanced Preprocessing Strategies for KG Data (SOTA Implementation).
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import polars as pl
from scipy import sparse

from pff.shared import logger
from pff_rust import (
    sorted_jaccard_similarity,
    string_to_ngram_hashes,
)

from .strategies import PreprocessingStrategy, ProcessingResult


class HubDownsamplingStrategy(PreprocessingStrategy):
    """Downsample edges from hub nodes to reduce their dominance."""

    def __init__(
        self,
        percentile: float = 0.99,
        max_edges_per_hub: int | None = None,
        sampling_factor: float = 0.1,
        seed: int = 42,
    ) -> None:
        """Execute init.



        Args:

            percentile: Optional input value.

            max_edges_per_hub: Optional input value.

            sampling_factor: Optional input value.

            seed: Optional input value.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        self.percentile = percentile
        self.max_edges_per_hub = max_edges_per_hub
        self.sampling_factor = sampling_factor
        self.seed = seed

    @property
    def name(self) -> str:
        """Execute name.



        Returns:

            Return value produced by the callable.

        """

        return "hub_downsampling"

    def process(self, df: pl.DataFrame) -> ProcessingResult:
        """Execute process.



        Args:

            df: Input value used by this callable.



        Returns:

            Return value produced by the callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        initial_count = len(df)
        if initial_count == 0:
            return ProcessingResult(data=df, stats={"initial_triples": 0})

        degrees = (
            pl.concat(
                [df.select(pl.col("s").alias("e")), df.select(pl.col("o").alias("e"))]
            )
            .group_by("e")
            .len()
            .rename({"len": "degree"})
        )

        _q = degrees["degree"].quantile(self.percentile)
        threshold = float(_q) if _q is not None else 0.0  # type: ignore
        hubs = degrees.filter(pl.col("degree") >= threshold)

        if len(hubs) == 0:
            return ProcessingResult(
                data=df, stats={"initial_triples": initial_count, "n_hubs": 0}
            )

        limit = self.max_edges_per_hub or int((degrees["degree"].median() or 0.0) * 2)  # type: ignore

        df_with_hubs = df.join(
            hubs.select(pl.col("e").alias("s"), pl.lit(True).alias("_h_s")),
            on="s",
            how="left",
        ).join(
            hubs.select(pl.col("e").alias("o"), pl.lit(True).alias("_h_o")),
            on="o",
            how="left",
        )

        df_with_hubs = df_with_hubs.with_columns(
            [
                pl.col("_h_s").fill_null(False),
                pl.col("_h_o").fill_null(False),
            ]
        )

        normal = df_with_hubs.filter(~pl.col("_h_s") & ~pl.col("_h_o")).drop(
            ["_h_s", "_h_o"]
        )
        hub_triples = df_with_hubs.filter(pl.col("_h_s") | pl.col("_h_o")).drop(
            ["_h_s", "_h_o"]
        )

        sampled_hub = (
            hub_triples.with_columns(
                pl.int_range(0, pl.len()).shuffle(seed=self.seed).alias("_r")
            )
            .with_columns(
                [
                    pl.col("_r").rank().over("s").alias("_rs"),
                    pl.col("_r").rank().over("o").alias("_ro"),
                ]
            )
            .filter((pl.col("_rs") <= limit) & (pl.col("_ro") <= limit))
            .drop(["_r", "_rs", "_ro"])
        )

        result_df = pl.concat([normal, sampled_hub])

        stats = {
            "initial_triples": initial_count,
            "final_triples": len(result_df),
            "triples_removed": initial_count - len(result_df),
            "n_hubs": len(hubs),
            "hub_threshold": threshold,
        }
        logger.info(f"[HUB DOWNSAMPLING] {len(hubs)} hubs processados")
        return ProcessingResult(data=result_df, stats=stats)


class SemanticInverseStrategy(PreprocessingStrategy):
    """Add inverse relations with semantic naming."""

    DEFAULT_SEMANTIC_INVERSES: dict[str, str] = {
        "worksIn": "employs",
        "worksFor": "employs",
        "memberOf": "hasMember",
        "hasParent": "hasChild",
        "parentOf": "childOf",
        "partOf": "hasPart",
    }

    def __init__(
        self,
        semantic_mappings: dict | None = None,
        fallback_suffix: str = "_inv",
        case_insensitive: bool = True,
    ):
        """Execute init.



        Args:

            semantic_mappings: Optional input value.

            fallback_suffix: Optional input value.

            case_insensitive: Optional input value.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        self.mappings = {**self.DEFAULT_SEMANTIC_INVERSES, **(semantic_mappings or {})}
        self.fallback_suffix = fallback_suffix
        self.case_insensitive = case_insensitive
        self._lookup = (
            {k.lower(): v for k, v in self.mappings.items()}
            if case_insensitive
            else self.mappings
        )

    @property
    def name(self) -> str:
        """Execute name.



        Returns:

            Return value produced by the callable.

        """

        return "semantic_inverse"

    def process(self, df: pl.DataFrame) -> ProcessingResult:
        """Execute process.



        Args:

            df: Input value used by this callable.



        Returns:

            Return value produced by the callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        unique_rels = df["p"].unique().to_list()
        inv_map = {
            r: self._lookup.get(
                r.lower() if self.case_insensitive else r, f"{r}{self.fallback_suffix}"
            )
            for r in unique_rels
        }

        inv_df = df.select(
            [
                pl.col("o").alias("s"),
                pl.col("p").replace(inv_map).alias("p"),
                pl.col("s").alias("o"),
            ]
        )

        result_df = pl.concat([df, inv_df])
        stats = {"initial_triples": len(df), "final_triples": len(result_df)}
        metadata = {"inverse_mapping": inv_map}
        return ProcessingResult(data=result_df, stats=stats, metadata=metadata)


@dataclass
class EntityCluster:
    """Represent EntityCluster."""

    canonical: str
    members: set[str] = field(default_factory=set)
    similarity: float = 0.0


class EntityResolutionStrategy(PreprocessingStrategy):
    """Resolve duplicate entities using Rust-accelerated similarity."""

    def __init__(
        self,
        min_similarity: float = 0.8,
        blocking_key_length: int = 3,
        ngram_size: int = 3,
        max_cluster_size: int = 100,
        canonical_strategy: str = "shortest",
    ):
        """Execute init.



        Args:

            min_similarity: Optional input value.

            blocking_key_length: Optional input value.

            ngram_size: Optional input value.

            max_cluster_size: Optional input value.

            canonical_strategy: Optional input value.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        self.min_similarity = min_similarity
        self.blocking_key_length = blocking_key_length
        self.ngram_size = ngram_size
        self.max_cluster_size = max_cluster_size
        self.canonical_strategy = canonical_strategy

    @property
    def name(self) -> str:
        """Execute name.



        Returns:

            Return value produced by the callable.

        """

        return "entity_resolution"

    def _create_blocks(self, entities: list[str]) -> dict[str, list[str]]:
        """Execute create blocks.



        Args:

            entities: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        blocks = defaultdict(list)
        for e in entities:
            blocks[str(e)[: self.blocking_key_length].lower()].append(e)
        return blocks

    def _select_canonical(self, members: set[str], counts: dict[str, int]) -> str:
        """Execute select canonical.



        Args:

            members: Input value used by this callable.

            counts: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        if self.canonical_strategy == "shortest":
            return min(members, key=len)
        if self.canonical_strategy == "longest":
            return max(members, key=len)
        if self.canonical_strategy == "most_frequent":
            return max(members, key=lambda e: counts.get(e, 0))
        return min(members, key=len)

    def _cluster_entities(
        self, blocks: dict[str, list[str]], counts: dict[str, int]
    ) -> list[EntityCluster]:
        """Execute cluster entities.



        Args:

            blocks: Input value used by this callable.

            counts: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        clusters = []
        processed = set()
        for block_entities in blocks.values():
            if len(block_entities) < 2:
                continue
            hashes = [
                string_to_ngram_hashes(str(e), self.ngram_size) for e in block_entities
            ]
            hash_sizes = [len(h) for h in hashes]
            for i, e1 in enumerate(block_entities):
                if e1 in processed:
                    continue
                members = {e1}
                size_i = hash_sizes[i]
                for j in range(i + 1, len(block_entities)):
                    e2 = block_entities[j]
                    size_j = hash_sizes[j]
                    min_size = min(size_i, size_j)
                    max_size = max(size_i, size_j)
                    # Safe upper bound for Jaccard; skip pairs that can never reach threshold.
                    if max_size == 0 or (min_size / max_size) < self.min_similarity:
                        continue
                    if (
                        e2 not in processed
                        and sorted_jaccard_similarity(hashes[i], hashes[j])
                        >= self.min_similarity
                    ):
                        members.add(e2)
                        if len(members) >= self.max_cluster_size:
                            break
                if len(members) > 1:
                    clusters.append(
                        EntityCluster(
                            self._select_canonical(members, counts),
                            members,
                            self.min_similarity,
                        )
                    )
                    processed.update(members)
        return clusters

    def process(self, df: pl.DataFrame) -> ProcessingResult:
        """Execute process.



        Args:

            df: Input value used by this callable.



        Returns:

            Return value produced by the callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        ents = (
            pl.concat([df.select(pl.col("s")), df.select(pl.col("o").alias("s"))])
            .unique()["s"]
            .to_list()
        )
        if len(ents) < 10:
            return ProcessingResult(data=df, stats={"clusters_found": 0})

        counts_df = (
            pl.concat(
                [df.select(pl.col("s").alias("e")), df.select(pl.col("o").alias("e"))]
            )
            .group_by("e")
            .len()
        )
        counts = dict(zip(counts_df["e"], counts_df["len"]))

        clusters = self._cluster_entities(self._create_blocks(ents), counts)
        if not clusters:
            return ProcessingResult(data=df, stats={"clusters_found": 0})

        mapping = {
            m: c.canonical for c in clusters for m in c.members if m != c.canonical
        }
        result = df.with_columns(
            [pl.col("s").replace(mapping), pl.col("o").replace(mapping)]
        ).unique()

        return ProcessingResult(
            data=result,
            stats={"clusters_found": len(clusters)},
            metadata={"clusters": clusters, "entity_mapping": mapping},
        )


class RelationCardinalityClassifier(PreprocessingStrategy):
    """Classify relations by cardinality pattern."""

    def __init__(self, threshold: float = 1.5):
        """Execute init.



        Args:

            threshold: Optional input value.

        """

        self.threshold = threshold

    @property
    def name(self) -> str:
        """Execute name.



        Returns:

            Return value produced by the callable.

        """

        return "relation_cardinality"

    def process(self, df: pl.DataFrame) -> ProcessingResult:
        """Execute process.



        Args:

            df: Input value used by this callable.



        Returns:

            Return value produced by the callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        hpt = (
            df.group_by(["p", "o"])
            .agg(pl.n_unique("s").alias("n"))
            .group_by("p")
            .agg(pl.mean("n").alias("ahpt"))
        )
        tph = (
            df.group_by(["p", "s"])
            .agg(pl.n_unique("o").alias("n"))
            .group_by("p")
            .agg(pl.mean("n").alias("atph"))
        )

        card_df = hpt.join(tph, on="p")
        dist = {"1:1": 0, "1:N": 0, "N:1": 0, "N:N": 0}
        mapping = {}

        for r in card_df.iter_rows(named=True):
            mh, mt = r["ahpt"] > self.threshold, r["atph"] > self.threshold
            c = (
                "1:1"
                if not mh and not mt
                else "1:N" if not mh else "N:1" if not mt else "N:N"
            )
            dist[c] += 1
            mapping[r["p"]] = c

        return ProcessingResult(
            data=df,
            stats={"cardinality_distribution": dist},
            metadata={"cardinality_mapping": mapping},
        )


class PathCountingStrategy(PreprocessingStrategy):
    """Count k-hop paths for entities."""

    def __init__(self, max_hops: int = 2):
        """Execute init.



        Args:

            max_hops: Optional input value.

        """

        self.max_hops = max_hops

    @property
    def name(self) -> str:
        """Execute name.



        Returns:

            Return value produced by the callable.

        """

        return "path_counting"

    def process(self, df: pl.DataFrame) -> ProcessingResult:
        """Execute process.



        Args:

            df: Input value used by this callable.



        Returns:

            Return value produced by the callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        ents = (
            pl.concat([df.select(pl.col("s")), df.select(pl.col("o").alias("s"))])
            .unique()
            .sort("s")["s"]
            .to_list()
        )
        e2i = {e: i for i, e in enumerate(ents)}
        n = len(ents)
        if n == 0:
            return ProcessingResult(data=df, stats={})

        r, c = df["s"].replace(e2i).to_numpy(), df["o"].replace(e2i).to_numpy()
        adj = sparse.csr_matrix(
            (np.ones(len(r)), (r, c)), shape=(n, n), dtype=np.float32
        )
        adj_sym = (adj + adj.T).tocsr()
        adj_sym.data[:] = 1

        path_data = {"entity": ents}
        curr = adj_sym.copy()
        for h in range(1, self.max_hops + 1):
            path_data[f"{h}_hop_paths"] = list(  # type: ignore
                np.array(curr.sum(axis=1)).flatten().astype(int)
            )
            if h < self.max_hops:
                curr = curr @ adj_sym

        path_df = pl.DataFrame(path_data)
        stats = {
            f"avg_{h}_hop": path_df[f"{h}_hop_paths"].mean()
            for h in range(1, self.max_hops + 1)
        }
        stats["n_entities"] = n
        return ProcessingResult(
            data=df, stats=stats, metadata={"path_features": path_df}
        )


class TextualizationStrategy(PreprocessingStrategy):
    """Generate BERT-ready text from triples."""

    DEFAULT_TEMPLATES = {
        "worksIn": "{head} works in {tail}",
        "hasChild": "{head} has child {tail}",
    }

    def __init__(
        self,
        templates: dict | None = None,
        default_template: str = "{head} {relation} {tail}",
        humanize_relation: bool = True,
    ):
        """Execute init.



        Args:

            templates: Optional input value.

            default_template: Optional input value.

            humanize_relation: Optional input value.

        """

        self.templates = {**self.DEFAULT_TEMPLATES, **(templates or {})}  # type: ignore[assignment]
        self.default_template = default_template
        self.humanize_relation = humanize_relation

    @property
    def name(self) -> str:
        """Execute name.



        Returns:

            Return value produced by the callable.

        """

        return "textualization"

    def _humanize(self, s: str) -> str:
        res = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(s))
        return re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", res).lower()

    def process(self, df: pl.DataFrame) -> ProcessingResult:
        """Execute process.



        Args:

            df: Input value used by this callable.



        Returns:

            Return value produced by the callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        if len(df) == 0:
            return ProcessingResult(data=df, stats={})
        unique_rels = df["p"].unique().to_list()
        h_rels = (
            {r: self._humanize(r) for r in unique_rels}
            if self.humanize_relation
            else {r: str(r) for r in unique_rels}
        )

        temp_df = df.with_columns(pl.col("p").replace(h_rels).alias("_rt"))
        f_fmt = (
            self.default_template.replace("{head}", "{}")
            .replace("{relation}", "{}")
            .replace("{tail}", "{}")
        )
        expr = pl.format(f_fmt, pl.col("s"), pl.col("_rt"), pl.col("o"))

        for r, t in self.templates.items():
            if r not in unique_rels:
                continue
            fmt = (
                t.replace("{head}", "{}")
                .replace("{tail}", "{}")
                .replace("{relation}", "{}")
            )
            args = [
                pl.col("s") if "{head}" in t else None,
                pl.col("o") if "{tail}" in t else None,
            ]
            expr = (
                pl.when(pl.col("p") == r)
                .then(pl.format(fmt, *[a for a in args if a is not None]))
                .otherwise(expr)
            )

        result = temp_df.with_columns(expr.alias("text")).drop("_rt")
        stats = {
            "total_triples": len(df),
            "template_coverage": sum(1 for r in unique_rels if r in self.templates)
            / len(unique_rels),
        }
        metadata = {
            "templates_used": list(self.templates.keys()),
            "unmapped_relations": [r for r in unique_rels if r not in self.templates],
        }
        return ProcessingResult(data=result, stats=stats, metadata=metadata)


__all__ = [
    "HubDownsamplingStrategy",
    "SemanticInverseStrategy",
    "EntityResolutionStrategy",
    "RelationCardinalityClassifier",
    "PathCountingStrategy",
    "TextualizationStrategy",
    "EntityCluster",
]
